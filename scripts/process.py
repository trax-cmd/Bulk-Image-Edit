#!/usr/bin/env python3
"""Remove TraxNYC branding from product photos and upscale to 1200x1200.

Usage:
  python3 scripts/process.py <input_dir> <output_dir> [--size 1200] [--format jpg|png]
         [--logo assets/trax_logo_template.png] [--ruler-text assets/trax_ruler_text_template.png]

Pipeline per image:
  1. Find every "TraxNYC" word printed on the ruler (ruler shots only) by
     template matching and fill it with the ruler's own background. The image
     is padded with white first so a word cut off at the frame edge still
     matches.
  2. Find the small grey corner stamp by masked template matching. Inpaint
     only the stamp's own stroke pixels, so anything underneath is kept.
  3. Find the large coloured eye-and-lettering stamp (any size, either bottom
     corner) by multi-scale masked matching and inpaint it. The share of its
     border that touched non-background pixels is reported as `overlap`, so a
     stamp that sat on the product can be reviewed by a human.
  4. Upscale with Lanczos + light unsharp mask, pad to a square with the
     photo's own background colour, save.
Nothing is ever drawn on top of the image.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upscale import EXTS, border_color  # noqa: E402

from PIL import ImageFilter, ImageOps  # noqa: E402


def to_rgb(img: Image.Image) -> Image.Image:
    """Flatten any transparency onto white; return an RGB image."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def load_template(path: Path):
    t = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if t is None:
        raise SystemExit(f"cannot read template {path}")
    return t


def match_all(gray, tpl, thresh, mask=None, prior=None, prior_thresh=0.15):
    """Return list of (x, y, score) for every non-overlapping match >= thresh.

    mask: optional template mask so only the logo's own strokes are compared
    (lets a stamp over a busy background still match).
    prior: optional (dx, dy) offset of the stamp from the bottom-right corner;
    if nothing clears `thresh`, the stamp is accepted at that spot when its
    score there clears `prior_thresh`.
    """
    th, tw = tpl.shape
    if mask is not None:
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    else:
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    hits = []
    res = res.copy()
    while True:
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < thresh:
            break
        x, y = loc
        hits.append((x, y, float(mx)))
        # suppress neighbourhood
        x0, y0 = max(0, x - tw // 2), max(0, y - th // 2)
        res[y0:y + th // 2 + 1, x0:x + tw // 2 + 1] = -1
    if not hits and prior is not None:
        px, py = gray.shape[1] - prior[0], gray.shape[0] - prior[1]
        if 0 <= py < res.shape[0] and 0 <= px < res.shape[1]:
            sc = float(res[py, px])
            if sc >= prior_thresh:
                hits.append((px, py, sc))
    return hits


# Where the stamp sits, as an offset of its top-left corner from the
# bottom-right corner of the photo (measured on the 500x380 catalogue shots).
LOGO_PRIOR = (100, 79)


LOGOB_PRIOR = (204, 160)          # top-left of the coloured stamp, from the bottom-right corner
LOGOB_SCALES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3)


def _masked_match(gray, tpl, mask):
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _maroon_fraction(bgr, red_mask_tpl, x, y, scale):
    """Share of the stamp's maroon eye pixels that really are maroon in the image.

    Maroon: red well above green, green and blue similar, not bright. Gold and
    skin tones fail the green-blue balance, white and grey fail the red lift.
    """
    th, tw = max(1, round(red_mask_tpl.shape[0] * scale)), max(1, round(red_mask_tpl.shape[1] * scale))
    m = cv2.resize(red_mask_tpl, (tw, th), interpolation=cv2.INTER_NEAREST) > 0
    h, w = min(th, bgr.shape[0] - y), min(tw, bgr.shape[1] - x)
    if h <= 0 or w <= 0:
        return 0.0
    win = bgr[y:y + h, x:x + w].astype(int)
    m = m[:h, :w]
    if m.sum() < 10:
        return 0.0
    b, g, r = win[..., 0][m], win[..., 1][m], win[..., 2][m]
    ok = (r - g > 35) & (np.abs(g - b) < 30) & (r < 235)
    return float(ok.mean())


def _top_k(res, k, th, tw):
    out = []
    res = res.copy()
    for _ in range(k):
        _, mx, _, (x, y) = cv2.minMaxLoc(res)
        if mx <= -1:
            break
        out.append((x, y, float(mx)))
        res[max(0, y - th // 2):y + th // 2 + 1, max(0, x - tw // 2):x + tw // 2 + 1] = -1
    return out


def find_logo_b(bgr, tpl_bgr, thresh=0.38, color_thresh=0.5, prior_thresh=0.25, topk=3):
    """Locate the large coloured TraxNYC stamp at any scale, anywhere in the lower
    60% of the frame, including stamps cut off by the frame edge.

    A candidate counts only if its matching score clears `thresh` AND the eye
    region is actually maroon in the image (`color_thresh`), which rejects
    look-alike patterns on gold pieces. Returns (x, y, scale, score) or None.
    """
    H, W = bgr.shape[:2]
    tgray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    core = (tgray < 235).astype(np.uint8) * 255          # eye + lettering, no soft shadow
    tb, tg_, tr = cv2.split(tpl_bgr.astype(int))
    red_mask = ((tr - np.maximum(tg_, tb)) > 40).astype(np.uint8)
    pad = int(tgray.shape[1] * 0.6)
    y_off = max(0, int(H * 0.4) - pad)
    padded = cv2.copyMakeBorder(bgr[y_off:], 0, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    best = None
    for sc in LOGOB_SCALES:
        th, tw = round(tgray.shape[0] * sc), round(tgray.shape[1] * sc)
        if th >= gray.shape[0] or tw >= gray.shape[1] or th < 8:
            continue
        t = cv2.resize(tgray, (tw, th), interpolation=cv2.INTER_AREA)
        m = cv2.resize(core, (tw, th), interpolation=cv2.INTER_NEAREST)
        if m.sum() == 0:
            continue
        res = _masked_match(gray, t, m)
        for x, y, score in _top_k(res, topk, th, tw):
            if score < thresh:
                continue
            if _maroon_fraction(padded, red_mask, x, y, sc) < color_thresh:
                continue
            if best is None or score > best[3]:
                best = (x - pad, y + y_off, sc, score)
    if best is None:
        # usual bottom-right placement at full size, lower bar but colour still required
        px, py = W - LOGOB_PRIOR[0], H - LOGOB_PRIOR[1]
        if px >= 0 and py >= 0:
            th, tw = tgray.shape
            r = 8
            x0, y0 = max(0, px - r), max(0, py - r)
            win = cv2.cvtColor(bgr[y0:py + th + r, x0:px + tw + r], cv2.COLOR_BGR2GRAY)
            if win.shape[0] >= th and win.shape[1] >= tw:
                res = _masked_match(win, tgray, core)
                _, mx, _, (wx, wy) = cv2.minMaxLoc(res)
                if mx >= prior_thresh and _maroon_fraction(bgr, red_mask, x0 + wx, y0 + wy, 1.0) >= color_thresh:
                    best = (x0 + wx, y0 + wy, 1.0, float(mx))
    return best


def stamp_mask(shape, tpl_bgr, x, y, scale, white_thresh=248, dilate=3):
    """Binary mask of where the coloured stamp (incl. its soft shadow) sits."""
    th, tw = round(tpl_bgr.shape[0] * scale), round(tpl_bgr.shape[1] * scale)
    t = cv2.resize(tpl_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    m = (t.min(axis=2) < white_thresh).astype(np.uint8) * 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    mask = np.zeros(shape[:2], np.uint8)
    sx, sy = max(0, -x), max(0, -y)                 # part of the stamp outside the frame
    x, y = max(0, x), max(0, y)
    h, w = min(th - sy, shape[0] - y), min(tw - sx, shape[1] - x)
    if h > 0 and w > 0:
        mask[y:y + h, x:x + w] = m[sy:sy + h, sx:sx + w]
    return mask


def over_product(bgr, mask, ring=4, dark=235):
    """Fraction of pixels in a thin ring just outside `mask` that are not background."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    outer = cv2.dilate(mask, k)
    ringpx = (outer > 0) & (mask == 0)
    if ringpx.sum() == 0:
        return 0.0
    return float(((bgr.min(axis=2) < dark) & ringpx).sum() / ringpx.sum())


def find_logo(bgr, logo_tpl, thresh=0.45, stroke_thresh=250, prior_thresh=0.25):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stroke = (logo_tpl < stroke_thresh).astype(np.uint8) * 255
    # The grey stamp only appears on the 500px-wide catalogue frames; the
    # position fallback is restricted to those so it cannot fire elsewhere.
    prior = LOGO_PRIOR if 495 <= gray.shape[1] <= 505 else None
    return match_all(gray, logo_tpl, thresh, mask=stroke, prior=prior, prior_thresh=prior_thresh)


def remove_logo(bgr, logo_tpl, hits, stroke_thresh=250, dilate=1):
    if not hits:
        return bgr, None
    th, tw = logo_tpl.shape
    mask = np.zeros(bgr.shape[:2], np.uint8)
    stroke = (logo_tpl < stroke_thresh).astype(np.uint8) * 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        stroke = cv2.dilate(stroke, k)
    for x, y, _ in hits:
        h = min(th, mask.shape[0] - y)
        w = min(tw, mask.shape[1] - x)
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], stroke[:h, :w])
    return cv2.inpaint(bgr, mask, 2, cv2.INPAINT_TELEA), mask


def remove_ruler_text(bgr, text_tpl, thresh=0.45, pad=80, margin=2):
    th, tw = text_tpl.shape
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    hits = match_all(gray, text_tpl, thresh)
    if not hits:
        return bgr, []
    out = padded.copy()
    for x, y, _ in hits:
        y0, y1 = max(0, y - margin), y + th + margin
        x0, x1 = max(0, x - margin), x + tw + margin
        box = out[y0:y1, x0:x1]
        light = box[box.min(axis=2) > 235]
        fill = np.median(light, axis=0) if len(light) else np.array([255, 255, 255])
        out[y0:y1, x0:x1] = fill.astype(np.uint8)
    out = out[pad:-pad, pad:-pad]
    return out, [(x - pad, y - pad, s) for x, y, s in hits]


def upscale(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    pad = border_color(img)
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    if scale > 1:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    canvas = Image.new("RGB", (size, size), pad)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def process_image(path: Path, logo_tpl, text_tpl, size: int, no_upscale: bool = False, logo_b_tpl=None):
    """Run the whole pipeline on one file.

    Returns (original, final, info) where info has: logo_hits, text_hits,
    logo_b (x, y, scale, score) or None, overlap (0..1 fraction of the removed
    stamp's border that touched non-background pixels).
    """
    pil = to_rgb(ImageOps.exif_transpose(Image.open(path)))
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    logo_hits = find_logo(bgr, logo_tpl)                     # detect on the untouched image
    logo_b = find_logo_b(bgr, logo_b_tpl) if logo_b_tpl is not None else None
    bgr, text_hits = remove_ruler_text(bgr, text_tpl)
    overlap = 0.0
    bgr, mask_a = remove_logo(bgr, logo_tpl, logo_hits)
    if mask_a is not None:
        overlap = max(overlap, over_product(bgr, mask_a))
    if logo_b is not None:
        x, y, sc, _ = logo_b
        mask_b = stamp_mask(bgr.shape, logo_b_tpl, x, y, sc)
        overlap = max(overlap, over_product(bgr, mask_b))
        bgr = cv2.inpaint(bgr, mask_b, 3, cv2.INPAINT_TELEA)
    clean = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    final = clean if no_upscale else upscale(clean, size)
    info = {"logo_hits": logo_hits, "text_hits": text_hits, "logo_b": logo_b, "overlap": round(overlap, 3)}
    return pil, final, info


def save_image(img: Image.Image, dst: Path, fmt: str, quality: int = 92) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        img.save(dst, "PNG", optimize=True)
    else:
        img.save(dst, "JPEG", quality=quality, optimize=True, progressive=True, subsampling=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=92)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--logo", default=str(root / "assets/trax_logo_template.png"))
    ap.add_argument("--ruler-text", default=str(root / "assets/trax_ruler_text_template.png"))
    ap.add_argument("--logo-b", default=str(root / "assets/trax_logoB_template.png"))
    ap.add_argument("--no-upscale", action="store_true", help="only strip branding, keep size")
    a = ap.parse_args()

    logo_tpl = load_template(Path(a.logo))
    text_tpl = load_template(Path(a.ruler_text))
    logo_b_tpl = cv2.imread(a.logo_b, cv2.IMREAD_COLOR)
    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images found under {in_dir}", file=sys.stderr)
        return 1

    for p in files:
        pil, final, info = process_image(p, logo_tpl, text_tpl, a.size, a.no_upscale, logo_b_tpl)
        logo_hits, text_hits = info["logo_hits"], info["text_hits"]

        rel = p.relative_to(in_dir).with_suffix("." + a.format)
        save_image(final, out_dir / rel, a.format, a.quality)
        lg = ", ".join(f"({x},{y}) {s:.2f}" for x, y, s in logo_hits) or "none"
        tx = ", ".join(f"({x},{y}) {s:.2f}" for x, y, s in text_hits) or "none"
        lb = info["logo_b"]
        lb = f"({lb[0]},{lb[1]}) x{lb[2]:.2f} {lb[3]:.2f}" if lb else "none"
        print(f"{rel}: {pil.size[0]}x{pil.size[1]} -> {final.size[0]}x{final.size[1]} | stamp A: {lg} | stamp B: {lb} | ruler text: {tx} | overlap {info['overlap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
