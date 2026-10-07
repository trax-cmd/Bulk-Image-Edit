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
  2. Find the corner TraxNYC eye logo by masked template matching. Inpaint
     only the logo's own stroke pixels, so anything underneath is kept.
  3. Upscale with Lanczos + light unsharp mask, pad to a square with the
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


def find_logo(bgr, logo_tpl, thresh=0.35, stroke_thresh=250):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stroke = (logo_tpl < stroke_thresh).astype(np.uint8) * 255
    return match_all(gray, logo_tpl, thresh, mask=stroke, prior=LOGO_PRIOR)


def remove_logo(bgr, logo_tpl, hits, stroke_thresh=250, dilate=1):
    if not hits:
        return bgr
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
    return cv2.inpaint(bgr, mask, 2, cv2.INPAINT_TELEA)


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
    ap.add_argument("--no-upscale", action="store_true", help="only strip branding, keep size")
    a = ap.parse_args()

    logo_tpl = load_template(Path(a.logo))
    text_tpl = load_template(Path(a.ruler_text))
    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images found under {in_dir}", file=sys.stderr)
        return 1

    for p in files:
        pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        # Ruler text first: the corner stamp can overlap the last word, and the
        # stamp inpaints cleanly only once the text under it is gone.
        logo_hits = find_logo(bgr, logo_tpl)       # detect on the untouched image
        bgr, text_hits = remove_ruler_text(bgr, text_tpl)
        bgr = remove_logo(bgr, logo_tpl, logo_hits)
        clean = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        final = clean if a.no_upscale else upscale(clean, a.size)

        rel = p.relative_to(in_dir).with_suffix("." + a.format)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a.format == "png":
            final.save(dst, "PNG", optimize=True)
        else:
            final.save(dst, "JPEG", quality=a.quality, optimize=True, progressive=True, subsampling=0)
        lg = ", ".join(f"({x},{y}) {s:.2f}" for x, y, s in logo_hits) or "none"
        tx = ", ".join(f"({x},{y}) {s:.2f}" for x, y, s in text_hits) or "none"
        print(f"{rel}: {pil.size[0]}x{pil.size[1]} -> {final.size[0]}x{final.size[1]} | logo: {lg} | ruler text: {tx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
