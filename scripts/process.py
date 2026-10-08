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

import unblend as U

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


_LOGO_A_BGR = {}


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
    if gray.shape[0] <= th or gray.shape[1] <= tw:
        return []
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
LOGOB_SCALES = (1.1, 1.15, 1.2, 1.05, 1.0, 1.25, 1.3, 1.4, 1.5, 0.95, 0.9)   # the stamp is always near full size


def _masked_match(gray, tpl, mask):
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _scaled_mask(mask, scale):
    th, tw = max(1, round(mask.shape[0] * scale)), max(1, round(mask.shape[1] * scale))
    return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST) > 0


def _window(img, x, y, m):
    """Image window under a template mask placed at (x, y); clipped to the frame
    on every side (x or y may be negative when the stamp is cut off)."""
    th, tw = m.shape
    sx, sy = max(0, -x), max(0, -y)
    x, y = max(0, x), max(0, y)
    h, w = min(th - sy, img.shape[0] - y), min(tw - sx, img.shape[1] - x)
    if h <= 0 or w <= 0:
        return None, None
    return img[y:y + h, x:x + w], m[sy:sy + h, sx:sx + w]


def _maroon_fraction(bgr, red_mask_tpl, x, y, scale):
    """Share of the stamp's maroon eye pixels that really are maroon in the image.

    Maroon: red well above green, green and blue similar, not bright. Gold and
    most metal fail the green-blue balance, white and grey fail the red lift.
    (Skin passes, which is why a lettering check follows.)
    """
    win, m = _window(bgr, x, y, _scaled_mask(red_mask_tpl, scale))
    if win is None or m.sum() < 60:
        return 0.0
    win = win.astype(int)
    b, g, r = win[..., 0][m], win[..., 1][m], win[..., 2][m]
    ok = (r - g >= 28) & (np.abs(g - b) <= 22) & (r >= 100) & (r <= 205)
    return float(ok.mean())


def _lettering_contrast(gray, letter_mask_tpl, blank_mask_tpl, x, y, scale):
    """How much darker the 'TraxNYC' lettering is than the stamp's blank areas, in the image."""
    wl, ml = _window(gray, x, y, _scaled_mask(letter_mask_tpl, scale))
    wb, mb = _window(gray, x, y, _scaled_mask(blank_mask_tpl, scale))
    if wl is None or wb is None or ml.sum() < 30 or mb.sum() < 30:
        return 0.0
    return float(wb[mb].mean() - wl[ml].mean())


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


def _find_logo_b_masked(bgr, tpl_bgr, thresh=0.45, big_thresh=0.6, color_thresh=0.8, letter_thresh=18, topk=5):
    """Locate the large coloured TraxNYC stamp anywhere in the lower 60% of the
    frame, including stamps cut off by the frame edge.

    A candidate counts only if its matching score clears the threshold AND the
    eye region is maroon in the image AND the lettering is darker than the
    stamp's blank areas. Frames 900px or wider are modern photography that
    never carries the stamp, so they need a much higher score.
    Returns (x, y, scale, score) or None.
    """
    H, W = bgr.shape[:2]
    if W >= 900:
        thresh = big_thresh
    tgray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    core = (tgray < 235).astype(np.uint8) * 255          # eye + lettering, no soft shadow
    tb, tg_, tr = cv2.split(tpl_bgr.astype(int))
    red_mask = ((tr - np.maximum(tg_, tb)) > 40).astype(np.uint8)
    letters = ((tgray < 120) & (red_mask == 0)).astype(np.uint8)
    blank = (tgray > 250).astype(np.uint8)
    pad = int(tgray.shape[1] * 0.6)
    y_off = max(0, int(H * 0.4) - pad)
    padded = cv2.copyMakeBorder(bgr[y_off:], 0, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    best = None
    for sc in LOGOB_SCALES:
        th, tw = round(tgray.shape[0] * sc), round(tgray.shape[1] * sc)
        if th >= gray.shape[0] or tw >= gray.shape[1]:
            continue
        t = cv2.resize(tgray, (tw, th), interpolation=cv2.INTER_AREA)
        m = cv2.resize(core, (tw, th), interpolation=cv2.INTER_NEAREST)
        res = _masked_match(gray, t, m)
        for x, y, score in _top_k(res, topk, th, tw):
            if score < thresh:
                continue
            maroon = _maroon_fraction(padded, red_mask, x, y, sc)
            if maroon < color_thresh or (score < 0.35 and maroon < 0.8):
                continue
            if _lettering_contrast(gray, letters, blank, x, y, sc) < letter_thresh:
                continue
            if best is None or score > best[3]:
                best = (x - pad, y + y_off, sc, score)
    if best is None:
        return None
    # refine the scale around the best hit, keeping the stamp centre fixed
    x, y, sc, score = best
    base = sc
    cx, cy = x + tgray.shape[1] * sc / 2, y + tgray.shape[0] * sc / 2
    for ds in (-0.075, -0.05, -0.025, 0.025, 0.05, 0.075):
        s2 = base + ds
        th, tw = round(tgray.shape[0] * s2), round(tgray.shape[1] * s2)
        if th < 8 or tw < 8:
            continue
        x2, y2 = round(cx - tw / 2) + pad, round(cy - th / 2) - y_off
        r = 4
        x0, y0 = max(0, x2 - r), max(0, y2 - r)
        win = gray[y0:y2 + th + r, x0:x2 + tw + r]
        if win.shape[0] < th or win.shape[1] < tw:
            continue
        t = cv2.resize(tgray, (tw, th), interpolation=cv2.INTER_AREA)
        m = cv2.resize(core, (tw, th), interpolation=cv2.INTER_NEAREST)
        res = _masked_match(win, t, m)
        _, mx, _, (wx, wy) = cv2.minMaxLoc(res)
        if mx > score:
            score, sc, x, y = float(mx), s2, x0 + wx - pad, y0 + wy + y_off
    return (x, y, sc, score)


LETTERING_OFFSET = (100, 114)     # top-left of the lettering band inside the stamp template
LETTERING_SCALES = (1.0, 1.1, 1.2, 1.3, 0.9, 1.4, 1.5, 1.6, 1.7)


def find_logo_b(bgr, tpl_bgr, lettering_tpl, thresh=0.68, big_thresh=0.85, color_thresh=0.5, topk=3):
    """Locate the large coloured TraxNYC stamp via its 'TraxNYC' lettering.

    The lettering is opaque and carries its own white halo, so it looks the
    same on any background and gives a reliable position and scale (the
    masked whole-stamp match rewards oversized templates). Each candidate is
    confirmed by the maroon colour of the eye region, then the scale is
    refined. Falls back to the masked whole-stamp match when nothing is
    found. Returns (x, y, scale, score) or None.
    """
    H, W = bgr.shape[:2]
    if W >= 900:
        thresh = big_thresh
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tb, tg_, tr = cv2.split(tpl_bgr.astype(int))
    red_mask = ((tr - np.maximum(tg_, tb)) > 40).astype(np.uint8)
    pad = int(lettering_tpl.shape[1] * 1.2)
    y_off = max(0, int(H * 0.35) - pad)
    gray = cv2.copyMakeBorder(gray_full[y_off:], 0, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    padded = cv2.copyMakeBorder(bgr[y_off:], 0, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ox, oy = LETTERING_OFFSET

    def stamp_origin(lx, ly, sc):
        return round(lx - ox * sc), round(ly - oy * sc)

    def score_at(sc, win_x0=None, win_y0=None, win=None):
        th, tw = round(lettering_tpl.shape[0] * sc), round(lettering_tpl.shape[1] * sc)
        g = gray if win is None else win
        if th >= g.shape[0] or tw >= g.shape[1] or th < 6:
            return None
        t = cv2.resize(lettering_tpl, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        return cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED), th, tw

    best = None
    for sc in LETTERING_SCALES:
        r = score_at(sc)
        if r is None:
            continue
        res, th, tw = r
        for lx, ly, score in _top_k(res, topk, th, tw):
            if score < thresh:
                continue
            sx, sy = stamp_origin(lx, ly, sc)
            if _maroon_fraction(padded, red_mask, sx, sy, sc) < color_thresh:
                continue
            if best is None or score > best[3]:
                best = (lx, ly, sc, score)
    if best is None and W < 900:
        # second pass (catalogue frames only): the stamp's usual corner
        # placements, lower bar, colour still required
        for px_off, dx_lo, dx_hi in ((W - LOGOB_PRIOR[0], -40, 40), (-100, -30, 30)):
            for sc in (0.9, 1.0, 1.1, 1.2, 1.3):
                th, tw = round(lettering_tpl.shape[0] * sc), round(lettering_tpl.shape[1] * sc)
                lx_c = px_off + ox * sc + pad
                ly_c = (H - LOGOB_PRIOR[1]) + oy * sc - y_off
                x0, x1 = int(lx_c + dx_lo), int(lx_c + tw + dx_hi)
                y0, y1 = int(ly_c - 30), int(ly_c + th + 30)
                x0, y0 = max(0, x0), max(0, y0)
                win = gray[y0:y1, x0:x1]
                if win.shape[0] <= th or win.shape[1] <= tw:
                    continue
                t = cv2.resize(lettering_tpl, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
                res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, (wx, wy) = cv2.minMaxLoc(res)
                if mx < 0.55:
                    continue
                lx, ly = x0 + wx, y0 + wy
                sx, sy = stamp_origin(lx, ly, sc)
                if _maroon_fraction(padded, red_mask, sx, sy, sc) < 0.25:
                    continue
                if best is None or mx > best[3]:
                    best = (lx, ly, sc, float(mx))
    if best is None:
        return _find_logo_b_masked(bgr, tpl_bgr)
    lx, ly, sc, score = best
    base = sc
    cx, cy = lx + lettering_tpl.shape[1] * sc / 2, ly + lettering_tpl.shape[0] * sc / 2
    for ds in (-0.05, -0.025, 0.025, 0.05):
        s2 = base + ds
        th, tw = round(lettering_tpl.shape[0] * s2), round(lettering_tpl.shape[1] * s2)
        x2, y2 = round(cx - tw / 2), round(cy - th / 2)
        r = 4
        x0, y0 = max(0, x2 - r), max(0, y2 - r)
        win = gray[y0:y2 + th + r, x0:x2 + tw + r]
        rr = score_at(s2, win=win)
        if rr is None:
            continue
        res, th, tw = rr
        _, mx, _, (wx, wy) = cv2.minMaxLoc(res)
        if mx > score:
            score, sc, lx, ly = float(mx), s2, x0 + wx, y0 + wy
    sx, sy = stamp_origin(lx, ly, sc)
    return (sx - pad, sy + y_off, sc, float(score))


def stamp_mask(shape, tpl_bgr, x, y, scale, white_thresh=253, dilate=None):
    """Binary mask of where the coloured stamp sits, including its soft shadow.

    The shadow fades to within a couple of levels of white, so the threshold
    is generous and the mask is dilated well past the visible edge; a stamp
    left half-removed shows as a grey ghost.
    """
    th, tw = round(tpl_bgr.shape[0] * scale), round(tpl_bgr.shape[1] * scale)
    t = cv2.resize(tpl_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    m = (t.min(axis=2) < white_thresh).astype(np.uint8) * 255
    if dilate is None:
        dilate = round(6 * scale) + 2
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


def clear_background_ghost(bgr, mask, ring=6, light=238):
    """After inpainting a stamp on plain background, flatten any leftover grey.

    Looks at a ring just outside the mask: where that ring is background
    (light), the inpainted patch is set to the ring's own colour so no tint
    propagates in. Where the ring touches the product the inpaint is kept.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    outer = cv2.dilate(mask, k)
    ringpx = (outer > 0) & (mask == 0)
    if ringpx.sum() == 0:
        return bgr
    ring_vals = bgr[ringpx]
    if float((ring_vals.min(axis=1) >= light).mean()) < 0.98:
        return bgr                                   # stamp touched the product; leave inpaint as is
    out = bgr.copy()
    out[mask > 0] = np.median(ring_vals, axis=0).astype(np.uint8)
    return out


_INPAINTER = None


def set_inpainter(fn):
    """fn(bgr, mask_uint8) -> bgr. Used for the stamp core instead of OpenCV's
    Telea fill (which smears textured metal). None restores the default."""
    global _INPAINTER
    _INPAINTER = fn


def rebuild(bgr, mask, grow=2, scale=2, ctx=110):
    """Rebuild the masked pixels with the model, working on a crop around
    the mask upsampled `scale` times: at the catalogue's 450-500px frame
    size the model keeps chain links, stones and ruler ticks far crisper and
    more regular that way. Falls back to OpenCV's fill without a model."""
    if not mask.any():
        return bgr
    if _INPAINTER is None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        return cv2.inpaint(bgr, cv2.dilate(mask, k) if grow else mask, 3, cv2.INPAINT_TELEA)
    if scale == 1:
        return _INPAINTER(bgr, mask, grow=grow)
    ys, xs = np.where(mask > 0)
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - ctx), min(H, ys.max() + ctx + 1)
    x0, x1 = max(0, xs.min() - ctx), min(W, xs.max() + ctx + 1)
    crop, m = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    c2 = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    m2 = cv2.resize(m, (c2.shape[1], c2.shape[0]), interpolation=cv2.INTER_NEAREST)
    o2 = _INPAINTER(c2, m2, grow=grow * scale)
    o = cv2.resize(o2, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    sel = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 3, 2 * grow + 3))) > 0
    out = bgr.copy()
    sub = out[y0:y1, x0:x1]
    sub[sel] = o[sel]
    out[y0:y1, x0:x1] = sub
    return out


def _inpaint_core(bgr, mask, radius=3):
    if _INPAINTER is not None and mask.any():
        return _INPAINTER(bgr, mask)
    return cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)


def lama_inpainter(weights=None):
    """Build a LaMa-based inpainter (see lama.py); weights default to
    $LAMA_WEIGHTS or models/big-lama.pt next to the repo."""
    import os
    from lama import Lama
    weights = weights or os.environ.get("LAMA_WEIGHTS") or str(Path(__file__).resolve().parent.parent / "models/big-lama.pt")
    return Lama(weights)


def _stamp_b_opaque_template(tpl_bgr):
    """Template-space mask of the coloured stamp's opaque parts: the eye
    (maroon plus its enclosed white) and the lettering."""
    tg = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    tb, tg_, tr = cv2.split(tpl_bgr.astype(int))
    maroon = ((tr - np.maximum(tg_, tb)) > 40)
    letters = (tg < 150) & ~maroon
    eye = cv2.morphologyEx(maroon.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    eye = cv2.dilate(eye, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # the lettering carries an opaque white glow about 5px wide
    let = cv2.dilate(letters.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # the flame's thin spike above the eye has a hard dark edge that an undo
    # cannot align to the pixel: it is narrow, so it is rebuilt instead
    ys = np.where(maroon.any(axis=1))[0]
    eye_top = int(ys.min()) if len(ys) else 0
    hull = U.stamp_hull(tg)
    # the needle: the rows above the point where the flame widens
    widths = (hull > 0).sum(axis=1)
    wide = np.where(widths >= 25)[0]
    needle_end = int(wide.min()) + 2 if len(wide) else eye_top + 4
    tip = np.zeros_like(hull)
    tip[:needle_end] = hull[:needle_end]
    # the needle fades out upward; the faint part (template value up to 252)
    # near the hull belongs to it, and it is extrapolated a little further
    # along its own axis
    near_hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    faint = ((tg < 253) & near_hull).astype(np.uint8) * 255
    faint[needle_end:] = 0
    tip = np.maximum(tip, faint)
    ys, xs = np.where(tip > 0)
    if len(ys) > 10:
        y_top = int(ys.min())
        band = (ys <= y_top + 12)
        if band.sum() >= 2 and (ys[band].max() > y_top):
            # axis from the lowest to the highest rows of the top band
            lo = ys[band] >= y_top + 6
            hi = ys[band] <= y_top + 3
            if lo.any() and hi.any():
                x_lo, y_lo = xs[band][lo].mean(), ys[band][lo].mean()
                x_hi, y_hi = xs[band][hi].mean(), ys[band][hi].mean()
                dx, dy = x_hi - x_lo, y_hi - y_lo
                n = max(1e-3, (dx * dx + dy * dy) ** 0.5)
                ex, ey = int(round(x_hi + dx / n * 14)), int(round(y_hi + dy / n * 14))
                cv2.line(tip, (int(round(x_hi)), int(round(y_hi))), (ex, ey), 255, 3)
    tip = cv2.dilate(tip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return np.maximum(np.maximum(eye, let), tip)


def _stamp_b_highlight_template(tpl_bgr, grey=70.0):
    """Template-space mask of the flame's light highlight: inside the stamp's
    silhouette, light, and low alpha under the grey assumption. On a product
    this is an opaque light streak that must be rebuilt, not undone."""
    tg = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    hull = U.stamp_hull(tg) > 0
    a = U.alpha_from_template(tg, grey)
    m = hull & (tg > 200) & (a < 0.3)
    return m.astype(np.uint8) * 255


def remove_stamp_b(bgr, tpl_bgr, x, y, scale, grey=70.0, light=215, use_model=True, tol=22):
    """Remove the coloured stamp.

    1. Pixels that look exactly like the stamp on plain white (every channel
       within `tol` of the template) are background under the stamp: they
       become the paper tone.
    2. Translucent parts over a product (flame body, drop shadow) are undone
       arithmetically, so the product under them is the real product.
    3. Opaque parts (eye, lettering and its glow, the flame highlight on a
       product) are rebuilt by the inpainter.
    4. Light, flat leftovers anywhere in the stamp's footprint are snapped to
       the paper tone.
    Returns (image, rebuilt mask).
    """
    opaque_tpl = _stamp_b_opaque_template(tpl_bgr)
    out, inp, trans = U.unblend(bgr, tpl_bgr, x, y, scale, grey_colour=grey, max_alpha=0.62,
                                opaque_mask_tpl=opaque_tpl, grow_opaque=0)
    tg_placed = U.place(bgr.shape, cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
                        x, y, scale, fill=255.0)
    footprint = U.place(bgr.shape, U.stamp_hull(cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32),
                        x, y, scale, interp=cv2.INTER_NEAREST) > 127
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(6 * scale) + 3,) * 2)
    footprint = cv2.dilate(footprint.astype(np.uint8), k) > 0
    hl = U.place(bgr.shape, _stamp_b_highlight_template(tpl_bgr, grey), x, y, scale,
                 interp=cv2.INTER_NEAREST) > 127
    on_product = is_product(out, dark=225)
    # an opaque light part of the stamp (flame highlight, glow) on a product
    # survives the undo as a bright patch: brighter than the surrounding
    # product where the template itself is light
    g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    local = cv2.medianBlur(g_out, 21)
    bright = footprint & (g_out.astype(int) > local.astype(int) + 12) & (tg_placed > 185) & on_product
    extra = ((hl & on_product) | bright).astype(np.uint8) * 255
    extra = cv2.dilate(extra, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inp = np.maximum(inp, extra)
    trans[inp > 0] = 0
    region = np.maximum(inp, trans)
    paper = U.paper_tone(bgr, region, ring=12, light=light)
    # 1. stamp over background: compare the original with the stamp on white
    expected = np.stack([U.place(bgr.shape, tpl_bgr[..., i].astype(np.float32), x, y, scale, fill=255.0)
                         for i in range(3)], axis=-1)
    like_stamp = (np.abs(bgr.astype(np.int16) - expected.astype(np.int16)) <= tol).all(axis=2) & footprint
    undone_light = (out.min(axis=2) >= 215) & ((out.max(axis=2).astype(np.int16) - out.min(axis=2).astype(np.int16)) <= 18)
    out[like_stamp & undone_light & (inp == 0)] = paper
    # 4a. light flat leftovers in the undone area
    out = U.snap_background(out, trans, paper)
    # 3. rebuild the opaque parts (on plain background a paper fill is exact
    #    and the model is not needed)
    if inp.any():
        if product_near(bgr, inp, ring=6) < 15 and product_near(out, inp, ring=6) < 15:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out[cv2.dilate(inp, k3) > 0] = paper
        elif use_model:
            out = rebuild(out, inp, grow=2)
        else:
            out = cv2.inpaint(out, inp, 3, cv2.INPAINT_TELEA)
        # 4b. a rebuilt patch on background comes out faintly grey: snap it
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = U.snap_background(out, cv2.dilate(inp, k2), paper, light=230, flat=7.0)
    # hairlines: only where the stamp's own translucent pixels were (plus 2px)
    out = U.clean_thin_residue(out, footprint.astype(np.uint8) * 255, paper)
    out = U.flatten_near_paper(out, footprint.astype(np.uint8) * 255, paper)
    return out, inp


def is_product(bgr, dark=215, texture=10.0):
    """Pixels that are product rather than plain background: darker than
    paper in some channel, or locally textured (white metal and stones can be
    as bright as paper but are never flat)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (7, 7))
    sq = cv2.blur(g * g, (7, 7))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return (bgr.min(axis=2) < dark) | (std > texture)


def product_near(bgr, mask, ring=6, dark=215):
    """Number of product pixels in a ring just outside `mask` (how much of
    the product the removal had to work next to)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    ringpx = (cv2.dilate(mask, k) > 0) & (mask == 0)
    return int((is_product(bgr, dark) & ringpx).sum())


def over_product(bgr, mask, inner=3, outer=8, dark=220):
    """Fraction of pixels in a ring 3-8px outside `mask` that are not background.

    The ring starts a few pixels out so the stamp's own faint shadow does not
    count, and `dark` tolerates slightly off-white backgrounds.
    """
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1))
    ringpx = (cv2.dilate(mask, ko) > 0) & (cv2.dilate(mask, ki) == 0)
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


def remove_logo(bgr, logo_tpl, hits, stroke_thresh=250, dilate=2, logo_tpl_bgr=None):
    """Remove the grey stamp.

    - On plain background: paper fill of the strokes.
    - Next to or over a product: the model rebuilds the strokes, grown 2px,
      on a 2x upsampled crop.
      (An arithmetic undo of this stamp was tried and rejected: its strokes
      are one or two pixels wide, so after JPEG compression the undo leaves
      ghost letters on smooth metal and speckle on pave.)
    Returns (image, mask of what was rebuilt).
    """
    if not hits:
        return bgr, None
    th, tw = logo_tpl.shape
    stroke0 = (logo_tpl < stroke_thresh).astype(np.uint8) * 255
    mask = np.zeros(bgr.shape[:2], np.uint8)
    for x, y, _ in hits:
        h = min(th, mask.shape[0] - y)
        w = min(tw, mask.shape[1] - x)
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], stroke0[:h, :w])
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near = (cv2.dilate(mask, k2) > 0)
    prod = is_product(bgr)
    if int((prod & near & (mask == 0)).sum()) < 15:
        paper = U.paper_tone(bgr, mask, ring=12)
        out = bgr.copy()
        out[cv2.dilate(mask, k2) > 0] = paper
        return out, mask
    return rebuild(bgr, mask, grow=dilate), mask


RULER_TEXT_SCALES = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.75, 1.9, 2.1, 2.3, 0.9, 0.8)


def _looks_like_ticks(gray_win, dark=200):
    """True when a candidate window is a row of ruler tick marks rather than
    lettering: ticks are many tall, thin bars (median blob height over 60% of
    the window, width under 7%), letters are shorter and wider blobs."""
    d = (gray_win < dark).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    st = st[1:]
    st = st[st[:, 4] >= max(3, 0.002 * gray_win.size)]
    if len(st) < 6:
        return False
    med_h = float(np.median(st[:, 3]) / gray_win.shape[0])
    med_w = float(np.median(st[:, 2]) / gray_win.shape[1])
    return med_h > 0.58 and med_w < 0.07


def remove_ruler_text(bgr, text_tpl, thresh=0.42, part_thresh=0.52, pad=120, margin=3):
    """Find every 'TraxNYC' word printed on a ruler, at any of the sizes the
    catalogue uses, and fill it with the ruler's own background.

    Words cut off by the frame edge are found with partial templates (the
    left part of the word must touch the right edge, the right part the left
    edge, the top part the bottom edge), which only count when they do touch
    that edge.
    """
    H, W = bgr.shape[:2]
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    band_top = int(H * 0.3) + pad                      # rulers sit in the lower part of the frame
    sub = gray[band_top:, :]
    TH, TW = text_tpl.shape
    parts = [  # (name, template slice, acceptance test on the hit box in padded coords)
        ("full", text_tpl, lambda x, y, h, w: True),
        ("left", text_tpl[:, : int(TW * 0.45)], lambda x, y, h, w: x + w >= W + pad - 6),
        ("left30", text_tpl[:, : int(TW * 0.3)], lambda x, y, h, w: x + w >= W + pad - 6),
        ("left20", text_tpl[:, : int(TW * 0.2)], lambda x, y, h, w: x + w >= W + pad - 6),
        ("right", text_tpl[:, int(TW * 0.55):], lambda x, y, h, w: x <= pad + 6),
        ("right30", text_tpl[:, int(TW * 0.7):], lambda x, y, h, w: x <= pad + 6),
        ("top", text_tpl[: int(TH * 0.55), :], lambda x, y, h, w: y + h >= H + pad - 4),
        ("top35", text_tpl[: int(TH * 0.35), :], lambda x, y, h, w: y + h >= H + pad - 4),
    ]
    cands = []
    for sc in RULER_TEXT_SCALES:
        for name, part, accept in parts:
            th, tw = round(part.shape[0] * sc), round(part.shape[1] * sc)
            if th >= sub.shape[0] or tw >= sub.shape[1] or th < 6 or tw < 10:
                continue
            t = cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            need = thresh if name == "full" else (part_thresh + 0.08 if name in ("left30", "left20", "right30", "top35") else part_thresh)
            for x, y, score in _top_k(res, 6, th, tw):
                yy = y + band_top
                # a partial word is also accepted at a lower score when it sits
                # on the same baseline, at the same size, as a whole word
                if score < min(need, 0.45) or not accept(x, yy, th, tw):
                    continue
                # letter tops of a bottom-cut word are thin strokes too, so the
                # tick test only applies to candidates that show whole letters
                if name not in ("top", "top35") and _looks_like_ticks(gray[yy:yy + th, x:x + tw]):
                    continue
                cands.append((x, yy, sc, score, th, tw, name, part, need))
    full = [c for c in cands if c[6] == "full" and c[3] >= c[8]]
    hits = []
    for c in cands:
        x, y, sc, score, th, tw, name, part, need = c
        ok = score >= need
        if not ok and name != "full":
            ok = any(abs(y + th - (fy + fth)) <= 0.2 * th and abs(sc - fsc) <= 0.16 for fx, fy, fsc, _, fth, *_ in full)
        if ok:
            hits.append(c[:8])
    if not hits:
        return bgr, []
    # full-word hits win over partials; among partials the one showing more
    # of the word wins, so a short piece cannot land on the wrong letters
    hits.sort(key=lambda h: (h[6] != "full", -h[5], -h[3]))
    kept = []
    for h in hits:
        x, y, sc, score, th, tw, name, part = h
        if any(x < k[0] + k[5] and x + tw > k[0] and y < k[1] + k[4] and y + th > k[1] for k in kept):
            continue
        kept.append(h)
    out = padded.copy()
    for x, y, sc, score, th, tw, name, part in kept:
        y0, y1 = max(0, y - margin), min(out.shape[0], y + th + margin)
        x0, x1 = max(0, x - margin), min(out.shape[1], x + tw + margin)
        # the ruler's own paper tone: light pixels in a ring around the word
        ring = 6
        ry0, ry1 = max(0, y0 - ring), min(out.shape[0], y1 + ring)
        rx0, rx1 = max(0, x0 - ring), min(out.shape[1], x1 + ring)
        region = padded[ry0:ry1, rx0:rx1]
        inner = np.zeros(region.shape[:2], bool)
        inner[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
        # only real image pixels count as paper, never the white padding
        inside = np.zeros(region.shape[:2], bool)
        inside[max(0, pad - ry0):max(0, H + pad - ry0), max(0, pad - rx0):max(0, W + pad - rx0)] = True
        ringpx = region[~inner & inside]
        ringpx = ringpx[ringpx.min(axis=1) > 170]
        fill = (np.median(ringpx, axis=0) if len(ringpx) >= 20 else np.array([255, 255, 255])).astype(np.uint8)
        fill_gray = float(0.114 * fill[0] + 0.587 * fill[1] + 0.299 * fill[2])
        # the glyphs are taken from the image itself (ink darker than the
        # paper inside the word's box), so a word printed a little larger or
        # a pixel off from the template is still covered entirely; the
        # template ink, generously grown, limits that to the word's own
        # letters, and ink reaching in from above the box (ruler digits,
        # tick marks) is left alone
        box_g = gray[y0:y1, x0:x1]
        bh = y1 - y0
        # where the template says the letters are (grown 3px, and 5px for the
        # outer limit of what may be filled)
        tpl_ink = np.zeros((bh, x1 - x0), np.uint8)
        t = (cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC) < 200).astype(np.uint8) * 255
        oy, ox = y - y0, x - x0
        h_, w_ = min(th, tpl_ink.shape[0] - oy), min(tw, tpl_ink.shape[1] - ox)
        if h_ > 0 and w_ > 0:
            tpl_ink[oy:oy + h_, ox:ox + w_] = t[:h_, :w_]
        tpl_ink3 = cv2.dilate(tpl_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        tpl_ink = cv2.dilate(tpl_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        strong = (box_g < fill_gray - 35).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
        protect = np.zeros(strong.shape, np.uint8)
        touch = np.zeros(strong.shape, np.uint8)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for i in range(1, n):
            top, hgt = st[i, 1], st[i, 3]
            if top != 0:
                continue
            comp = (lab == i).astype(np.uint8) * 255
            if top + hgt < 0.5 * bh:
                # a digit's bottom reaching into the box from above
                protect = np.maximum(protect, comp)
            else:
                # something tall reaching in from above: a product lying over
                # the word. Its pixels away from the letter positions are
                # kept; the letter strokes attached to it are rebuilt by the
                # model after the paper fill, so the product keeps a clean
                # edge where the letters touched it
                body = cv2.bitwise_and(comp, cv2.bitwise_not(tpl_ink3))
                protect = np.maximum(protect, body)
                near_body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
                touch = np.maximum(touch, cv2.bitwise_and(cv2.dilate(cv2.bitwise_and(comp, tpl_ink3), k5), near_body))
        protect = cv2.dilate(protect, k5) > 0
        # everything darker than the paper near the letters goes: the ink,
        # its soft edges and the JPEG ringing around it
        gm = (box_g < fill_gray - 5) & ~protect & (tpl_ink > 0)
        # the paper under the word: the local average of the paper pixels
        # around each glyph, so the fill follows the ruler's own shading
        reg = padded[ry0:ry1, rx0:rx1].astype(np.float32)
        reg_g = gray[ry0:ry1, rx0:rx1]
        gm_reg = np.zeros(reg_g.shape, bool)
        gm_reg[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = gm
        wgt = ((reg_g > fill_gray - 12) & ~gm_reg & inside).astype(np.float32)
        local = None
        for ksz in (15, 31, 61):
            den = cv2.blur(wgt, (ksz, ksz))
            num = cv2.blur(reg * wgt[..., None], (ksz, ksz))
            cand = num / np.maximum(den, 1e-6)[..., None]
            if local is None:
                local, have = cand, den > 0.05
            else:
                local[~have] = cand[~have]
                have |= den > 0.05
        local[~have] = fill
        # the paper's own grain goes on top of the smooth fill, so the filled
        # letters do not stand out as a flatter, cleaner patch
        py_, px_ = np.where(wgt > 0)
        resid = (reg - local)[py_, px_] if len(py_) else np.zeros((1, 3), np.float32)
        rng = np.random.default_rng(int(x) * 7919 + int(y))
        gy, gx = np.where(gm)
        pick = np.clip(resid[rng.integers(0, len(resid), size=len(gy))], -5, 5)
        sub_local = local[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]
        box = out[y0:y1, x0:x1]
        box[gy, gx] = np.clip(sub_local[gy, gx] + pick, 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1] = box
        touch = cv2.bitwise_and(touch, (tpl_ink > 0).astype(np.uint8) * 255)
        if touch.any():
            full = np.zeros(out.shape[:2], np.uint8)
            full[y0:y1, x0:x1] = touch
            out = rebuild(out, full, grow=1)
    out = out[pad:-pad, pad:-pad]
    return out, [(x - pad, y - pad, round(score, 2)) for x, y, sc, score, th, tw, name, part in kept]


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


def process_image(path: Path, logo_tpl, text_tpl, size: int, no_upscale: bool = False, logo_b_tpl=None, lettering_tpl=None,
                  precomputed=None):
    """Run the whole pipeline on one file.

    Returns (original, final, info) where info has: logo_hits, text_hits,
    logo_b (x, y, scale, score) or None, overlap (fraction of the rebuilt
    area's border that touched the product) and product_px (number of
    product pixels next to anything that was rebuilt).
    """
    pil = to_rgb(ImageOps.exif_transpose(Image.open(path)))
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if precomputed is not None:
        # detections from an earlier run of the same detectors
        logo_hits = precomputed.get("a") or []
        logo_b = precomputed.get("b")
        text_hits = []
        if precomputed.get("ruler_words", 0) > 0:
            bgr, text_hits = remove_ruler_text(bgr, text_tpl)
    else:
        logo_hits = find_logo(bgr, logo_tpl)                 # detect on the untouched image
        logo_b = find_logo_b(bgr, logo_b_tpl, lettering_tpl) if logo_b_tpl is not None else None
        bgr, text_hits = remove_ruler_text(bgr, text_tpl)
    overlap, product_px = 0.0, 0
    if logo_b is not None:
        x, y, sc, _ = logo_b
        before = bgr
        bgr, mask_b = remove_stamp_b(bgr, logo_b_tpl, x, y, sc)
        if mask_b is not None and mask_b.any():
            overlap = max(overlap, over_product(before, mask_b))
            product_px += product_near(before, mask_b)
    before = bgr
    bgr, mask_a = remove_logo(bgr, logo_tpl, logo_hits, logo_tpl_bgr=_LOGO_A_BGR.get("img"))
    if mask_a is not None:
        overlap = max(overlap, over_product(before, mask_a))
        product_px += product_near(before, mask_a)
    clean = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    final = clean if no_upscale else upscale(clean, size)
    info = {"logo_hits": logo_hits, "text_hits": text_hits, "logo_b": logo_b,
            "overlap": round(overlap, 3), "product_px": product_px}
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
    ap.add_argument("--lettering", default=str(root / "assets/trax_logoB_lettering.png"))
    ap.add_argument("--no-upscale", action="store_true", help="only strip branding, keep size")
    ap.add_argument("--lama", action="store_true", help="use the LaMa model to rebuild what was under a stamp")
    a = ap.parse_args()
    if a.lama:
        set_inpainter(lama_inpainter())

    logo_tpl = load_template(Path(a.logo))
    _LOGO_A_BGR["img"] = cv2.imread(a.logo, cv2.IMREAD_COLOR)
    text_tpl = load_template(Path(a.ruler_text))
    logo_b_tpl = cv2.imread(a.logo_b, cv2.IMREAD_COLOR)
    lettering_tpl = load_template(Path(a.lettering))
    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images found under {in_dir}", file=sys.stderr)
        return 1

    for p in files:
        pil, final, info = process_image(p, logo_tpl, text_tpl, a.size, a.no_upscale, logo_b_tpl, lettering_tpl)
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
