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


def remove_stamp_b(bgr, tpl_bgr, x, y, scale, light=215):
    """Remove the coloured stamp with as little collateral damage as possible.

    Core (eye, lettering, dense shadow) is always inpainted. The faint outer
    shadow is only flattened on pixels that are themselves background, so a
    product the stamp touches keeps its detail. Returns (image, core mask).
    """
    def around(white, dil):
        cx, cy = x + tpl_bgr.shape[1] * scale / 2, y + tpl_bgr.shape[0] * scale / 2
        m = np.zeros(bgr.shape[:2], np.uint8)
        for s2 in (scale * 0.94, scale, scale * 1.06):
            tw, th = tpl_bgr.shape[1] * s2, tpl_bgr.shape[0] * s2
            m |= stamp_mask(bgr.shape, tpl_bgr, round(cx - tw / 2), round(cy - th / 2), s2, white_thresh=white, dilate=dil)
        return m
    core = around(235, round(2.5 * scale) + 1)
    full = around(253, round(6 * scale) + 2)
    halo = (full > 0) & (core == 0)
    # shadow-tinted product pixels under the translucent parts are rebuilt
    # too (a model fill reproduces the metal; a paper fill would not)
    if _INPAINTER is not None:
        tinted = halo & (bgr.min(axis=2) < light)
        core = core | (tinted.astype(np.uint8) * 255)
        halo = (full > 0) & (core == 0)
    out = _inpaint_core(bgr, core, 3)
    out = clear_background_ghost(out, core)
    # background fill colour: light pixels in a ring just outside the full mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    ringpx = (cv2.dilate(full, k) > 0) & (full == 0)
    ring_vals = bgr[ringpx]
    ring_light = ring_vals[ring_vals.min(axis=1) >= light]
    fill = np.median(ring_light, axis=0).astype(np.uint8) if len(ring_light) else np.array([255, 255, 255], np.uint8)
    bg_px = halo & (bgr.min(axis=2) >= light)
    out[bg_px] = fill
    return out, core


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


def remove_logo(bgr, logo_tpl, hits, stroke_thresh=250, dilate=2):
    """Inpaint the grey stamp's strokes. Over a row of ruler tick marks the
    mask is the bare strokes with a 1px inpaint radius, so each tick is
    rebuilt from its own neighbours instead of smeared into a blotch;
    elsewhere a wider mask also clears the faint halo."""
    if not hits:
        return bgr, None
    th, tw = logo_tpl.shape
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stroke0 = (logo_tpl < stroke_thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    stroke_wide = cv2.dilate(stroke0, k)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    on_ticks = False
    for x, y, _ in hits:
        h = min(th, mask.shape[0] - y)
        w = min(tw, mask.shape[1] - x)
        under = gray[y:y + h, x:x + w]
        lower = under[int(h * 0.7):]
        # a dense dark structure under the lettering (ruler ticks and their
        # baseline, dark metal) is better served by the thin mask
        ticks = lower.size > 0 and float((lower < 150).mean()) >= 0.25
        on_ticks = on_ticks or ticks
        stroke = stroke0 if ticks else stroke_wide
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], stroke[:h, :w])
    if _INPAINTER is not None and not on_ticks:
        return _INPAINTER(bgr, mask), mask
    return cv2.inpaint(bgr, mask, 1 if on_ticks else 2, cv2.INPAINT_TELEA), mask


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
        ("left", text_tpl[:, : int(TW * 0.45)], lambda x, y, h, w: x + w >= W + pad - 4),
        ("left30", text_tpl[:, : int(TW * 0.3)], lambda x, y, h, w: x + w >= W + pad - 4),
        ("right", text_tpl[:, int(TW * 0.55):], lambda x, y, h, w: x <= pad + 4),
        ("top", text_tpl[: int(TH * 0.55), :], lambda x, y, h, w: y + h >= H + pad - 4),
        ("top35", text_tpl[: int(TH * 0.35), :], lambda x, y, h, w: y + h >= H + pad - 4),
    ]
    hits = []
    for sc in RULER_TEXT_SCALES:
        for name, part, accept in parts:
            th, tw = round(part.shape[0] * sc), round(part.shape[1] * sc)
            if th >= sub.shape[0] or tw >= sub.shape[1] or th < 6 or tw < 10:
                continue
            t = cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            need = thresh if name == "full" else (part_thresh + 0.08 if name in ("left30", "top35") else part_thresh)
            for x, y, score in _top_k(res, 6, th, tw):
                yy = y + band_top
                if score < need or not accept(x, yy, th, tw):
                    continue
                # letter tops of a bottom-cut word are thin strokes too, so the
                # tick test only applies to candidates that show whole letters
                if name not in ("top", "top35") and _looks_like_ticks(gray[yy:yy + th, x:x + tw]):
                    continue
                hits.append((x, yy, sc, score, th, tw, name))
    if not hits:
        return bgr, []
    hits.sort(key=lambda h: (h[6] != "full", -h[3]))   # full-word hits win over partials
    kept = []
    for h in hits:
        x, y, sc, score, th, tw, name = h
        if any(x < k[0] + k[5] and x + tw > k[0] and y < k[1] + k[4] and y + th > k[1] for k in kept):
            continue
        kept.append(h)
    out = padded.copy()
    for x, y, sc, score, th, tw, name in kept:
        y0, y1 = max(0, y - margin), y + th + margin
        x0, x1 = max(0, x - margin), x + tw + margin
        # fill with the ruler's own paper tone: the median of the non-ink
        # pixels in a ring just outside the word
        ring = 6
        ry0, ry1 = max(0, y0 - ring), min(out.shape[0], y1 + ring)
        rx0, rx1 = max(0, x0 - ring), min(out.shape[1], x1 + ring)
        region = padded[ry0:ry1, rx0:rx1]
        inner = np.zeros(region.shape[:2], bool)
        inner[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
        ringpx = region[~inner]
        ringpx = ringpx[ringpx.min(axis=1) > 170]
        fill = np.median(ringpx, axis=0) if len(ringpx) >= 20 else np.array([255, 255, 255])
        out[y0:y1, x0:x1] = fill.astype(np.uint8)
    out = out[pad:-pad, pad:-pad]
    return out, [(x - pad, y - pad, round(score, 2)) for x, y, sc, score, th, tw, name in kept]


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


def process_image(path: Path, logo_tpl, text_tpl, size: int, no_upscale: bool = False, logo_b_tpl=None, lettering_tpl=None):
    """Run the whole pipeline on one file.

    Returns (original, final, info) where info has: logo_hits, text_hits,
    logo_b (x, y, scale, score) or None, overlap (0..1 fraction of the removed
    stamp's border that touched non-background pixels).
    """
    pil = to_rgb(ImageOps.exif_transpose(Image.open(path)))
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    logo_hits = find_logo(bgr, logo_tpl)                     # detect on the untouched image
    logo_b = find_logo_b(bgr, logo_b_tpl, lettering_tpl) if logo_b_tpl is not None else None
    bgr, text_hits = remove_ruler_text(bgr, text_tpl)
    overlap = 0.0
    bgr, mask_a = remove_logo(bgr, logo_tpl, logo_hits)
    if mask_a is not None:
        overlap = max(overlap, over_product(bgr, mask_a))
    if logo_b is not None:
        x, y, sc, _ = logo_b
        before = bgr
        bgr, core_b = remove_stamp_b(bgr, logo_b_tpl, x, y, sc)
        overlap = max(overlap, over_product(before, core_b))
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
    ap.add_argument("--lettering", default=str(root / "assets/trax_logoB_lettering.png"))
    ap.add_argument("--no-upscale", action="store_true", help="only strip branding, keep size")
    ap.add_argument("--lama", action="store_true", help="use the LaMa model to rebuild what was under a stamp")
    a = ap.parse_args()
    if a.lama:
        set_inpainter(lama_inpainter())

    logo_tpl = load_template(Path(a.logo))
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
