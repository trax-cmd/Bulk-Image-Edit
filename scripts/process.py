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
from bridge import bridge_bands

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
LETTERING_SCALES = (1.0, 1.1, 1.2, 1.3, 0.9, 1.4, 1.5, 1.6, 1.7, 1.9, 2.1, 2.3)   # up to 2.3 on the portrait frames


def _flame_grey_fraction(bgr, flame_mask_tpl, x, y, scale):
    """Share of the stamp's flame body (silhouette minus the eye) that is
    grey in the image: the translucent grey flame over paper or metal is
    neutral, while a ruby, a garnet or a pink photo under a false match is
    coloured throughout."""
    win, m = _window(bgr, x, y, _scaled_mask(flame_mask_tpl, scale))
    if win is None or m.sum() < 60:
        return 0.0
    win = win.astype(int)
    b, g, r = win[..., 0][m], win[..., 1][m], win[..., 2][m]
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    gray = (r + g + b) // 3
    return float(((chroma <= 25) & (gray >= 60) & (gray <= 240)).mean())


def _stamp_b_confirmed(padded, tpl_bgr, lettering_tpl, lx, ly, sx, sy, sc, score):
    """Beyond the maroon of the eye: the lettering box is neutral and the
    flame body is largely grey (or the lettering match is very strong)."""
    th, tw = round(lettering_tpl.shape[0] * sc), round(lettering_tpl.shape[1] * sc)
    box = padded[max(0, ly):ly + th, max(0, lx):lx + tw].astype(np.int16)
    if box.size == 0:
        return False
    chroma = box.max(axis=2) - box.min(axis=2)
    g = box.mean(axis=2)
    bright = float(((g > 200) & (chroma <= 30)).mean())      # the letters' white
    dark = float((g < 110).mean())                            # their outline, never a solid dark mass
    if np.percentile(chroma, 25) > 10 or bright < 0.2:
        return False
    if score >= 0.75 or dark <= 0.3:
        return True
    tg = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    tb, tg_, tr = cv2.split(tpl_bgr.astype(int))
    red = ((tr - np.maximum(tg_, tb)) > 20).astype(np.uint8)
    flame = (U.stamp_hull(tg) > 0) & (cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) == 0)
    flame[LETTERING_OFFSET[1] - 4:] = False          # not the lettering rows
    return _flame_grey_fraction(padded, flame.astype(np.uint8), sx, sy, sc) >= 0.5


def find_logo_b(bgr, tpl_bgr, lettering_tpl, thresh=0.5, big_thresh=0.85, color_thresh=0.5, topk=3):
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
    y_off = 0                      # the stamp sits at the top of some frames
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
            # on a grey frame the translucent eye prints as a muted pink, so
            # a strong lettering match needs less of it
            need = color_thresh if score < 0.6 else min(color_thresh, 0.3)
            mf = _maroon_fraction(padded, red_mask, sx, sy, sc)
            if mf < need:
                continue
            # a candidate the proven rules accept (usual sizes, the eye's
            # maroon in full, below the top of the frame) stands; one found
            # only by the wider search (a larger size, a muted eye, the top
            # of the frame) must also show the lettering's own colours
            usual = sc <= 1.7 and mf >= color_thresh and ly >= max(0, int(H * 0.35) - pad)
            if not usual and not _stamp_b_confirmed(padded, tpl_bgr, lettering_tpl, lx, ly, sx, sy, sc, score):
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
        fb = _find_logo_b_masked(bgr, tpl_bgr)
        if fb is None:
            return None
        # the whole-stamp match is confirmed the same way
        fx, fy, fsc, fscore = fb
        lx, ly = round(fx + ox * fsc) + pad, round(fy + oy * fsc) - y_off
        if _stamp_b_confirmed(padded, tpl_bgr, lettering_tpl, lx, ly, fx + pad, fy - y_off, fsc, fscore):
            return fb
        return None
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
    # the eye's soft reddish edge: an undo that assumes a grey overlay
    # leaves a green-blue cast there on a product, so it is rebuilt too
    reddish = ((tr - np.maximum(tg_, tb)) > 20).astype(np.uint8) * 255
    eye = np.maximum(eye, cv2.dilate(reddish, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))
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
    m = hull & (tg > 238)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m


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
    `grey` is the overlay's assumed grey level: on plain paper the undo is
    exact whatever it is; on a product it sets how much is taken away
    (lighter values over-darken polished metal and chain links).
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
    # the flame's bright core is an opaque white streak that the template
    # (white on white) cannot show: on a product it survives the undo as a
    # patch clearly brighter than its surroundings where the template is
    # light. Only sizeable patches count, so a chain link's own sparkle is
    # not rebuilt (that cost whole runs of links before).
    g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # the product's own local brightness (background excluded), so a white
    # streak on skin or at a band's edge is measured against the product
    pm = on_product.astype(np.float32)
    dens = cv2.blur(pm, (31, 31))
    local = cv2.blur(g_out * pm, (31, 31)) / np.maximum(dens, 1e-3)
    # the streak can reach beyond the template's visible outline (white on
    # white), so look a little outside the footprint too, but only inside
    # the product's silhouette and only in patches joined to the stamp
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(12 * scale) + 1,) * 2)
    near_foot = cv2.dilate(footprint.astype(np.uint8), k_near) > 0
    silhouette = cv2.morphologyEx(on_product.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    # the streak is flat white; a product's own bright, sparkling edge is not
    mean5 = cv2.blur(g_out, (5, 5))
    std5 = np.sqrt(np.maximum(cv2.blur(g_out * g_out, (5, 5)) - mean5 * mean5, 0))
    bright = near_foot & silhouette & (g_out > local + 20) & (dens > 0.3)
    n_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    keep_b = np.zeros(n_b, bool)
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i in range(1, n_b):
        area, bw, bh = st_b[i, 4], st_b[i, 2], st_b[i, 3]
        # the streak is a patch of limited size; a huge patch is background
        # between parts of the product that the silhouette closed over
        if area < 30 * scale * scale or area > 800 * scale * scale or max(bw, bh) > 70 * scale:
            continue
        comp = lab_b == i
        around = (cv2.dilate(comp.astype(np.uint8), k15) > 0) & ~comp & on_product
        ref = float(np.median(std5[around])) if around.sum() >= 20 else 4.0
        d_local = float((g_out[comp] - local[comp]).mean())
        # on a smooth product (skin, polished metal) any clearly lighter
        # patch is the streak; on a textured one (chain, pave) only a flat
        # near-white patch is, never the product's own sparkle
        if (ref < 12 and d_local >= 25) or (float(g_out[comp].mean()) >= 225 and float(std5[comp].mean()) < 12):
            keep_b[i] = True
    bright = keep_b[lab_b]
    extra = ((hl & on_product) | bright).astype(np.uint8) * 255
    extra = cv2.dilate(extra, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inp = np.maximum(inp, extra)
    trans[inp > 0] = 0
    region = np.maximum(inp, trans)
    paper = U.paper_tone(bgr, region, ring=12, light=light)
    # what lies under the stamp's translucent parts once they are undone:
    # paper (white, neutral) or not (skin, a backdrop)
    under = (trans > 0) & (inp == 0)
    fp = out[under].astype(np.int16)
    paper_share = float(((fp.min(axis=1) >= light) & (fp.max(axis=1) - fp.min(axis=1) <= 20)).mean()) if len(fp) >= 50 else 1.0
    no_paper = paper_share < 0.5
    if no_paper and use_model:
        # on skin or a dark backdrop the arithmetic undo of the translucent
        # parts (which assumes paper under them) leaves a pale ghost of the
        # flame and its shadow: the whole footprint is rebuilt instead
        inp = np.maximum(inp, footprint.astype(np.uint8) * 255)
        trans[:] = 0
        out = bgr.copy()
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
        if plain_background(out, inp):
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out[cv2.dilate(inp, k3) > 0] = paper
        elif use_model:
            # a thin smooth band (a hoop, a shank) cut by the opaque parts is
            # joined up first; the model then only blends the join's edges
            out, inp_m, _ = bridge_bands(out, inp)
            out = rebuild(out, inp_m, grow=2)
            inp = inp_m
        else:
            out = cv2.inpaint(out, inp, 3, cv2.INPAINT_TELEA)
        # 4b. a rebuilt patch on background comes out faintly grey: snap it
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = U.snap_background(out, cv2.dilate(inp, k2), paper, light=230, flat=7.0)
    # hairlines: only where the stamp's own translucent pixels were (plus 2px)
    out = U.clean_thin_residue(out, footprint.astype(np.uint8) * 255, paper, max_chroma=40)
    out = U.flatten_near_paper(out, footprint.astype(np.uint8) * 255, paper)
    # a fleck of the flame's tip beyond the template's reach: a small blob
    # of non-paper near the stamp that touches no product is residue
    prod_out = is_product(out).astype(np.uint8)
    n_p, lab_p, st_p, _ = cv2.connectedComponentsWithStats(prod_out, connectivity=8)
    # (a light blob of some size with no dark pixel at all is the stamp's
    # shadow printed off the template, never a piece of jewellery, which
    # always carries dark edges or facets)
    g_res = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    big = np.zeros(n_p, bool)
    big[1:] = st_p[1:, 4] >= 40
    tb_, tg__, tr_ = cv2.split(out.astype(np.int16))
    maroon_px = ((tr_ - np.maximum(tg__, tb_)) > 40) & (tg__ < 100)   # dark red, not gold
    for i in range(1, n_p):
        if big[i] and st_p[i, 4] <= 600:
            comp_i = lab_p == i
            # a light blob, or a maroon one (a piece of the eye printed off
            # the template), is never jewellery
            if int(g_res[comp_i].min()) > 150 or float(maroon_px[comp_i].mean()) > 0.5:
                big[i] = False
    big_near = cv2.dilate(big[lab_p].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    fleck = np.zeros(out.shape[:2], bool)
    for i in range(1, n_p):
        if big[i]:
            continue
        comp = lab_p == i
        if near_foot[comp].all() and not big_near[comp].any():
            fleck |= comp
    if fleck.any():
        fleck = cv2.dilate(fleck.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        out[fleck] = paper
    # everything near the stamp that is not paper and is not joined to real
    # jewellery (which has dark pixels, or is large) is the stamp's
    # leftover: its shadow printed off the template, the model's faintly
    # grey paper or a blob it grew from the eye's glow, an undone edge
    if int(paper.min()) >= 235:
        dark_any = out.min(axis=2) < 150
        prod2 = is_product(out).astype(np.uint8)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(prod2, connectivity=8)
        real = np.zeros(n2, bool)
        real[np.unique(lab2[dark_any & (prod2 > 0)])] = True
        # a large pale thing is real too (a blurred shank), but only when it
        # continues beyond the stamp's reach: the eye's unblended shadow is
        # a large pale blob that lies wholly within it
        beyond = np.zeros(n2, bool)
        beyond[np.unique(lab2[(prod2 > 0) & ~near_foot])] = True
        real[1:] |= (st2[1:, 4] >= 1500) & beyond[1:]
        real[0] = False
        real_near = cv2.dilate(real[lab2].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        dev = (np.abs(out.astype(np.int16) - paper.astype(np.int16)).max(axis=2) > 8).astype(np.uint8)
        # only a patch that lies wholly within the stamp's reach: a pale
        # blurred product continues beyond it and is kept
        nd, dlab, dst_, _ = cv2.connectedComponentsWithStats(dev, connectivity=8)
        inside = np.ones(nd, bool)
        inside[np.unique(dlab[(dev > 0) & ~near_foot])] = False
        inside[1:] &= dst_[1:, 4] <= 4000       # a broad soft shading is not a leftover
        inside[0] = False
        left = inside[dlab] & ~real_near
        if left.any():
            left = cv2.dilate(left.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
            out[left & ~real_near] = paper
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


def remove_logo(bgr, logo_tpl, hits, stroke_thresh=250, dilate=1, logo_tpl_bgr=None):
    """Remove the grey stamp.

    - On plain background: paper fill of the strokes.
    - Next to or over a product: the model rebuilds the strokes, grown 1px,
      on a 2x upsampled crop (grown 2px it flattened pave facets).
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
    tplg = np.full(bgr.shape[:2], 255, np.float32)
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    for x, y, _ in hits:
        h = min(th, mask.shape[0] - y)
        w = min(tw, mask.shape[1] - x)
        st = stroke0[:h, :w] > 0
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], stroke0[:h, :w])
        # the template is the stamp printed on white; on a grey or tinted
        # ruler face the same translucent stamp prints darker in proportion
        # to the face, so the expected stroke level is scaled by the face's
        # tone (the light majority of the box outside the strokes)
        box = gray0[y:y + h, x:x + w][~st]
        light = box[box > 120]
        tone = 255.0
        if len(light) >= 50 and len(light) >= 0.3 * len(box):
            tone = float(np.median(light))
            if tone >= 235:
                tone = 255.0
        elif len(box) >= 50:
            # a dark backdrop: the stamp lightens it, nothing under it is ink
            tone = float(np.median(box))
            light = box
        tplg[y:y + h, x:x + w] = np.minimum(tplg[y:y + h, x:x + w], logo_tpl[:h, :w].astype(np.float32) * (tone / 255.0))
        if tone < 235:
            # the stamp's white glow around its strokes is invisible on white
            # paper (and so absent from the template) but lightens a tinted
            # face: pixels next to the strokes lighter than the face go too
            face = light[np.abs(light - tone) <= 16]
            sig = float(np.clip(face.std() if len(face) >= 20 else 20.0, 1.0, 20.0))
            near = cv2.dilate(stroke0[:h, :w], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            glow = near & (gray0[y:y + h, x:x + w] > tone + max(5.0, 1.5 * sig))
            mask[y:y + h, x:x + w][glow] = 255
    # a pixel clearly darker than the stamp's own stroke would be on the
    # face at that spot is ink or product showing through the translucent
    # stamp (a ruler digit, a dark link): it is kept, so the model only
    # bridges the stroke's own width through it instead of redrawing the digit
    kept = ((mask > 0) & (gray0 < tplg - 30)).astype(np.uint8)
    # a dark speck of a few pixels inside a stroke is the stamp's own
    # (JPEG-darkened) ink, not a digit or a link showing through
    n, lab, stats, _ = cv2.connectedComponentsWithStats(kept, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            kept[lab == i] = 0
    kept = kept > 0
    mask[kept] = 0
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if plain_background(bgr, mask, dark=110):
        # on plain background the paper fill is exact; the model's paper
        # comes out faintly grey. Light flat leftovers of the strokes' edges
        # (a stamp printed a pixel off the template) are snapped to paper
        paper = U.paper_tone(bgr, mask, ring=12)
        out = bgr.copy()
        out[cv2.dilate(mask, k2) > 0] = paper
        k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        out = U.snap_background(out, cv2.dilate(mask, k4), paper, light=225, flat=8.0)
        return out, mask
    out = rebuild(bgr, mask, grow=dilate)
    return settle_face(out, bgr, mask, kept), mask


def settle_face(out, src, mask, kept, grow=4, tol_k=3.0, tol_min=8, share=0.6, damp=0.3):
    """After the model rebuilt the strokes of a stamp lying on a plain face
    (white paper or a grey or tinted ruler face), the rebuilt pixels and the
    face's JPEG ringing around the strokes are a little rougher than the
    face itself (std 4-5 against a face's 2), which the upscale's sharpening
    shows as a faint ghost of the stamp. Where the surroundings are a plain
    face, the deviations from its tone inside the strokes and a `grow` px
    band around them are damped. Pixels not close to the face tone (ink,
    product, their anti-aliased edges), the ink and product kept out of
    the mask with a 3px margin, and anything whose neighbourhood is not
    mostly face-toned are left alone."""
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (mask > 0).astype(np.uint8)
    far = (cv2.dilate(m, k(14)) > 0) & (cv2.dilate(m, k(8)) == 0)
    gsrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vals = gsrc[far]
    if len(vals) < 100:
        return out
    hist, edges = np.histogram(vals, bins=np.arange(0, 260, 4))
    mode = float(edges[int(np.argmax(hist))] + 2)
    facepx = far & (np.abs(gsrc - mode) <= 12)
    if mode < 120 or facepx.sum() < share * far.sum():
        return out
    tone = np.median(src[facepx].astype(np.float32), axis=0)
    sigma = float(np.clip(gsrc[facepx].std(), 1.0, 20.0))
    if sigma > 6.0:
        return out                      # skin or a textured surface: its grain is not a ghost
    tol = max(tol_min, tol_k * sigma)
    region = (cv2.dilate(m, k(grow)) > 0) & (cv2.dilate(kept.astype(np.uint8), k(3)) == 0)
    dev = out.astype(np.float32) - tone
    adev = np.abs(dev).max(axis=2)
    near = adev <= tol
    frac = cv2.blur(near.astype(np.float32), (7, 7))
    # a lump the model left (a few pixels, up to ~20 levels off) counts as
    # face too when its neighbourhood is mostly face-toned
    sel = region & (adev <= max(2.0 * tol, 18.0)) & (frac >= share)
    res = out.copy()
    res[sel] = np.clip(tone + damp * dev[sel], 0, 255).astype(np.uint8)
    return res


def plain_background(bgr, mask, dark=215, ring=6, far=(4, 10), dev=15, max_dark=15, max_dev=30):
    """True when nothing but plain background lies next to `mask`: fewer
    than `max_dark` pixels darker than `dark` within `ring` px of it, and
    fewer than `max_dev` pixels in a farther ring (`far` px out, beyond the
    stamp's own anti-aliased edge) that differ from the paper tone by more
    than `dev` levels (a pale, blurred product is neither dark nor
    textured, but it is not paper either)."""
    m = (mask > 0).astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring1 = (cv2.dilate(m, k(ring)) > 0) & (m == 0)
    if int(((bgr.min(axis=2) < dark) & ring1).sum()) >= max_dark:
        return False
    ring2 = (cv2.dilate(m, k(far[1])) > 0) & (cv2.dilate(m, k(far[0])) == 0)
    paper = U.paper_tone(bgr, mask, ring=12).astype(np.int16)
    d = np.abs(bgr.astype(np.int16) - paper).max(axis=2)
    return int(((d > dev) & ring2).sum()) < max_dev


RULER_TEXT_SCALES = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.75, 1.9, 2.1, 2.3, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5)


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


_RULER_DBG = None  # set to a list to collect per-word intermediates (debugging only)


def remove_ruler_text(bgr, text_tpl, thresh=0.42, part_thresh=0.52, pad=120, margin=4, hits=None, detect_only=False):
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
    if hits is not None:
        # the words were found on the untouched image (before the stamps
        # were removed, which can smear a word under a stamp beyond
        # recognition); only the removal runs here
        return _remove_ruler_words(bgr, text_tpl, hits, pad, margin, padded, gray, TH, TW)
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
            # the small sizes are rows of repeated whole words: a piece of a
            # word a few pixels tall matches too much else
            if sc < 0.8 and name != "full":
                continue
            th, tw = round(part.shape[0] * sc), round(part.shape[1] * sc)
            if th >= sub.shape[0] or tw >= sub.shape[1] or th < 6 or tw < 10:
                continue
            t = cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            need = thresh if name == "full" else (part_thresh + 0.08 if name in ("left30", "left20", "right30", "top35") else part_thresh)
            for x, y, score in _top_k(res, 8, th, tw):
                yy = y + band_top
                if score < (0.36 if name == "full" else 0.45):
                    continue
                # letter tops of a bottom-cut word are thin strokes too, so the
                # tick test only applies to candidates that show whole letters
                if name not in ("top", "top35") and _looks_like_ticks(gray[yy:yy + th, x:x + tw]):
                    continue
                # a partial template normally only counts at the frame edge it
                # belongs to; off the edge it is kept as a candidate for the
                # aligned acceptance below
                cands.append((x, yy, sc, score, th, tw, name, part, need if accept(x, yy, th, tw) else 9.0))
    def aligned(c, others):
        x, y, sc, score, th, tw = c[:6]
        # below full size the anchor word must itself be a confident match
        return any(abs(y + th - (oy + oth)) <= 0.2 * th and abs(sc - osc) <= 0.16 and (sc >= 1.0 or osc_score >= 0.55)
                   for ox, oy, osc, osc_score, oth, *_ in others)
    SMALL = ("left30", "left20", "right30")
    # the whole word and the large partials are accepted on their own score;
    # the small partials (two or three letters) match ruler digits too, so
    # they only count beside a whole word on the same baseline
    # a small word (under 0.8 of the template) counts only at a good score
    # and beside another small word of the same size on the same baseline:
    # at that size a lone match is as likely a scrap of product or shadow
    def looks_like_word(c, partial=False):
        # the printed word has tall letters (T, N, Y, C) and short ones
        # (r, a, x) side by side; a row of stones, a chain, a "mm" label or
        # blank paper does not. Only the smaller sizes need this: a large
        # template is specific enough on its own
        x, y, sc, score, th, tw = c[:6]
        if sc >= 1.0:
            return True
        win = gray[y:y + th, x:x + tw].astype(np.int16)
        if win.size == 0:
            return False
        fill = float(np.median(win))
        ink = (win < fill - 40).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        st = st[1:]
        st = st[st[:, 4] >= 2]
        hs = st[:, 3] / float(th)
        st, hs = st[hs >= 0.15], hs[hs >= 0.15]      # specks are not letters
        if not (3 <= len(st) <= 14):
            return False
        hmax = float(hs.max())
        if hmax < 0.35:
            return False
        tall = int((hs >= 0.75 * hmax).sum())
        short = int(((hs >= 0.45 * hmax) & (hs < 0.75 * hmax)).sum())
        span = (st[:, 0] + st[:, 2]).max() - st[:, 0].min()
        if partial:
            # a word half hidden by the product shows a few letters
            return len(st) >= 2 and tall >= 1 and span >= 0.4 * tw
        if sc < 0.8:
            # a small print must show most of its letters (a stud and its
            # post make three blobs of the right heights)
            return len(st) >= 4 and tall >= 2 and short >= 2 and span >= 0.6 * tw
        return tall >= 2 and short >= 1 and span >= 0.6 * tw
    def has_ink(c):
        # a small template's correlation spikes on blank paper and on
        # product texture: the window must hold ink in a word's proportion
        x, y, sc, score, th, tw = c[:6]
        win = gray[y:y + th, x:x + tw].astype(np.int16)
        if win.size == 0:
            return False
        fill = float(np.median(win))
        frac = float((win < fill - 40).mean())
        return 0.05 <= frac <= 0.5
    def small_ok(c):
        return c[3] >= 0.5 and has_ink(c) and (c[3] >= 0.6 or looks_like_word(c))
    def direct_ok(c):
        if c[6] in SMALL or c[3] < c[8]:
            return False
        if c[2] < 0.8:
            return small_ok(c)
        if c[6] == "full" and c[2] < 1.0:
            # the mid sizes match stones, studs and digit rows at 0.4-0.5
            return c[3] >= 0.5 and (c[3] >= 0.55 or looks_like_word(c))
        return True
    direct = [c[:8] for c in cands if direct_ok(c)]
    full_direct = [c for c in direct if c[6] == "full"]
    hits = list(direct)
    # a second word on the same baseline, at the same size, as an accepted
    # word is accepted at a lower score: a word half hidden by the product, a
    # word printed in a lighter colour, or a piece cut off by the frame
    for c in cands:
        x, y, sc, score, th, tw, name, part, need = c
        if c[:8] in hits:
            continue
        if name in SMALL:
            ok = need < 9 and score >= 0.5 and aligned(c, full_direct)
        elif name == "full":
            ok = score >= (thresh if sc < 0.8 else 0.36) and aligned(c, direct) and (sc >= 0.8 or has_ink(c)) \
                and (score >= 0.55 or looks_like_word(c, partial=True))
        else:
            ok = score >= 0.45 and aligned(c, direct)
        if ok:
            hits.append(c[:8])
    # a partial word whose other half is inside the frame: look for the whole
    # word anchored on it, at a low score since the position is pinned
    extra = []
    for x, y, sc, score, th, tw, name, part in hits:
        if name == "full":
            continue
        fth, ftw = round(TH * sc), round(TW * sc)
        if name.startswith("left"):
            ax = x
        elif name.startswith("right"):
            ax = x + tw - ftw
        else:
            continue
        ay = y
        if ax < pad - 6 or ax + ftw > W + pad + 6:
            continue
        ry0, ry1 = max(0, ay - 4), min(gray.shape[0], ay + fth + 4)
        rx0, rx1 = max(0, ax - 4), min(gray.shape[1], ax + ftw + 4)
        win = gray[ry0:ry1, rx0:rx1]
        if win.shape[0] <= fth or win.shape[1] <= ftw:
            continue
        t = cv2.resize(text_tpl, (ftw, fth), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx >= 0.3 and not _looks_like_ticks(win[loc[1]:loc[1] + fth, loc[0]:loc[0] + ftw]):
            extra.append((rx0 + loc[0], ry0 + loc[1], sc, float(mx), fth, ftw, "full", text_tpl))
    hits = extra + hits
    if not hits:
        return [] if detect_only else (bgr, [])
    # full-word hits win over partials; among partials the one showing more
    # of the word wins, so a short piece cannot land on the wrong letters
    hits.sort(key=lambda h: (h[6] != "full", -h[5], -h[3]))
    kept = []
    for h in hits:
        x, y, sc, score, th, tw, name, part = h
        if any(x < k[0] + k[5] and x + tw > k[0] and y < k[1] + k[4] and y + th > k[1] for k in kept):
            continue
        kept.append(h)
    # refine each word's size and position: the size steps above are coarse,
    # and a word matched a size step too small leaves the bottoms of its
    # letters outside the fill
    refined = []
    for x, y, sc, score, th, tw, name, part in kept:
        best = (score, x, y, sc, th, tw)
        for f_ in (0.92, 0.95, 0.97, 1.0, 1.03, 1.06, 1.09, 1.12):
            s2 = sc * f_
            th2, tw2 = round(part.shape[0] * s2), round(part.shape[1] * s2)
            wy0, wy1 = max(0, y - 6), min(gray.shape[0], y + th2 + 6)
            wx0, wx1 = max(0, x - 6), min(gray.shape[1], x + tw2 + 6)
            win = gray[wy0:wy1, wx0:wx1]
            if win.shape[0] <= th2 or win.shape[1] <= tw2 or th2 < 6 or tw2 < 10:
                continue
            t2 = cv2.resize(part, (tw2, th2), interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(win, t2, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0] + 0.01:
                best = (float(mx), wx0 + loc[0], wy0 + loc[1], s2, th2, tw2)
        refined.append((best[1], best[2], best[3], best[0], best[4], best[5], name, part))
    # a small print's match improves markedly once its size is refined; a
    # stud, a shadow or a scrap of chain matched at the coarse size does not
    kept = [h for h in refined if not (h[2] < 0.8 and h[6] == "full" and h[3] < 0.58)]
    if detect_only:
        return kept
    return _remove_ruler_words(bgr, text_tpl, kept, pad, margin, padded, gray, TH, TW)


def _remove_ruler_words(bgr, text_tpl, kept, pad, margin, padded, gray, TH, TW):
    """Fill the ruler words in `kept` (as found by remove_ruler_text)."""
    H, W = bgr.shape[:2]
    out = padded.copy()

    def paper_tone(x, y, th, tw):
        """The ruler's own paper tone around a word: the median of the light
        pixels in a ring around it (real image pixels only, never the white
        padding). Returns (box, ring box, fill colour, fill grey)."""
        y0, y1 = max(0, y - margin), min(out.shape[0], y + th + margin)
        x0, x1 = max(0, x - margin), min(out.shape[1], x + tw + margin)
        ring = 6
        ry0, ry1 = max(0, y0 - ring), min(out.shape[0], y1 + ring)
        rx0, rx1 = max(0, x0 - ring), min(out.shape[1], x1 + ring)
        region = padded[ry0:ry1, rx0:rx1]
        inner = np.zeros(region.shape[:2], bool)
        inner[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
        inside = np.zeros(region.shape[:2], bool)
        inside[max(0, pad - ry0):max(0, H + pad - ry0), max(0, pad - rx0):max(0, W + pad - rx0)] = True
        ringpx = region[~inner & inside]
        # the paper is the ring's lighter majority (a ruler face may be grey
        # or tinted, so no fixed brightness is assumed); its noise level
        # scales the ink thresholds below
        if len(ringpx) >= 20:
            rg = 0.114 * ringpx[:, 0] + 0.587 * ringpx[:, 1] + 0.299 * ringpx[:, 2]
            # the paper's level is the ring's most common one (its mode), so
            # the ticks, digits and letters in the ring do not pull it down
            # and the face's own darker grain does not pull it up
            hist, edges = np.histogram(rg, bins=np.arange(0, 260, 4))
            mode = float(edges[int(np.argmax(hist))] + 2)
            light = ringpx[np.abs(rg - mode) <= 12]
            fill = np.median(light, axis=0).astype(np.uint8)
            wide = rg[np.abs(rg - mode) <= 24]
            sigma = float(min(20.0, max(1.0, wide.std())))
        else:
            fill, sigma = np.array([255, 255, 255], np.uint8), 1.0
        fill_gray = float(0.114 * fill[0] + 0.587 * fill[1] + 0.299 * fill[2])
        return (x0, y0, x1, y1), (rx0, ry0, rx1, ry1), fill, fill_gray, sigma

    def ink_per_letter(x, y, th, tw, part, sc, fill_gray, sigma=1.0):
        """The median colour of the darker third of each of the word's
        letters, as placed by the template (one entry per letter found)."""
        t = (cv2.resize(part, (tw, th), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC) < 200).astype(np.uint8) * 255
        gh, gw = min(th, gray.shape[0] - y), min(tw, gray.shape[1] - x)
        if gh <= 0 or gw <= 0:
            return []
        g_win = gray[y:y + gh, x:x + gw]
        b_win = padded[y:y + gh, x:x + gw]
        core = cv2.dilate(t[:gh, :gw], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ng, glab = cv2.connectedComponents(core, connectivity=8)
        strong = g_win < fill_gray - max(35.0, 4 * sigma)
        if int((strong & (core > 0)).sum()) < 8:
            # a word printed in a pale ink (light cyan on white): take the
            # darkest of what there is
            strong = g_win < fill_gray - max(8.0, 2 * sigma)
        cols = []
        for gi in range(1, ng):
            m = (glab == gi) & strong
            if m.sum() < 8:
                continue
            darker = m & (g_win <= np.percentile(g_win[m], 35))
            cols.append(np.median(b_win[darker if darker.sum() >= 4 else m], axis=0))
        return cols

    # The words of one ruler share one ink colour. Every word gives an
    # estimate; the word whose letters agree best wins (a product lying
    # over some letters of a word spoils that word's estimate)
    ink_global, best = None, (-1, -1)
    for x, y, sc, score, th, tw, name, part in kept:
        _, _, fill_, fg_, sg_ = paper_tone(x, y, th, tw)
        cols = ink_per_letter(x, y, th, tw, part, sc, fg_, sg_)
        if len(cols) < 2:
            continue
        arr = np.array(cols)
        med = np.median(arr, axis=0)
        close_ = np.abs(arr - med).max(axis=1) <= 35
        key = (int(close_.sum()), len(cols))
        if key > best:
            best, ink_global = key, np.median(arr[close_], axis=0) if close_.sum() >= 2 else med

    touched_words = []
    for x, y, sc, score, th, tw, name, part in kept:
        (x0, y0, x1, y1), (rx0, ry0, rx1, ry1), fill, fill_gray, paper_sigma = paper_tone(x, y, th, tw)
        # offsets below the paper tone that mean ink, scaled by the paper's
        # own noise (a grey ruler face carries ten levels of it, clipped
        # white paper almost none)
        off_np = max(12.0, 2.5 * paper_sigma)      # anything that is not paper
        off_core = max(15.0, 3.0 * paper_sigma)    # the ink's core
        off_dark = max(30.0, 4.0 * paper_sigma)    # clearly dark
        off_strong = max(35.0, 4.0 * paper_sigma)
        # only ink and its halo are filled, not the darker half of the
        # paper's own noise: filling that lifted the whole area a couple of
        # levels and replaced the paper's texture with speckle
        ink_thr = fill_gray - max(3.0, 1.5 * paper_sigma)
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
        # the template's size steps are coarse, so the far letters of a long
        # word can sit several pixels off: grow with the word's width
        g_ = 2 * (5 + int(round(0.03 * tw))) + 1
        tpl_ink = cv2.dilate(tpl_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g_, g_)))
        strong = (box_g < fill_gray - off_strong).astype(np.uint8)
        # Everything that is not paper, in a window around the word, grouped
        # into shapes. A shape that lives inside the word's zone is letters;
        # a shape that reaches beyond it is a ruler digit or tick coming in
        # from above, or a product lying over the word. Inside such a shape
        # the letters' own colour tells the letters from the product.
        ftw_ = round(TW * sc)
        cy0, cy1 = max(0, y0 - 24), min(gray.shape[0], y1 + 24)
        cx0, cx1 = max(0, x0 - ftw_ - 24), min(gray.shape[1], x1 + ftw_ + 24)
        gray_w = gray[cy0:cy1, cx0:cx1]
        bgr_w = padded[cy0:cy1, cx0:cx1].astype(np.int16)
        nonpaper = (gray_w < fill_gray - off_np).astype(np.uint8)
        blobs = cv2.morphologyEx(nonpaper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
        zone = np.zeros(blobs.shape, bool)
        zy0, zy1 = max(0, y0 - 8 - cy0), min(blobs.shape[0], y1 + 8 - cy0)
        if name.startswith("left"):
            zx0, zx1 = max(0, x0 - 12 - cx0), min(blobs.shape[1], x + ftw_ + 12 - cx0)
        elif name.startswith("right"):
            zx0, zx1 = max(0, x + tw - ftw_ - 12 - cx0), min(blobs.shape[1], x1 + 12 - cx0)
        else:
            zx0, zx1 = max(0, x0 - 12 - cx0), min(blobs.shape[1], x1 + 12 - cx0)
        zone[zy0:zy1, zx0:zx1] = True
        tpl3_w = np.zeros(blobs.shape, np.uint8)
        tpl3_w[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = tpl_ink3
        # the glyphs of the whole word, placed where this (possibly partial)
        # match puts it, so letters beyond a partial template are known too
        fth_ = round(TH * sc)
        t_full = (cv2.resize(text_tpl, (ftw_, fth_), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC) < 200).astype(np.uint8) * 255
        fx_ = x + tw - ftw_ if name.startswith("right") else x
        glyph_w = np.zeros(blobs.shape, np.uint8)
        gy0, gx0 = y - cy0, fx_ - cx0
        sy_, sx_ = max(0, -gy0), max(0, -gx0)
        gh, gw = min(fth_ - sy_, blobs.shape[0] - max(0, gy0)), min(ftw_ - sx_, blobs.shape[1] - max(0, gx0))
        if gh > 0 and gw > 0:
            glyph_w[max(0, gy0):max(0, gy0) + gh, max(0, gx0):max(0, gx0) + gw] = t_full[sy_:sy_ + gh, sx_:sx_ + gw]
        glyph_w = cv2.dilate(glyph_w, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        gh, gw = min(th, blobs.shape[0] - (y - cy0)), min(tw, blobs.shape[1] - (x - cx0))
        in_frac = np.zeros(n)
        for i in range(1, n):
            comp = lab == i
            in_frac[i] = float((comp & zone).sum()) / max(1, int(comp.sum()))
        letters_only = (in_frac >= 0.9)[lab] & (lab > 0)
        # the ink colour: the median over the letters of each letter's own
        # median colour (its darker half), so a product lying over a few of
        # the letters does not pull the estimate towards its own colour
        glyph_core = np.zeros(blobs.shape, np.uint8)
        if gh > 0 and gw > 0:
            glyph_core[y - cy0:y - cy0 + gh, x - cx0:x - cx0 + gw] = t[:gh, :gw]
        glyph_core = cv2.dilate(glyph_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ng, glab = cv2.connectedComponents(glyph_core, connectivity=8)
        strong_w = gray_w < fill_gray - off_strong
        if int((strong_w & (glyph_core > 0)).sum()) < 8:
            strong_w = gray_w < fill_gray - max(8.0, 2 * paper_sigma)
        per_letter = []
        for gi in range(1, ng):
            gm_ = (glab == gi) & strong_w
            if gm_.sum() < 8:
                continue
            gp = gray_w[gm_]
            darker = gm_ & (gray_w <= np.percentile(gp, 35))
            per_letter.append(np.median(bgr_w[darker if darker.sum() >= 4 else gm_], axis=0))
        letter_colour = np.median(np.array(per_letter), axis=0) if len(per_letter) >= 2 else None
        if ink_global is not None:
            letter_colour = ink_global
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        prot_w = np.zeros(blobs.shape, bool)
        touch_w = np.zeros(blobs.shape, bool)
        prod_w = np.zeros(blobs.shape, bool)   # the product (or digit) itself
        word_touched = False
        if letter_colour is not None:
            # a letter pixel is the ink seen through some amount of
            # anti-aliasing: it lies on the line from the paper colour to the
            # ink colour. A product pixel of another hue does not.
            paper_v = fill.astype(np.float32)
            v = letter_colour.astype(np.float32) - paper_v
            d = bgr_w.astype(np.float32) - paper_v
            tt = (d * v).sum(axis=2) / max(1.0, float((v * v).sum()))
            resid = np.sqrt(((d - tt[..., None] * v) ** 2).sum(axis=2))
            # the tolerance grows with darkness: the darkest pixels of a
            # letter drift furthest from a line through an estimate
            dn = np.sqrt((d * d).sum(axis=2))
            letter_like = (resid <= np.maximum(40.0, 0.2 * dn)) & (tt >= -0.15) & (tt <= 2.2)
        else:
            letter_like = np.zeros(blobs.shape, bool)
        shape_log = []
        for i in range(1, n):
            if in_frac[i] >= 0.9:
                continue
            comp = lab == i
            rows = np.where((comp & zone).any(axis=1))[0]
            from_above = (comp[:zy0] if zy0 > 0 else np.zeros((0, 1), bool)).any()
            depth = (rows.max() - zy0 + 1) / max(1, zy1 - zy0) if len(rows) else 0.0
            outside = comp & ~zone
            # judged on the shape's pixels that are at least half as dark as
            # the ink (a pave piece's bright stones lie on every ink line)
            judge = outside & (nonpaper > 0) & (tt > 0.5) if letter_colour is not None else outside
            like_out = float(letter_like[judge].mean()) if judge.sum() >= 30 else 0.0
            shape_log.append(dict(i=i, px=int(comp.sum()), in_frac=float(in_frac[i]), from_above=bool(from_above),
                                  depth=float(depth), judge=int(judge.sum()), like_out=like_out))
            if (from_above and depth < 0.5) or letter_colour is None:
                # a digit's or tick's foot: kept whole
                prot_w |= cv2.dilate(comp.astype(np.uint8), k5) > 0
                continue
            # The part of the shape that continues into the zone from outside
            # without crossing a letter (the shape's pixels joined to its
            # outside part through non-glyph pixels) is the product's or the
            # digit's own body, whatever its colour: letters only lie where
            # the template puts them. A digit merged with the letters by the
            # closing is told from them this way; the paper-like pixels the
            # closing added next to the letters are not part of it
            # the flow runs through the shape's own ink and gaps of up to
            # two pixels (a pave piece's bright stones), not through the
            # wider bridges the closing laid between the letters and a tick
            # or digit nearby: those would carry a misplaced letter edge
            # into the body
            # ... and only through clearly dark pixels: a ruler's paper shades
            # by more than the non-paper margin across a window, and that
            # shading must not carry the flow from a tick to a letter
            dark_w = (gray_w < fill_gray - off_dark).astype(np.uint8)
            tight = cv2.morphologyEx(dark_w, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
            flow = (comp & tight & ~glyph_w).astype(np.uint8)
            _, flab = cv2.connectedComponents(flow, connectivity=8)
            ids = np.unique(flab[outside & (flow > 0)])
            reach = np.isin(flab, ids[ids > 0]) & (flow > 0)
            reach_body = reach & ((nonpaper > 0) | (tpl3_w == 0))
            outside_dark = outside & (dark_w > 0)
            if like_out > 0.5:
                # the product is of the ink's own colour (or a black digit
                # over black letters): colour cannot tell them apart. The
                # ink on and just around the template's glyph shapes, away
                # from the shape's own body, is rebuilt by the model; the
                # rest is kept
                # (one pixel of margin: letters right against the product are
                # the model's to rebuild, or they stay as stray marks)
                body = cv2.dilate((outside_dark | reach_body).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
                hole = comp & zone & (cv2.dilate(glyph_w.astype(np.uint8), k7) > 0) & (gray_w < fill_gray - 1) & ~body
                prot_w |= comp & ~hole
                touch_w |= hole
                prod_w |= outside_dark | reach_body
                word_touched = word_touched or int((comp & zone & ~body).sum()) > 40
                continue
            # a product of another colour over the word: its own-coloured
            # pixels and anything within 3px of them are product, letter-
            # coloured pixels away from them are letters
            core = comp & ~letter_like & (nonpaper > 0)
            # stray ink pixels off the ink line (JPEG colour noise) are not
            # product: only patches of some size count (a pave piece's thin
            # dark gaps between stones are long, so they survive)
            nc, clab, cst, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
            keep_c = np.zeros(nc, bool)
            keep_c[1:] = cst[1:, 4] >= 6
            # ... and only when joined to the product's body: a dark patch on
            # a glyph with no product next to it is a letter's own dark core
            # (a lightly printed word's strokes run darker than its ink
            # estimate in places)
            at_body = cv2.dilate((outside | reach_body).astype(np.uint8), k5) > 0
            touching = np.zeros(nc, bool)
            touching[np.unique(clab[at_body & core])] = True
            keep_c &= touching
            core = keep_c[clab]
            near = cv2.dilate(core.astype(np.uint8), k7) > 0
            # pixels of the ink's colour at least half as dark as the ink are
            # letters even near the product; right against it (within 3px of
            # its own-coloured pixels) they are left to the model instead,
            # since on a lightly printed word a product's half-tones pass
            # for ink
            ink_sure = comp & zone & letter_like & (tt > 0.5) & (nonpaper > 0) & ~core & ~near & ~reach_body
            # ink right against the product takes on some of its colour in
            # the JPEG: dark pixels roughly on the ink line, on the glyphs,
            # are rebuilt by the model rather than kept
            loose = (resid <= np.maximum(60.0, 0.35 * dn)) & (tt > 0.5) & (tt <= 2.2) & (nonpaper > 0)
            band = np.zeros(blobs.shape, bool)
            band[max(0, y - 2 - cy0):min(blobs.shape[0], y + th + 2 - cy0), :] = True
            glyph9 = cv2.dilate(glyph_w.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
            mend = comp & zone & band & glyph9 & loose & near & ~ink_sure & ~reach_body
            touch_w |= mend
            prot_w |= (outside | near | reach_body) & ~ink_sure & ~mend
            prod_w |= (outside & (dark_w > 0)) | core | reach_body
            word_touched = word_touched or int(((near | reach_body) & zone & band & glyph9).sum()) > 40
        # the model is only needed where the letters meet the product: away
        # from it the paper fill is exact, and the model's paper comes out
        # faintly mottled
        near_prod = cv2.dilate(prod_w.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        nt, tlab = cv2.connectedComponents(touch_w.astype(np.uint8), connectivity=8)
        keep_t = np.zeros(nt, bool)
        keep_t[np.unique(tlab[near_prod & touch_w])] = True
        keep_t[0] = False
        touch_w &= keep_t[tlab]
        protect_wide = prot_w
        protect = protect_wide[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]
        touch = (touch_w[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] & ~protect).astype(np.uint8) * 255
        box_bgr = padded[y0:y1, x0:x1]
        # everything darker than the paper near the letters goes: the ink,
        # its soft edges and the JPEG ringing around it
        # the ink, and every pixel within 3px of its core (its soft halo,
        # whatever the pixel's own level), so the filled band carries the
        # paper's mean tone rather than the mean of its darker half
        ink_core = ((box_g < fill_gray - off_core) & ~protect & (tpl_ink > 0)).astype(np.uint8)
        near_ink = cv2.dilate(ink_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        gm = ((box_g < ink_thr) | near_ink) & ~protect & (tpl_ink > 0)
        if name.startswith("left") or name.startswith("right"):
            # a partial that is not at the frame edge may show more letters
            # than its template: fill ink of the same colour as the matched
            # letters along the rest of the word's extent
            ftw = round(TW * sc)
            if name.startswith("left"):
                ex0, ex1 = x1, min(out.shape[1], x + ftw + margin)
            else:
                ex0, ex1 = max(0, x + tw - ftw - margin), x0
            if ex1 > ex0 and letter_colour is not None:
                ext_g = gray[y0:y1, ex0:ex1]
                close = letter_like[y0 - cy0:y1 - cy0, ex0 - cx0:ex1 - cx0] & (ext_g < fill_gray - max(20.0, 3.0 * paper_sigma))
                close = cv2.dilate(close.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
                close &= ext_g < fill_gray - max(5.0, paper_sigma)
                close &= ~protect_wide[y0 - cy0:y1 - cy0, ex0 - cx0:ex1 - cx0]
                # grow the working box to cover the extension
                nx0, nx1 = min(x0, ex0), max(x1, ex1)
                gm_new = np.zeros((bh, nx1 - nx0), bool)
                gm_new[:, x0 - nx0:x1 - nx0] = gm
                gm_new[:, ex0 - nx0:ex1 - nx0] |= close
                pr_new = protect_wide[y0 - cy0:y1 - cy0, nx0 - cx0:nx1 - cx0].copy()
                # what the model must rebuild in the extension too (a
                # product over the letters beyond the partial template)
                tch_new = (touch_w[y0 - cy0:y1 - cy0, nx0 - cx0:nx1 - cx0] & ~pr_new).astype(np.uint8) * 255
                ti_new = np.zeros((bh, nx1 - nx0), np.uint8)
                ti_new[:, x0 - nx0:x1 - nx0] = tpl_ink
                ti_new[:, ex0 - nx0:ex1 - nx0] = 255
                x0, x1, gm, protect, touch, tpl_ink = nx0, nx1, gm_new, pr_new, tch_new, ti_new
                box_g = gray[y0:y1, x0:x1]
                rx0, rx1 = max(0, x0 - 6), min(out.shape[1], x1 + 6)
        # the paper under the word: the local average of the paper pixels
        # around each glyph, so the fill follows the ruler's own shading
        reg = padded[ry0:ry1, rx0:rx1].astype(np.float32)
        reg_g = gray[ry0:ry1, rx0:rx1]
        gm_reg = np.zeros(reg_g.shape, bool)
        gm_reg[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = gm
        inside = np.zeros(reg_g.shape, bool)
        inside[max(0, pad - ry0):max(0, H + pad - ry0), max(0, pad - rx0):max(0, W + pad - rx0)] = True
        # clean paper only: not the letters' halo, which would pull the fill
        # a few levels down into a faint ghost of the word
        halo = cv2.dilate(gm_reg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        # (the paper's own darker grain counts, or the fill comes out a
        # shade lighter than the paper around it and the word shows as a
        # faint light ghost)
        wgt = ((reg_g > fill_gray - off_core) & ~halo & inside).astype(np.float32)
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
        pick = 0.6 * np.clip(resid[rng.integers(0, len(resid), size=len(gy))], -4, 4)
        sub_local = local[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]
        box = out[y0:y1, x0:x1]
        box[gy, gx] = np.clip(sub_local[gy, gx] + pick, 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1] = box
        touch = cv2.bitwise_and(touch, (tpl_ink > 0).astype(np.uint8) * 255)
        if touch.any():
            full = np.zeros(out.shape[:2], np.uint8)
            full[y0:y1, x0:x1] = touch
            out = rebuild(out, full, grow=1)
        touched_words.append(word_touched)
        if _RULER_DBG is not None:
            _RULER_DBG.append(dict(name=name, sc=sc, score=score, x=x - pad, y=y - pad, th=th, tw=tw,
                                   box=(x0 - pad, y0 - pad, x1 - pad, y1 - pad), win=(cx0 - pad, cy0 - pad, cx1 - pad, cy1 - pad),
                                   zone=zone, lab=lab, in_frac=in_frac, prot=prot_w, touch=touch_w, gm=gm,
                                   letter_like=letter_like, nonpaper=nonpaper > 0, glyph=glyph_w, shapes=shape_log,
                                   letter_colour=letter_colour, fill=fill, fill_gray=fill_gray, word_touched=word_touched))
    out = out[pad:-pad, pad:-pad]
    # each hit: (x, y, score, product_over_word)
    return out, [(x - pad, y - pad, round(score, 2), bool(t)) for (x, y, sc, score, th, tw, name, part), t in zip(kept, touched_words)]


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
        ruler_words = precomputed.get("ruler_words", 0) > 0
    else:
        logo_hits = find_logo(bgr, logo_tpl)                 # detect on the untouched image
        logo_b = find_logo_b(bgr, logo_b_tpl, lettering_tpl) if logo_b_tpl is not None else None
        ruler_words = True
    # on a frame carrying the grey stamp, a weak match of the coloured stamp
    # is the grey stamp's own lettering (the same face) or a piece of the
    # product: only a strong match counts there
    if logo_hits and logo_b is not None and logo_b[3] < 0.8:
        logo_b = None
    # the ruler words are found on the untouched image (a stamp over a word
    # is removed first, and what the model leaves of the word under it may
    # no longer match the template) and again once the stamps are gone (a
    # stamp over a word can hide it from the first search); an earlier
    # run's count is not trusted for this
    word_hits = remove_ruler_text(bgr, text_tpl, detect_only=True)
    if logo_hits:
        # the grey stamp's own small "TRAXNYC" matches the word template:
        # a word lying over the stamp's box is the stamp (removed with it)
        th_a, tw_a = logo_tpl.shape
        def over_stamp(h):
            x, y, th, tw = h[0] - 120, h[1] - 120, h[4], h[5]
            if h[2] >= 0.7:
                return False                  # a ruler word under the stamp is larger
            for ax, ay, _ in logo_hits:
                ix = max(0, min(x + tw, ax + tw_a) - max(x, ax)); iy = max(0, min(y + th, ay + th_a) - max(y, ay))
                if ix * iy >= 0.3 * th * tw:
                    return True
            return False
        word_hits = [h for h in word_hits if not over_stamp(h)]
    overlap, product_px = 0.0, 0
    # the stamps go first: a stamp lying over a ruler word would otherwise be
    # taken for a product over the word (its strokes kept, the word under
    # them rebuilt around them, and the stamp's own removal then left a blob)
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
    text_hits = []
    if logo_hits or logo_b is not None:
        more = remove_ruler_text(bgr, text_tpl, detect_only=True)
        for h in more:
            if all(abs(h[0] - k[0]) > 20 or abs(h[1] - k[1]) > 10 for k in word_hits):
                word_hits.append(h)
    if word_hits:
        bgr, text_hits = remove_ruler_text(bgr, text_tpl, hits=word_hits)
    clean = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    final = clean if no_upscale else upscale(clean, size)
    info = {"logo_hits": logo_hits, "text_hits": text_hits, "logo_b": logo_b,
            "overlap": round(overlap, 3), "product_px": product_px,
            "ruler_product": any(len(h) > 3 and h[3] for h in text_hits)}
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
        tx = ", ".join(f"({h[0]},{h[1]}) {h[2]:.2f}" for h in text_hits) or "none"
        lb = info["logo_b"]
        lb = f"({lb[0]},{lb[1]}) x{lb[2]:.2f} {lb[3]:.2f}" if lb else "none"
        print(f"{rel}: {pil.size[0]}x{pil.size[1]} -> {final.size[0]}x{final.size[1]} | stamp A: {lg} | stamp B: {lb} | ruler text: {tx} | overlap {info['overlap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
