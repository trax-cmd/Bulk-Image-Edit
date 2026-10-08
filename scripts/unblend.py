"""Undo a translucent stamp instead of rebuilding what is under it.

The stamp was composited onto the photo: observed = (1-a)*product + a*colour.
The template (the same stamp on plain white) gives a per-pixel alpha once a
stamp colour is assumed: on white, T = (1-a)*255 + a*colour, so
a = (255 - T) / (255 - colour). The product is then
product = (observed - a*colour) / (1 - a), which is reliable where a is small
or moderate. Opaque parts (a high, the maroon eye, the lettering and its
glow) cannot be undone and are left for the inpainter.
"""
import cv2
import numpy as np


def alpha_from_template(tpl_gray: np.ndarray, colour: float) -> np.ndarray:
    a = (255.0 - tpl_gray.astype(np.float32)) / max(1.0, 255.0 - colour)
    return np.clip(a, 0.0, 1.0)


def place(shape, tpl, x, y, scale, interp=cv2.INTER_LINEAR, fill=0.0):
    """Resize `tpl` by `scale` and paste it at (x, y) into a float canvas of
    `shape` (2-D), clipping at the frame edges. Returns the canvas and the
    canvas-space slice of the pasted region."""
    th, tw = max(1, round(tpl.shape[0] * scale)), max(1, round(tpl.shape[1] * scale))
    t = cv2.resize(tpl.astype(np.float32), (tw, th), interpolation=interp)
    canvas = np.full(shape[:2], fill, np.float32)
    sx, sy = max(0, -x), max(0, -y)
    px, py = max(0, x), max(0, y)
    h, w = min(th - sy, shape[0] - py), min(tw - sx, shape[1] - px)
    if h > 0 and w > 0:
        canvas[py:py + h, px:px + w] = t[sy:sy + h, sx:sx + w]
    return canvas


def unblend(bgr: np.ndarray, tpl_bgr: np.ndarray, x: int, y: int, scale: float,
            grey_colour: float = 70.0, max_alpha: float = 0.62,
            opaque_mask_tpl: np.ndarray = None, grow_opaque: int = 4,
            denoise: bool = True):
    """Return (image with the translucent stamp undone, mask of pixels that
    still need inpainting, mask of pixels that were unblended).

    Alpha comes from the template's grey level under the assumed stamp grey.
    The stamp colour is then derived per channel from the template so that on
    plain white the undo is exact (no colour cast on background); over a
    product the correction strength depends only on `grey_colour`.
    opaque_mask_tpl: template-space mask (uint8) of parts that are opaque by
    construction (maroon eye, lettering); grown by `grow_opaque` px for the
    lettering's white glow.
    """
    tpl = tpl_bgr.astype(np.float32)
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    a_t = alpha_from_template(tpl_gray, grey_colour)
    # per-channel stamp colour so that (1-a)*255 + a*C == T exactly
    with np.errstate(divide="ignore", invalid="ignore"):
        c_t = (tpl - (1.0 - a_t)[..., None] * 255.0) / np.maximum(a_t, 1e-3)[..., None]
    c_t = np.where((a_t > 0.02)[..., None], np.clip(c_t, 0, 255), 0.0)
    a = place(bgr.shape, a_t, x, y, scale)
    c = np.stack([place(bgr.shape, c_t[..., i], x, y, scale) for i in range(3)], axis=-1)
    opaque = np.zeros(bgr.shape[:2], bool)
    if opaque_mask_tpl is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_opaque + 1, 2 * grow_opaque + 1))
        om = cv2.dilate(opaque_mask_tpl, k)
        opaque = place(bgr.shape, om, x, y, scale, interp=cv2.INTER_NEAREST) > 127
    too_dense = a > max_alpha
    inpaint = opaque | too_dense
    translucent = (a > 0.02) & ~inpaint
    img = bgr.astype(np.float32)
    aa = a[..., None]
    rec = (img - aa * c) / np.maximum(1e-3, 1.0 - aa)
    rec = np.clip(rec, 0, 255)
    if denoise:
        # the division amplifies JPEG noise in proportion to a: smooth with a
        # strength that follows a, edges preserved
        sm = cv2.bilateralFilter(rec.astype(np.uint8), 5, 30, 5).astype(np.float32)
        w = np.clip((a - 0.15) / 0.45, 0, 1)[..., None]
        rec = rec * (1 - w) + sm * w
    out = np.where(translucent[..., None], rec, img)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (inpaint.astype(np.uint8) * 255), (translucent.astype(np.uint8) * 255)


def stamp_hull(tpl_gray: np.ndarray, ink=235, close=9) -> np.ndarray:
    """Filled silhouette of the stamp in template space (uint8 0/255)."""
    m = (tpl_gray < ink).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(255 - m, connectivity=4)
    # fill enclosed holes (components of background not touching the border)
    H, W = m.shape
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if x > 0 and y > 0 and x + w < W and y + h < H:
            m[lab == i] = 255
    return m


def snap_background(out: np.ndarray, region: np.ndarray, paper: np.ndarray,
                    light=236, flat=6.0) -> np.ndarray:
    """Within `region`, set light and locally flat pixels (background with a
    faint residue) to the paper tone; textured or coloured pixels are kept."""
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    sq = cv2.blur(g * g, (5, 5))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    light_px = out.min(axis=2) > light
    snap = (region > 0) & light_px & (std < flat)
    res = out.copy()
    res[snap] = paper
    return res


def paper_tone(bgr: np.ndarray, region: np.ndarray, ring=12, light=215) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    ringpx = (cv2.dilate(region, k) > 0) & (region == 0)
    vals = bgr[ringpx]
    vals = vals[vals.min(axis=1) >= light]
    if len(vals) < 20:
        return np.array([255, 255, 255], np.uint8)
    return np.median(vals, axis=0).astype(np.uint8)


def clean_thin_residue(out: np.ndarray, region: np.ndarray, paper: np.ndarray,
                       light=180, white=236, share=0.75, win=15, max_chroma=14) -> np.ndarray:
    """Within `region`, a light, neutral pixel whose neighbourhood is mostly
    paper is a hairline of stamp residue on background: snap it to the paper
    tone. Coloured or dark pixels (product) are never touched."""
    is_white = (out.min(axis=2) > white).astype(np.float32)
    frac = cv2.blur(is_white, (win, win))
    chroma = out.max(axis=2).astype(np.int16) - out.min(axis=2).astype(np.int16)
    snap = (region > 0) & (out.min(axis=2) > light) & (frac >= share) & (chroma <= max_chroma)
    res = out.copy()
    res[snap] = paper
    return res


def flatten_near_paper(out: np.ndarray, region: np.ndarray, paper: np.ndarray,
                       tol=14, share=0.6, win=7) -> np.ndarray:
    """Within `region`, pixels within `tol` of the paper tone whose
    neighbourhood is mostly near-paper are background with faint noise or a
    seam: set them to the paper tone exactly."""
    near = (np.abs(out.astype(np.int16) - paper.astype(np.int16)) <= tol).all(axis=2)
    frac = cv2.blur(near.astype(np.float32), (win, win))
    snap = (region > 0) & near & (frac >= share)
    res = out.copy()
    res[snap] = paper
    return res
