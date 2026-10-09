"""Bridge a thin band of product (a ring's hoop, a chain, a shank) across a
hole that an opaque stamp part left in it.

The inpainting model continues a thin band only a short way into a hole and
cannot join it up across the width of the coloured stamp's lettering: the
hoop ends in a point and resumes on the far side. Here the band's two ends
at the hole's edge are found, paired by direction and width, and joined by
a smooth tube whose cross-section blends from one end's colour profile to
the other's; the model then only blends the tube's edges.
"""
import math

import cv2
import numpy as np


def _contacts(prod, mask, gray, reach=14, min_px=3):
    """Product pixels touching the hole: for each contact component, its
    centroid, the band's unit direction pointing into the hole, its width
    and its mean colour (measured on the band within `reach` px)."""
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = (cv2.dilate(mask, k5) > 0) & (mask == 0)
    contact = (ring & prod).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(contact, connectivity=8)
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    inside = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    found = []
    for i in range(1, n):
        if st[i, 4] < min_px:
            continue
        cx, cy = cen[i]
        local = prod & (mask == 0) & ((xx - cx) ** 2 + (yy - cy) ** 2 <= reach * reach)
        if local.sum() < 20:
            continue
        pts = np.stack([xx[local] - cx, yy[local] - cy], axis=1).astype(np.float64)
        cov = pts.T @ pts / len(pts)
        w_, v_ = np.linalg.eigh(cov)
        d = v_[:, 1]                       # principal axis (band direction)
        nrm = np.array([-d[1], d[0]])
        width = float(np.percentile(pts @ nrm, 97) - np.percentile(pts @ nrm, 3))
        # the band is long compared with its width along the axis
        length = float(np.percentile(pts @ d, 97) - np.percentile(pts @ d, 3))
        if length < 1.3 * width:
            continue
        # orient d into the hole: the band's own pixels lie behind the contact
        if (pts @ d).mean() > 0:
            d = -d
        px, py = int(round(cx + d[0] * 3)), int(round(cy + d[1] * 3))
        found.append({"c": np.array([cx, cy]), "d": d, "w": width, "n": len(pts), "tex": float(gray[local].std())})
    return found


def _profile(img, prod, c, d, width, back=(2, 8), samples=None):
    """Mean cross-section colour profile of the band behind a contact: at
    offsets u across the band (-width/2..width/2) the colours sampled at
    rows `back` px back from the contact along -d are averaged. Each row is
    re-centred on the band (a curved band drifts off the contact's axis)."""
    nrm = np.array([-d[1], d[0]])
    us = np.linspace(-width / 2, width / 2, samples or max(3, int(round(width * 2))))
    wide = np.linspace(-width, width, int(round(width * 4)) + 1)
    prof = np.zeros((len(us), 3), np.float64)
    cnt = 0
    rows = []
    H, W = img.shape[:2]
    pm = prod.astype(np.float32)
    for k in range(back[0], back[1] + 1):
        base = c - d * k
        xw = (base[0] + nrm[0] * wide).astype(np.float32).reshape(1, -1)
        yw = (base[1] + nrm[1] * wide).astype(np.float32).reshape(1, -1)
        if xw.min() < 0 or xw.max() > W - 1 or yw.min() < 0 or yw.max() > H - 1:
            continue
        on = cv2.remap(pm, xw, yw, cv2.INTER_NEAREST)[0] > 0.5
        if on.sum() < 2:
            continue
        shift = float(wide[on].mean())
        xs = (base[0] + nrm[0] * (us + shift)).astype(np.float32).reshape(1, -1)
        ys = (base[1] + nrm[1] * (us + shift)).astype(np.float32).reshape(1, -1)
        row = cv2.remap(img, xs, ys, cv2.INTER_LINEAR)[0].astype(np.float64)
        prof += row
        rows.append(row)
        cnt += 1
    if cnt < 3:
        return us, None, None
    rows = np.stack(rows)
    # how much the cross-section changes from row to row along the band:
    # nil for a smooth tube, large for a chain's links or a pave band
    along = float(rows.std(axis=0).mean())
    return us, prof / cnt, along


def bridge_bands(img, mask, max_width=14, max_gap=200, max_angle=40.0, prod_dark=235, max_along=32.0, debug=None):
    """Paint tubes joining the ends of thin bands cut by `mask`.
    Returns (image, mask with the tubes' cores removed, number of bridges)."""
    m = (mask > 0).astype(np.uint8)
    prod = (img.min(axis=2) < prod_dark) & (m == 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cts = _contacts(prod, m, gray)
    # only a narrow band (a polished hoop, a shank, a wire)
    cts = [c for c in cts if c["w"] <= max_width]
    out = img.copy()
    painted = np.zeros(mask.shape, np.uint8)
    used = set()
    pairs = []
    H, W = mask.shape
    hole = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    for i in range(len(cts)):
        for j in range(i + 1, len(cts)):
            a, b = cts[i], cts[j]
            v = b["c"] - a["c"]
            gap = float(np.hypot(*v))
            if gap < 6 or gap > max_gap:
                continue
            u = v / gap
            ang_a = math.degrees(math.acos(np.clip(a["d"] @ u, -1, 1)))
            ang_b = math.degrees(math.acos(np.clip(b["d"] @ (-u), -1, 1)))
            if ang_a > max_angle or ang_b > max_angle:
                continue
            if max(a["w"], b["w"]) > 1.8 * min(a["w"], b["w"]):
                continue
            # the joining curve must run through the hole
            L = gap
            p0, p3 = a["c"], b["c"]
            p1, p2 = p0 + a["d"] * (L / 3), p3 + b["d"] * (L / 3)
            ins = 0
            for s_ in range(1, 10):
                t = s_ / 10
                p = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
                px, py = int(round(p[0])), int(round(p[1]))
                ins += bool(0 <= px < W and 0 <= py < H and hole[py, px])
            if ins < 5:
                continue
            pairs.append((ang_a + ang_b + gap / 10.0, i, j))
    pairs.sort()
    n_br = 0
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        a, b = cts[i], cts[j]
        us_a, pa, va = _profile(img, prod, a["c"], a["d"], a["w"])
        us_b, pb, vb = _profile(img, prod, b["c"], b["d"], b["w"])
        if pa is None or pb is None:
            continue
        # a smooth band (its cross-section barely changes along it), with
        # real contrast (not a pale leftover of the stamp), the same colour
        # at both ends
        reason = None
        if va > max_along or vb > max_along:
            reason = "along"
        elif min(pa.min(), pb.min()) > 190:
            reason = "pale"
        elif np.abs(pa.mean(axis=0) - pb.mean(axis=0)).max() > 50:
            reason = "colour"
        if reason:
            if debug is not None:
                debug.append({"reject": reason, "along": (va, vb), "w": (a["w"], b["w"]), "dark": (float(pa.min()), float(pb.min())),
                              "cdiff": float(np.abs(pa.mean(axis=0) - pb.mean(axis=0)).max())})
            continue
        used.add(i); used.add(j)
        n_br += 1
        # cubic Bezier from a to b leaving along a's direction, arriving against b's
        L = float(np.hypot(*(b["c"] - a["c"])))
        p0, p3 = a["c"], b["c"]
        p1, p2 = p0 + a["d"] * (L / 3), p3 + b["d"] * (L / 3)
        N = max(8, int(L * 2))
        acc = np.zeros(img.shape, np.float64)
        hit = np.zeros(mask.shape, np.float64)
        ns = 32
        pa_r = np.stack([np.interp(np.linspace(0, 1, ns), np.linspace(0, 1, len(pa)), pa[:, ch]) for ch in range(3)], axis=1)
        pb_r = np.stack([np.interp(np.linspace(0, 1, ns), np.linspace(0, 1, len(pb)), pb[:, ch]) for ch in range(3)], axis=1)
        for s in range(N + 1):
            t = s / N
            p = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
            dp = 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2)
            nd = np.hypot(*dp)
            if nd < 1e-6:
                continue
            tang = dp / nd
            nrm = np.array([-tang[1], tang[0]])
            w = (1 - t) * a["w"] + t * b["w"]
            bl = t * t * (3 - 2 * t)
            prof = (1 - bl) * pa_r + bl * pb_r
            for q in range(ns):
                uq = (q / (ns - 1) - 0.5) * w
                for extra in (-0.5, 0.0, 0.5):
                    pt = p + nrm * (uq + extra)
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= px < W and 0 <= py < H:
                        acc[py, px] += prof[q]
                        hit[py, px] += 1
        got = hit > 0
        col = np.zeros(img.shape, np.float64)
        col[got] = acc[got] / hit[got][:, None]
        # only the hole itself is painted (the band outside it is real)
        tube = got & (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0)
        out[tube] = np.clip(col[tube], 0, 255).astype(np.uint8)
        painted[tube] = 255
        if debug is not None:
            debug.append({"a": a, "b": b, "L": L, "along": (va, vb)})
    if n_br == 0:
        return img, mask, 0
    core = cv2.erode(painted, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    new_mask = mask.copy()
    new_mask[core > 0] = 0
    return out, new_mask, n_br
