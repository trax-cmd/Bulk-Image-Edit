#!/usr/bin/env python3
"""List images that an earlier run damaged, by comparing its output with the
corrected output wherever the ORIGINAL photo shows product (not background).

Usage: damage_list.py <report_new.csv> <input_root> <old_output_root> <out.csv> [--limit N]

For each image: the two 1200x1200 outputs are compared on the pixels where
both the original (upscaled the same way) and the corrected output show
product (min channel < 215, and not within 2px of background). A pixel counts as changed when the two outputs
differ by more than 24 levels in any channel. Output columns:
  changed_product_px  number of such pixels in the old output
  changed_share       changed_product_px / product pixels in the stamp region
  bbox                x0,y0,x1,y1 of the changed product pixels (1200px output coordinates)
  verdict             damaged (>= 1000 px) / touched (200-999) / clean
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("input_root")
    ap.add_argument("old_root")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.report)) if not r["error"] and (r["stamp_a"] or r["stamp_b"])]
    if a.limit:
        rows = rows[: a.limit]
    out_rows = []
    for r in rows:
        src = Path(a.input_root) / r["item"] / r["file"]
        new_p = Path(r["output"])
        old_p = Path(a.old_root) / r["item"] / (src.stem + ".jpg")
        if not (new_p.exists() and old_p.exists()):
            continue
        orig = cv2.imread(str(src))
        new = cv2.imread(str(new_p))
        old = cv2.imread(str(old_p))
        if orig is None or new is None or old is None:
            continue
        H, W = orig.shape[:2]
        s = 1200 / max(W, H)
        iw, ih = round(W * s), round(H * s)
        ox, oy = (1200 - iw) // 2, (1200 - ih) // 2
        up = cv2.resize(orig, (iw, ih), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((1200, 1200, 3), 255, np.uint8)
        canvas[oy:oy + ih, ox:ox + iw] = up
        # product: darker than paper both in the original and in the corrected
        # output (so the stamp's own dark pixels over plain background do not
        # count), away from the product's edge
        product = ((canvas.min(axis=2) < 215) & (new.min(axis=2) < 215)).astype(np.uint8)
        product = cv2.erode(product, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        diff = np.abs(old.astype(np.int16) - new.astype(np.int16)).max(axis=2) > 24
        changed = diff & product
        n = int(changed.sum())
        # restrict denominator to the neighbourhood of changes (stamp region)
        region = cv2.dilate(changed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
        denom = int((product & region).sum())
        share = round(n / denom, 3) if denom else 0.0
        verdict = "damaged" if n >= 1000 else ("touched" if n >= 200 else "clean")
        bbox = ""
        if n:
            ys, xs = np.where(changed)
            bbox = f"{int(xs.min())},{int(ys.min())},{int(xs.max()) + 1},{int(ys.max()) + 1}"
        out_rows.append({"item": r["item"], "file": r["file"], "old_output": str(old_p), "new_output": str(new_p),
                         "changed_product_px": n, "changed_share": share, "bbox": bbox,
                         "verdict": verdict, "product_px": r["product_px"], "flags": r["flags"]})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    c = {}
    for o in out_rows:
        c[o["verdict"]] = c.get(o["verdict"], 0) + 1
    print(f"{len(out_rows)} compared: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
