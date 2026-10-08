#!/usr/bin/env python3
"""Build QA contact sheets: for each report row, a zoomed BEFORE | AFTER crop
around every area that was rebuilt or undone (stamp footprints, ruler
words), six images per sheet, each cell numbered.

Usage: qa_sheets.py <report.csv> <input_root> <sheet_dir> [--min-product-px N]
       [--sample-others N] [--per-sheet 6] [--seed 1]
Writes <sheet_dir>/sheet_NNNN.jpg and <sheet_dir>/cells.csv (sheet, cell,
item, file, crop box) so verdicts can be mapped back to files.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("input_root")
    ap.add_argument("sheet_dir")
    ap.add_argument("--min-product-px", type=int, default=15)
    ap.add_argument("--sample-others", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    tplA = cv2.imread(str(root / "assets/trax_logo_template.png"), 0)
    tplB = cv2.imread(str(root / "assets/trax_logoB_template.png"))
    rows = list(csv.DictReader(open(a.report)))
    rows = [r for r in rows if not r["error"] and (r["stamp_a"] or r["stamp_b"] or int(r["ruler_words"] or 0) > 0)]
    hard = [r for r in rows if int(r["product_px"] or 0) >= a.min_product_px]
    others = [r for r in rows if int(r["product_px"] or 0) < a.min_product_px]
    random.seed(a.seed)
    random.shuffle(others)
    chosen = hard + others[: a.sample_others]
    random.shuffle(chosen)
    if a.limit:
        chosen = chosen[: a.limit]
    in_root, out_dir = Path(a.input_root), Path(a.sheet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 440, 440
    cols, per = 2, a.per_sheet
    rows_per = (per + cols - 1) // cols
    index = []
    sheet_no = 0
    for s in range(0, len(chosen), per):
        group = chosen[s:s + per]
        sheet_no += 1
        sh = Image.new("RGB", (cols * (2 * cell_w + 30) + 10, rows_per * (cell_h + 34) + 10), (235, 235, 235))
        d = ImageDraw.Draw(sh)
        for k, r in enumerate(group):
            src = in_root / r["item"] / r["file"]
            img = Image.open(src).convert("RGB")
            W, H = img.size
            boxes = []
            if r["stamp_b"]:
                x, y = map(int, r["stamp_b"].split(","))
                sc = float(r["stamp_b_scale"])
                boxes.append((x + int(70 * sc), y + int(45 * sc), x + round(tplB.shape[1] * sc) + 12, y + round(tplB.shape[0] * sc) + 14))
            if r["stamp_a"]:
                x, y = map(int, r["stamp_a"].split(","))
                boxes.append((x - 20, y - 20, x + tplA.shape[1] + 20, y + tplA.shape[0] + 20))
            if not boxes:
                boxes.append((0, int(H * 0.55), W, H))
            x0 = max(0, min(b[0] for b in boxes)); y0 = max(0, min(b[1] for b in boxes))
            x1 = min(W, max(b[2] for b in boxes)); y1 = min(H, max(b[3] for b in boxes))
            # square-ish crop, at least 120px, centred on the box
            side = max(120, x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            x1, y1 = min(W, x0 + side), min(H, y0 + side)
            x0, y0 = max(0, x1 - side), max(0, y1 - side)
            before = img.crop((x0, y0, x1, y1))
            out = Image.open(r["output"]).convert("RGB")
            s1200 = 1200 / max(W, H); iw, ih = round(W * s1200), round(H * s1200); ox, oy = (1200 - iw) // 2, (1200 - ih) // 2
            after = out.crop((ox + round(x0 * s1200), oy + round(y0 * s1200), ox + round(x1 * s1200), oy + round(y1 * s1200)))
            X = (k % cols) * (2 * cell_w + 30) + 10; Y = (k // cols) * (cell_h + 34) + 10
            sh.paste(before.resize((cell_w, cell_h), Image.LANCZOS), (X, Y + 24))
            sh.paste(after.resize((cell_w, cell_h), Image.LANCZOS), (X + cell_w + 10, Y + 24))
            d.rectangle((X, Y + 2, X + 2 * cell_w + 10, Y + 22), fill=(60, 60, 60))
            d.text((X + 4, Y + 6), f"CELL {k + 1}   BEFORE (left)  |  AFTER (right)", fill=(255, 255, 255))
            index.append({"sheet": f"sheet_{sheet_no:04d}.jpg", "cell": k + 1, "item": r["item"], "file": r["file"],
                          "output": r["output"], "crop": f"{x0},{y0},{x1},{y1}", "product_px": r["product_px"], "flags": r["flags"]})
        sh.save(out_dir / f"sheet_{sheet_no:04d}.jpg", quality=88)
    with open(out_dir / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader(); w.writerows(index)
    print(f"{sheet_no} sheets, {len(index)} cells ({len(hard)} with product next to a rebuilt area, {min(len(others), a.sample_others)} sampled others)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
