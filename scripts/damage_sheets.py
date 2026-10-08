#!/usr/bin/env python3
"""Contact sheets of the images an earlier run damaged: for each row of a
damage_list.py CSV with the wanted verdict, a zoomed crop around the damaged
area showing ORIGINAL | EARLIER OUTPUT | CORRECTED OUTPUT.

Usage: damage_sheets.py <damage.csv> <input_root> <sheet_dir> [--verdict damaged]
       [--per-sheet 4] [--limit N] [--sort px]
Writes sheet_NNNN.jpg and cells.csv (sheet, cell, item, file, bbox).
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("damage")
    ap.add_argument("input_root")
    ap.add_argument("sheet_dir")
    ap.add_argument("--verdict", default="damaged")
    ap.add_argument("--per-sheet", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sort", default="px", choices=["px", "name"])
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.damage)) if r["verdict"] == a.verdict and r["bbox"]]
    if a.sort == "px":
        rows.sort(key=lambda r: -int(r["changed_product_px"]))
    else:
        rows.sort(key=lambda r: (r["item"], r["file"]))
    if a.limit:
        rows = rows[: a.limit]
    out_dir = Path(a.sheet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile = 400
    index = []
    sheet_no = 0
    for s in range(0, len(rows), a.per_sheet):
        group = rows[s:s + a.per_sheet]
        sheet_no += 1
        sh = Image.new("RGB", (3 * tile + 40, len(group) * (tile + 34) + 10), (235, 235, 235))
        d = ImageDraw.Draw(sh)
        for k, r in enumerate(group):
            orig = Image.open(Path(a.input_root) / r["item"] / r["file"]).convert("RGB")
            W, H = orig.size
            sc = 1200 / max(W, H)
            ox, oy = (1200 - round(W * sc)) // 2, (1200 - round(H * sc)) // 2
            old = Image.open(r["old_output"]).convert("RGB")
            new = Image.open(r["new_output"]).convert("RGB")
            x0, y0, x1, y1 = map(int, r["bbox"].split(","))
            # square crop around the damaged area, at least 260px, with margin
            side = max(260, int(max(x1 - x0, y1 - y0) * 1.6))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            bx0, by0 = max(0, min(1200 - side, cx - side // 2)), max(0, min(1200 - side, cy - side // 2))
            box = (bx0, by0, bx0 + side, by0 + side)
            # the same area in the original's own pixels
            obox = ((bx0 - ox) / sc, (by0 - oy) / sc, (bx0 + side - ox) / sc, (by0 + side - oy) / sc)
            o_tile = Image.new("RGB", (side, side), (255, 255, 255))
            cx0, cy0 = max(0, int(obox[0])), max(0, int(obox[1]))
            cx1, cy1 = min(W, int(obox[2])), min(H, int(obox[3]))
            if cx1 > cx0 and cy1 > cy0:
                part = orig.crop((cx0, cy0, cx1, cy1)).resize((round((cx1 - cx0) * sc), round((cy1 - cy0) * sc)), Image.LANCZOS)
                o_tile.paste(part, (round(cx0 * sc + ox - bx0), round(cy0 * sc + oy - by0)))
            tiles = [o_tile, old.crop(box), new.crop(box)]
            y = 10 + k * (tile + 34)
            d.rectangle((10, y, 3 * tile + 30, y + 22), fill=(40, 40, 40))
            label = f"CELL {k + 1}   ORIGINAL | EARLIER OUTPUT | CORRECTED      {r['item'][:48]} / {r['file'][:40]}"
            d.text((14, y + 4), label, fill=(255, 255, 255))
            for j, t in enumerate(tiles):
                sh.paste(t.resize((tile, tile), Image.LANCZOS), (10 + j * (tile + 10), y + 24))
            index.append({"sheet": f"sheet_{sheet_no:04d}.jpg", "cell": k + 1, "item": r["item"], "file": r["file"],
                          "bbox": r["bbox"], "changed_product_px": r["changed_product_px"]})
        sh.save(out_dir / f"sheet_{sheet_no:04d}.jpg", quality=90)
    with open(out_dir / "cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "cell", "item", "file", "bbox", "changed_product_px"])
        w.writeheader(); w.writerows(index)
    print(f"{sheet_no} sheets, {len(index)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
