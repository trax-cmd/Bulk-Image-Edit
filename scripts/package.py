#!/usr/bin/env python3
"""Package finished images for hand-off.

Usage: python3 scripts/package.py <output_root> <manifest_images.csv> <dest_dir> [--chunk-mb 450]

Writes:
  <dest_dir>/images_part01.zip, part02, ...   item folders, each zip <= chunk size
  <dest_dir>/image_manifest.csv               one row per output image, joined to the
                                              Shopify product id / position from the
                                              Drive manifest, plus the review flags
"""
import argparse
import csv
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("manifest")
    ap.add_argument("dest")
    ap.add_argument("--report", default=None, help="report.csv from batch.py (for flags)")
    ap.add_argument("--chunk-mb", type=int, default=450)
    a = ap.parse_args()
    out_root, dest = Path(a.output_root), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)

    flags = {}
    if a.report:
        for r in csv.DictReader(open(a.report)):
            flags[(r["item"], Path(r["file"]).stem)] = (r["flags"], r["overlap"])

    rows = []
    for m in csv.DictReader(open(a.manifest, encoding="utf-8-sig")):
        folder, fname = m["file"].split("/", 1)
        out = out_root / folder / (Path(fname).stem + ".jpg")
        fl, ov = flags.get((folder, Path(fname).stem), ("", ""))
        rows.append({"item_number": m["item_number"], "shopify_product_id": m["shopify_product_id"],
                     "position": int(m["position"]), "folder": folder, "output_file": out.name,
                     "exists": out.exists(), "source_width": m["width"], "source_height": m["height"],
                     "flags": fl, "overlap": ov, "old_source_url": m["source_url"]})
    def item_key(r):
        n = r["item_number"]
        return (0, int(n)) if n.isdigit() else (1, n)
    rows.sort(key=lambda r: (item_key(r), r["position"]))
    with open(dest / "image_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    missing = [r for r in rows if not r["exists"]]
    print(f"manifest: {len(rows)} rows, {len(missing)} missing outputs")

    limit = a.chunk_mb * 1024 * 1024
    part, size, zf = 0, 0, None
    folders = sorted(p for p in out_root.iterdir() if p.is_dir())
    for folder in folders:
        files = sorted(folder.glob("*.jpg"))
        fsize = sum(f.stat().st_size for f in files)
        if zf is None or size + fsize > limit:
            if zf:
                zf.close()
            part += 1
            zf = zipfile.ZipFile(dest / f"images_part{part:02d}.zip", "w", zipfile.ZIP_STORED)
            size = 0
        for f in files:
            zf.write(f, f"{folder.name}/{f.name}")
        size += fsize
    if zf:
        zf.close()
    print(f"{part} zip parts in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
