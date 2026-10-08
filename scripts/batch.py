#!/usr/bin/env python3
"""Run the branding-removal + upscale pipeline over a whole catalogue export.

Usage: python3 scripts/batch.py <input_root> <output_root> [--workers N] [--report report.csv]
       [--only "10570 - 14K Gold Milano Cross Crucifix 3"] [--limit N]

Input layout: <input_root>/<item folder>/<image files>. Output mirrors it as
<output_root>/<item folder>/<name>.jpg (always 1200x1200 JPEG). Every file
gets a row in the report CSV with what was detected, plus a `flags` column
naming anything a human should spot-check:
  no_stamp            neither stamp found (fine if the source had none)
  weak_stamp_a/b      stamp accepted on position only or with a low score
  stamp_over_product  300+ product pixels sat next to a rebuilt area: look at it
  stamp_near_product  15-299 product pixels next to a rebuilt area
  ruler_text          TraxNYC words removed from a ruler shot (count in column)
  error               the file could not be processed (message in `error`)
Already-finished outputs are skipped, so the run can be resumed.
"""
import argparse
import csv
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
from process import EXTS, load_template, process_image, save_image, set_inpainter, lama_inpainter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_TPL = {}


_LAMA = {"on": False}
_PRE = {}
_COPY = {"root": None}


def _templates():
    if _LAMA["on"] and "inpainter" not in _TPL:
        _TPL["inpainter"] = lama_inpainter()
        set_inpainter(_TPL["inpainter"])
    if not _TPL or "logo" not in _TPL:
        _TPL["logo"] = load_template(ROOT / "assets/trax_logo_template.png")
        _TPL["text"] = load_template(ROOT / "assets/trax_ruler_text_template.png")
        _TPL["logo_b"] = cv2.imread(str(ROOT / "assets/trax_logoB_template.png"), cv2.IMREAD_COLOR)
        _TPL["lettering"] = load_template(ROOT / "assets/trax_logoB_lettering.png")
        import process as _P
        _P._LOGO_A_BGR["img"] = cv2.imread(str(ROOT / "assets/trax_logo_template.png"), cv2.IMREAD_COLOR)
    return _TPL["logo"], _TPL["text"], _TPL["logo_b"], _TPL["lettering"]


def work(args):
    src, dst = args
    row = {"item": src.parent.name, "file": src.name, "output": str(dst), "width": "", "height": "",
           "stamp_a": "", "stamp_a_score": "", "stamp_b": "", "stamp_b_scale": "", "stamp_b_score": "",
           "ruler_words": 0, "overlap": "", "product_px": "", "flags": "", "error": ""}
    try:
        pre = _PRE.get((src.parent.name, src.name))
        if pre is not None and _COPY["root"] is not None and not pre[0].get("a") and not pre[0].get("b"):
            # nothing was removed from this image apart from ruler words: unchanged
            old = _COPY["root"] / src.parent.name / (src.stem + ".jpg")
            if old.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, dst)
                row.update({k: pre[1][k] for k in pre[1] if k in row and k not in ("output",)})
                row["output"] = str(dst)
                return row
        logo_tpl, text_tpl, logo_b_tpl, lettering_tpl = _templates()
        orig, final, info = process_image(src, logo_tpl, text_tpl, 1200, logo_b_tpl=logo_b_tpl, lettering_tpl=lettering_tpl,
                                          precomputed=pre[0] if pre is not None else None)
        save_image(final, dst, "jpg")
        row["width"], row["height"] = orig.size
        flags = []
        logo_hits, text_hits, lb = info["logo_hits"], info["text_hits"], info["logo_b"]
        if logo_hits:
            x, y, s = max(logo_hits, key=lambda h: h[2])
            row["stamp_a"], row["stamp_a_score"] = f"{x},{y}", round(s, 3)
            if s < 0.5:
                flags.append("weak_stamp_a")
        if lb:
            row["stamp_b"], row["stamp_b_scale"], row["stamp_b_score"] = f"{lb[0]},{lb[1]}", lb[2], round(lb[3], 3)
            if lb[3] < 0.5:
                flags.append("weak_stamp_b")
        if not logo_hits and not lb:
            flags.append("no_stamp")
        row["overlap"] = info["overlap"]
        row["product_px"] = info["product_px"]
        if info["product_px"] >= 300:
            flags.append("stamp_over_product")
        elif info["product_px"] >= 15:
            flags.append("stamp_near_product")
        row["ruler_words"] = len(text_hits)
        if text_hits:
            flags.append("ruler_text")
        row["flags"] = " ".join(flags)
    except Exception as e:  # noqa: BLE001
        row["flags"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_root")
    ap.add_argument("output_root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", default=None)
    ap.add_argument("--only", default=None, help="process just this item folder")
    ap.add_argument("--limit", type=int, default=None, help="stop after N files")
    ap.add_argument("--lama", action="store_true", help="use the LaMa model to rebuild what was under a stamp")
    ap.add_argument("--list", default=None, help="CSV with item,file columns: process only these")
    ap.add_argument("--from-report", default=None, help="earlier report.csv: reuse its stamp/ruler detections")
    ap.add_argument("--copy-unchanged", default=None, help="earlier output root: copy files that had no stamp instead of re-processing")
    a = ap.parse_args()
    _LAMA["on"] = a.lama
    _PRE.clear()
    if a.from_report:
        for r in csv.DictReader(open(a.from_report)):
            if r["error"]:
                continue
            pre = {"ruler_words": int(r["ruler_words"] or 0)}
            if r["stamp_a"]:
                x, y = map(int, r["stamp_a"].split(","))
                pre["a"] = [(x, y, float(r["stamp_a_score"] or 0))]
            if r["stamp_b"]:
                x, y = map(int, r["stamp_b"].split(","))
                pre["b"] = (x, y, float(r["stamp_b_scale"]), float(r["stamp_b_score"] or 0))
            _PRE[(r["item"], r["file"])] = (pre, r)
    _COPY["root"] = Path(a.copy_unchanged) if a.copy_unchanged else None

    in_root, out_root = Path(a.input_root), Path(a.output_root)
    report = Path(a.report) if a.report else out_root / "report.csv"
    wanted = None
    if a.list:
        wanted = {(r["item"], r["file"]) for r in csv.DictReader(open(a.list))}
    jobs = []
    for src in sorted(in_root.rglob("*")):
        if src.suffix.lower() not in EXTS or not src.is_file():
            continue
        if a.only and src.parent.name != a.only:
            continue
        if wanted is not None and (src.parent.name, src.name) not in wanted:
            continue
        dst = out_root / src.relative_to(in_root).with_suffix(".jpg")
        if dst.exists():
            continue
        jobs.append((src, dst))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} files to process", flush=True)
    if not jobs:
        return 0

    fields = ["item", "file", "output", "width", "height", "stamp_a", "stamp_a_score", "stamp_b",
              "stamp_b_scale", "stamp_b_score", "ruler_words", "overlap", "product_px", "flags", "error"]
    new = not report.exists()
    report.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(report, "a", newline="") as fh, ProcessPoolExecutor(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for fut in as_completed(pool.submit(work, j) for j in jobs):
            w.writerow(fut.result())
            fh.flush()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
