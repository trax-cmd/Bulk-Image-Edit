#!/usr/bin/env python3
"""Run the branding-removal + upscale pipeline over a whole catalogue export.

Usage: python3 scripts/batch.py <input_root> <output_root> [--workers N] [--report report.csv]
       [--only "10570 - 14K Gold Milano Cross Crucifix 3"] [--limit N]

Input layout: <input_root>/<item folder>/<image files>. Output mirrors it as
<output_root>/<item folder>/<name>.jpg (always 1200x1200 JPEG). Every file
gets a row in the report CSV with what was detected, plus a `flags` column
naming anything a human should spot-check:
  no_logo         no corner stamp found (fine if the source had none)
  weak_logo       stamp accepted on position only or with a low score
  odd_size        source is not the usual 500x380 catalogue frame
  ruler_text      TraxNYC words removed from a ruler shot (count in column)
  error           the file could not be processed (message in `error`)
Already-finished outputs are skipped, so the run can be resumed.
"""
import argparse
import csv
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process import EXTS, load_template, process_image, save_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_TPL = {}


def _templates():
    if not _TPL:
        _TPL["logo"] = load_template(ROOT / "assets/trax_logo_template.png")
        _TPL["text"] = load_template(ROOT / "assets/trax_ruler_text_template.png")
    return _TPL["logo"], _TPL["text"]


def work(args):
    src, dst = args
    row = {"item": src.parent.name, "file": src.name, "output": str(dst), "width": "", "height": "",
           "logo_x": "", "logo_y": "", "logo_score": "", "ruler_words": 0, "flags": "", "error": ""}
    try:
        logo_tpl, text_tpl = _templates()
        orig, final, logo_hits, text_hits = process_image(src, logo_tpl, text_tpl, 1200)
        save_image(final, dst, "jpg")
        row["width"], row["height"] = orig.size
        flags = []
        if logo_hits:
            x, y, s = max(logo_hits, key=lambda h: h[2])
            row["logo_x"], row["logo_y"], row["logo_score"] = x, y, round(s, 3)
            if s < 0.5:
                flags.append("weak_logo")
        else:
            flags.append("no_logo")
        if orig.size != (500, 380):
            flags.append("odd_size")
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
    a = ap.parse_args()

    in_root, out_root = Path(a.input_root), Path(a.output_root)
    report = Path(a.report) if a.report else out_root / "report.csv"
    jobs = []
    for src in sorted(in_root.rglob("*")):
        if src.suffix.lower() not in EXTS or not src.is_file():
            continue
        if a.only and src.parent.name != a.only:
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

    fields = ["item", "file", "output", "width", "height", "logo_x", "logo_y", "logo_score",
              "ruler_words", "flags", "error"]
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
