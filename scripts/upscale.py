#!/usr/bin/env python3
"""Upscale product images to a 1200x1200 square.

Usage: python3 scripts/upscale.py <input_dir> <output_dir> [--size 1200] [--format jpg|png]

- Keeps aspect ratio; pads to a square with the image's own border color
  (sampled from the edges, white for typical product shots).
- Resamples with Lanczos and applies a light unsharp mask so upscaled
  edges stay crisp.
- Never adds any overlay, watermark or logo.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic"}


def border_color(img: Image.Image) -> tuple:
    """Median color of the outer 2px frame, used as padding color."""
    w, h = img.size
    px = img.load()
    samples = []
    for x in range(w):
        for y in (0, 1, h - 2, h - 1):
            samples.append(px[x, y])
    for y in range(h):
        for x in (0, 1, w - 2, w - 1):
            samples.append(px[x, y])
    chans = list(zip(*samples))
    return tuple(sorted(c)[len(c) // 2] for c in chans)


def process(src: Path, dst: Path, size: int, fmt: str, quality: int) -> dict:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    orig = img.size
    if img.mode in ("RGBA", "LA", "P"):
        # Flatten transparency onto white.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    pad = border_color(img)
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    if scale > 1:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))

    canvas = Image.new("RGB", (size, size), pad)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))

    dst.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        canvas.save(dst, "PNG", optimize=True)
    else:
        canvas.save(dst, "JPEG", quality=quality, optimize=True, progressive=True,
                    subsampling=0)
    return {"src": str(src), "dst": str(dst), "orig": orig, "pad": pad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"no images found under {in_dir}", file=sys.stderr)
        return 1
    for p in files:
        rel = p.relative_to(in_dir).with_suffix("." + a.format)
        info = process(p, out_dir / rel, a.size, a.format, a.quality)
        print(f"{info['orig'][0]}x{info['orig'][1]} -> {a.size}x{a.size}  pad={info['pad']}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
