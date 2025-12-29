#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image, ImageDraw
import os
import sys


def parse_bbox_row(row: list) -> Tuple[str, str, float, float, float, float]:
    """
    Expected CSV row format:
    filename,label,x_or_cx,y_or_cy,width,height
    Example: 00010815_006.png,Mass,311.1822,241.5313,146.7733,256
    """
    filename = row[0].strip()
    label = row[1].strip()
    x = float(row[2])
    y = float(row[3])
    w = float(row[4])
    h = float(row[5])
    return filename, label, x, y, w, h


def to_box(x: float, y: float, w: float, h: float, fmt: str) -> Tuple[int, int, int, int]:
    """
    Convert coordinates to a PIL box (left, top, right, bottom).
    fmt: 'tlwh' for top-left (x,y,width,height), 'cwh' for center (cx,cy,width,height)
    """
    if fmt == 'tlwh':
        left = x
        top = y
        right = x + w
        bottom = y + h
    elif fmt == 'cwh':
        left = x - w / 2.0
        top = y - h / 2.0
        right = x + w / 2.0
        bottom = y + h / 2.0
    else:
        raise ValueError("Unknown fmt. Use 'tlwh' or 'cwh'.")
    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def draw_bbox(img_path: Path, box: Tuple[int, int, int, int], label: Optional[str], out_path: Path, color: str = 'red', width: int = 3):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=color, width=width)
    if label:
        # Draw a small label above the box
        x0, y0, x1, y1 = box
        text_pos = (x0 + 4, max(0, y0 - 18))
        draw.text(text_pos, label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding box on an image.')
    parser.add_argument('--image', type=str, help='Path to a single image file (PNG/JPG). Optional in CSV batch mode.')
    parser.add_argument('--coords', type=float, nargs=4, metavar=('X', 'Y', 'W', 'H'), help='Coords as x y w h (or cx cy w h).')
    parser.add_argument('--label', type=str, default='Mass', help='Optional label to draw.')
    parser.add_argument('--format', type=str, choices=['tlwh', 'cwh'], default='tlwh', help='Coordinate format: top-left (tlwh) or center (cwh).')
    parser.add_argument('--csv', type=str, help='Optional CSV file to read bbox from (like BBox_List_2017.csv).')
    parser.add_argument('--csv-filename', type=str, help='Filename key to pick the row (e.g., 00010815_006.png). If omitted, all rows will be processed.')
    parser.add_argument('--image-root', type=str, help='Root folder to search images by filename when running on CSV batch.')
    parser.add_argument('--color', type=str, default='red', help='Rectangle color.')
    parser.add_argument('--width', type=int, default=3, help='Rectangle line width.')
    parser.add_argument('--output', type=str, help='Output image path. Default: <image> with _bbox suffix.')
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs when processing CSV batch.')
    parser.add_argument('--only-label', type=str, default=None, help='When processing CSV, only visualize rows with this label (e.g., Mass).')

    args = parser.parse_args()

    # Single image mode
    if args.image and args.coords and not args.csv:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        label = args.label
        x, y, w, h = args.coords
        box = to_box(x, y, w, h, args.format)
        out_path = Path(args.output) if args.output else img_path.with_name(img_path.stem + '_bbox.png')
        draw_bbox(img_path, box, label, out_path, color=args.color, width=args.width)
        print(f"Saved: {out_path}")
        return

    # CSV batch or single-by-csv mode
    if not args.csv:
        raise ValueError('For batch mode, provide --csv. For single mode, provide --image and --coords.')

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Helper to locate image by filename
    def find_image(filename: str, explicit_image: Optional[str], image_root: Optional[str]) -> Optional[Path]:
        if explicit_image:
            p = Path(explicit_image)
            return p if p.exists() else None
        if image_root:
            root = Path(image_root)
            # search recursively for filename
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    return Path(dirpath) / filename
            return None
        return None

    processed = 0
    target_label = args.only_label
    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                filename, lbl, x, y, w, h = parse_bbox_row(row)
            except Exception:
                continue

            if args.csv_filename and filename != args.csv_filename:
                continue
            if target_label and lbl != target_label:
                continue

            img_path = find_image(filename, args.image, args.image_root)
            if not img_path:
                print(f"WARN: image not found for {filename}", file=sys.stderr)
                continue

            box = to_box(x, y, w, h, args.format)
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_path = out_dir / (Path(filename).stem + '_bbox.png')
            else:
                out_path = img_path.with_name(Path(filename).stem + '_bbox.png')
            try:
                draw_bbox(img_path, box, lbl, out_path, color=args.color, width=args.width)
                print(f"Saved: {out_path}")
                processed += 1
            except Exception as e:
                print(f"ERROR: failed {filename}: {e}", file=sys.stderr)

    print(f"Done. Processed: {processed}")


if __name__ == '__main__':
    main()
