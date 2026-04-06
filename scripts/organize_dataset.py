#!/usr/bin/env python3
"""Organize galaxy images into class folders expected by train_galaxy_classifier.py.

Input CSV format:
filename,label
123.jpg,spiral
456.jpg,elliptical
...
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

VALID_LABELS = {"spiral", "elliptical", "irregular"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize galaxy dataset into class folders.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory with source images")
    parser.add_argument("--labels-csv", type=Path, required=True, help="CSV with columns: filename,label")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output class-folder root")
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move files into class folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(VALID_LABELS):
        (args.output_dir / label).mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    invalid = 0

    with args.labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"filename", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain columns: filename,label")

        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip().lower()

            if label not in VALID_LABELS:
                invalid += 1
                continue

            src = args.images_dir / filename
            if not src.exists():
                missing += 1
                continue

            dst = args.output_dir / label / src.name
            if args.mode == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            copied += 1

    print(f"Done. Processed: {copied}, missing files: {missing}, invalid labels: {invalid}")


if __name__ == "__main__":
    main()
