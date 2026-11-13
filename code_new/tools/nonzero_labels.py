#!/usr/bin/env python3
"""
Find label files whose glaucoma or atrophy value is not 0.0.

Each .csv looks like:
    glaucoma,atrophy,valid_image
    0.0,0.0,1.0

Usage:
  python find_nonzero_labels.py FOLDER
  python find_nonzero_labels.py FOLDER -o to_check.txt

The output .txt uses the same format as the list for move_listed_files.py
(e.g. "360_right_0.png"), so you can feed it straight into that script.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

COLUMNS = ["glaucoma", "atrophy"]   # columns that must be 0.0 to be ignored


def natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def check_file(path: Path):
    """Return {column: value} for non-zero columns, or None if file is fine."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("no data rows")
    hits = {}
    for row in rows:
        for col in COLUMNS:
            if col not in row:
                raise ValueError(f"missing column '{col}'")
            value = float(row[col])
            if value != 0.0:
                hits[col] = value
    return hits or None


def main():
    parser = argparse.ArgumentParser(description="List files with non-zero glaucoma/atrophy.")
    parser.add_argument("folder", type=Path, help="folder with the .csv label files")
    parser.add_argument("-o", "--output", type=Path, help="save the names to this .txt file")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"Folder not found: {args.folder}")

    found = []
    files = sorted(args.folder.glob("*.csv"), key=lambda p: natural_key(p.name))
    for path in files:
        try:
            hits = check_file(path)
        except Exception as e:
            print(f"ERROR  {path.name}: {e}")
            continue
        if hits:
            name = path.name[:-4]            # "360_right_0.png.csv" -> "360_right_0.png"
            found.append(name)
            details = ", ".join(f"{k}={v}" for k, v in hits.items())
            print(f"{name}   ({details})")

    print(f"\n{len(found)} of {len(files)} files have non-zero {' / '.join(COLUMNS)}")

    if args.output:
        args.output.write_text("\n".join(found) + ("\n" if found else ""), encoding="utf-8")
        print(f"Saved list to {args.output}")


if __name__ == "__main__":
    main()