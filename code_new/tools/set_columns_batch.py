#!/usr/bin/env python3
"""
Set valid_image in specific label files to values from a dictionary.
Files not in the dictionary are left untouched.

Edit UPDATES below, then run:
  python set_valid_image.py FOLDER
  python set_valid_image.py FOLDER --dry-run     (show changes, write nothing)

Keys can be written with or without ".csv":
  "360_right_0.png" and "360_right_0.png.csv" both work.
"""

import argparse
import csv
import sys
from pathlib import Path

# ---- EDIT THIS -------------------------------------------------------------
UPDATES = {
    # "1255_left_1.png": 0.5,
    # "1255_left_2.png ": 0.0,
    # "44_left_0.png": 0.75,
"62_left_2.png" : 0.2,
"163_left_0.png" : 0.5,
"1150_right_1.png" : 0.6,
"1271_right_0.png" : 0.5,
"1349_left_0.png" : 0.4,
"1349_left_1.png" : 0.1,
"1572_left_1.png" : 0.1,
"1615_right_2.png" : 0.1,
"2041_left_0.png" : 0.4,
"2600_right_1.png" : 0.6,
"2931_right_1.png" : 0.8,
"3071_left_0.png" : 0.8,
"3284_left_0.png" : 0.75,
"3312_right_0.png" : 0.7,
"3312_right_1.png" : 0.8,
"4259_right_0.png" : 0.9,
}
COLUMN = "valid_image"
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Set valid_image for listed files.")
    parser.add_argument("folder", type=Path, help="folder with the .csv label files")
    parser.add_argument("--dry-run", action="store_true", help="show changes without saving")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"Folder not found: {args.folder}")

    changed, missing = 0, []
    for key, new_value in UPDATES.items():
        name = key if key.endswith(".csv") else key + ".csv"
        path = args.folder / name
        if not path.is_file():
            print(f"MISSING {name}")
            missing.append(name)
            continue

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        if not fieldnames or COLUMN not in fieldnames:
            print(f"ERROR  {name}: no '{COLUMN}' column")
            continue

        new_text = str(float(new_value))     # keeps the "1.0" style
        old_values = [row[COLUMN] for row in rows]
        for row in rows:
            row[COLUMN] = new_text

        if not args.dry_run:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

        verb = "WOULD SET" if args.dry_run else "SET"
        print(f"{verb}  {name}: {COLUMN} {', '.join(old_values)} -> {new_text}")
        changed += 1

    print(f"\n{'Would change' if args.dry_run else 'Changed'}: {changed}, missing: {len(missing)}")


if __name__ == "__main__":
    main()