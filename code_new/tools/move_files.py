#!/usr/bin/env python3
"""
Move files named in a .txt list from one folder to another.

Steps:
  1. Read the list, drop blanks/duplicates, sort it naturally
     (44_left_0 < 102_left_0, left < right, _0 < _1 < _2)
     and save the sorted list back to the .txt file.
  2. Go through the sorted list one by one, look for the file in the
     source folder and move it to the destination folder.
     A list entry "360_right_0.png" matches "360_right_0.png.csv"
     (and also the exact name "360_right_0.png" if that exists).

Usage:
  python move_listed_files.py list.txt SOURCE_FOLDER DEST_FOLDER
  python move_listed_files.py list.txt SOURCE_FOLDER DEST_FOLDER --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Extensions to try after the name from the list. "" = exact name.
SUFFIXES = [".csv", ""]


def natural_key(name: str):
    """Sort key that compares numbers as numbers: 2 < 10 < 102."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def read_and_sort_list(list_file: Path) -> list[str]:
    lines = list_file.read_text(encoding="utf-8").splitlines()
    names = {line.strip() for line in lines if line.strip()}
    sorted_names = sorted(names, key=natural_key)
    # Save the ordered list back to the file
    list_file.write_text("\n".join(sorted_names) + "\n", encoding="utf-8")
    return sorted_names


def main():
    parser = argparse.ArgumentParser(description="Move files listed in a txt file.")
    parser.add_argument("list_file", type=Path, help="txt file with one file name per line")
    parser.add_argument("source", type=Path, help="folder to move files from")
    parser.add_argument("dest", type=Path, help="folder to move files to")
    parser.add_argument("--dry-run", action="store_true",
                        help="only show what would be moved")
    args = parser.parse_args()

    if not args.list_file.is_file():
        sys.exit(f"List file not found: {args.list_file}")
    if not args.source.is_dir():
        sys.exit(f"Source folder not found: {args.source}")
    args.dest.mkdir(parents=True, exist_ok=True)

    # 1. Order the list
    names = read_and_sort_list(args.list_file)
    print(f"Sorted {len(names)} names and saved them to {args.list_file}\n")

    # 2. Check and move one by one
    moved, missing, skipped = 0, [], 0
    for name in names:
        found = False
        for suffix in SUFFIXES:
            src = args.source / (name + suffix)
            if not src.is_file():
                continue
            found = True
            dst = args.dest / src.name
            if dst.exists():
                print(f"SKIP   {src.name} (already exists in destination)")
                skipped += 1
                continue
            if args.dry_run:
                print(f"WOULD MOVE  {src.name}")
            else:
                shutil.move(str(src), str(dst))
                print(f"MOVED  {src.name}")
            moved += 1
        if not found:
            print(f"MISSING {name}")
            missing.append(name)

    print("\n--- Summary ---")
    print(f"{'Would move' if args.dry_run else 'Moved'}: {moved}")
    print(f"Skipped (already in destination): {skipped}")
    print(f"Not found: {len(missing)}")


if __name__ == "__main__":
    main()