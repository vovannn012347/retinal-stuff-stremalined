"""Copy the crops of every photo that produced MORE THAN ONE island into a separate folder for review.

    python code_new/tools/collect_multi_islands.py training-data/eye-classifier/odir_local_v2/images review/multi

Crops are named <photo>_<k>.<ext> (k = 0, 1, 2 ...). A photo counts when it has <photo>_0 and at least one
more (<photo>_1, _2 ...); then ALL its crops are copied, so you see them side by side. Originals stay where
they are - fix the label files in the dataset's labels/ folder, not the copies.

Options:
    --min N        photos with at least N crops (default 2)
    --labels DIR   label folder (default: the 'labels' folder next to src, if it exists); the matching
                   label files (<crop>.csv) are copied into <out>/labels for reference
    --no-labels    do not copy labels
    --move         move instead of copy (careful: the trainer then no longer sees those crops)
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r'^(.*)_(\d+)$')
EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='folder with the crops (e.g. <dataset>/images)')
    ap.add_argument('out', help='review folder (created)')
    ap.add_argument('--min', type=int, default=2, help='at least this many crops per photo')
    ap.add_argument('--labels', default=None, help='labels folder to copy matching <crop>.csv files from')
    ap.add_argument('--no-labels', action='store_true')
    ap.add_argument('--move', action='store_true')
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.is_dir():
        sys.exit(f"not a folder: {src}")
    groups = defaultdict(dict)
    for f in src.iterdir():
        if f.is_file() and f.suffix.lower() in EXT:
            m = NAME.match(f.stem)
            if m:
                groups[m.group(1)][int(m.group(2))] = f
    picked = {k: v for k, v in groups.items() if 0 in v and len(v) >= a.min}
    out.mkdir(parents=True, exist_ok=True)
    lbl_src = Path(a.labels) if a.labels else src.parent / 'labels'
    if a.no_labels or not lbl_src.is_dir():
        if a.labels:
            sys.exit(f"labels folder not found: {lbl_src}")
        lbl_src = None
    if lbl_src:
        (out / 'labels').mkdir(exist_ok=True)
    op = shutil.move if a.move else shutil.copy2
    n = n_lbl = 0
    for stem in sorted(picked):
        for k in sorted(picked[stem]):
            f = picked[stem][k]
            op(str(f), str(out / f.name))
            n += 1
            if lbl_src:
                lf = lbl_src / (f.name + '.csv')
                if lf.is_file():
                    shutil.copy2(lf, out / 'labels' / lf.name)
                    n_lbl += 1
    (out / 'photos.txt').write_text('\n'.join(sorted(picked)), encoding='utf-8')
    print(f"{len(picked)} photos with >= {a.min} crops, {n} files {'moved' if a.move else 'copied'} -> {out}")
    if lbl_src:
        print(f"{n_lbl} label files copied from {lbl_src} -> {out / 'labels'} (edit the originals, not these)")
    else:
        print("no labels folder found next to the images - pass --labels <folder>")


if __name__ == '__main__':
    main()
