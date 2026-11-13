"""Copy the extra-island crops (<stem>_1, _2, ...) into a separate folder for review - only those whose
<stem>_0 also exists - optionally together with that _0 crop and the label files.

    python code_new/tools/collect_extra_islands.py training-data/eye-classifier/odir_local_v2/images ^
        --out training-data/eye-classifier/review_extra_islands --with-first --labels

Files are COPIED (originals untouched); --move moves them instead. Any image extension works
(.png from make_classifier_crops.py, .jpg from older crops).
"""
import argparse
import re
import shutil
from pathlib import Path

IMG = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
NAME = re.compile(r'^(.+)_(\d+)$')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', help='folder with the crops')
    ap.add_argument('--out', required=True)
    ap.add_argument('--with-first', action='store_true', help='also copy the matching _0 crop (for comparison)')
    ap.add_argument('--labels', action='store_true',
                    help='also copy <crop>.csv from the sibling labels/ folder into <out>/labels')
    ap.add_argument('--move', action='store_true', help='move instead of copy')
    a = ap.parse_args()

    src, out = Path(a.images), Path(a.out)
    files = {p.stem: p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG}
    picked = set()
    for stem, p in files.items():
        m = NAME.match(stem)
        if m and int(m.group(2)) >= 1 and f"{m.group(1)}_0" in files:
            picked.add(stem)
            if a.with_first:
                picked.add(f"{m.group(1)}_0")
    out.mkdir(parents=True, exist_ok=True)
    lbl_src, lbl_out = src.parent / 'labels', out / 'labels'
    op = shutil.move if a.move else shutil.copy2
    n_lbl = 0
    for stem in sorted(picked):
        p = files[stem]
        op(str(p), str(out / p.name))
        if a.labels:
            lf = lbl_src / (p.name + '.csv')
            if lf.is_file():
                lbl_out.mkdir(exist_ok=True)
                op(str(lf), str(lbl_out / lf.name))
                n_lbl += 1
    extra = sum(1 for s in picked if not s.endswith('_0'))
    print(f"{extra} extra-island crops" + (f" + {len(picked) - extra} matching _0 crops" if a.with_first else '')
          + (f", {n_lbl} label files" if a.labels else '') + f" {'moved' if a.move else 'copied'} -> {out}")


if __name__ == '__main__':
    main()
