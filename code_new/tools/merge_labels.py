"""Merge reviewed label files back into a classifier dataset before training.

    python code_new/tools/merge_labels.py training-data/eye-classifier/odir_local_x9_v3

Takes every labels_*/ folder of the dataset (e.g. labels_correct/, labels_iccorect/; or the folders given
with --from), checks each file and copies it into <dataset>/labels/, replacing the old one. The images
next to them (images_*/) are copied into <dataset>/images/ when that crop is missing there.
Replaced label files are kept in <dataset>/labels_backup_<time>/ first.

Every value of glaucoma, atrophy and valid_image must be a number in 0..1 (0.5 etc. allowed); a file
that fails is reported and NOT copied. At the end the whole dataset (images/ + labels/) is counted:
    glaucoma > 0, atrophy > 0, valid_image = 1, valid_image = 0, 0 < valid_image < 1, images without
    a label (the trainer skips them), labels without an image.
The trainer uses exactly the images in images/ that have a label in labels/.

Options:  --from DIR [DIR ...]   label folders to merge (default: <dataset>/labels_*)
          --dry-run              only check and count, change nothing
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ORDER = ('glaucoma', 'atrophy', 'valid_image')
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def read(f: Path):
    """-> (values dict, error or None)"""
    try:
        with open(f, encoding='utf-8-sig', newline='') as fh:
            rows = list(csv.DictReader(fh))
    except Exception as e:                                     # noqa: BLE001
        return None, f"unreadable ({e})"
    if not rows:
        return None, "empty"
    out = {}
    for k in ORDER:
        v = (rows[0].get(k) or '').strip().replace(',', '.')
        try:
            x = float(v)
        except ValueError:
            return None, f"{k} = {v!r} is not a number"
        if not 0.0 <= x <= 1.0:
            return None, f"{k} = {x} is outside 0..1"
        out[k] = x
    return out, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='dataset folder with images/ and labels/')
    ap.add_argument('--from', dest='sources', nargs='+', default=None, help='label folders to merge')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    ds = Path(a.dataset)
    labels, images = ds / 'labels', ds / 'images'
    if not images.is_dir():
        sys.exit(f"no images/ folder in {ds}")
    labels.mkdir(exist_ok=True)
    sources = [Path(s) for s in a.sources] if a.sources else sorted(
        p for p in ds.iterdir() if p.is_dir() and p.name.startswith('labels_') and not p.name.startswith('labels_backup'))
    if not sources:
        print("no labels_* folders to merge - only counting")

    # 1. collect + check (a later folder wins if a crop is in two; reported)
    merged, src_of, bad, dup = {}, {}, [], []
    for src in sources:
        for f in sorted(src.glob('*.csv')):
            vals, err = read(f)
            if err:
                bad.append(f"{src.name}/{f.name}: {err}")
                continue
            if f.name in merged and merged[f.name] != vals:
                dup.append(f"{f.name}: {src_of[f.name].name} -> {src.name}")
            merged[f.name], src_of[f.name] = vals, src
    print(f"{len(merged)} reviewed label files from {', '.join(s.name for s in sources) or '-'}; "
          f"{len(bad)} rejected, {len(dup)} in more than one folder (the later one is used)")
    for line in bad[:30]:
        print("  REJECTED " + line)
    for line in dup[:30]:
        print("  TWICE    " + line)

    # 2. copy (labels; images that are missing in images/)
    replaced = added = img_copied = img_missing = 0
    backup = ds / f"labels_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    for name in sorted(merged):
        dst = labels / name
        crop = name[:-4]                                        # 1010_left_1.png
        if not (images / crop).is_file():
            cands = [src_of[name].parent / ('images' + src_of[name].name[len('labels'):])] + \
                    [s.parent / ('images' + s.name[len('labels'):]) for s in sources]
            img_src = next((p / crop for p in cands if (p / crop).is_file()), None)
            if img_src is None:
                img_missing += 1
                print(f"  NO IMAGE {crop} (label kept out)")
                continue
            if not a.dry_run:
                shutil.copy2(img_src, images / crop)
            img_copied += 1
        if dst.exists():
            if not a.dry_run:
                backup.mkdir(exist_ok=True)
                shutil.copy2(dst, backup / name)
            replaced += 1
        else:
            added += 1
        if not a.dry_run:
            shutil.copy2(src_of[name] / name, dst)
    verb = 'would be' if a.dry_run else 'were'
    print(f"labels: {replaced} {verb} replaced (old ones in {backup.name}/), {added} added; "
          f"images: {img_copied} copied into images/, {img_missing} labels without any image")

    # 3. count the dataset as the trainer will see it
    imgs = {p.name for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT}
    stats = dict.fromkeys(['used', 'glaucoma>0', 'glaucoma>=0.5', 'atrophy>0', 'atrophy>=0.5',
                           'valid=1', 'valid=0', '0<valid<1', 'invalid_label_files'], 0)
    lbl_no_img = 0
    for f in labels.glob('*.csv'):
        if f.name[:-4] not in imgs:
            lbl_no_img += 1
            continue
        vals, err = read(f)
        if err:
            stats['invalid_label_files'] += 1
            continue
        stats['used'] += 1
        stats['glaucoma>0'] += vals['glaucoma'] > 0
        stats['glaucoma>=0.5'] += vals['glaucoma'] >= 0.5
        stats['atrophy>0'] += vals['atrophy'] > 0
        stats['atrophy>=0.5'] += vals['atrophy'] >= 0.5
        v = vals['valid_image']
        stats['valid=1' if v == 1 else 'valid=0' if v == 0 else '0<valid<1'] += 1
    no_lbl = sum(1 for i in imgs if not (labels / (i + '.csv')).exists())
    print(f"\ndataset {ds.name}: {len(imgs)} images, {stats['used']} with a valid label -> used for training")
    for k in ('glaucoma>0', 'glaucoma>=0.5', 'atrophy>0', 'atrophy>=0.5', 'valid=1', 'valid=0', '0<valid<1'):
        print(f"  {k:<14} {stats[k]}")
    print(f"  images without a label (skipped): {no_lbl}; labels without an image: {lbl_no_img}; "
          f"broken label files in labels/: {stats['invalid_label_files']}")
    if a.dry_run:
        print("\n(dry run - nothing was changed)")


if __name__ == '__main__':
    main()
