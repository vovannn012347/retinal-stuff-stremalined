"""Classifier dataset from original photos: definitor -> every disc candidate (island) -> crop + label.

    python code_new/make_classifier_crops.py ^
        --model  training-runs/definitor_cpu_from_x16/final_model.pth ^
        --images artifacts/retina_images_prepared/kaggle-images ^
        --labels artifacts/retina_images_prepared/retina_classify/nerves_cropped_out_labels ^
        --out    training-data/eye-classifier/odir_local_v2

Output (<out>/, a dataset folder for classifier_*.yaml):
    images/<stem>_<k>.png          crop k of the photo (k = 0 is the largest island), cut from the ORIGINAL
                                   photo (no CLAHE), disc-scale normalised (side = margin x disc diameter)
    labels/<stem>_<k>.png.csv      glaucoma,atrophy,valid_image
    crops.csv                      every crop: source photo, island, status (ok | partial | irregular),
                                   completeness, the label files used and the label written
    missing_labels.txt             photos that had islands but no label file (their crops get no label
                                   file and are skipped by the trainer until you add one)
    no_islands.txt                 photos where the definitor found nothing

Labels come from the earlier labelled crops (--labels), named <photo stem>_<n>.<ext>.csv: all files of
a photo are combined - glaucoma and atrophy take the LARGEST value over its files (a 0.5 "suspected"
stays 0.5; files of old crops marked valid_image = 0 are ignored unless there is no other) - and
valid_image is written as 1 for every crop. Go through crops.csv / the images and set
valid_image to 0 (and glaucoma/atrophy to 0) for crops that are not the disc.
--folder-labels instead takes glaucoma / atrophy from the photo's folder name (e.g. raw_data/glaucoma/).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical.config import resolve_path            # noqa: E402
from surgical.data import list_images              # noqa: E402
from surgical import disc_crop as C                 # noqa: E402
from surgical import pipeline as P                  # noqa: E402

ORDER = ['glaucoma', 'atrophy', 'valid_image']
_ISLAND = re.compile(r'^(.*)_(\d+)$')


def index_labels(folder: Path):
    """label files by photo stem: '<stem>_<n>.<ext>.csv' (one per old crop) and '<stem>.<ext>.csv'"""
    by_island, exact = {}, {}
    for f in sorted(folder.rglob('*.csv')):
        base = f.name[:-4]                                   # 1138_left_0.jpg
        if '.' in base:
            base = base.rsplit('.', 1)[0]                    # 1138_left_0
        exact.setdefault(base, []).append(f)
        m = _ISLAND.match(base)
        if m:
            by_island.setdefault(m.group(1), []).append(f)
    return by_island, exact


def read_label(f: Path):
    with open(f, encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    out = {}
    for n in ORDER:
        try:
            out[n] = float(rows[0].get(n) or 0.0)
        except ValueError:
            out[n] = 0.0
    return out


def combined_label(files):
    vals = [v for v in (read_label(f) for f in files) if v is not None]
    if not vals:
        return None
    valid = [v for v in vals if v['valid_image'] >= 0.5]       # old crops marked "not the disc" don't count
    vals = valid or vals
    return {'glaucoma': max(v['glaucoma'] for v in vals), 'atrophy': max(v['atrophy'] for v in vals),
            'valid_image': 1.0}


def folder_label(path: Path, root: Path):
    parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
    return {'glaucoma': float('glaucoma' in parts), 'atrophy': float('atrophy' in parts), 'valid_image': 1.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='definitor model file')
    ap.add_argument('--images', required=True, help='original photos (searched recursively)')
    ap.add_argument('--labels', default=None, help='folder with the earlier label files (<stem>_<n>.<ext>.csv)')
    ap.add_argument('--folder-labels', action='store_true', help='labels from folder names glaucoma/ atrophy/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256, help='crop size in px')
    ap.add_argument('--margin', type=float, default=C.DEFAULTS['margin'], help='crop side = margin x disc diameter')
    ap.add_argument('--max-islands', type=int, default=3, help='at most this many crops per photo (largest first)')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--fallback-rel', type=float, default=P.ISLAND_DEFAULTS['fallback_rel'],
                    help='dim photos: if nothing passes the threshold, use this x the photo\'s max probability (0 = off)')
    ap.add_argument('--min-disc-frac', type=float, default=P.CROP_DEFAULTS['min_disc_frac'],
                    help='crop diameter never below this x the retina width (small islands = part of a disc)')
    ap.add_argument('--min-area-frac', type=float, default=P.ISLAND_DEFAULTS['min_area_frac'])
    ap.add_argument('--prepare-json', default=None)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    img_dir, out = resolve_path(a.images), resolve_path(a.out)
    photos = list_images(img_dir)
    if not photos:
        sys.exit(f"no images in {img_dir}")
    by_island, exact = index_labels(resolve_path(a.labels)) if a.labels else ({}, {})
    if a.labels:
        print(f"label files for {len(set(by_island) | set(exact))} photo stems in {a.labels}")
    runner = P.DefinitorRunner(resolve_path(a.model), a.device, resolve_path(a.prepare_json) if a.prepare_json else None)
    thr = runner.threshold if a.threshold is None else a.threshold
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    cparams = {'size': a.size, 'margin': a.margin, 'min_disc_frac': a.min_disc_frac}

    rows, missing, empty = [], [], []
    for n, path in enumerate(photos, 1):
        if a.folder_labels:
            label, used = folder_label(path, img_dir), ['<folder>']
        else:
            files = by_island.get(path.stem) or exact.get(path.stem) or []
            label, used = combined_label(files), [f.name for f in files]
        rgb = np.asarray(Image.open(path).convert('RGB'))
        islands = P.find_islands(runner.probability(rgb), thr, None, a.min_area_frac, a.fallback_rel)[:a.max_islands]
        if not islands:
            empty.append(path.name)
        elif label is None:
            missing.append(path.name)
        for isl in islands:
            info, crop = P.crop_island(rgb, isl, cparams)
            if crop is None:
                continue
            name = f"{path.stem}_{isl.index}.png"
            Image.fromarray(crop).save(out / 'images' / name)
            if label is not None:
                (out / 'labels' / (name + '.csv')).write_text(
                    'glaucoma,atrophy,valid_image\n' + ','.join(f"{label[k]:.1f}" for k in ORDER) + '\n',
                    encoding='utf-8')
            rows.append({'crop': name, 'source': str(path.relative_to(img_dir)), 'island': isl.index,
                         'islands_in_photo': len(islands), 'mean_prob': f"{isl.mean_prob:.3f}", 'fallback': int(isl.fallback),
                         'status': info.status, 'completeness': f"{info.completeness:.3f}",
                         'diameter_px': f"{info.diameter_px:.1f}",
                         'crop_diameter_px': f"{info.crop_diameter_px:.1f}", 'enlarged': int(info.enlarged),
                         'glaucoma': '' if label is None else label['glaucoma'],
                         'atrophy': '' if label is None else label['atrophy'],
                         'valid_image': '' if label is None else label['valid_image'],
                         'label_files': ' '.join(used)})
        if n % 20 == 0 or n == len(photos):
            print(f"  {n}/{len(photos)} photos, {len(rows)} crops")

    with open(out / 'crops.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['crop'])
        w.writeheader(); w.writerows(rows)
    (out / 'missing_labels.txt').write_text('\n'.join(missing), encoding='utf-8')
    (out / 'no_islands.txt').write_text('\n'.join(empty), encoding='utf-8')
    extra = sum(1 for r in rows if r['island'] > 0)
    enl = sum(1 for r in rows if r['enlarged'])
    print(f"done: {len(rows)} crops ({extra} from second/third islands - check those first), "
          f"{enl} crops enlarged to the minimum disc size, "
          f"{len(missing)} photos without labels, {len(empty)} without islands -> {out}")


if __name__ == '__main__':
    main()
