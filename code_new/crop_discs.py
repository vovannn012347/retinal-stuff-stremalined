"""Cut optic-disc crops for the classifier from the ORIGINAL photos (not the colour-enhanced ones).

    python code_new/crop_discs.py
    python code_new/crop_discs.py --images <photos> --masks <masks> --out <folder> --size 256 --margin 1.75

For every photo with a mask (<stem>.mask.<ext>, at the photo's resolution or any - it is rescaled):
  * the disc = largest blob of the mask; a circle is fitted to the part of its outline that is not
    on the edge of the photo / the black border of the retina;
  * completeness = visible disc area / circle area (1.0 when the disc does not touch the edge);
  * crop = square centred on the circle, side = margin x disc diameter, black padding past the
    photo, resized to size x size  ->  same disc scale for every camera magnification.

Output: <out>/images/<stem>.png (all, for review), optional <out>/masks/, and <out>/crops.csv with
status (ok | partial | irregular | no_disc), the reason, completeness, diameter, centre, solidity.
For classifier training, 'partial' crops are good valid_image = 0 examples.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical.config import resolve_path            # noqa: E402
from surgical.data import list_images              # noqa: E402
from surgical import disc_crop as C                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='training-data/source-input/input_for_definitor')
    ap.add_argument('--masks', default='training-data/eye-nerve-definitor/training-masks')
    ap.add_argument('--out', default='training-data/classifier/disc_crops_256')
    ap.add_argument('--size', type=int, default=C.DEFAULTS['size'])
    ap.add_argument('--margin', type=float, default=C.DEFAULTS['margin'], help='crop side in disc diameters')
    ap.add_argument('--min-completeness', type=float, default=C.DEFAULTS['min_completeness'])
    ap.add_argument('--min-solidity', type=float, default=C.DEFAULTS['min_solidity'])
    ap.add_argument('--black-level', type=int, default=C.DEFAULTS['black_level'])
    ap.add_argument('--save-masks', action='store_true', help='also write the cropped masks')
    ap.add_argument('--only-ok', action='store_true', help='write image files only for status ok')
    a = ap.parse_args()

    img_dir, mask_dir, out = resolve_path(a.images), resolve_path(a.masks), resolve_path(a.out)
    for d in (img_dir, mask_dir):
        if not d.is_dir():
            sys.exit(f"folder not found: {d}")
    masks = {p.stem[:-5]: p for p in list_images(mask_dir) if p.stem.endswith('.mask')}
    p = {'size': a.size, 'margin': a.margin, 'min_completeness': a.min_completeness,
         'min_solidity': a.min_solidity, 'black_level': a.black_level}
    (out / 'images').mkdir(parents=True, exist_ok=True)
    if a.save_masks:
        (out / 'masks').mkdir(parents=True, exist_ok=True)

    rows, counts = [], {}
    for ip in list_images(img_dir):
        mp = masks.get(ip.stem)
        if mp is None:
            continue
        rgb = np.asarray(Image.open(ip).convert('RGB'))
        mk = Image.open(mp).convert('L')
        if mk.size != (rgb.shape[1], rgb.shape[0]):
            mk = mk.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
        mk = np.asarray(mk)
        info = C.analyse(rgb, mk, p)
        counts[info.status] = counts.get(info.status, 0) + 1
        crop_file = ''
        if info.status != 'no_disc' and (info.status == 'ok' or not a.only_ok):
            c, mc = C.crop(rgb, info, p, mk if a.save_masks else None)
            crop_file = f"{ip.stem}.png"
            Image.fromarray(c).save(out / 'images' / crop_file)
            if mc is not None:
                Image.fromarray(mc).save(out / 'masks' / f"{ip.stem}.mask.png")
        rows.append({'image': ip.name, 'crop_file': crop_file, **info.as_row()})

    with open(out / 'crops.csv', 'w', newline='', encoding='utf-8') as f:
        fields = ['image', 'crop_file', 'status', 'reason', 'completeness', 'diameter_px', 'center_x', 'center_y',
                  'touches_edge', 'solidity', 'area_px']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / 'crop.json').write_text(json.dumps({'created': datetime.now().isoformat(timespec='seconds'),
                                               'source_images': str(img_dir), 'source_masks': str(mask_dir),
                                               'params': p, 'counts': counts}, ensure_ascii=False, indent=1),
                                   encoding='utf-8')
    d = [r['diameter_px'] for r in rows if r['status'] != 'no_disc']
    print(f"{len(rows)} photos with masks: {counts}")
    if d:
        print(f"disc diameter {min(d):.0f}-{max(d):.0f} px (median {float(np.median(d)):.0f}) -> crops of "
              f"{a.margin:g} diameters resized to {a.size} px")
    for r in rows:
        if r['status'] != 'ok':
            print(f"  {r['image']}: {r['status']} - {r['reason']}")
    print(f"-> {out}")


if __name__ == '__main__':
    main()
