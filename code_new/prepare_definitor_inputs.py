"""Prepare definitor training inputs: crop the black border, HSV-S equalisation + CLAHE-LAB, resize.

    python code_new/prepare_definitor_inputs.py
    python code_new/prepare_definitor_inputs.py --images <folder> --masks <folder> --out <folder> --size 512

Defaults: images training-data/source-input/input_for_definitor, masks
training-data/eye-nerve-definitor/training-masks, output training-data/eye-nerve-definitor/prepared_512/
with images/ (PNG) and masks/ (<stem>.mask.png). Masks get the same crop and are resized with
nearest-neighbour, so they stay pixel-aligned with their images. Images without a mask are
processed too (useful later for inference checks); masks without an image are skipped.
prepare.json records the parameters - the inference script must use the same ones.

Source files are never changed. The output folder is rebuilt from scratch on every run
(so masks you removed from the source disappear from the prepared set too).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical.config import resolve_path            # noqa: E402
from surgical.data import list_images              # noqa: E402
from surgical import enhance as E                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='training-data/source-input/input_for_definitor')
    ap.add_argument('--masks', default='training-data/eye-nerve-definitor/training-masks')
    ap.add_argument('--out', default='training-data/eye-nerve-definitor/prepared_512')
    ap.add_argument('--size', type=int, default=512, help='output is size x size (divisible by 16 for the definitor)')
    ap.add_argument('--no-hsv-s', action='store_true', help='skip the S-channel equalisation')
    ap.add_argument('--no-clahe', action='store_true', help='skip CLAHE')
    ap.add_argument('--clip-limit', type=float, default=2.0)
    ap.add_argument('--tile', type=int, default=16)
    ap.add_argument('--black-level', type=int, default=10)
    ap.add_argument('--no-crop', action='store_true', help='keep the black border')
    a = ap.parse_args()

    img_dir, mask_dir, out = resolve_path(a.images), resolve_path(a.masks), resolve_path(a.out)
    if not img_dir.is_dir():
        sys.exit(f"image folder not found: {img_dir}")
    masks = {}
    if mask_dir and mask_dir.is_dir():
        for p in list_images(mask_dir):
            if p.stem.endswith('.mask'):
                masks[p.stem[:-5]] = p
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    # start clean: a mask removed from the source must not survive in the prepared set
    stale = list((out / 'images').glob('*.png')) + list((out / 'masks').glob('*.mask.png'))
    for f in stale:
        f.unlink()
    if stale:
        print(f"removed {len(stale)} files of the previous preparation")
    params = {'hsv_s': not a.no_hsv_s, 'clahe': not a.no_clahe, 'clip_limit': a.clip_limit, 'tile': a.tile,
              'black_level': a.black_level, 'crop_to_retina': not a.no_crop, 'size': a.size}

    n_img = n_mask = n_empty = 0
    for p in list_images(img_dir):
        rgb = np.asarray(Image.open(p).convert('RGB'))
        box = None if a.no_crop else E.retina_box(rgb, a.black_level)
        if box:
            rgb = rgb[box[1]:box[3], box[0]:box[2]]
        enh = E.enhance(rgb, params['hsv_s'], params['clahe'], a.clip_limit, a.tile, a.black_level)
        Image.fromarray(enh).resize((a.size, a.size), Image.LANCZOS).save(out / 'images' / f"{p.stem}.png")
        n_img += 1
        m = masks.get(p.stem)
        if m is None:
            continue
        mk = Image.open(m).convert('L')
        if mk.size != (p_size := Image.open(p).size):
            mk = mk.resize(p_size, Image.NEAREST)
        mk = np.asarray(mk)
        if box:
            mk = mk[box[1]:box[3], box[0]:box[2]]
        mk = (mk > 127).astype(np.uint8) * 255
        n_empty += int(mk.max() == 0)
        Image.fromarray(mk).resize((a.size, a.size), Image.NEAREST).save(out / 'masks' / f"{p.stem}.mask.png")
        n_mask += 1
        if n_img % 20 == 0:
            print(f"  {n_img} images...")

    (out / 'prepare.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'), 'source_images': str(img_dir),
        'source_masks': str(mask_dir), 'params': params, 'images': n_img, 'masks': n_mask,
        'empty_masks': n_empty}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"{n_img} images -> {out / 'images'}")
    print(f"{n_mask} masks -> {out / 'masks'} ({n_empty} empty; {len(masks) - n_mask} masks had no image)")
    print(f"parameters: {params}")
    print("train with datasets: images/masks pointing at the folders above (definitor_base.yaml does by default)")


if __name__ == '__main__':
    main()
