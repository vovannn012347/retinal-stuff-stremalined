"""Run a trained definitor over a folder of ORIGINAL fundus photos (no preparation needed).

    python code_new/run_definitor.py --model training-runs/definitor_cpu/accepted/cycle_02.pth ^
        --images artifacts/retina_images_prepared/kaggle-images --out training-runs/definitor_output

Output (<out>/):
    masks/<stem>.mask.png      every disc candidate (island) white on black, at the photo's resolution
                               (crop_discs.py and make_classifier_crops.py read these)
    preview/<stem>.png         photo | probability (black 0 .. white 1), small, for a quick look
    islands.csv                one row per island: area, mean probability, centre, diameter, crop status
    no_islands.txt             photos where nothing was found

The photo goes through exactly the preparation used for training (retina crop, HSV-S + CLAHE-LAB,
512 px, model size); see surgical/pipeline.py. Subfolders of --images are searched too.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical.config import resolve_path            # noqa: E402
from surgical.data import list_images              # noqa: E402
from surgical import pipeline as P                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='definitor model file (accepted/cycle_XX.pth or final_model.pth)')
    ap.add_argument('--images', required=True, help='folder with original photos (searched recursively)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prepare-json', default=None,
                    help='prepare.json of the training data (default: the standard HSV-S + CLAHE, 512 px)')
    ap.add_argument('--threshold', type=float, default=None, help='probability threshold (default: the model\'s)')
    ap.add_argument('--open-frac', type=float, default=P.ISLAND_DEFAULTS['open_frac'])
    ap.add_argument('--fallback-rel', type=float, default=P.ISLAND_DEFAULTS['fallback_rel'],
                    help='dim photos: if nothing passes the threshold, use this x the photo\'s max probability (0 = off)')
    ap.add_argument('--min-area-frac', type=float, default=P.ISLAND_DEFAULTS['min_area_frac'],
                    help='smallest island, as a fraction of the photo area')
    ap.add_argument('--preview-size', type=int, default=256, help='0 = no previews')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    img_dir, out = resolve_path(a.images), resolve_path(a.out)
    photos = list_images(img_dir)
    if not photos:
        sys.exit(f"no images in {img_dir}")
    runner = P.DefinitorRunner(resolve_path(a.model), a.device, resolve_path(a.prepare_json) if a.prepare_json else None)
    thr = runner.threshold if a.threshold is None else a.threshold
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    if a.preview_size:
        (out / 'preview').mkdir(exist_ok=True)
    print(f"{len(photos)} photos, model input {runner.size} px, threshold {thr}")

    rows, empty, t0 = [], [], time.time()
    for n, path in enumerate(photos, 1):
        rgb = np.asarray(Image.open(path).convert('RGB'))
        prob = runner.probability(rgb)
        isl = P.find_islands(prob, thr, a.open_frac, a.min_area_frac, a.fallback_rel)
        union = np.zeros(prob.shape, np.uint8)
        for i in isl:
            union[i.mask] = 255
            info, _ = P.crop_island(rgb, i)
            rows.append({'image': path.name, 'island': i.index, 'area_px': i.area_px,
                         'mean_prob': f"{i.mean_prob:.3f}", 'fallback': int(i.fallback), 'center_x': f"{info.center_x:.1f}",
                         'center_y': f"{info.center_y:.1f}", 'diameter_px': f"{info.diameter_px:.1f}",
                         'status': info.status, 'completeness': f"{info.completeness:.3f}",
                         'solidity': f"{info.solidity:.3f}"})
        if not isl:
            empty.append(path.name)
        Image.fromarray(union).save(out / 'masks' / f"{path.stem}.mask.png")
        if a.preview_size:
            s = a.preview_size
            left = Image.fromarray(rgb).resize((s, s), Image.BILINEAR)
            right = Image.fromarray((prob * 255).round().astype(np.uint8)).resize((s, s), Image.BILINEAR).convert('RGB')
            sheet = Image.new('RGB', (2 * s, s)); sheet.paste(left, (0, 0)); sheet.paste(right, (s, 0))
            sheet.save(out / 'preview' / f"{path.stem}.png")
        if n % 20 == 0 or n == len(photos):
            print(f"  {n}/{len(photos)}  ({(time.time() - t0) / n:.2f} s per photo)")

    with open(out / 'islands.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'island', 'area_px', 'mean_prob', 'fallback', 'center_x', 'center_y',
                                          'diameter_px', 'status', 'completeness', 'solidity'])
        w.writeheader(); w.writerows(rows)
    (out / 'no_islands.txt').write_text('\n'.join(empty), encoding='utf-8')
    per = {}
    for r in rows:
        per[r['image']] = per.get(r['image'], 0) + 1
    multi = sum(1 for v in per.values() if v > 1)
    print(f"done: {len(rows)} islands in {len(per)} photos; {multi} photos with more than one; "
          f"{len(empty)} with none (no_islands.txt) -> {out}")


if __name__ == '__main__':
    main()
