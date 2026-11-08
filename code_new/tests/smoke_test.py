"""End-to-end run on a synthetic dataset with a tiny model (a few minutes on CPU).

Checks: learn -> gate -> accept -> dropout ramp -> selection at 0.2 -> penalty -> cut -> recovery
-> accept, two datasets with switching, interrupt + --resume, evaluate.py on the result.

    python code_new/tests/smoke_test.py [--keep]
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parents[1]


def make_dataset(root: Path, n_patients: int, seed: int):
    """Colour-coded labels: red disc = glaucoma, green ring = atrophy, blue background = valid."""
    rng = np.random.default_rng(seed)
    (root / 'images').mkdir(parents=True); (root / 'labels').mkdir()
    yy, xx = np.mgrid[0:48, 0:48]
    r = np.sqrt((yy - 24) ** 2 + (xx - 24) ** 2)
    for pid in range(n_patients):
        for eye in ('left', 'right'):
            g, a, v = rng.random() < 0.4, rng.random() < 0.3, rng.random() < 0.7
            img = rng.integers(0, 60, (48, 48, 3)).astype(np.float32)
            if v:
                img[..., 2] += 120
            if g:
                img[r < 10, 0] += 170
            if a:
                img[(r > 12) & (r < 18), 1] += 170
            name = f"{pid}_{eye}_0.jpg"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / 'images' / name, quality=95)
            (root / 'labels' / (name + '.csv')).write_text(
                f"glaucoma,atrophy,valid_image\n{int(g)},{int(a)},{int(v)}\n", encoding='utf-8')


def run(args, cwd):
    print('$', ' '.join(str(a) for a in args), flush=True)
    p = subprocess.run([sys.executable] + [str(a) for a in args], cwd=cwd)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='surgical_smoke_'))
    try:
        make_dataset(tmp / 'ds_a', 90, 1)
        make_dataset(tmp / 'ds_b', 90, 2)
        cfg = tmp / 'smoke.yaml'
        cfg.write_text(f"""extends: {CODE / 'configs' / 'surgical_base.yaml'}
run: {{name: smoke, output_root: {tmp / 'runs'}, seed: 7}}
device: {{use_cuda_if_available: false}}
datasets:
  - {{name: ds_a, path: {tmp / 'ds_a'}}}
  - {{name: ds_b, path: {tmp / 'ds_b'}}}
preprocess: {{size: 32}}
augmentation: {{randaug_magnitude: 0, rotation_degrees: 0, color_jitter: [0, 0, 0, 0]}}
folds: {{k: 3, passes_per_dataset: 3, test_fraction: 0.1}}
model: {{init: scratch, base: 8, hidden: 16}}
training: {{batch_size: 16, num_workers: 0, lr: 3.0e-3, pos_weight: [1.0, 1.0, 1.0]}}
surgical: {{min_learn_passes: 2, max_learn_passes: 60, max_recover_passes: 40, max_cycles: 2,
           cut_ratio: 0.25, min_channels: 2, target_param_ratio: 0.01}}
logging: {{print_every_steps: 1000}}
""", encoding='utf-8')
        train = CODE / 'train_surgical.py'
        # part 1: stop early, then resume to the end
        assert run([train, '--config', cfg, '--max-steps', 20], tmp) == 0
        assert run([train, '--config', cfg, '--resume'], tmp) == 0
        rd = tmp / 'runs' / 'smoke'
        rows = list(csv.DictReader(open(rd / 'log.csv', encoding='utf-8')))
        actions = [r['action'] for r in rows]
        print('actions:', {x: actions.count(x) for x in set(actions)})
        assert actions.count('accept') >= 3, 'expected accepts for cycle 0, 1, 2'
        assert actions.count('cut') >= 2
        sel_rows = [r for r in rows if r['sel_norm_ratio'] not in ('', 'nan')]
        assert sel_rows and float(sel_rows[0]['dropout']) >= 0.2 - 1e-9
        ratios = [float(r['sel_norm_ratio']) for r in sel_rows if r['phase'] == 'ramp']
        print(f"selected/kept |w| over the first penalty window: {ratios[0]:.3f} -> {min(ratios):.4f}")
        assert min(ratios) < 0.05, 'penalty did not shrink the selected units'
        datasets = {r['dataset'] for r in rows}
        assert datasets == {'ds_a', 'ds_b'}, datasets
        assert (rd / 'final_model.pth').is_file()
        assert sorted(p.name for p in (rd / 'accepted').glob('*.pth'))[:3] == ['cycle_00.pth', 'cycle_01.pth',
                                                                                'cycle_02.pth']
        assert run([CODE / 'evaluate.py', '--model', rd / 'final_model.pth', '--dataset', tmp / 'ds_a',
                    '--split', rd / 'split_ds_a.json'], tmp) == 0
        assert run([CODE / 'plot_log.py', rd], tmp) == 0
        print(f"SMOKE TEST OK  ({len(rows)} passes)  run folder: {rd}")
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
