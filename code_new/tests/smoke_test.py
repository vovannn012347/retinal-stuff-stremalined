"""End-to-end runs on synthetic data with tiny models (a few minutes on CPU), for both tasks.

Checks: learn -> gate -> accept -> dropout ramp -> selection at 0.2 -> penalty -> cut -> recovery
-> accept; dataset switching; mistake replay; interrupt + --resume; evaluate.py on the result.

    python code_new/tests/smoke_test.py [--keep] [--only classifier|definitor]
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


def make_classifier_ds(root: Path, n_patients: int, seed: int):
    """Colour-coded labels: red disc = glaucoma, green ring = atrophy, blue background = valid."""
    rng = np.random.default_rng(seed)
    (root / 'images').mkdir(parents=True); (root / 'labels').mkdir()
    yy, xx = np.mgrid[0:48, 0:48]
    r = np.sqrt((yy - 24) ** 2 + (xx - 24) ** 2)
    for pid in range(n_patients):
        for eye in ('left', 'right'):
            g, a, v = rng.random() < 0.4, rng.random() < 0.3, rng.random() < 0.7
            img = rng.integers(0, 60, (48, 48, 3)).astype(np.float32)
            if v: img[..., 2] += 120
            if g: img[r < 10, 0] += 170
            if a: img[(r > 12) & (r < 18), 1] += 170
            name = f"{pid}_{eye}_0.jpg"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / 'images' / name, quality=95)
            (root / 'labels' / (name + '.csv')).write_text(
                f"glaucoma,atrophy,valid_image\n{int(g)},{int(a)},{int(v)}\n", encoding='utf-8')


def make_definitor_ds(root: Path, n: int, seed: int, hard_every: int = 7):
    """Fundus-like: black border, round retina, a bright disc at a random place = the mask.
    Every `hard_every`-th image has a dim disc (harder) so mistake replay has something to repeat."""
    rng = np.random.default_rng(seed)
    (root / 'images').mkdir(parents=True); (root / 'masks').mkdir()
    H = 80
    yy, xx = np.mgrid[0:H, 0:H + 16]
    for i in range(n):
        img = np.zeros((H, H + 16, 3), np.float32)
        retina = (yy - H / 2) ** 2 + (xx - (H + 16) / 2) ** 2 < (H / 2 - 2) ** 2
        img[retina] = [110, 50, 30]
        img += rng.normal(0, 8, img.shape) * retina[..., None]
        cy, cx = rng.integers(25, H - 25), rng.integers(30, H - 14)
        rad = rng.integers(6, 10)
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
        img[disc] += 60 if i % hard_every == 0 else 120
        name = f"{i:03d}ПАЦ{i % 2 + 1:02d}" if i < n // 2 else f"eye_case_{i}"
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / 'images' / f"{name}.jpg", quality=95)
        Image.fromarray((disc * 255).astype(np.uint8)).save(root / 'masks' / f"{name}.mask.png")


def run(args, cwd, env=None):
    import os
    print('$', ' '.join(str(a) for a in args), flush=True)
    e = dict(os.environ, **(env or {}))
    return subprocess.run([sys.executable] + [str(a) for a in args], cwd=cwd, env=e).returncode


def check_run(rd: Path, datasets):
    rows = list(csv.DictReader(open(rd / 'log.csv', encoding='utf-8')))
    actions = [r['action'] for r in rows]
    print('  actions:', {x: actions.count(x) for x in sorted(set(actions))})
    assert actions.count('accept') >= 3 and actions.count('cut') >= 2, actions
    ramp = [r for r in rows if r['phase'] == 'ramp' and r['sel_norm_ratio'] not in ('', 'nan')]
    assert ramp and float(ramp[0]['dropout']) >= 0.2 - 1e-9
    ratios = [float(r['sel_norm_ratio']) for r in ramp]
    print(f"  selected/kept |w| during the penalty: {ratios[0]:.3f} -> {min(ratios):.4f}")
    assert min(ratios) < 0.1, 'penalty did not shrink the selected units'
    assert {r['dataset'] for r in rows} == set(datasets)
    extra = sum(int(r['replay_extra'] or 0) for r in rows)
    print(f"  mistake replay: {extra} extra training items in total")
    assert extra > 0 and (rd / 'mistakes.csv').is_file()
    assert (rd / 'final_model.pth').is_file()
    return rows


def smoke_classifier(tmp):
    make_classifier_ds(tmp / 'ds_a', 90, 1); make_classifier_ds(tmp / 'ds_b', 90, 2)
    cfg = tmp / 'cls.yaml'
    cfg.write_text(f"""extends: {CODE / 'configs' / 'classifier_base.yaml'}
run: {{name: cls, output_root: {tmp / 'runs'}, seed: 7}}
device: {{use_cuda_if_available: false}}
datasets:
  - {{name: ds_a, path: {tmp / 'ds_a'}}}
  - {{name: ds_b, path: {tmp / 'ds_b'}}}
preprocess: {{size: 32}}
augmentation: {{randaug_magnitude: 0, rotation_degrees: 0, color_jitter: [0, 0, 0, 0]}}
folds: {{k: 3, passes_per_dataset: 3, test_fraction: 0.1}}
model: {{init: scratch, base: 16, hidden: 32}}
training: {{batch_size: 16, effective_batch_size: 16, num_workers: 0, lr: 3.0e-3, new_layers_lr_mult: 1,
           pos_weight: [1.0, 1.0, 1.0], warmup_steps: 5}}
surgical: {{min_learn_passes: 2, max_learn_passes: 200, max_recover_passes: 60, max_cycles: 2,
           cut_ratio: 0.25, min_channels: 2, target_param_ratio: 0.01}}
logging: {{print_every_steps: 1000}}
""", encoding='utf-8')
    train = CODE / 'train_surgical.py'
    # first part on a "GPU" that holds only 4 images: batch 16 must fall back to 4 x 4 accumulation
    assert run([train, '--config', cfg, '--max-steps', 20], tmp, {'SURGICAL_TEST_OOM': '4'}) == 0
    ev = (tmp / 'runs' / 'cls' / 'events.log').read_text(encoding='utf-8')
    assert 'batch 4 x 4 accumulation' in ev, 'OOM back-off did not happen'
    assert run([train, '--config', cfg, '--resume'], tmp) == 0
    rd = tmp / 'runs' / 'cls'
    check_run(rd, ['ds_a', 'ds_b'])
    rows = list(csv.DictReader(open(rd / 'log.csv', encoding='utf-8')))
    assert rows[-1]['batch'] == '4x4' and rows[-1]['gate_source'] == 'test', (rows[-1]['batch'], rows[-1]['gate_source'])
    assert run([CODE / 'evaluate.py', '--config', cfg, '--model', rd / 'final_model.pth',
                '--split', rd / 'split_ds_a.json'], tmp) == 0
    assert run([CODE / 'plot_log.py', rd], tmp) == 0
    print("CLASSIFIER SMOKE OK")


def smoke_definitor(tmp):
    make_definitor_ds(tmp / 'dd_a', 60, 1); make_definitor_ds(tmp / 'dd_b', 60, 2)
    cfg = tmp / 'def.yaml'
    cfg.write_text(f"""extends: {CODE / 'configs' / 'definitor_base.yaml'}
run: {{name: deff, output_root: {tmp / 'runs'}, seed: 7}}
device: {{use_cuda_if_available: false}}
datasets:
  - {{name: dd_a, images: {tmp / 'dd_a' / 'images'}, masks: {tmp / 'dd_a' / 'masks'}}}
  - {{name: dd_b, images: {tmp / 'dd_b' / 'images'}, masks: {tmp / 'dd_b' / 'masks'}}}
preprocess: {{size: 48}}
augmentation: {{rotation_degrees: 0}}
folds: {{k: 3, passes_per_dataset: 3, test_fraction: 0.15, min_steps_per_step: 4}}
model: {{init: scratch, base: 4}}
training: {{batch_size: 4, effective_batch_size: 8, num_workers: 0, lr: 3.0e-3, new_layers_lr_mult: 1, warmup_steps: 5}}
gate: {{recall: 0.90, precision: 0.60, boundary_tolerance_px: 1}}
surgical: {{min_learn_passes: 2, max_learn_passes: 120, max_recover_passes: 60, max_cycles: 2,
           cut_ratio: 0.25, min_channels: 2, target_param_ratio: 0.01}}
logging: {{print_every_steps: 1000}}
""", encoding='utf-8')
    train = CODE / 'train_surgical.py'
    assert run([train, '--config', cfg, '--max-steps', 10], tmp) == 0
    rd = tmp / 'runs' / 'deff'
    assert list((rd / 'backups').glob('state_pass_*.pth')), 'no backup written'
    # simulate a hard crash in the middle of writing state_last.pth: resume must fall back to a backup
    data = (rd / 'state_last.pth').read_bytes()
    (rd / 'state_last.pth').write_bytes(data[:len(data) // 3])
    assert run([train, '--config', cfg, '--resume'], tmp) == 0
    assert 'resuming from backup' in (rd / 'events.log').read_text(encoding='utf-8')
    assert len(list((rd / 'backups').glob('state_pass_*.pth'))) <= 5
    check_run(rd, ['dd_a', 'dd_b'])
    assert run([CODE / 'evaluate.py', '--config', cfg, '--model', rd / 'final_model.pth',
                '--split', rd / 'split_dd_a.json'], tmp) == 0
    imgs = sorted((tmp / 'dd_a' / 'images').glob('*.jpg'))[:2]
    assert run([CODE / 'evaluate.py', '--config', cfg, '--model', rd / 'final_model.pth', '--images', *imgs,
                '--out', tmp / 'pred'], tmp) == 0
    assert len(list((tmp / 'pred').glob('*.mask.png'))) == 2
    assert run([CODE / 'plot_log.py', rd], tmp) == 0
    assert run([CODE / 'lr_find.py', '--config', cfg, '--steps', 60], tmp) == 0
    assert (tmp / 'runs' / 'lr_find_deff' / 'lr_find.png').is_file()
    print("DEFINITOR SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--only', choices=['classifier', 'definitor'])
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='surgical_smoke_'))
    try:
        if a.only != 'definitor':
            smoke_classifier(tmp)
        if a.only != 'classifier':
            smoke_definitor(tmp)
        print(f"SMOKE TEST OK  ({tmp})")
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
