"""Learning-rate range test: estimate a good lr for a config on your data (a few minutes on a GPU).

The lr grows exponentially from --start to --end over --steps optimizer steps while the loss is
recorded. The suggested lr is where the smoothed loss falls fastest, capped at a tenth of the lr
with the lowest loss.

    python code_new/lr_find.py --config code_new/configs/definitor_gpu.yaml
    -> training-runs/lr_find_<run name>/lr_find.csv, lr_find.png  and the suggestion printed

Put the result into training.lr. new_layers_lr_mult stays as in the config: the test sweeps the
encoder lr with the new layers at lr * mult, the same as in training.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surgical.config import load_config, resolve_path            # noqa: E402
from surgical import data as D                                   # noqa: E402
from surgical.tasks import get_task, _TaskDataset                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--start', type=float, default=1e-7)
    ap.add_argument('--end', type=float, default=1e-1)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    torch.manual_seed(int(cfg.run.seed))
    task = get_task(cfg)
    dev = torch.device('cuda' if cfg.device.use_cuda_if_available and torch.cuda.is_available() else 'cpu')
    use_amp = dev.type == 'cuda'
    model = (task.load_model(resolve_path(cfg.model.checkpoint)) if cfg.model.init == 'checkpoint'
             else task.build_model()).to(dev).train()
    model.set_dropout(float(cfg.surgical.learn_dropout))
    samples = task.scan(cfg.datasets[0])
    store = D.ImageStore(task.load, True)
    bs = int(cfg.training.batch_size)
    accum = max(1, round(int(cfg.training.get('effective_batch_size') or bs) / bs))
    loader = torch.utils.data.DataLoader(_TaskDataset(samples, store, task.item, True), batch_size=bs, shuffle=True,
                                         num_workers=0, drop_last=len(samples) > bs)
    enc, new = model.param_groups()
    mult = float(cfg.training.get('new_layers_lr_mult', 1.0))
    opt = torch.optim.AdamW([{'params': enc}, {'params': new}], lr=a.start,
                            weight_decay=float(cfg.training.weight_decay), betas=tuple(cfg.training.betas))
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    gamma = (a.end / a.start) ** (1 / max(1, a.steps - 1))
    lrs, raw, smooth = [], [], []
    avg, best, it = 0.0, float('inf'), iter(loader)
    print(f"{len(samples)} samples, batch {bs} x {accum}, {a.steps} steps {a.start:g} -> {a.end:g} on {dev}")
    for step in range(a.steps):
        lr = a.start * gamma ** step
        opt.param_groups[0]['lr'], opt.param_groups[1]['lr'] = lr, lr * mult
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            try:
                x, y, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, _ = next(it)
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = task.loss(out, y).mean()
            scaler.scale(loss / accum).backward()
            tot += loss.item() / accum
        scaler.step(opt); scaler.update()
        avg = 0.9 * avg + 0.1 * tot
        sm = avg / (1 - 0.9 ** (step + 1))
        lrs.append(lr); raw.append(tot); smooth.append(sm)
        best = min(best, sm)
        if step > 10 and (sm > 4 * best or math.isnan(sm)):
            print(f"loss diverged at lr {lr:.2e} - stopping")
            break
        if step % 20 == 0:
            print(f"  step {step:4d} lr {lr:.2e} loss {sm:.4f}")

    lrs, smooth = np.array(lrs), np.array(smooth)
    i_min = int(np.argmin(smooth))
    lo = max(5, int(0.05 * len(smooth)))
    grads = np.gradient(smooth[lo:i_min + 1], np.log(lrs[lo:i_min + 1])) if i_min > lo + 2 else np.array([0.0])
    i_steep = lo + int(np.argmin(grads)) if len(grads) > 1 else i_min
    suggestion = min(lrs[i_steep], lrs[i_min] / 10)
    out_dir = resolve_path(cfg.run.output_root) / f"lr_find_{cfg.run.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'lr_find.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['lr', 'loss', 'loss_smoothed'])
        w.writerows(zip(lrs.tolist(), raw, smooth.tolist()))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.semilogx(lrs, smooth)
        plt.axvline(suggestion, color='g', ls='--', label=f'suggested {suggestion:.1e}')
        plt.axvline(lrs[i_min], color='r', ls=':', label=f'lowest loss {lrs[i_min]:.1e}')
        plt.xlabel('encoder lr (new layers x %g)' % mult); plt.ylabel('loss (smoothed)'); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / 'lr_find.png', dpi=110)
    except ImportError:
        pass
    print(f"\nlowest loss at lr {lrs[i_min]:.2e}; steepest descent at {lrs[i_steep]:.2e}")
    print(f"suggested training.lr: {suggestion:.1e}   (currently {float(cfg.training.lr):.1e})  -> {out_dir}")


if __name__ == '__main__':
    main()
