"""SurgicalSlicing: iterative train -> dropout ramp -> targeted penalty -> cut, with fold rotation.

    python code_new/train_surgical.py --config code_new/configs/laptop.yaml
    python code_new/train_surgical.py --config code_new/configs/gpu.yaml --set run.name=try2
    python code_new/train_surgical.py --config code_new/configs/gpu.yaml --resume      # continue a run

Outputs (training-runs/<run.name>/):
    accepted/cycle_XX.pth   model that passed the gate after XX cuts (deployable)
    final_model.pth         the last accepted model
    state_last.pth          full resume point (written after every fold step)
    log.csv                 one row per pass;  steps.csv  one row per fold step
    events.log              everything printed
    split_<dataset>.json    patient folds + held-out test patients
"""
import argparse
import csv
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surgical.config import load_config, resolve_path                     # noqa: E402
from surgical import data as D                                            # noqa: E402
from surgical import surgery as S                                         # noqa: E402
from surgical.metrics import compute_metrics, gate_values, gate_check     # noqa: E402
from surgical.model import (GlaucomaNet, default_channels, count_params, init_from_vgg16,  # noqa: E402
                            load_model_file, build_from_arch)
from surgical.phases import SurgicalSchedule                              # noqa: E402


# ── small utilities ───────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path: Path):
        self.f = open(path, 'a', encoding='utf-8')

    def __call__(self, *parts):
        line = ' '.join(str(p) for p in parts)
        stamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{stamp}] {line}", flush=True)
        self.f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")
        self.f.flush()


class CsvLog:
    def __init__(self, path: Path, fields):
        self.path, self.fields = path, fields
        new = not path.exists()
        self.f = open(path, 'a', encoding='utf-8', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction='ignore')
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in kw.items()})
        self.f.flush()


def fmt(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ── dataset bundle: samples, split, loaders for one dataset ───────────────────────────
class Bundle:
    def __init__(self, cfg, ds_cfg, run_dir: Path, device, log):
        root = resolve_path(ds_cfg['path'])
        self.name = ds_cfg.get('name') or root.name
        order = list(cfg.labels.order)
        log(f"[data] loading dataset '{self.name}' from {root}")
        self.samples = D.scan_dataset(root, order, ds_cfg.get('images', 'images'), ds_cfg.get('labels', 'labels'),
                                      cfg.labels.get('group_regex'), log)
        thr = float(cfg.labels.positive_threshold)
        self.test_idx, self.folds = D.load_or_make_split(
            self.samples, int(cfg.folds.k), float(cfg.folds.test_fraction), thr, int(cfg.run.seed),
            run_dir / f"split_{self.name}.json", log)
        pp = cfg.preprocess
        self.store = D.ImageStore(int(pp.size), pp.mode, float(pp.get('crop_fraction', 1.0)), bool(pp.cache_images))
        t0 = time.time()
        self.store.preload([s.path for s in self.samples])
        n_groups = len({s.group for s in self.samples})
        pos = np.sum(np.stack([s.labels for s in self.samples]) >= thr, axis=0)
        log(f"[data] {len(self.samples)} images, {n_groups} patients, positives per label "
            f"{dict(zip(order, pos.tolist()))}, folds {[len(f) for f in self.folds]}, test {len(self.test_idx)}"
            + (f", cached in {time.time() - t0:.1f}s" if pp.cache_images else ''))
        tr = cfg.training
        nw = int(tr.num_workers)
        common = dict(batch_size=int(tr.batch_size), num_workers=nw, pin_memory=device.type == 'cuda',
                      persistent_workers=nw > 0)
        self.train_sampler = D.SettableSampler(shuffle=True, seed=int(cfg.run.seed))
        self.eval_sampler = D.SettableSampler(shuffle=False, seed=0)
        self.train_loader = DataLoader(D.RetinaDataset(self.samples, self.store, D.build_augment(cfg.augmentation)),
                                       sampler=self.train_sampler, drop_last=len(self.samples) > 4 * int(tr.batch_size),
                                       **common)
        self.eval_loader = DataLoader(D.RetinaDataset(self.samples, self.store, None),
                                      sampler=self.eval_sampler, **common)


# ── train / eval ──────────────────────────────────────────────────────────────────────
def train_fold(model, bundle, fold_idx, opt, scaler, crit, device, use_amp, cfg, pass_info, selection):
    model.train()
    bundle.train_sampler.set(bundle.folds[fold_idx])
    l2 = cfg.surgical.penalty_mode == 'l2' and pass_info['apply_penalty'] and selection
    lam = pass_info['penalty'] * float(cfg.surgical.l2_scale)
    clip = float(cfg.training.grad_clip)
    total, n = 0.0, 0
    for _ in range(int(cfg.folds.repeats_per_step)):
        for x, y, _ in bundle.train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = crit(logits.float(), y)
            if l2:
                loss = loss + lam * S.l2_penalty(model, selection)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.shape[0]; n += x.shape[0]
    return total / max(1, n)


@torch.no_grad()
def predict(model, bundle, indices, device, use_amp):
    model.eval()
    bundle.eval_sampler.set(indices)
    probs, targets = [], []
    for x, y, _ in bundle.eval_loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy()); targets.append(y.numpy())
    if not probs:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    return np.concatenate(probs), np.concatenate(targets)


def make_optimizer(model, cfg, lr):
    tr = cfg.training
    return torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(tr.betas), weight_decay=float(tr.weight_decay))


def model_file(model, cfg, extra):
    return {'format': 'surgical-slicing-v1', 'arch': model.arch(),
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'label_order': list(cfg.labels.order), 'preprocess': cfg.preprocess.to_plain(),
            'decision_threshold': float(cfg.gate.decision_threshold),
            'created': datetime.now().isoformat(timespec='seconds'), **extra}


# ── main ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/surgical_base.yaml')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a config value')
    ap.add_argument('--resume', action='store_true', help='continue from <run>/state_last.pth')
    ap.add_argument('--max-steps', type=int, default=0, help='stop after N fold steps (testing)')
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out_root = resolve_path(cfg.run.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    gi = out_root / '.gitignore'
    if not gi.exists():
        gi.write_text('*\n', encoding='utf-8')
    run_dir = out_root / cfg.run.name
    state_path = run_dir / 'state_last.pth'
    if run_dir.exists() and state_path.exists() and not args.resume:
        sys.exit(f"Run folder {run_dir} already has a training state. Use --resume to continue it "
                 f"or --set run.name=<new name> for a new run.")
    (run_dir / 'accepted').mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / 'events.log')
    import yaml
    (run_dir / ('config_used.yaml' if not args.resume else 'config_resume.yaml')).write_text(
        yaml.safe_dump(cfg.to_plain(), allow_unicode=True, sort_keys=False), encoding='utf-8')

    # device
    dv = cfg.device
    device = torch.device('cuda' if dv.use_cuda_if_available and torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(int(dv.cpu_threads) or os.cpu_count() or 1)
    else:
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == 'cuda' and str(dv.amp).lower() in ('auto', 'true', '1')
    log(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else
                              f" threads={torch.get_num_threads()}") + f" amp={use_amp} torch={torch.__version__}")
    set_seed(int(cfg.run.seed))

    sc = cfg.surgical
    order = list(cfg.labels.order)
    datasets = list(cfg.datasets)
    rotation = D.FoldRotation(len(datasets), int(cfg.folds.k), int(cfg.folds.passes_per_dataset), int(cfg.run.seed))
    schedule = SurgicalSchedule(sc)
    selection, pass_info, pass_buf = {}, None, {'probs': [], 'targets': [], 'loss': []}
    last_accepted, orig_params, step = None, None, 0
    lr = float(cfg.training.lr)
    accepted_lr = lr

    # model
    if args.resume:
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        model = build_from_arch(st['arch']); model.load_state_dict(st['model'])
        rotation.load_state_dict(st['rotation']); schedule.load_state_dict(st['schedule'])
        selection, pass_info, pass_buf = st['selection'], st['pass_info'], st['pass_buf']
        last_accepted, orig_params, step, lr = st['last_accepted'], st['orig_params'], st['step'], st['lr']
        accepted_lr = st.get('accepted_lr', lr)
        log(f"resumed: pass {rotation.total_passes} step {step} cycle {schedule.cycle} phase {schedule.phase}")
    else:
        init = cfg.model.init
        if init == 'checkpoint':
            model = load_model_file(resolve_path(cfg.model.checkpoint), input_size=int(cfg.preprocess.size))
            log(f"model from checkpoint {cfg.model.checkpoint}")
        else:
            model = GlaucomaNet(default_channels(int(cfg.model.base)), int(cfg.model.hidden), len(order),
                                int(cfg.preprocess.size))
            if init == 'vgg16':
                init_from_vgg16(model); log("model initialised from VGG16 (ImageNet)")
            elif init != 'scratch':
                sys.exit(f"model.init must be vgg16 | checkpoint | scratch, got {init!r}")
        if model.num_classes != len(order) or model.input_size != int(cfg.preprocess.size):
            sys.exit(f"model has {model.num_classes} outputs / input {model.input_size}, config wants "
                     f"{len(order)} / {cfg.preprocess.size}")
        orig_params = count_params(model)
    model.to(device)
    opt = make_optimizer(model, cfg, lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if args.resume:
        opt.load_state_dict(st['optimizer'])
        if st.get('scaler') and use_amp:
            scaler.load_state_dict(st['scaler'])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.training.pos_weight, dtype=torch.float32, device=device))
    log(f"model: channels={model.channels} hidden={model.hidden} params={count_params(model):,} "
        f"({count_params(model) / orig_params:.1%} of start)")

    pass_fields = ['time', 'pass', 'cycle', 'phase', 'dataset', 'dropout', 'penalty', 'holding', 'lr', 'params',
                   'param_ratio', 'loss', 'fold_acc', 'fold_recall', 'test_acc', 'test_recall', 'gate_passed',
                   'sel_norm_ratio', 'action'] + [f'fold_{n}_{m}' for n in order for m in ('acc', 'recall')] \
        + [f'test_{n}_{m}' for n in order for m in ('acc', 'recall')]
    pass_log = CsvLog(run_dir / 'log.csv', pass_fields)
    step_log = CsvLog(run_dir / 'steps.csv', ['time', 'step', 'pass', 'dataset', 'train_fold', 'test_fold',
                                              'loss', 'acc', 'recall', 'dropout', 'penalty'])

    bundles = {}

    def bundle(i):
        if i not in bundles:
            for k in list(bundles):          # keep only the current dataset in memory
                del bundles[k]
            bundles[i] = Bundle(cfg, datasets[i], run_dir, device, log)
        return bundles[i]

    def save_state():
        atomic_save({'arch': model.arch(), 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                     'scaler': scaler.state_dict() if use_amp else None, 'rotation': rotation.state_dict(),
                     'schedule': schedule.state_dict(), 'selection': selection, 'pass_info': pass_info,
                     'pass_buf': pass_buf, 'last_accepted': last_accepted, 'orig_params': orig_params,
                     'step': step, 'lr': lr, 'accepted_lr': accepted_lr}, state_path)

    done, reason = False, ''
    t_pass = time.time()
    try:
        while not done:
            # ── start of a pass ────────────────────────────────────────────────────────
            if pass_info is None:
                pass_info = schedule.begin_pass()
                model.set_dropout(pass_info['dropout'], list(sc.dropout_layers))
                if pass_info['select_now']:
                    selection = S.select_units(model, schedule.cut_ratio, int(sc.min_channels),
                                               bool(sc.cut_fc_hidden), sc.importance)
                    n_sel = sum(len(v) for v in selection.values())
                    log(f"[cycle {schedule.cycle}] dropout reached {pass_info['dropout']:.2f}: selected {n_sel} "
                        f"units to cut (ratio {schedule.cut_ratio:.3f}): "
                        + ', '.join(f"{k}:{len(v)}" for k, v in selection.items()))
                pass_buf = {'probs': [], 'targets': [], 'loss': []}
                t_pass = time.time()

            # ── one fold step ──────────────────────────────────────────────────────────
            ds_i, tr_f, te_f = rotation.current()
            b = bundle(ds_i)
            loss = train_fold(model, b, tr_f, opt, scaler, crit, device, use_amp, cfg, pass_info, selection)
            probs, targets = predict(model, b, b.folds[te_f], device, use_amp)
            m = compute_metrics(probs, targets, order, float(cfg.labels.positive_threshold),
                                float(cfg.gate.decision_threshold))
            acc, rec = gate_values(m, cfg.gate.accuracy_labels, cfg.gate.recall_labels)
            pass_buf['probs'].append(probs); pass_buf['targets'].append(targets); pass_buf['loss'].append(loss)
            step += 1
            step_log.row(time=datetime.now().isoformat(timespec='seconds'), step=step,
                         **{'pass': rotation.total_passes + 1}, dataset=b.name, train_fold=tr_f, test_fold=te_f,
                         loss=loss, acc=acc, recall=rec, dropout=pass_info['dropout'], penalty=pass_info['penalty'])
            if step % max(1, int(cfg.logging.print_every_steps)) == 0:
                log(f"  step {step:5d} | pass {rotation.total_passes + 1} [{b.name}] train f{tr_f} -> test f{te_f} "
                    f"| loss {loss:.4f} | acc {fmt(acc)} recall {fmt(rec)} | drop {pass_info['dropout']:.2f} "
                    f"pen {pass_info['penalty']:.3f}")
            pass_done, switched = rotation.advance()

            # ── end of a pass ──────────────────────────────────────────────────────────
            if pass_done:
                if pass_info['apply_penalty'] and sc.penalty_mode == 'shrink' and selection:
                    S.shrink(model, selection, pass_info['penalty'])
                sel_ratio = S.selected_norm_ratio(model, selection) if selection else float('nan')
                fm = compute_metrics(np.concatenate(pass_buf['probs']), np.concatenate(pass_buf['targets']), order,
                                     float(cfg.labels.positive_threshold), float(cfg.gate.decision_threshold))
                f_acc, f_rec = gate_values(fm, cfg.gate.accuracy_labels, cfg.gate.recall_labels)
                tm, t_acc, t_rec = {}, float('nan'), float('nan')
                if b.test_idx:
                    tp, tt = predict(model, b, b.test_idx, device, use_amp)
                    tm = compute_metrics(tp, tt, order, float(cfg.labels.positive_threshold),
                                         float(cfg.gate.decision_threshold))
                    t_acc, t_rec = gate_values(tm, cfg.gate.accuracy_labels, cfg.gate.recall_labels)
                g_acc, g_rec = (t_acc, t_rec) if cfg.gate.source == 'test' and b.test_idx else (f_acc, f_rec)
                passed, degraded = gate_check(g_acc, g_rec, cfg.gate)
                phase_before, cycle_before = schedule.phase, schedule.cycle
                action = schedule.end_pass(passed, degraded)
                params = count_params(model)
                pass_log.row(time=datetime.now().isoformat(timespec='seconds'), **{'pass': rotation.total_passes},
                             cycle=cycle_before, phase=phase_before, dataset=b.name, dropout=pass_info['dropout'],
                             penalty=pass_info['penalty'], holding=int(pass_info['holding']), lr=lr, params=params,
                             param_ratio=params / orig_params, loss=float(np.mean(pass_buf['loss'])),
                             fold_acc=f_acc, fold_recall=f_rec, test_acc=t_acc, test_recall=t_rec,
                             gate_passed=int(passed), sel_norm_ratio=sel_ratio, action=action,
                             **{f'fold_{n}_{k}': fm.get(f'{n}_{k}', float('nan')) for n in order for k in ('acc', 'recall')},
                             **{f'test_{n}_{k}': tm.get(f'{n}_{k}', float('nan')) for n in order for k in ('acc', 'recall')})
                log(f"== pass {rotation.total_passes} done ({time.time() - t_pass:.0f}s) cycle {cycle_before} "
                    f"{phase_before}{' [hold]' if pass_info['holding'] else ''} | folds acc {fmt(f_acc)} recall "
                    f"{fmt(f_rec)} | test acc {fmt(t_acc)} recall {fmt(t_rec)} | gate {'OK' if passed else '--'} "
                    f"| drop {pass_info['dropout']:.2f} pen {pass_info['penalty']:.3f}"
                    + (f" | sel/kept |w| {sel_ratio:.3f}" if selection else '') + f" -> {action}")

                if action == 'accept':
                    path = run_dir / 'accepted' / f"cycle_{schedule.cycle:02d}.pth"
                    torch.save(model_file(model, cfg, {
                        'cycle': schedule.cycle, 'params': params, 'orig_params': orig_params,
                        'metrics': {'fold_acc': f_acc, 'fold_recall': f_rec, 'test_acc': t_acc, 'test_recall': t_rec,
                                    'fold': fm, 'test': tm}}), path)
                    last_accepted, accepted_lr = str(path), lr
                    schedule.after_accept()
                    log(f"[cycle {schedule.cycle}] ACCEPTED -> {path.name} ({params:,} params, "
                        f"{params / orig_params:.1%} of start)")
                    if schedule.cycle >= int(sc.max_cycles):
                        done, reason = True, f"max_cycles ({sc.max_cycles}) reached"
                    elif params <= float(sc.target_param_ratio) * orig_params:
                        done, reason = True, f"target size reached ({params / orig_params:.1%} of start)"
                    else:
                        schedule.start_ramp()
                elif action == 'cut':
                    before = count_params(model)
                    model = S.cut(model, selection).to(device)
                    selection = {}
                    schedule.after_cut()
                    lr = max(float(cfg.training.lr_min), lr * float(cfg.training.lr_cycle_decay))
                    opt = make_optimizer(model, cfg, lr)
                    after = count_params(model)
                    log(f"[cycle {schedule.cycle}] CUT: {before:,} -> {after:,} params ({after / orig_params:.1%} of "
                        f"start), channels {model.channels}, hidden {model.hidden}, lr {lr:.2e}")
                elif action == 'fail':
                    if last_accepted is None:
                        done, reason = True, (f"gate not reached in {schedule.max_learn()} passes - raise "
                                              f"surgical.max_learn_passes or check the data/gate")
                    else:
                        log(f"[cycle {schedule.cycle}] recovery failed in {schedule.max_learn()} passes -> back to "
                            f"{Path(last_accepted).name}")
                        model = load_model_file(last_accepted).to(device)
                        selection = {}
                        if schedule.retry_smaller():
                            lr = accepted_lr
                            opt = make_optimizer(model, cfg, lr)
                            log(f"  retry {schedule.retries} with cut ratio {schedule.cut_ratio:.3f}")
                        else:
                            done, reason = True, "recovery failed and no retry left"
                if switched:
                    log(f"[data] {cfg.folds.passes_per_dataset} passes done -> next dataset")
                pass_info = None

            if cfg.logging.save_state_every_step or pass_done:
                save_state()
            if args.max_steps and step >= args.max_steps:
                done, reason = True, f"--max-steps {args.max_steps}"
    except KeyboardInterrupt:
        log("interrupted - saving state (continue with --resume)")
        save_state()
        return 130

    save_state()
    log(f"finished: {reason}")
    if last_accepted:
        shutil.copyfile(last_accepted, run_dir / 'final_model.pth')
        log(f"final model: {run_dir / 'final_model.pth'} (= {Path(last_accepted).name})")
    else:
        log("no model passed the gate yet - there is no final model")
    return 0


if __name__ == '__main__':
    sys.exit(main())
