"""SurgicalSlicing: iterative train -> dropout ramp -> targeted penalty -> cut, with fold rotation
and mistake replay. One trainer for both steps (config key `task`):

    task: definitor   - optic nerve disc segmentation   (configs/definitor_*.yaml)
    task: classifier  - glaucoma classifier on disc crops (configs/classifier_*.yaml)

    python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml
    python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml --resume
    python code_new/train_surgical.py --config code_new/configs/classifier_laptop.yaml --set run.name=try2

Outputs (training-runs/<run.name>/):
    accepted/cycle_XX.pth   model that passed the gate after XX cuts (deployable)
    final_model.pth         the last accepted model
    state_last.pth          full resume point (written after every fold step)
    log.csv / steps.csv     one row per pass / per fold step
    mistakes.csv            samples ordered by error (what mistake replay repeats)
    events.log              everything printed
    split_<dataset>.json    patient folds + held-out test patients
"""
import argparse
import csv
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
from surgical.phases import SurgicalSchedule                              # noqa: E402
from surgical.tasks import get_task, load_model_any                       # noqa: E402


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, 'a', encoding='utf-8')

    def __call__(self, *parts):
        line = ' '.join(str(p) for p in parts)
        print(f"[{datetime.now():%H:%M:%S}] {line}", flush=True)
        self.f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {line}\n")
        self.f.flush()


class CsvLog:
    def __init__(self, path: Path, fields):
        new = not path.exists()
        self.f = open(path, 'a', encoding='utf-8', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction='ignore')
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in kw.items()})
        self.f.flush()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Bundle:
    """One dataset: samples, patient split, image cache and the two loaders."""

    def __init__(self, cfg, task, ds_cfg, run_dir: Path, device, log):
        self.name = ds_cfg.get('name') or 'dataset'
        log(f"[data] loading dataset '{self.name}'")
        self.samples = task.scan(ds_cfg)
        self.test_idx, self.folds = D.load_or_make_split(
            self.samples, int(cfg.folds.k), float(cfg.folds.test_fraction), int(cfg.run.seed),
            run_dir / f"split_{self.name}.json", log)
        self.store = D.ImageStore(task.load, bool(cfg.preprocess.get('cache_images', True)))
        t0 = time.time()
        self.store.preload(self.samples)
        log(f"[data] {len(self.samples)} samples, {len({s.group for s in self.samples})} patients, "
            f"folds {[len(f) for f in self.folds]}, test {len(self.test_idx)}"
            + (f", cached in {time.time() - t0:.1f}s" if cfg.preprocess.get('cache_images', True) else ''))
        tr = cfg.training
        nw = int(tr.num_workers)
        common = dict(batch_size=int(tr.batch_size), num_workers=nw, pin_memory=device.type == 'cuda',
                      persistent_workers=nw > 0)
        self.train_sampler = D.SettableSampler(shuffle=True, seed=int(cfg.run.seed))
        self.eval_sampler = D.SettableSampler(shuffle=False, seed=0)
        self.train_loader = DataLoader(D_dataset(self.samples, self.store, task, True), sampler=self.train_sampler,
                                       **common)
        self.eval_loader = DataLoader(D_dataset(self.samples, self.store, task, False), sampler=self.eval_sampler,
                                      **common)


def D_dataset(samples, store, task, train):
    from surgical.tasks import _TaskDataset
    return _TaskDataset(samples, store, task.item, train)


def train_fold(model, task, b, indices, opt, scaler, device, use_amp, cfg, pass_info, selection):
    model.train()
    b.train_sampler.set(indices)
    l2 = cfg.surgical.penalty_mode == 'l2' and pass_info['apply_penalty'] and selection
    lam = pass_info['penalty'] * float(cfg.surgical.l2_scale)
    clip = float(cfg.training.grad_clip)
    total, n = 0.0, 0
    for _ in range(int(cfg.folds.repeats_per_step)):
        for x, y, _ in b.train_loader:
            if x.shape[0] < 2 and n > 0:           # a 1-sample tail batch adds noise only
                continue
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = task.loss(logits, y).mean()
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
def evaluate(model, task, b, indices, device, use_amp, acc, tracker=None):
    """Adds predictions to `acc`; per-sample losses go to the mistake tracker. Returns mean loss."""
    model.eval()
    b.eval_sampler.set(indices)
    losses = []
    for x, y, idx in b.eval_loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        per = task.loss(logits, y)
        acc.add(logits, y)
        losses += per.tolist()
        if tracker is not None:
            tracker.update([b.samples[i].path for i in idx.tolist()], task.sample_error(logits, y).tolist())
    return float(np.mean(losses)) if losses else float('nan')


def make_optimizer(model, cfg, lr):
    tr = cfg.training
    return torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(tr.betas), weight_decay=float(tr.weight_decay))


def model_file(model, cfg, task, extra):
    return {'format': 'surgical-slicing-v1', 'arch': model.arch(),
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'preprocess': cfg.preprocess.to_plain(), 'decision_threshold': float(cfg.gate.decision_threshold),
            'created': datetime.now().isoformat(timespec='seconds'), **task.model_extra(), **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a config value')
    ap.add_argument('--resume', action='store_true', help='continue from <run>/state_last.pth')
    ap.add_argument('--max-steps', type=int, default=0, help='stop after N fold steps (testing)')
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out_root = resolve_path(cfg.run.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not (out_root / '.gitignore').exists():
        (out_root / '.gitignore').write_text('*\n', encoding='utf-8')
    run_dir = out_root / cfg.run.name
    state_path = run_dir / 'state_last.pth'
    if state_path.exists() and not args.resume:
        sys.exit(f"Run folder {run_dir} already has a training state. Use --resume to continue it "
                 f"or --set run.name=<new name> for a new run.")
    (run_dir / 'accepted').mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / 'events.log')
    import yaml
    (run_dir / ('config_resume.yaml' if args.resume else 'config_used.yaml')).write_text(
        yaml.safe_dump(cfg.to_plain(), allow_unicode=True, sort_keys=False), encoding='utf-8')

    dv = cfg.device
    device = torch.device('cuda' if dv.use_cuda_if_available and torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(int(dv.cpu_threads) or os.cpu_count() or 1)
    else:
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == 'cuda' and str(dv.amp).lower() in ('auto', 'true', '1')
    log(f"task={cfg.task} device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda'
                                               else f" threads={torch.get_num_threads()}")
        + f" amp={use_amp} torch={torch.__version__}")
    set_seed(int(cfg.run.seed))

    task = get_task(cfg, log)
    sc = cfg.surgical
    datasets = list(cfg.datasets)
    rotation = D.FoldRotation(len(datasets), int(cfg.folds.k), int(cfg.folds.passes_per_dataset), int(cfg.run.seed))
    schedule = SurgicalSchedule(sc)
    tracker = D.MistakeTracker(cfg.get('replay', {}))
    selection, pass_info, pass_buf = {}, None, {'loss': [], 'extra': 0}
    last_accepted, orig_params, step = None, None, 0
    lr = accepted_lr = float(cfg.training.lr)

    if args.resume:
        st = torch.load(state_path, map_location='cpu', weights_only=False)
        model = load_from_state(st)
        rotation.load_state_dict(st['rotation']); schedule.load_state_dict(st['schedule'])
        tracker.load_state_dict(st.get('tracker', {}))
        selection, pass_info, pass_buf = st['selection'], st['pass_info'], st['pass_buf']
        last_accepted, orig_params, step, lr = st['last_accepted'], st['orig_params'], st['step'], st['lr']
        accepted_lr = st.get('accepted_lr', lr)
        log(f"resumed: pass {rotation.total_passes} step {step} cycle {schedule.cycle} phase {schedule.phase}")
    else:
        if cfg.model.init == 'checkpoint':
            model = task.load_model(resolve_path(cfg.model.checkpoint))
            log(f"model from checkpoint {cfg.model.checkpoint}")
        elif cfg.model.init in ('vgg16', 'scratch'):
            model = task.build_model()
        else:
            sys.exit(f"model.init must be vgg16 | checkpoint | scratch, got {cfg.model.init!r}")
        orig_params = count_params(model)
    model.to(device)
    opt = make_optimizer(model, cfg, lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if args.resume:
        opt.load_state_dict(st['optimizer'])
        if st.get('scaler') and use_amp:
            scaler.load_state_dict(st['scaler'])
    log(f"model: {type(model).__name__} params={count_params(model):,} "
        f"({count_params(model) / orig_params:.1%} of start)")

    mf = task.metric_fields()
    pass_log = CsvLog(run_dir / 'log.csv', ['time', 'pass', 'cycle', 'phase', 'dataset', 'dropout', 'penalty',
                                            'holding', 'lr', 'params', 'param_ratio', 'loss', 'test_loss',
                                            'replay_extra', 'gate_passed', 'sel_norm_ratio', 'action']
                      + [f'fold_{k}' for k in mf] + [f'test_{k}' for k in mf])
    step_log = CsvLog(run_dir / 'steps.csv', ['time', 'step', 'pass', 'dataset', 'train_fold', 'test_fold',
                                              'train_items', 'replay_extra', 'loss', 'test_loss', 'dropout',
                                              'penalty'] + mf)
    bundles = {}

    def bundle(i):
        if i not in bundles:
            bundles.clear()
            bundles[i] = Bundle(cfg, task, datasets[i], run_dir, device, log)
        return bundles[i]

    def save_state():
        atomic_save({'model_class': type(model).__name__, 'arch': model.arch(), 'model': model.state_dict(),
                     'optimizer': opt.state_dict(), 'scaler': scaler.state_dict() if use_amp else None,
                     'rotation': rotation.state_dict(), 'schedule': schedule.state_dict(),
                     'tracker': tracker.state_dict(), 'selection': selection, 'pass_info': pass_info,
                     'pass_buf': pass_buf, 'last_accepted': last_accepted, 'orig_params': orig_params,
                     'step': step, 'lr': lr, 'accepted_lr': accepted_lr}, state_path)

    done, reason = False, ''
    fold_acc = task.new_accumulator()
    t_pass = time.time()
    try:
        while not done:
            if pass_info is None:                                     # ── start of a pass
                pass_info = schedule.begin_pass()
                model.set_dropout(pass_info['dropout'], list(sc.dropout_layers) if sc.get('dropout_layers') else None)
                if pass_info['select_now']:
                    selection = S.select_units(model, schedule.cut_ratio, int(sc.min_channels),
                                               task.cut_exclude(), sc.importance)
                    log(f"[cycle {schedule.cycle}] dropout reached {pass_info['dropout']:.2f}: selected "
                        f"{sum(len(v) for v in selection.values())} units to cut (ratio {schedule.cut_ratio:.3f}): "
                        + ', '.join(f"{k}:{len(v)}" for k, v in selection.items()))
                pass_buf = {'loss': [], 'extra': 0}
                fold_acc = task.new_accumulator()
                t_pass = time.time()

            ds_i, tr_f, te_f = rotation.current()                    # ── one fold step
            b = bundle(ds_i)
            train_idx, extra = tracker.expand(b.folds[tr_f], b.samples)
            loss = train_fold(model, task, b, train_idx, opt, scaler, device, use_amp, cfg, pass_info, selection)
            step_acc = task.new_accumulator()
            test_loss = evaluate(model, task, b, b.folds[te_f], device, use_amp, _Both(step_acc, fold_acc), tracker)
            sm = step_acc.result()
            pass_buf['loss'].append(loss); pass_buf['extra'] += extra
            step += 1
            step_log.row(time=datetime.now().isoformat(timespec='seconds'), step=step,
                         **{'pass': rotation.total_passes + 1}, dataset=b.name, train_fold=tr_f, test_fold=te_f,
                         train_items=len(train_idx), replay_extra=extra, loss=loss, test_loss=test_loss,
                         dropout=pass_info['dropout'], penalty=pass_info['penalty'], **{k: sm.get(k) for k in mf})
            if step % max(1, int(cfg.logging.print_every_steps)) == 0:
                log(f"  step {step:5d} | pass {rotation.total_passes + 1} [{b.name}] train f{tr_f}"
                    f"{f' (+{extra} replay)' if extra else ''} -> test f{te_f} | loss {loss:.4f} | {task.short(sm)}"
                    f" | drop {pass_info['dropout']:.2f} pen {pass_info['penalty']:.3f}")
            pass_done, switched = rotation.advance()

            if pass_done:                                             # ── end of a pass
                if pass_info['apply_penalty'] and sc.penalty_mode == 'shrink' and selection:
                    S.shrink(model, selection, pass_info['penalty'])
                sel_ratio = S.selected_norm_ratio(model, selection) if selection else float('nan')
                fm = fold_acc.result()
                tm, t_loss = {}, float('nan')
                if b.test_idx:
                    ta = task.new_accumulator()
                    t_loss = evaluate(model, task, b, b.test_idx, device, use_amp, ta)
                    tm = ta.result()
                gm = tm if cfg.gate.source == 'test' and tm else fm
                passed, degraded = task.gate(gm)
                phase_before, cycle_before = schedule.phase, schedule.cycle
                action = schedule.end_pass(passed, degraded)
                params = count_params(model)
                pass_log.row(time=datetime.now().isoformat(timespec='seconds'), **{'pass': rotation.total_passes},
                             cycle=cycle_before, phase=phase_before, dataset=b.name, dropout=pass_info['dropout'],
                             penalty=pass_info['penalty'], holding=int(pass_info['holding']), lr=lr, params=params,
                             param_ratio=params / orig_params, loss=float(np.mean(pass_buf['loss'])),
                             test_loss=t_loss, replay_extra=pass_buf['extra'], gate_passed=int(passed),
                             sel_norm_ratio=sel_ratio, action=action,
                             **{f'fold_{k}': fm.get(k) for k in mf}, **{f'test_{k}': tm.get(k) for k in mf})
                tracker.write_csv(run_dir / 'mistakes.csv')
                log(f"== pass {rotation.total_passes} ({time.time() - t_pass:.0f}s) cycle {cycle_before} "
                    f"{phase_before}{' [hold]' if pass_info['holding'] else ''} | folds {task.short(fm)}"
                    + (f" | test {task.short(tm)}" if tm else '')
                    + f" | gate {'OK' if passed else '--'} | drop {pass_info['dropout']:.2f} "
                      f"pen {pass_info['penalty']:.3f}" + (f" | sel/kept |w| {sel_ratio:.3f}" if selection else '')
                    + (f" | replay +{pass_buf['extra']}" if pass_buf['extra'] else '') + f" -> {action}")

                if action == 'accept':
                    path = run_dir / 'accepted' / f"cycle_{schedule.cycle:02d}.pth"
                    torch.save(model_file(model, cfg, task, {
                        'cycle': schedule.cycle, 'params': params, 'orig_params': orig_params,
                        'metrics': {'fold': fm, 'test': tm}}), path)
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
                    log(f"[cycle {schedule.cycle}] CUT: {before:,} -> {after:,} params ({after / orig_params:.1%} "
                        f"of start), lr {lr:.2e}")
                elif action == 'fail':
                    if last_accepted is None:
                        done, reason = True, (f"gate not reached in {schedule.max_learn()} passes - raise "
                                              f"surgical.max_learn_passes or check the data/gate")
                    else:
                        log(f"[cycle {schedule.cycle}] recovery failed in {schedule.max_learn()} passes -> back to "
                            f"{Path(last_accepted).name}")
                        model = load_model_any(last_accepted).to(device)
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


class _Both:
    """feed one evaluation into two accumulators (this step + the whole pass)"""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def add(self, logits, y):
        self.a.add(logits, y); self.b.add(logits, y)


def load_from_state(st):
    if st.get('model_class') == 'DefinitorNet':
        from surgical.definitor_model import DefinitorNet
        m = DefinitorNet(**st['arch'])
    else:
        from surgical.model import build_from_arch
        m = build_from_arch(st['arch'])
    m.load_state_dict(st['model'])
    return m


if __name__ == '__main__':
    sys.exit(main())
