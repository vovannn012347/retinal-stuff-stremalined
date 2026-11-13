"""SurgicalSlicing: iterative train -> dropout ramp -> targeted penalty -> cut, with fold rotation
and mistake replay. One trainer for both steps (config key `task`):

    task: definitor   - optic nerve disc segmentation   (configs/definitor_*.yaml)
    task: classifier  - glaucoma classifier on disc crops (configs/classifier_*.yaml)

    python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml
    python code_new/train_surgical.py --config code_new/configs/definitor_gpu.yaml --resume
    python code_new/train_surgical.py --config ... --resume --resume-from training-runs/<run>/backups/state_pass_0012.pth
    python code_new/train_surgical.py --config code_new/configs/classifier_laptop.yaml --set run.name=try2

Outputs (training-runs/<run.name>/):
    accepted/cycle_XX.pth   model that passed the gate after XX cuts (deployable)
    final_model.pth         the last accepted model
    state_last.pth          full resume point (written after every fold step)
    slicing_registry.json   every cut: layer, removed indices (current + original numbering), kept indices
    backups/state_pass_*.pth  copies every logging.backup_every_passes passes (the newest backup_keep kept);
                            --resume falls back to them automatically if state_last.pth is damaged
    log.csv / steps.csv     one row per pass / per fold step
    mistakes.csv            samples ordered by error (what mistake replay repeats)
    test_per_image.csv      every pass: recall / precision / error of every held-out image
    worst/*.png + .csv      the worst held-out images (input | output | true mask), every
                            logging.worst_every_passes passes and at every accepted model
    best_in_phase.pth       gate.mode plateau: best weights of the current learn phase
    samples/pass_NNNN_images.png + pass_NNNN_masks.png   definitor: inputs and predicted masks (white on
                            black) of fixed held-out images + the current worst mistakes, same grid;
                            every logging.sample_every_passes passes; accepted_cycle_XX_* at each accept
    events.log              everything printed
    split_<dataset>.json    patient folds + held-out test patients
"""
import argparse
import csv
import json
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
        new = not path.exists() or path.stat().st_size == 0
        if not new:   # resumed run: keep the columns the file already has (a newer code version may log more)
            with open(path, encoding='utf-8', newline='') as f:
                head = next(csv.reader(f), None)
            if head:
                fields = head
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
    """write to a temp file, flush to disk, then rename: a crash mid-write never damages the old file"""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_resume_state(run_dir: Path, explicit, log):
    """state_last.pth, else the newest readable backup (after a hard crash the last write may be damaged)."""
    cands = [Path(explicit)] if explicit else \
        [run_dir / 'state_last.pth'] + sorted((run_dir / 'backups').glob('state_pass_*.pth'), reverse=True)
    for p in cands:
        if not p.is_file():
            continue
        try:
            st = torch.load(p, map_location='cpu', weights_only=False)
            if p.name != 'state_last.pth':
                log(f"resuming from backup {p.name}")
            return st
        except Exception as e:                      # truncated / corrupted file
            log(f"cannot read {p.name} ({type(e).__name__}: {e}) - trying an older one")
    sys.exit(f"no readable training state in {run_dir} (state_last.pth / backups/)")


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
        self.cfg, self.task, self.device = cfg, task, device
        self.train_sampler = D.SettableSampler(shuffle=True, seed=int(cfg.run.seed))
        self.eval_sampler = D.SettableSampler(shuffle=False, seed=0)
        self.set_batch(int(cfg.training.batch_size))

    def set_batch(self, bs):
        self.batch_size = bs
        nw = int(self.cfg.training.num_workers)
        # label learning changes the training targets between steps: workers must be restarted to see them
        persistent = nw > 0 and getattr(self.task, 'label_learner', None) is None
        common = dict(batch_size=bs, num_workers=nw, pin_memory=self.device.type == 'cuda', persistent_workers=persistent)
        self.train_loader = DataLoader(D_dataset(self.samples, self.store, self.task, True),
                                       sampler=self.train_sampler, **common)
        self.eval_loader = DataLoader(D_dataset(self.samples, self.store, self.task, False),
                                      sampler=self.eval_sampler, **common)


def D_dataset(samples, store, task, train):
    from surgical.tasks import _TaskDataset
    return _TaskDataset(samples, store, task.item, train)


class LrControl:
    """Per-group learning rates: encoder (pretrained) = lr, new layers = lr * new_layers_lr_mult.
    Linear warm-up over the first warmup_steps optimizer steps of every new optimizer
    (start, after a cut, after a retry)."""

    def __init__(self, cfg):
        self.mult = float(cfg.training.get('new_layers_lr_mult', 1.0))
        self.warmup = int(cfg.training.get('warmup_steps', 0))
        self.steps = 0

    def reset(self):
        self.steps = 0

    def apply(self, opt, lr):
        scale = min(1.0, (self.steps + 1) / self.warmup) if self.warmup > 0 else 1.0
        for g in opt.param_groups:
            g['lr'] = lr * (self.mult if g.get('name') == 'new' else 1.0) * scale
        self.steps += 1


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()


def train_fold(model, task, b, indices, opt, scaler, device, use_amp, cfg, pass_info, selection, lr, lrc, accum):
    model.train()
    b.train_sampler.set(indices)
    l2 = cfg.surgical.penalty_mode == 'l2' and pass_info['apply_penalty'] and selection
    lam = pass_info['penalty'] * float(cfg.surgical.l2_scale)
    clip = float(cfg.training.grad_clip)
    eff = b.batch_size * accum
    repeats = int(cfg.folds.repeats_per_step)
    min_steps = int(cfg.folds.get('min_steps_per_step', 0))
    if min_steps and indices:
        repeats = max(repeats, math.ceil(min_steps * eff / len(indices)))
    total, n, micro, opt_steps = 0.0, 0, 0, 0

    def step():
        nonlocal opt_steps
        lrc.apply(opt, lr)
        if clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        opt_steps += 1

    opt.zero_grad(set_to_none=True)
    fake_limit = int(os.environ.get('SURGICAL_TEST_OOM', 0))       # test hook: pretend the GPU holds N images
    if fake_limit and b.batch_size > fake_limit:
        raise RuntimeError(f'CUDA out of memory (simulated: SURGICAL_TEST_OOM={fake_limit})')
    for _ in range(repeats):
        for x, y, _ in b.train_loader:
            if x.shape[0] < 2 <= b.batch_size and n > 0:     # a 1-sample tail batch adds noise only
                continue
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = task.loss(logits, y).mean()
            if l2:
                loss = loss + lam * S.l2_penalty(model, selection)
            scaler.scale(loss / accum).backward()
            micro += 1
            if micro % accum == 0:
                step()
            total += loss.item() * x.shape[0]; n += x.shape[0]
    if micro % accum:
        step()
    return total / max(1, n), opt_steps


@torch.no_grad()
def evaluate(model, task, b, indices, device, use_amp, acc, tracker=None, per_image=None, learner=None, passes=0):
    """Adds predictions to `acc`; per-sample errors go to the mistake tracker; per-image results are
    appended to `per_image` as (sample index, dict) if a list is given. Returns mean loss."""
    model.eval()
    b.eval_sampler.set(indices)
    losses, paths, errs = [], [], []
    for x, y, idx in b.eval_loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        per = task.loss(logits, y)
        acc.add(logits, y)
        losses += per.tolist()
        if tracker is not None:
            paths += [b.samples[i].path for i in idx.tolist()]
            errs += task.sample_error(logits, y).tolist()
        if per_image is not None:
            per_image += list(zip(idx.tolist(), task.per_image(logits, y)))
        if learner is not None and learner.enabled:
            learner.update([b.samples[i] for i in idx.tolist()], torch.sigmoid(logits.float()).cpu().numpy(), passes)
    if tracker is not None:
        tracker.update(paths, errs)
    return float(np.mean(losses)) if losses else float('nan')


def make_optimizer(model, cfg, lr):
    """AdamW, two groups: pretrained encoder and newly initialised layers (see LrControl)."""
    tr = cfg.training
    enc, new = model.param_groups()
    return torch.optim.AdamW([{'params': enc, 'name': 'enc'}, {'params': new, 'name': 'new'}], lr=lr,
                             betas=tuple(tr.betas), weight_decay=float(tr.weight_decay))


def model_file(model, cfg, task, extra):
    return {'format': 'surgical-slicing-v1', 'arch': model.arch(),
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'preprocess': cfg.preprocess.to_plain(), 'decision_threshold': float(cfg.gate.decision_threshold),
            'created': datetime.now().isoformat(timespec='seconds'), **task.model_extra(), **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a config value')
    ap.add_argument('--resume', action='store_true', help='continue from <run>/state_last.pth (or the newest backup)')
    ap.add_argument('--resume-from', metavar='PATH', help='with --resume: a specific state / backup file')
    ap.add_argument('--max-steps', type=int, default=0, help='stop after N fold steps (testing)')
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out_root = resolve_path(cfg.run.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not (out_root / '.gitignore').exists():
        (out_root / '.gitignore').write_text('*\n', encoding='utf-8')
    run_dir = out_root / cfg.run.name
    state_path = run_dir / 'state_last.pth'
    if (state_path.exists() or (run_dir / 'backups').is_dir()) and not args.resume:
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
    learner = D.LabelLearner(cfg.labels.get('learning', {}) if task.name == 'classifier' else {},
                             getattr(task, 'order', []))
    if learner.enabled:
        task.label_learner = learner
    selection, pass_info, pass_buf = {}, None, {'loss': [], 'extra': 0}
    registry, channel_map = [], {}  # slicing registry: every cut, in current and original channel numbering
    last_accepted, orig_params, step = None, None, 0
    lr = accepted_lr = float(cfg.training.lr)
    lrc = LrControl(cfg)
    phys_batch = int(cfg.training.batch_size)
    eff_batch = int(cfg.training.get('effective_batch_size') or phys_batch)
    accum = max(1, round(eff_batch / phys_batch))
    plateau = {'best': float('inf'), 'wait': 0}
    pg = {}                       # plateau gate state of the current learn phase (gate.mode: plateau)
    accepted_score = None         # smoothed gate score of the last accepted model

    def selection_mode():
        """what decides the selected channels; a resumed selection made another way is redone"""
        imp = str(sc.importance)
        return f"{imp}:{sc.get('index_pattern', 'interleaved')}" if imp == 'index' else imp

    def select_now_units():
        return S.select_units(model, schedule.cut_ratio, int(sc.min_channels), task.cut_exclude(),
                              str(sc.importance), str(sc.get('index_pattern', 'interleaved')))

    if args.resume:
        st = load_resume_state(run_dir, args.resume_from, log)
        model = load_from_state(st)
        rotation.load_state_dict(st['rotation']); schedule.load_state_dict(st['schedule'])
        tracker.load_state_dict(st.get('tracker', {}))
        learner.load_state_dict(st.get('label_learner', {}))
        selection, pass_info, pass_buf = st['selection'], st['pass_info'], st['pass_buf']
        last_accepted, orig_params, step, lr = st['last_accepted'], st['orig_params'], st['step'], st['lr']
        accepted_lr = st.get('accepted_lr', lr)
        phys_batch, accum = st.get('phys_batch', phys_batch), st.get('accum', accum)
        plateau = st.get('plateau', plateau)
        pg, accepted_score = st.get('pg', {}), st.get('accepted_score')
        lrc.steps = st.get('lr_steps', 0)
        registry, channel_map = st.get('registry', []), st.get('channel_map', {})
        stored_mode = st.get('selection_mode', 'l1')   # states from before index selection used l1
        if selection and stored_mode != selection_mode():
            # the channels were chosen another way (e.g. l1 before index selection existed): choose them
            # again from the current config and restart the penalty; dropout keeps its value
            schedule.cut_ratio = float(sc.cut_ratio) * float(sc.retry_ratio_factor) ** schedule.retries
            old_n = sum(len(v) for v in selection.values())
            selection = select_now_units()
            schedule.penalty = 0.0
            if pass_info:
                pass_info['penalty'], pass_info['apply_penalty'] = 0.0, False
            log(f"selection was made by '{stored_mode}' ({old_n} units); re-selected by '{selection_mode()}' "
                f"with ratio {schedule.cut_ratio:.3f}: {sum(len(v) for v in selection.values())} units "
                f"({', '.join(f'{k}:{len(v)}' for k, v in selection.items())}); penalty restarts from 0 "
                f"at dropout {schedule.dropout:.2f}")
        if not selection:   # nothing selected yet: take cut_ratio from the (possibly edited) config
            schedule.cut_ratio = float(sc.cut_ratio) * float(sc.retry_ratio_factor) ** schedule.retries
        log(f"resumed: pass {rotation.total_passes} step {step} cycle {schedule.cycle} phase {schedule.phase}")
    else:
        if cfg.model.init == 'checkpoint':
            model = task.load_model(resolve_path(cfg.model.checkpoint))
            log(f"model from checkpoint {cfg.model.checkpoint}")
            ck = torch.load(resolve_path(cfg.model.checkpoint), map_location='cpu', weights_only=False)
            if isinstance(ck, dict) and ck.get('slicing_registry') is not None:
                # continue slicing an accepted model: keep its cut history in the ORIGINAL channel numbering
                registry, channel_map = list(ck['slicing_registry']), dict(ck.get('channel_map') or {})
                log(f"  continuing its slicing registry: {len(registry)} earlier cut(s)")
        elif cfg.model.init in ('vgg16', 'scratch'):
            model = task.build_model()
        else:
            sys.exit(f"model.init must be vgg16 | checkpoint | scratch, got {cfg.model.init!r}")
        orig_params = count_params(model)
        if cfg.model.init == 'checkpoint' and isinstance(ck, dict) and ck.get('orig_params'):
            orig_params = int(ck['orig_params'])     # sizes stay relative to the original full model
    model.to(device)
    opt = make_optimizer(model, cfg, lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if args.resume:
        opt.load_state_dict(st['optimizer'])
        if st.get('scaler') and use_amp:
            scaler.load_state_dict(st['scaler'])
    log(f"model: {type(model).__name__} params={count_params(model):,} "
        f"({count_params(model) / orig_params:.1%} of start)")
    log(f"batch: {phys_batch} x {accum} accumulation = {phys_batch * accum} per optimizer step; lr {lr:.2e} "
        f"(new layers x{lrc.mult:g}), weight decay {float(cfg.training.weight_decay):g}")

    mf = task.metric_fields()
    pass_log = CsvLog(run_dir / 'log.csv', ['time', 'pass', 'cycle', 'phase', 'dataset', 'dropout', 'penalty',
                                            'holding', 'lr', 'batch', 'params', 'param_ratio', 'loss', 'test_loss',
                                            'gate_source', 'overfit_gap',
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
            if bundles[i].batch_size != phys_batch:
                bundles[i].set_batch(phys_batch)
        return bundles[i]

    def save_state():
        atomic_save({'model_class': type(model).__name__, 'arch': model.arch(), 'model': model.state_dict(),
                     'optimizer': opt.state_dict(), 'scaler': scaler.state_dict() if use_amp else None,
                     'rotation': rotation.state_dict(), 'schedule': schedule.state_dict(),
                     'tracker': tracker.state_dict(), 'label_learner': learner.state_dict(), 'selection': selection, 'pass_info': pass_info,
                     'pass_buf': pass_buf, 'last_accepted': last_accepted, 'orig_params': orig_params,
                     'step': step, 'lr': lr, 'accepted_lr': accepted_lr, 'phys_batch': phys_batch, 'accum': accum,
                     'plateau': plateau, 'lr_steps': lrc.steps, 'pg': pg, 'accepted_score': accepted_score,
                     'registry': registry, 'channel_map': channel_map, 'selection_mode': selection_mode()},
                    state_path)

    def write_registry():
        tmp = run_dir / 'slicing_registry.json.tmp'
        tmp.write_text(json.dumps({'format': 'surgical-slicing-registry-v1', 'cuts': registry,
                                   'channel_map': channel_map}, indent=1), encoding='utf-8')
        os.replace(tmp, run_dir / 'slicing_registry.json')

    def backup_state():
        every = int(cfg.logging.get('backup_every_passes', 0))
        if every <= 0 or rotation.total_passes % every:
            return
        bdir = run_dir / 'backups'
        bdir.mkdir(exist_ok=True)
        dst = bdir / f"state_pass_{rotation.total_passes:04d}.pth"
        tmp = dst.with_suffix('.pth.tmp')
        shutil.copyfile(state_path, tmp)
        os.replace(tmp, dst)
        for old in sorted(bdir.glob('state_pass_*.pth'))[:-max(1, int(cfg.logging.get('backup_keep', 5)))]:
            old.unlink()

    def sample_indices(b):
        """fixed images (same every time, so the progress is comparable) + current worst mistakes"""
        n_fixed, n_worst = int(cfg.logging.get('sample_count', 8)), int(cfg.logging.get('sample_worst', 4))
        pool = b.test_idx or [i for f in b.folds for i in f]
        pool = sorted(pool, key=lambda i: Path(b.samples[i].path).name)
        step_ = max(1, len(pool) // max(1, n_fixed))
        fixed = pool[::step_][:n_fixed]
        pos = {s.path: i for i, s in enumerate(b.samples)}
        worst = [pos[p] for p, r in sorted(tracker.rec.items(), key=lambda kv: -kv[1]['err'])
                 if p in pos and pos[p] not in fixed][:n_worst]
        return fixed + worst

    def write_samples(b, name, title):
        every = int(cfg.logging.get('sample_every_passes', 1))
        if every <= 0:
            return
        try:
            task.render_samples(model, b.samples, b.store, sample_indices(b), device,
                                run_dir / 'samples' / f"{name}.png", title)
        except Exception as e:                                    # never stop training for a picture
            log(f"[samples] could not write {name}.png: {type(e).__name__}: {e}")

    last_test_rows = {'rows': [], 'bundle': None}

    def write_test_rows(b, rows, sched):
        """per-image results of the held-out patients, appended every pass -> test_per_image.csv"""
        last_test_rows['rows'], last_test_rows['bundle'] = rows, b
        path = run_dir / 'test_per_image.csv'
        new = not path.exists()
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['pass', 'cycle', 'phase', 'dataset', 'image', 'recall', 'precision', 'error',
                            'disc_px_true', 'disc_px_found'])
            for i, r in rows:
                w.writerow([rotation.total_passes, sched.cycle, sched.phase, b.name, Path(b.samples[i].path).name,
                            f"{r['recall']:.4f}", f"{r['precision']:.4f}", f"{r['error']:.4f}",
                            r.get('disc_px_true', ''), r.get('disc_px_found', '')])

    def write_worst(b, name, force=False):
        """worst/<name>.png + .csv: the worst held-out images (input | output | true mask)"""
        every = int(cfg.logging.get('worst_every_passes', 5))
        if not force and (every <= 0 or rotation.total_passes % every):
            return
        rows = [r for r in last_test_rows['rows']] if last_test_rows['bundle'] is b else []
        if not rows:
            return
        rows.sort(key=lambda r: -r[1]['error'])
        rows = rows[:int(cfg.logging.get('worst_count', 8))]
        wdir = run_dir / 'worst'
        wdir.mkdir(exist_ok=True)
        with open(wdir / f"{name}.csv", 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'recall', 'precision', 'error', 'path'])
            for i, r in rows:
                w.writerow([Path(b.samples[i].path).name, f"{r['recall']:.4f}", f"{r['precision']:.4f}",
                            f"{r['error']:.4f}", b.samples[i].path])
        try:
            task.render_samples(model, b.samples, b.store, [i for i, _ in rows], device, wdir / f"{name}.png",
                                f"worst held-out images, {name}", with_truth=True)
        except Exception as e:
            log(f"[worst] could not write {name}.png: {type(e).__name__}: {e}")

    def plateau_gate(m, floors_ok):
        """gate.mode plateau. Score = task.score (definitor: F2, recall-weighted), smoothed over passes.
        Learn phase: accept the best model of the phase once the score has not improved for
        plateau_passes passes - if it meets the floors (gate.recall / precision / max_spill) and, after a
        cut, is within max_drop of the accepted model's score; give up (-> retry smaller cut) if a
        plateau is reached below that. Ramp: degraded = score below accepted - max_drop - tolerance."""
        g = cfg.gate
        s = task.score(m)
        s = 0.0 if s is None or math.isnan(s) else s
        pg['ema'] = s if pg.get('ema') is None else 0.5 * pg['ema'] + 0.5 * s
        max_drop, tol = float(g.get('max_drop', 0.02)), float(g.get('degrade_tolerance', 0.02))
        if schedule.phase != 'learn':
            ref = accepted_score if accepted_score is not None else pg['ema']
            return False, pg['ema'] < ref - max_drop - tol, False
        if pg['ema'] > pg.get('best', -1.0) + float(g.get('plateau_min_delta', 0.002)):
            pg.update(best=pg['ema'], since=0, best_floors_ok=bool(floors_ok))
            snap = run_dir / 'best_in_phase.pth'
            atomic_save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                         'fold': fm_holder['fm'], 'test': fm_holder['tm'], 'pass': rotation.total_passes,
                         'score': pg['ema']}, snap)
            pg['snapshot'] = str(snap)
        else:
            pg['since'] = pg.get('since', 0) + 1
        plateaued = pg['since'] >= int(g.get('plateau_passes', 6))
        target = None if accepted_score is None else accepted_score - max_drop
        good = pg.get('best_floors_ok', False) and (target is None or pg['best'] >= target)
        log(f"[gate] score {s:.4f} smoothed {pg['ema']:.4f}, best {pg['best']:.4f} "
            f"({pg['since']} passes without improvement)" + (f", target {target:.4f}" if target is not None else ''))
        give_up = plateaued and not good and target is not None
        return plateaued and good, False, give_up

    fm_holder = {'fm': {}, 'tm': {}}
    done, reason = False, ''
    fold_acc = task.new_accumulator()
    t_pass = time.time()
    try:
        while not done:
            if pass_info is None:                                     # ── start of a pass
                pass_info = schedule.begin_pass()
                model.set_dropout(pass_info['dropout'], list(sc.dropout_layers) if sc.get('dropout_layers') else None)
                if pass_info['select_now']:
                    selection = select_now_units()
                    if not selection:
                        done, reason = True, f"nothing left to cut (every layer at min_channels {sc.min_channels})"
                        pass_info = None
                        break
                    log(f"[cycle {schedule.cycle}] dropout reached {pass_info['dropout']:.2f}: selected "
                        f"{sum(len(v) for v in selection.values())} units to cut (ratio {schedule.cut_ratio:.3f}): "
                        + ', '.join(f"{k}:{len(v)}" for k, v in selection.items()))
                pass_buf = {'loss': [], 'extra': 0, 'tloss': []}
                fold_acc = task.new_accumulator()
                t_pass = time.time()

            ds_i, tr_f, te_f = rotation.current()                    # ── one fold step
            b = bundle(ds_i)
            train_idx, extra = tracker.expand(b.folds[tr_f], b.samples)
            while True:                                               # retry the step with a smaller batch on OOM
                try:
                    loss, _ = train_fold(model, task, b, train_idx, opt, scaler, device, use_amp, cfg, pass_info,
                                         selection, lr, lrc, accum)
                    break
                except RuntimeError as e:
                    if not is_oom(e) or phys_batch <= 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    phys_batch //= 2
                    accum *= 2
                    b.set_batch(phys_batch)
                    log(f"[memory] out of memory -> batch {phys_batch} x {accum} accumulation "
                        f"(same {phys_batch * accum} per optimizer step); repeating this step")
            step_acc = task.new_accumulator()
            test_loss = evaluate(model, task, b, b.folds[te_f], device, use_amp, _Both(step_acc, fold_acc), tracker,
                                 learner=learner if schedule.phase == 'learn' else None,   # not while the ramp degrades on purpose
                                 passes=rotation.total_passes)
            sm = step_acc.result()
            pass_buf['loss'].append(loss); pass_buf['extra'] += extra
            pass_buf.setdefault('tloss', []).append(test_loss)
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
                tm, t_loss, test_rows = {}, float('nan'), []
                if b.test_idx:
                    ta = task.new_accumulator()
                    test_rows = []
                    t_loss = evaluate(model, task, b, b.test_idx, device, use_amp, ta, per_image=test_rows)
                    tm = ta.result()
                    write_test_rows(b, test_rows, schedule)
                fm_holder['fm'], fm_holder['tm'] = fm, tm
                g_src = 'test' if cfg.gate.source == 'test' and tm else 'folds'
                gm = tm if g_src == 'test' else fm
                passed, degraded = task.gate(gm)
                give_up, immediate = False, False
                gmode = str(cfg.gate.get('mode', 'average'))
                if gmode == 'plateau':
                    passed, degraded, give_up = plateau_gate(gm, passed)
                    immediate = True
                elif gmode == 'average':
                    # thresholds on the mean of the last `window` passes of this phase (noise-proof)
                    hist = pg.setdefault('hist', [])
                    hist.append({k: v for k, v in gm.items() if isinstance(v, (int, float))})
                    del hist[:-int(cfg.gate.get('window', 5))]
                    keys = set().union(*hist)
                    avg = {k: float(np.nanmean([h.get(k, float('nan')) for h in hist])) for k in keys}
                    passed, degraded = task.gate(avg)
                    passed = passed and len(hist) >= min(int(cfg.gate.get('window', 5)), 3)
                    immediate = True
                    log(f"[gate] mean of the last {len(hist)} passes ({g_src}): {task.short(avg)}")
                elif gmode != 'absolute':
                    sys.exit(f"gate.mode must be average | plateau | absolute, got {gmode!r}")
                gap = (fm.get('gate_recall', float('nan')) - tm.get('gate_recall', float('nan'))) if tm else float('nan')
                # reduce-on-plateau, learn phase only. Signal: the next-fold test loss of the pass (every
                # sample once, far less noisy than the few held-out patients), smoothed over passes.
                if schedule.phase == 'learn':
                    raw = float(np.nanmean(pass_buf['tloss'])) if pass_buf.get('tloss') else float('nan')
                    if not math.isnan(raw):
                        sm = plateau.get('ema')
                        plateau['ema'] = raw if sm is None else 0.7 * sm + 0.3 * raw
                    cur = plateau.get('ema', float('nan'))
                    if not math.isnan(cur):
                        if cur < plateau['best'] * (1 - float(cfg.training.get('plateau_min_delta', 0.01))):
                            plateau['best'], plateau['wait'] = cur, 0
                        else:
                            plateau['wait'] += 1
                            if plateau['wait'] >= int(cfg.training.get('plateau_passes', 12)) and \
                                    plateau.get('cuts', 0) < int(cfg.training.get('plateau_max_reductions', 3)) and \
                                    lr > float(cfg.training.lr_min):
                                lr = max(float(cfg.training.lr_min), lr * float(cfg.training.get('plateau_factor', 0.5)))
                                plateau['wait'], plateau['cuts'] = 0, plateau.get('cuts', 0) + 1
                                log(f"[lr] no improvement for {cfg.training.get('plateau_passes', 12)} passes -> "
                                    f"lr {lr:.2e} (reduction {plateau['cuts']} of "
                                    f"{cfg.training.get('plateau_max_reductions', 3)} in this phase)")
                phase_before, cycle_before = schedule.phase, schedule.cycle
                action = schedule.end_pass(passed, degraded, give_up, immediate)
                params = count_params(model)
                pass_log.row(time=datetime.now().isoformat(timespec='seconds'), **{'pass': rotation.total_passes},
                             cycle=cycle_before, phase=phase_before, dataset=b.name, dropout=pass_info['dropout'],
                             penalty=pass_info['penalty'], holding=int(pass_info['holding']), lr=lr,
                             batch=f"{phys_batch}x{accum}", gate_source=g_src, overfit_gap=gap, params=params,
                             param_ratio=params / orig_params, loss=float(np.mean(pass_buf['loss'])),
                             test_loss=t_loss, replay_extra=pass_buf['extra'], gate_passed=int(passed),
                             sel_norm_ratio=sel_ratio, action=action,
                             **{f'fold_{k}': fm.get(k) for k in mf}, **{f'test_{k}': tm.get(k) for k in mf})
                tracker.write_csv(run_dir / 'mistakes.csv')
                if learner.enabled and learner.soft:
                    learner.write_csv(run_dir / 'refined_labels.csv', b.samples)
                write_worst(b, f"pass_{rotation.total_passes:04d}")
                every = int(cfg.logging.get('sample_every_passes', 1))
                if every > 0 and rotation.total_passes % every == 0:
                    write_samples(b, f"pass_{rotation.total_passes:04d}",
                                  f"pass {rotation.total_passes}, cycle {cycle_before} {phase_before}, "
                                  f"dropout {pass_info['dropout']:.2f}, params {params:,}")
                log(f"== pass {rotation.total_passes} ({time.time() - t_pass:.0f}s) cycle {cycle_before} "
                    f"{phase_before}{' [hold]' if pass_info['holding'] else ''} | folds {task.short(fm)}"
                    + (f" | test {task.short(tm)}" if tm else '')
                    + f" | gate {'OK' if passed else '--'} | drop {pass_info['dropout']:.2f} "
                      f"pen {pass_info['penalty']:.3f}" + (f" | sel/kept |w| {sel_ratio:.3f}" if selection else '')
                    + (f" | replay +{pass_buf['extra']}" if pass_buf['extra'] else '')
                    + (f" | suspect labels {tracker.n_suspect()}" if tracker.n_suspect() else '')
                    + (f" | fold-test recall gap {gap:+.3f}" if not math.isnan(gap) else '') + f" -> {action}")

                if action == 'accept':
                    if pg.get('snapshot') and Path(pg['snapshot']).is_file():
                        snap = torch.load(pg['snapshot'], map_location='cpu', weights_only=False)
                        model.load_state_dict(snap['model'])
                        fm, tm = snap['fold'], snap['test']
                        log(f"[gate] plateau reached: taking the best model of this phase (pass {snap['pass']}, "
                            f"score {snap['score']:.4f})")
                    accepted_score = pg.get('best') if pg else task.score(tm if tm else fm)
                    pg = {}
                    path = run_dir / 'accepted' / f"cycle_{schedule.cycle:02d}.pth"
                    torch.save(model_file(model, cfg, task, {
                        'cycle': schedule.cycle, 'params': params, 'orig_params': orig_params,
                        'metrics': {'fold': fm, 'test': tm},
                        'slicing_registry': registry, 'channel_map': channel_map}), path)
                    last_accepted, accepted_lr = str(path), lr
                    write_samples(b, f"accepted_cycle_{schedule.cycle:02d}",
                                  f"ACCEPTED cycle {schedule.cycle}, pass {rotation.total_passes}, params {params:,}")
                    write_worst(b, f"accepted_cycle_{schedule.cycle:02d}", force=True)
                    schedule.after_accept()
                    if learner.enabled and not learner.active:
                        learner.active = True
                        log("label learning starts now (first model accepted on the given labels)")
                    log(f"[cycle {schedule.cycle}] ACCEPTED -> {path.name} ({params:,} params, "
                        f"{params / orig_params:.1%} of start)")
                    if schedule.cycle >= int(sc.max_cycles):
                        done, reason = True, f"max_cycles ({sc.max_cycles}) reached"
                    elif params <= float(sc.target_param_ratio) * orig_params:
                        done, reason = True, f"target size reached ({params / orig_params:.1%} of start)"
                    else:
                        schedule.start_ramp()
                        pg = {}
                elif action == 'cut':
                    before = count_params(model)
                    rec = S.slice_record(model, selection, channel_map)
                    for r in rec:
                        channel_map[r['layer_index']] = r['kept_original_indices']
                    registry.append({'cycle': schedule.cycle, 'pass': rotation.total_passes,
                                     'cut_ratio': schedule.cut_ratio, 'importance': str(sc.importance),
                                     'layers': rec})
                    write_registry()
                    model = S.cut(model, selection).to(device)
                    selection = {}
                    schedule.after_cut()
                    lr = max(float(cfg.training.lr_min), accepted_lr * float(cfg.training.lr_cycle_decay))
                    opt = make_optimizer(model, cfg, lr); lrc.reset()
                    plateau = {'best': float('inf'), 'wait': 0}
                    pg = {}
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
                        acc = torch.load(last_accepted, map_location='cpu', weights_only=False)
                        registry = acc.get('slicing_registry', [])
                        channel_map = acc.get('channel_map', {})
                        write_registry()
                        if schedule.retry_smaller():
                            lr = accepted_lr
                            opt = make_optimizer(model, cfg, lr); lrc.reset()
                            plateau = {'best': float('inf'), 'wait': 0}
                            pg = {}
                            log(f"  retry {schedule.retries} with cut ratio {schedule.cut_ratio:.3f}")
                        else:
                            done, reason = True, "recovery failed and no retry left"
                if switched:
                    log(f"[data] {cfg.folds.passes_per_dataset} passes done -> next dataset")
                pass_info = None

            if cfg.logging.save_state_every_step or pass_done:
                save_state()
            if pass_done:
                backup_state()
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
