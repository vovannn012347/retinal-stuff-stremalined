"""Task-independent data plumbing: patient grouping, folds, fold rotation, mistake replay sampler.

Fold rotation (the training method):
  * a dataset is split into K folds by patient (stratified by a per-sample signature);
  * one fold step = train on fold order[j], then test on fold order[j+1];
  * a pass = all K folds trained once; then the fold order is shuffled;
  * after `passes_per_dataset` passes the next dataset in the list is used.
"""
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm', '.jfif')


@dataclass
class Sample:
    path: str                   # image path (also the key for caches and the mistake record)
    group: str                  # patient key
    sig: tuple = ()             # stratification signature (e.g. binarised labels)
    labels: Optional[np.ndarray] = None   # classifier
    mask: Optional[str] = None            # definitor: mask path
    extra: dict = field(default_factory=dict)


# ── Patient grouping ──────────────────────────────────────────────────────────────────
_ODIR = re.compile(r'^(\d+)_(left|right)(_\d+)*$', re.IGNORECASE)        # 1138_left_0 -> 1138
_LOCAL = re.compile(r'^\d{3}(\D+?)\d{2}(_\d+)*$')                       # 028ІГБ02_1_0 -> ІГБ


def group_key(filename: str, regex: Optional[str] = None) -> str:
    """Patient id from a file name. Local clinic files are numbered per eye (027ІГБ01, 028ІГБ02),
    so the initials are the patient key. Unknown names: the whole stem (one patient per file)."""
    stem = Path(filename).stem
    if stem.endswith('.mask'):
        stem = stem[:-5]
    if regex:
        m = re.search(regex, stem)
        return m.group(1) if m else stem
    stem = re.sub(r'_cropped(_\d+)*$', '', stem)
    m = _ODIR.match(stem)
    if m:
        return m.group(1)
    m = _LOCAL.match(stem)
    if m:
        return m.group(1)
    return stem


def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob('*')) if p.is_file() and p.name.lower().endswith(IMG_EXTENSIONS)]


# ── Folds ─────────────────────────────────────────────────────────────────────────────
def split_groups(samples: List[Sample], k: int, test_fraction: float, seed: int):
    """Stratified (by signature) patient-grouped split -> (test_idx, [fold_idx...])."""
    rng = random.Random(seed)
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group, []).append(i)
    buckets: Dict[tuple, List[str]] = {}
    for g, idx in groups.items():
        sigs = [samples[i].sig for i in idx]
        sig = tuple(max(v) for v in zip(*sigs)) if sigs and sigs[0] else ()
        buckets.setdefault(sig, []).append(g)

    test_groups = set()
    if test_fraction > 0:
        for sig in sorted(buckets):
            gs = sorted(buckets[sig])
            rng.shuffle(gs)
            test_groups.update(gs[:int(round(len(gs) * test_fraction))])

    fold_groups = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for sig in sorted(buckets):
        gs = sorted(g for g in buckets[sig] if g not in test_groups)
        rng.shuffle(gs)
        gs.sort(key=lambda g: -len(groups[g]))
        per_fold_sig = [0] * k
        for g in gs:
            j = min(range(k), key=lambda f: (per_fold_sig[f], fold_sizes[f]))
            fold_groups[j].append(g)
            per_fold_sig[j] += len(groups[g])
            fold_sizes[j] += len(groups[g])

    test_idx = sorted(i for g in test_groups for i in groups[g])
    folds = [sorted(i for g in fg for i in groups[g]) for fg in fold_groups]
    return test_idx, folds


def dataset_fingerprint(samples: List[Sample]) -> str:
    h = hashlib.sha1()
    for s in samples:
        h.update(Path(s.path).name.encode('utf-8'))
    return h.hexdigest()[:16]


def load_or_make_split(samples, k, test_fraction, seed, split_file: Path, log=print):
    names = [Path(s.path).name for s in samples]
    fp = dataset_fingerprint(samples)
    if split_file.is_file():
        data = json.loads(split_file.read_text(encoding='utf-8'))
        if data.get('fingerprint') == fp and data.get('k') == k:
            pos = {n: i for i, n in enumerate(names)}
            return [pos[n] for n in data['test']], [[pos[n] for n in f] for f in data['folds']]
        log(f"  [data] dataset changed since {split_file.name} was written -> new split")
    test_idx, folds = split_groups(samples, k, test_fraction, seed)
    if any(len(f) == 0 for f in folds):
        raise RuntimeError(f"{len({s.group for s in samples})} patients are not enough for {k} folds "
                           f"(+ test) - lower folds.k or folds.test_fraction")
    split_file.write_text(json.dumps({
        'fingerprint': fp, 'k': k, 'test_fraction': test_fraction, 'seed': seed,
        'test': [names[i] for i in test_idx], 'folds': [[names[i] for i in f] for f in folds],
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    return test_idx, folds


# ── Image cache ───────────────────────────────────────────────────────────────────────
class ImageStore:
    """Caches whatever `load_fn(sample)` returns (decoded, resized tensors), keyed by sample.path."""

    def __init__(self, load_fn: Callable, cache=True):
        self.load_fn, self.cache = load_fn, cache
        self._mem: Dict[str, object] = {}

    def get(self, sample: Sample):
        v = self._mem.get(sample.path)
        if v is None:
            v = self.load_fn(sample)
            if self.cache:
                self._mem[sample.path] = v
        return v

    def preload(self, samples):
        if self.cache:
            for s in samples:
                self.get(s)


def to_input(img_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [C,H,W] -> float normalised to [-1, 1] (same as the old trainers)."""
    return img_u8.float().div_(127.5).sub_(1.0)


# ── Mistake replay ────────────────────────────────────────────────────────────────────
class MistakeTracker:
    """Per-sample error history (error on a 0..1 scale, see the task's `sample_error`).
    Errors come from the "test on the next fold" evaluation (clean image, before the network
    trains on it in this pass), kept as a moving average. When a fold is trained, every sample whose
    error is >= min_error AND above the fold median (so early on, when everything is wrong, only the
    worse half) is repeated:  extra copies = min(max_extra, floor(gain * error / min_error)).
    With min_error 0.25: error 0.25 -> +1, 0.5 -> +2, >= 0.75 -> +3 (cap).
    Safety against wrong labels/masks: a sample that is still wrong (error >= suspect_error) while
    most of its fold is already learned (fold median < min_error), suspect_after evaluations in a
    row, is marked "suspect" and no longer repeated (listed first in mistakes.csv - check its
    label/mask); it is un-marked as soon as it is learned. Call update() once per evaluated fold."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get('enabled', True))
        self.ema = float(cfg.get('ema', 0.5))
        self.gain = float(cfg.get('gain', 1.0))
        self.max_extra = int(cfg.get('max_extra', 3))
        self.min_error = float(cfg.get('min_error', 0.25))
        self.suspect_error = float(cfg.get('suspect_error', 0.5))
        self.suspect_after = int(cfg.get('suspect_after', 8))
        self.rec: Dict[str, dict] = {}

    def update(self, paths, errors):
        errors = [float(e) for e in errors]
        fold_learned = len(errors) >= 2 and float(np.median(errors)) < self.min_error
        for p, e in zip(paths, errors):
            e = float(e)
            r = self.rec.get(p)
            if r is None:
                r = self.rec[p] = {'err': e, 'last': e, 'seen': 1, 'repeats': 0, 'bad_streak': 0, 'suspect': False}
            else:
                r['err'] = self.ema * r['err'] + (1 - self.ema) * e
                r['last'] = e
                r['seen'] += 1
            if e < self.suspect_error:
                r['bad_streak'] = 0
            elif fold_learned:
                r['bad_streak'] = r.get('bad_streak', 0) + 1
            r['suspect'] = self.suspect_after > 0 and r['bad_streak'] >= self.suspect_after

    def expand(self, indices, samples) -> (List[int], int):
        """-> (training index list with repeats, number of extra items)."""
        if not self.enabled:
            return list(indices), 0
        known = [self.rec[samples[i].path]['err'] for i in indices if samples[i].path in self.rec]
        med = float(np.median(known)) if len(known) >= 2 else 0.0
        out, extra_total = [], 0
        for i in indices:
            out.append(i)
            r = self.rec.get(samples[i].path)
            if r is None or r.get('suspect') or r['err'] < self.min_error or r['err'] <= med:
                continue
            extra = min(self.max_extra, int(self.gain * r['err'] / self.min_error))
            if extra > 0:
                out.extend([i] * extra)
                r['repeats'] += extra
                extra_total += extra
        return out, extra_total

    def write_csv(self, path: Path, top: int = 0):
        rows = sorted(self.rec.items(), key=lambda kv: (not kv[1].get('suspect'), -kv[1]['err']))
        if top:
            rows = rows[:top]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('path,suspect_label,error_ema,last_error,times_tested,wrong_in_a_row,extra_repeats\n')
            for p, r in rows:
                f.write(f'"{p}",{int(bool(r.get("suspect")))},{r["err"]:.6g},{r["last"]:.6g},{r["seen"]},'
                        f'{r.get("bad_streak", 0)},{r["repeats"]}\n')

    def n_suspect(self):
        return sum(1 for r in self.rec.values() if r.get('suspect'))

    def state_dict(self):
        return {'rec': self.rec}

    def load_state_dict(self, d):
        self.rec = d.get('rec', {})


class SettableSampler(Sampler):
    """Sampler whose index list is replaced between fold steps (keeps DataLoader workers alive)."""

    def __init__(self, shuffle: bool, seed: int):
        self.indices: List[int] = []
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def set(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        idx = list(self.indices)
        if self.shuffle:
            self.rng.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return len(self.indices)


# ── Fold rotation schedule ────────────────────────────────────────────────────────────
class FoldRotation:
    """Which dataset / fold to train and which fold to test next. State is plain JSON-able."""

    def __init__(self, n_datasets: int, k: int, passes_per_dataset: int, seed: int):
        self.n_datasets, self.k, self.passes_per_dataset = n_datasets, k, passes_per_dataset
        self.rng = random.Random(seed)
        self.dataset_idx = 0
        self.passes_on_dataset = 0
        self.total_passes = 0
        self.pos = 0
        self.order = list(range(k))
        self.rng.shuffle(self.order)

    def current(self):
        """-> (dataset_idx, train_fold, test_fold)"""
        return self.dataset_idx, self.order[self.pos], self.order[(self.pos + 1) % self.k]

    def advance(self):
        """Move to the next step. Returns (pass_finished, dataset_switched)."""
        self.pos += 1
        if self.pos < self.k:
            return False, False
        self.pos = 0
        self.total_passes += 1
        self.passes_on_dataset += 1
        last = self.order[-1]
        self.rng.shuffle(self.order)
        if self.k > 2 and self.order[0] == last:          # avoid training the same fold twice in a row
            self.order[0], self.order[1] = self.order[1], self.order[0]
        switched = False
        if self.passes_on_dataset >= self.passes_per_dataset and self.n_datasets > 1:
            self.dataset_idx = (self.dataset_idx + 1) % self.n_datasets
            self.passes_on_dataset = 0
            switched = True
        return True, switched

    def state_dict(self):
        return {'dataset_idx': self.dataset_idx, 'passes_on_dataset': self.passes_on_dataset,
                'total_passes': self.total_passes, 'pos': self.pos, 'order': self.order,
                'rng': self.rng.getstate()}

    def load_state_dict(self, d):
        self.dataset_idx = d['dataset_idx']
        self.passes_on_dataset = d['passes_on_dataset']
        self.total_passes = d['total_passes']
        self.pos = d['pos']
        self.order = list(d['order'])
        st = d['rng']
        self.rng.setstate((st[0], tuple(st[1]), st[2]))
