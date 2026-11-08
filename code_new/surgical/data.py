"""Datasets, patient-grouped folds and the fold rotation schedule.

Fold rotation (the training method):
  * a dataset is split into K folds by patient (stratified by label combination);
  * one fold step = train on fold order[j], then test on fold order[j+1];
  * a pass = all K folds trained once; then the fold order is shuffled;
  * after `passes_per_dataset` passes the next dataset in the list is used.
"""
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm', '.jfif')


@dataclass
class Sample:
    path: str
    labels: np.ndarray          # float32, label order from the config
    group: str                  # patient key


# ── Patient grouping ──────────────────────────────────────────────────────────────────
_TRAILING_INDEX = re.compile(r'(_\d+)+$')
_EYE_SUFFIX = re.compile(r'_(left|right)$', re.IGNORECASE)
_LOCAL_EYE = re.compile(r'^(\d+\D+?)\d{2}$')      # 028ІГБ02 -> 028ІГБ


def group_key(filename: str, regex: Optional[str] = None) -> str:
    stem = Path(filename).stem
    if regex:
        m = re.search(regex, stem)
        return m.group(1) if m else stem
    stem = _TRAILING_INDEX.sub('', stem)          # crop / variant indices: _0, _1_0
    stem = re.sub(r'_cropped$', '', stem)
    stem = _TRAILING_INDEX.sub('', stem)
    stem = _EYE_SUFFIX.sub('', stem)              # ODIR: 1138_left -> 1138
    m = _LOCAL_EYE.match(stem)                    # local clinic: 028ІГБ02 -> 028ІГБ
    if m:
        stem = m.group(1)
    return stem


# ── Scanning ──────────────────────────────────────────────────────────────────────────
def read_label_csv(path: Path, order: List[str]) -> Optional[np.ndarray]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    out = []
    for name in order:
        v = row.get(name, '')
        try:
            out.append(float(v) if v not in (None, '') else 0.0)
        except ValueError:
            out.append(0.0)
    return np.asarray(out, dtype=np.float32)


def scan_dataset(root: Path, order: List[str], images_sub='images', labels_sub='labels',
                 group_regex=None, log=print) -> List[Sample]:
    img_dir, lbl_dir = root / images_sub, root / labels_sub
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Dataset images folder not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Dataset labels folder not found: {lbl_dir}")
    samples, missing = [], 0
    for p in sorted(img_dir.rglob('*')):
        if not p.is_file() or not p.name.lower().endswith(IMG_EXTENSIONS):
            continue
        rel = p.relative_to(img_dir)
        cands = [lbl_dir / rel.parent / (p.name + '.csv'), lbl_dir / rel.parent / (p.stem + '.csv'),
                 lbl_dir / (p.name + '.csv')]
        lbl = next((c for c in cands if c.is_file()), None)
        if lbl is None:
            missing += 1
            continue
        labels = read_label_csv(lbl, order)
        if labels is None:
            missing += 1
            continue
        samples.append(Sample(str(p), labels, group_key(p.name, group_regex)))
    if missing:
        log(f"  [data] {root.name}: {missing} image(s) without a label file were skipped")
    if not samples:
        raise RuntimeError(f"No labelled images in {root}")
    return samples


# ── Folds ─────────────────────────────────────────────────────────────────────────────
def _signature(labels: np.ndarray, thr: float):
    return tuple(int(v >= thr) for v in labels)


def split_groups(samples: List[Sample], k: int, test_fraction: float, thr: float, seed: int):
    """Stratified (by label combination) patient-grouped split -> (test_idx, [fold_idx...])."""
    rng = random.Random(seed)
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group, []).append(i)
    # group signature = element-wise max of its samples' labels
    buckets: Dict[tuple, List[str]] = {}
    for g, idx in groups.items():
        sig = _signature(np.max(np.stack([samples[i].labels for i in idx]), axis=0), thr)
        buckets.setdefault(sig, []).append(g)

    test_groups = set()
    if test_fraction > 0:
        for sig in sorted(buckets):
            gs = sorted(buckets[sig])
            rng.shuffle(gs)
            n_test = int(round(len(gs) * test_fraction))
            test_groups.update(gs[:n_test])

    fold_groups = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for sig in sorted(buckets):
        gs = sorted(g for g in buckets[sig] if g not in test_groups)
        rng.shuffle(gs)
        gs.sort(key=lambda g: -len(groups[g]))            # big patients first, then fill
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


def load_or_make_split(samples, k, test_fraction, thr, seed, split_file: Path, log=print):
    names = [Path(s.path).name for s in samples]
    fp = dataset_fingerprint(samples)
    if split_file.is_file():
        data = json.loads(split_file.read_text(encoding='utf-8'))
        if data.get('fingerprint') == fp and data.get('k') == k:
            pos = {n: i for i, n in enumerate(names)}
            test_idx = [pos[n] for n in data['test']]
            folds = [[pos[n] for n in f] for f in data['folds']]
            return test_idx, folds
        log(f"  [data] dataset changed since {split_file.name} was written -> new split")
    test_idx, folds = split_groups(samples, k, test_fraction, thr, seed)
    split_file.write_text(json.dumps({
        'fingerprint': fp, 'k': k, 'test_fraction': test_fraction, 'seed': seed,
        'test': [names[i] for i in test_idx],
        'folds': [[names[i] for i in f] for f in folds],
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    return test_idx, folds


# ── Images ────────────────────────────────────────────────────────────────────────────
def load_image(path: str, size: int, mode: str = 'stretch', crop_fraction: float = 1.0) -> torch.Tensor:
    """-> uint8 tensor [3, size, size]."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if mode == 'center_crop':
        side = int(min(w, h) * max(0.05, min(1.0, crop_fraction)))
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    elif mode == 'pad':
        side = max(w, h)
        canvas = Image.new('RGB', (side, side), (0, 0, 0))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    elif mode != 'stretch':
        raise ValueError(f"preprocess.mode must be stretch | pad | center_crop, got {mode!r}")
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


class ImageStore:
    def __init__(self, size, mode, crop_fraction, cache=True):
        self.size, self.mode, self.crop_fraction, self.cache = size, mode, crop_fraction, cache
        self._mem: Dict[str, torch.Tensor] = {}

    def get(self, path: str) -> torch.Tensor:
        t = self._mem.get(path)
        if t is None:
            t = load_image(path, self.size, self.mode, self.crop_fraction)
            if self.cache:
                self._mem[path] = t
        return t

    def preload(self, paths):
        if self.cache:
            for p in paths:
                self.get(p)


def build_augment(cfg_aug):
    ops = []
    if cfg_aug.get('hflip', 0) > 0:
        ops.append(T.RandomHorizontalFlip(cfg_aug['hflip']))
    if cfg_aug.get('vflip', 0) > 0:
        ops.append(T.RandomVerticalFlip(cfg_aug['vflip']))
    if cfg_aug.get('rotation_degrees', 0) > 0:
        ops.append(T.RandomRotation(cfg_aug['rotation_degrees'], interpolation=T.InterpolationMode.BILINEAR))
    cj = cfg_aug.get('color_jitter')
    if cj and any(v > 0 for v in cj):
        ops.append(T.ColorJitter(*cj))
    if cfg_aug.get('randaug_magnitude', 0) > 0:
        ops.append(T.RandAugment(num_ops=2, magnitude=int(cfg_aug['randaug_magnitude']), num_magnitude_bins=31))
    return T.Compose(ops) if ops else None


def to_input(img_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [3,H,W] -> float normalised to [-1, 1] (same as the old trainer)."""
    return img_u8.float().div_(127.5).sub_(1.0)


class RetinaDataset(Dataset):
    def __init__(self, samples: List[Sample], store: ImageStore, augment=None):
        self.samples, self.store, self.augment = samples, store, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = self.store.get(s.path)
        if self.augment is not None:
            img = self.augment(img)
        return to_input(img), torch.from_numpy(s.labels), i


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
