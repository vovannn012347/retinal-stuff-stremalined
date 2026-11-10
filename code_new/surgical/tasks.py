"""The two trainable steps as tasks for the generic SurgicalSlicing trainer.

A task knows how to find samples, load/augment them, build/load its model, compute the
per-sample loss (also used as the error for mistake replay) and the quality gate.

  classifier : optic-disc crop -> glaucoma / atrophy / valid_image (multi-label)
  definitor  : fundus image -> optic nerve disc mask (segmentation)
"""
import csv
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from . import data as D
from .config import resolve_path


def get_task(cfg, log=print):
    kind = cfg.get('task', 'classifier')
    if kind == 'classifier':
        return ClassifierTask(cfg, log)
    if kind == 'definitor':
        return DefinitorTask(cfg, log)
    raise ValueError(f"task must be classifier | definitor, got {kind!r}")


def load_model_any(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cls = ckpt.get('model_class') if isinstance(ckpt, dict) else None
    if cls == 'DefinitorNet' or (isinstance(ckpt, dict) and 'nerve_definitor' in ckpt):
        from .definitor_model import load_definitor_file
        return load_definitor_file(path, map_location)
    from .model import load_model_file
    return load_model_file(path, map_location)


class _TaskDataset(Dataset):
    def __init__(self, samples, store, item_fn, train):
        self.samples, self.store, self.item_fn, self.train = samples, store, item_fn, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.item_fn(self.store.get(self.samples[i]), self.samples[i], self.train)
        return x, y, i


def _resize_u8(img: Image.Image, size, resample=Image.BILINEAR) -> torch.Tensor:
    img = img.resize((size, size), resample)
    a = np.asarray(img, dtype=np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    return torch.from_numpy(a.copy()).permute(2, 0, 1).contiguous()


# ══════════════════════════════════════════════════════════════════════════════════════
class ClassifierTask:
    name = 'classifier'

    def __init__(self, cfg, log):
        import torchvision.transforms as T
        self.cfg, self.log = cfg, log
        self.order = list(cfg.labels.order)
        self.thr = float(cfg.labels.positive_threshold)
        pp = cfg.preprocess
        self.size, self.mode, self.crop = int(pp.size), pp.mode, float(pp.get('crop_fraction', 1.0))
        a = cfg.augmentation
        ops = []
        if a.get('hflip', 0) > 0: ops.append(T.RandomHorizontalFlip(a['hflip']))
        if a.get('vflip', 0) > 0: ops.append(T.RandomVerticalFlip(a['vflip']))
        if a.get('rotation_degrees', 0) > 0:
            ops.append(T.RandomRotation(a['rotation_degrees'], interpolation=T.InterpolationMode.BILINEAR))
        cj = a.get('color_jitter')
        if cj and any(v > 0 for v in cj): ops.append(T.ColorJitter(*cj))
        if a.get('randaug_magnitude', 0) > 0:
            ops.append(T.RandAugment(num_ops=2, magnitude=int(a['randaug_magnitude']), num_magnitude_bins=31))
        self.augment = T.Compose(ops) if ops else None
        self.pos_weight = torch.tensor(cfg.training.pos_weight, dtype=torch.float32)

    # data
    def scan(self, ds_cfg) -> List[D.Sample]:
        root = resolve_path(ds_cfg['path'])
        img_dir, lbl_dir = root / ds_cfg.get('images', 'images'), root / ds_cfg.get('labels', 'labels')
        for d in (img_dir, lbl_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {d}")
        samples, missing = [], 0
        for p in D.list_images(img_dir):
            rel = p.relative_to(img_dir)
            cands = [lbl_dir / rel.parent / (p.name + '.csv'), lbl_dir / rel.parent / (p.stem + '.csv'),
                     lbl_dir / (p.name + '.csv')]
            lbl = next((c for c in cands if c.is_file()), None)
            labels = self._read_csv(lbl) if lbl else None
            if labels is None:
                missing += 1
                continue
            samples.append(D.Sample(str(p), D.group_key(p.name, self.cfg.labels.get('group_regex')),
                                    tuple(int(v >= self.thr) for v in labels), labels=labels))
        if missing:
            self.log(f"  [data] {root.name}: {missing} image(s) without a label file were skipped")
        if not samples:
            raise RuntimeError(f"No labelled images in {root}")
        pos = np.sum(np.stack([s.labels for s in samples]) >= self.thr, axis=0)
        self.log(f"  [data] positives per label {dict(zip(self.order, pos.tolist()))}")
        return samples

    def _read_csv(self, path: Path):
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        out = []
        for n in self.order:
            try:
                out.append(float(rows[0].get(n) or 0.0))
            except ValueError:
                out.append(0.0)
        return np.asarray(out, dtype=np.float32)

    def load(self, s: D.Sample):
        img = Image.open(s.path).convert('RGB')
        w, h = img.size
        if self.mode == 'center_crop':
            side = int(min(w, h) * max(0.05, min(1.0, self.crop)))
            l, t = (w - side) // 2, (h - side) // 2
            img = img.crop((l, t, l + side, t + side))
        elif self.mode == 'pad':
            side = max(w, h)
            c = Image.new('RGB', (side, side)); c.paste(img, ((side - w) // 2, (side - h) // 2)); img = c
        elif self.mode != 'stretch':
            raise ValueError(f"preprocess.mode must be stretch | pad | center_crop, got {self.mode!r}")
        return _resize_u8(img, self.size)

    def item(self, img, s, train):
        if train and self.augment is not None:
            img = self.augment(img)
        return D.to_input(img), torch.from_numpy(s.labels)

    # model
    def build_model(self):
        from .model import GlaucomaNet, default_channels, init_from_vgg16
        m = self.cfg.model
        model = GlaucomaNet(default_channels(int(m.base)), int(m.hidden), len(self.order), self.size)
        if m.init == 'vgg16':
            init_from_vgg16(model); self.log("model initialised from VGG16 (ImageNet)")
        return model

    def load_model(self, path):
        from .model import load_model_file
        model = load_model_file(path, input_size=self.size)
        if model.num_classes != len(self.order) or model.input_size != self.size:
            raise SystemExit(f"model has {model.num_classes} outputs / input {model.input_size}, config wants "
                             f"{len(self.order)} / {self.size}")
        return model

    def cut_exclude(self):
        return () if self.cfg.surgical.get('cut_fc_hidden', True) else ('fc.0',)

    def model_extra(self):
        return {'model_class': 'GlaucomaNet', 'label_order': self.order, 'task': 'classifier'}

    # loss / metrics
    def loss(self, logits, y):
        pw = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits.float(), y.float(), pos_weight=pw,
                                                  reduction='none').mean(1)          # per sample

    def sample_error(self, logits, y):
        """0..1 per sample: the worst label's distance from its target (>= 0.5 = that label is wrong)"""
        return (torch.sigmoid(logits.float()) - y.float()).abs().max(1).values

    def new_accumulator(self):
        return _ClsAcc(self)

    def gate(self, m):
        g = self.cfg.gate
        acc, rec = m['gate_main'], m['gate_recall']
        rec_ok = math.isnan(rec) or rec > g['recall']
        passed = not math.isnan(acc) and acc >= g['accuracy'] and rec_ok
        tol = g.get('degrade_tolerance', 0.02)
        degraded = math.isnan(acc) or acc < g['accuracy'] - tol or (not math.isnan(rec) and rec < g['recall'] - tol)
        return passed, degraded

    def metric_fields(self):
        return ['acc', 'recall'] + [f'{n}_{k}' for n in self.order for k in ('acc', 'recall')]

    def short(self, m):
        return f"acc {_f(m.get('acc'))} recall {_f(m.get('recall'))}"


class _ClsAcc:
    def __init__(self, t):
        self.t, self.p, self.y = t, [], []

    def add(self, logits, y):
        self.p.append(torch.sigmoid(logits.float()).cpu().numpy()); self.y.append(y.cpu().numpy())

    def result(self):
        t, g = self.t, self.t.cfg.gate
        if not self.p:
            return {'acc': float('nan'), 'recall': float('nan'), 'gate_main': float('nan'), 'gate_recall': float('nan')}
        p = np.concatenate(self.p) >= float(g.decision_threshold)
        y = np.concatenate(self.y) >= t.thr
        m = {'n': int(len(y))}
        for j, n in enumerate(t.order):
            tp = int(np.sum(p[:, j] & y[:, j])); fn = int(np.sum(~p[:, j] & y[:, j]))
            fp = int(np.sum(p[:, j] & ~y[:, j])); tn = int(np.sum(~p[:, j] & ~y[:, j]))
            m[f'{n}_acc'] = (tp + tn) / max(1, tp + tn + fp + fn)
            m[f'{n}_recall'] = tp / (tp + fn) if tp + fn else float('nan')
            m[f'{n}_precision'] = tp / (tp + fp) if tp + fp else float('nan')
            m[f'{n}_pos'] = tp + fn
        accs = [m[f'{n}_acc'] for n in g.accuracy_labels if f'{n}_acc' in m]
        recs = [m[f'{n}_recall'] for n in g.recall_labels if not math.isnan(m.get(f'{n}_recall', float('nan')))]
        m['acc'] = float(np.mean(accs)) if accs else float('nan')
        m['recall'] = float(min(recs)) if recs else float('nan')
        m['gate_main'], m['gate_recall'] = m['acc'], m['recall']
        return m


# ══════════════════════════════════════════════════════════════════════════════════════
class DefinitorTask:
    """recall    = correctly found disc pixels / all disc pixels      (TP / (TP + FN))  - the priority
       precision = correctly found disc pixels / all pixels predicted  (TP / (TP + FP))  ("accuracy" in our notes)"""
    name = 'definitor'

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        pp = cfg.preprocess
        self.size = int(pp.size)
        if self.size % 16:
            raise SystemExit(f"preprocess.size must be divisible by 16 for the definitor, got {self.size}")
        self.crop_retina = bool(pp.get('crop_to_retina', True))
        self.grayscale = bool(pp.get('grayscale', True))
        self.black_level = int(pp.get('black_level', 10))
        a = cfg.augmentation
        self.hflip, self.vflip = float(a.get('hflip', 0.5)), float(a.get('vflip', 0.5))
        self.rot = float(a.get('rotation_degrees', 0))
        self.bc = a.get('brightness_contrast', [0.15, 0.15])
        lo = cfg.loss
        self.pos_weight = float(lo.get('pos_weight', 3.0))
        self.tv_w = float(lo.get('tversky_weight', 1.0))
        self.tv_a, self.tv_b = float(lo.get('tversky_alpha', 0.3)), float(lo.get('tversky_beta', 0.7))

    # data
    def scan(self, ds_cfg) -> List[D.Sample]:
        img_dir, mask_dir = resolve_path(ds_cfg['images']), resolve_path(ds_cfg['masks'])
        for d in (img_dir, mask_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {d}")
        masks = {}
        for p in D.list_images(mask_dir):
            stem = p.stem
            if stem.endswith('.mask'):
                masks[stem[:-5]] = p
        samples, no_mask = [], 0
        for p in D.list_images(img_dir):
            m = masks.get(p.stem)
            if m is None:
                no_mask += 1
                continue
            samples.append(D.Sample(str(p), D.group_key(p.name, self.cfg.get('group_regex')), (), mask=str(m)))
        used = {Path(s.mask).name for s in samples}
        self.log(f"  [data] {len(samples)} image+mask pairs; {no_mask} images without a mask, "
                 f"{len(masks) - len(used)} masks without an image (skipped)")
        if not samples:
            raise RuntimeError(f"No image/mask pairs in {img_dir} + {mask_dir} (masks must be <image stem>.mask.<ext>)")
        return samples

    def load(self, s: D.Sample):
        img = Image.open(s.path).convert('RGB')
        mask = Image.open(s.mask).convert('L')
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        if self.crop_retina:
            g = np.asarray(img.convert('L'))
            ys, xs = np.nonzero(g > self.black_level)
            if len(xs):
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                img, mask = img.crop(box), mask.crop(box)
        if self.grayscale:
            img = img.convert('L').convert('RGB')
        return _resize_u8(img, self.size), _resize_u8(mask, self.size)

    def item(self, pair, s, train):
        img, mask = pair
        x = D.to_input(img)
        y = (mask.float() / 255.0 > 0.5).float()
        if train:
            if torch.rand(1).item() < self.hflip:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() < self.vflip:
                x, y = x.flip(-2), y.flip(-2)
            if self.rot > 0:
                import torchvision.transforms.functional as TF
                ang = float(torch.empty(1).uniform_(-self.rot, self.rot))
                x = TF.rotate(x, ang, interpolation=TF.InterpolationMode.BILINEAR, fill=-1.0)
                y = (TF.rotate(y, ang, interpolation=TF.InterpolationMode.NEAREST, fill=0.0) > 0.5).float()
            if self.bc and any(v > 0 for v in self.bc):
                b = float(torch.empty(1).uniform_(-self.bc[0], self.bc[0])) * 2
                c = 1.0 + float(torch.empty(1).uniform_(-self.bc[1], self.bc[1]))
                x = (x * c + b).clamp_(-1, 1)
        return x, y

    # model
    def build_model(self):
        from .definitor_model import DefinitorNet, default_widths, init_definitor_from_vgg16
        model = DefinitorNet(default_widths(int(self.cfg.model.base)))
        if self.cfg.model.init == 'vgg16':
            init_definitor_from_vgg16(model); self.log("encoder initialised from VGG16 (ImageNet)")
        return model

    def load_model(self, path):
        from .definitor_model import load_definitor_file
        return load_definitor_file(path)

    def cut_exclude(self):
        return ()

    def model_extra(self):
        return {'model_class': 'DefinitorNet', 'task': 'definitor'}

    # loss / metrics
    def loss(self, logits, y):
        logits, y = logits.float(), y.float()
        pw = torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction='none').mean((1, 2, 3))
        if self.tv_w <= 0:
            return bce
        p = torch.sigmoid(logits)
        tversky = 1 - self._tversky(p, y)
        return bce + self.tv_w * tversky

    def _tversky(self, p, y):
        """Tversky score per image. Smoothing = 1 % of the image area, so errors are measured on the
        scale of a disc (~3-7 % of the image) and an image with an empty mask is not "always wrong"."""
        s = 0.01 * p.shape[-1] * p.shape[-2]
        tp = (p * y).sum((1, 2, 3)); fp = (p * (1 - y)).sum((1, 2, 3)); fn = ((1 - p) * y).sum((1, 2, 3))
        return (tp + s) / (tp + self.tv_a * fp + self.tv_b * fn + s)

    def sample_error(self, logits, y):
        """0..1 per image: 1 - Tversky score of the thresholded prediction (pixels guessed right or wrong;
        a missed disc pixel weighs more than an extra one)"""
        p = (torch.sigmoid(logits.float()) >= float(self.cfg.gate.decision_threshold)).float()
        return 1 - self._tversky(p, y.float())

    def new_accumulator(self):
        return _SegAcc(float(self.cfg.gate.decision_threshold))

    def gate(self, m):
        g = self.cfg.gate
        rec, prec = m['recall'], m['precision']
        passed = not math.isnan(rec) and rec >= g['recall'] and not math.isnan(prec) and prec >= g['precision']
        tol = g.get('degrade_tolerance', 0.02)
        degraded = math.isnan(rec) or rec < g['recall'] - tol or prec < g['precision'] - tol
        return passed, degraded

    def metric_fields(self):
        return ['recall', 'precision', 'iou', 'dice', 'img_recall_min']

    def short(self, m):
        return (f"recall {_f(m.get('recall'))} precision {_f(m.get('precision'))} iou {_f(m.get('iou'))}"
                f" (worst image recall {_f(m.get('img_recall_min'))})")


class _SegAcc:
    def __init__(self, thr):
        self.thr = thr
        self.tp = self.fp = self.fn = 0
        self.img_rec = []

    def add(self, logits, y):
        p = torch.sigmoid(logits.float()) >= self.thr
        t = y >= 0.5
        tp = (p & t).sum((1, 2, 3)); fp = (p & ~t).sum((1, 2, 3)); fn = (~p & t).sum((1, 2, 3))
        self.tp += int(tp.sum()); self.fp += int(fp.sum()); self.fn += int(fn.sum())
        pos = (tp + fn)
        self.img_rec += [(a / b) for a, b in zip(tp.tolist(), pos.tolist()) if b > 0]

    def result(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        rec = tp / (tp + fn) if tp + fn else float('nan')
        prec = tp / (tp + fp) if tp + fp else (float('nan') if tp + fn == 0 else 0.0)
        m = {'recall': rec, 'precision': prec,
             'iou': tp / (tp + fp + fn) if tp + fp + fn else float('nan'),
             'dice': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan'),
             'img_recall_min': min(self.img_rec) if self.img_rec else float('nan')}
        m['gate_main'], m['gate_recall'] = prec, rec
        return m


def _f(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
