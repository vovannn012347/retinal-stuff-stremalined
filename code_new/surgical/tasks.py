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
        model = GlaucomaNet(default_channels(int(m.base)), int(m.hidden), len(self.order), self.size,
                            head=m.get('head', 'gap'))
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

    def render_samples(self, *args, **kwargs):          # classifier: no image samples
        return


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
        self.scale = float(a.get('scale', 0.0))           # +- fraction
        self.shift = float(a.get('shift', 0.0))           # +- fraction of the size
        self.bc = a.get('brightness_contrast', [0.15, 0.15])
        self.gamma = float(a.get('gamma', 0.0))           # gamma in [1/(1+g), 1+g]
        self.blur_p = float(a.get('blur_p', 0.0))
        self.tol = int(cfg.gate.get('boundary_tolerance_px', 0))
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
            if self.rot > 0 or self.scale > 0 or self.shift > 0:
                import torchvision.transforms.functional as TF
                u = lambda r: float(torch.empty(1).uniform_(-r, r))          # noqa: E731
                ang, sc = u(self.rot), 1.0 + u(self.scale)
                tr = [int(round(u(self.shift) * self.size)), int(round(u(self.shift) * self.size))]
                x = TF.affine(x, angle=ang, translate=tr, scale=sc, shear=[0.0],
                              interpolation=TF.InterpolationMode.BILINEAR, fill=-1.0)
                y = (TF.affine(y, angle=ang, translate=tr, scale=sc, shear=[0.0],
                               interpolation=TF.InterpolationMode.NEAREST, fill=0.0) > 0.5).float()
            if self.gamma > 0:
                g = float(torch.empty(1).uniform_(-1, 1))
                g = (1 + self.gamma) ** g
                x = ((x + 1) / 2).clamp_(0, 1).pow_(g).mul_(2).sub_(1)
            if self.blur_p > 0 and torch.rand(1).item() < self.blur_p:
                import torchvision.transforms.functional as TF
                x = TF.gaussian_blur(x, kernel_size=5, sigma=float(torch.empty(1).uniform_(0.5, 1.2)))
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
        g = self.cfg.gate
        return _SegAcc(float(g.decision_threshold), self.tol, bool(g.get('clean_prediction', True)),
                       float(g.get('open_frac', 0.03)))

    def gate(self, m):
        g = self.cfg.gate
        rec, prec = m['recall'], m['precision']
        spill, max_spill = m.get('spill', float('nan')), float(g.get('max_spill', 1.0))
        spill_ok = math.isnan(spill) or spill <= max_spill
        passed = (not math.isnan(rec) and rec >= g['recall'] and not math.isnan(prec) and prec >= g['precision']
                  and spill_ok)
        tol = g.get('degrade_tolerance', 0.02)
        degraded = math.isnan(rec) or rec < g['recall'] - tol or prec < g['precision'] - tol
        return passed, degraded

    def metric_fields(self):
        return ['recall', 'precision', 'strict_recall', 'strict_precision', 'iou', 'dice', 'img_recall_min',
                'spill', 'pieces']

    @torch.no_grad()
    def render_samples(self, model, samples, store, indices, device, out_path, title=''):
        """One PNG: per image [input as the network sees it | prediction]. Prediction panel:
        red = predicted probability, yellow line = cleaned mask (what the pipeline keeps),
        green line = true mask. Caption: per-image recall / precision of the cleaned mask."""
        from PIL import ImageDraw
        from .disc_crop import clean_mask
        import cv2
        model.eval()
        g = self.cfg.gate
        thr, open_frac = float(g.decision_threshold), float(g.get('open_frac', 0.03))
        font = _font(12)
        rows = []
        for i in indices:
            s = samples[i]
            x, y = self.item(store.get(s), s, False)
            prob = torch.sigmoid(model(x[None].to(device)).float())[0, 0].cpu().numpy()
            img = np.ascontiguousarray(((x.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))
            t = y[0].numpy() > 0.5
            c = clean_mask(prob >= thr, open_frac)
            over = img.astype(np.float32)
            over[..., 0] = over[..., 0] * (1 - prob) + 255 * prob
            over[..., 1:] *= (1 - 0.6 * prob[..., None])
            over = np.ascontiguousarray(over.clip(0, 255).astype(np.uint8))
            for m, col in ((c, (255, 230, 0)), (t, (0, 255, 0))):
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(over, cnts, -1, col, 1)
            tp, fn, fp = int((c & t).sum()), int((~c & t).sum()), int((c & ~t).sum())
            rec = f"{tp / (tp + fn):.2f}" if tp + fn else "no disc"
            prec = f"{tp / (tp + fp):.2f}" if tp + fp else "-"
            row = np.concatenate([img, over], 1)
            pil = Image.fromarray(row)
            ImageDraw.Draw(pil).text((4, 3), f"{Path(s.path).stem}  recall {rec}  precision {prec}",
                                     fill=(255, 255, 255), font=font)
            rows.append(np.asarray(pil))
        if not rows:
            return
        cols = 2 if len(rows) > 3 else 1
        while len(rows) % cols:
            rows.append(np.zeros_like(rows[0]))
        grid = np.concatenate([np.concatenate(rows[r:r + cols], 1) for r in range(0, len(rows), cols)], 0)
        head = np.zeros((18, grid.shape[1], 3), np.uint8)
        pil = Image.fromarray(np.concatenate([head, grid], 0))
        ImageDraw.Draw(pil).text((4, 2), title + "   red = probability, yellow = cleaned mask, green = true mask",
                                 fill=(255, 255, 255), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)

    def short(self, m):
        return (f"recall {_f(m.get('recall'))} precision {_f(m.get('precision'))} iou {_f(m.get('iou'))}"
                f" spill {_f(m.get('spill'))} pieces {_f(m.get('pieces'), 1)}"
                f" (worst image recall {_f(m.get('img_recall_min'))})")


class _SegAcc:
    """Pixel counts pooled over images. With boundary tolerance t > 0, pixels within t px of the
    true mask edge are not counted at all (the hand-drawn edge is uncertain by about that much);
    strict (untolerant) recall/precision are reported too."""

    def __init__(self, thr, tol=0, clean=True, open_frac=0.03):
        self.thr, self.tol, self.clean, self.open_frac = thr, tol, clean, open_frac
        self.c = {'tp': 0, 'fp': 0, 'fn': 0, 'stp': 0, 'sfp': 0, 'sfn': 0, 'raw': 0, 'kept': 0, 'pieces': 0, 'n': 0}
        self.img_rec = []

    def add(self, logits, y):
        raw = torch.sigmoid(logits.float()) >= self.thr
        t = y >= 0.5
        if self.clean:
            # what the pipeline will actually use: opening + largest blob (see disc_crop.clean_mask)
            import cv2
            from .disc_crop import clean_mask
            rn = raw[:, 0].cpu().numpy()
            cleaned = np.stack([clean_mask(a, self.open_frac) for a in rn])
            for a in rn:
                self.c['pieces'] += max(0, cv2.connectedComponents(a.astype(np.uint8), connectivity=8)[0] - 1)
            p = torch.from_numpy(cleaned)[:, None].to(raw.device)
            self.c['raw'] += int(raw.sum()); self.c['kept'] += int((p & raw).sum())
        else:
            p = raw
        self.c['n'] += raw.shape[0]
        valid = torch.ones_like(t)
        if self.tol > 0:
            k = 2 * self.tol + 1
            tf = t.float()
            dil = F.max_pool2d(tf, k, 1, self.tol) > 0.5
            ero = -F.max_pool2d(-tf, k, 1, self.tol) > 0.5
            valid = ~(dil & ~ero)
        tp = (p & t & valid).sum((1, 2, 3)); fp = (p & ~t & valid).sum((1, 2, 3)); fn = (~p & t & valid).sum((1, 2, 3))
        c = self.c
        c['tp'] += int(tp.sum()); c['fp'] += int(fp.sum()); c['fn'] += int(fn.sum())
        c['stp'] += int((p & t).sum()); c['sfp'] += int((p & ~t).sum()); c['sfn'] += int((~p & t).sum())
        self.img_rec += [(a / b) for a, b in zip(tp.tolist(), (tp + fn).tolist()) if b > 0]

    @staticmethod
    def _rp(tp, fp, fn):
        rec = tp / (tp + fn) if tp + fn else float('nan')
        prec = tp / (tp + fp) if tp + fp else (float('nan') if tp + fn == 0 else 0.0)
        return rec, prec

    def result(self):
        c = self.c
        rec, prec = self._rp(c['tp'], c['fp'], c['fn'])
        srec, sprec = self._rp(c['stp'], c['sfp'], c['sfn'])
        s = c['stp'] + c['sfp'] + c['sfn']
        m = {'recall': rec, 'precision': prec, 'strict_recall': srec, 'strict_precision': sprec,
             'iou': c['stp'] / s if s else float('nan'), 'dice': 2 * c['stp'] / (s + c['stp']) if s else float('nan'),
             'img_recall_min': min(self.img_rec) if self.img_rec else float('nan'),
             # spill: share of predicted pixels outside the kept disc blob (bridges, separate patches)
             'spill': 1 - c['kept'] / c['raw'] if c['raw'] else float('nan'),
             'pieces': c['pieces'] / c['n'] if c['n'] else float('nan')}
        m['gate_main'], m['gate_recall'] = prec, rec
        return m


def _font(size=12):
    """a TrueType font with Cyrillic (file names like 015МЛФ01); PIL's built-in font has none"""
    from PIL import ImageFont
    for name in ('arial.ttf', 'segoeui.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _f(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
