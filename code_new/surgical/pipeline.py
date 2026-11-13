"""Inference pipeline: original fundus photo -> definitor -> disc candidates (islands) -> disc crops.

    runner = DefinitorRunner('training-runs/definitor_cpu_from_x16/final_model.pth')
    prob = runner.probability(rgb)              # float32 [H, W] at the photo's own resolution
    for isl in find_islands(prob):              # every candidate, largest first
        info, crop = crop_island(rgb, isl)      # crop from the ORIGINAL photo (no CLAHE)

The steps repeat the training preparation exactly:
    retina box -> HSV-S + CLAHE-LAB (prepare_definitor_inputs.py) -> 512 px -> retina box again ->
    grey -> model size -> network -> probability, mapped back onto the original photo.
The definitor only says WHERE the disc candidates are; every island is a candidate and the
classifier's valid_image output rejects the wrong ones.

Uses torch, numpy, OpenCV and Pillow only (works with the ProcessorApplication executor: torch 2.3).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from . import disc_crop as C
from . import enhance as E

PREPARE_DEFAULTS = {'hsv_s': True, 'clahe': True, 'clip_limit': 2.0, 'tile': 16, 'black_level': 10,
                    'crop_to_retina': True, 'size': 512}
# A detected island can be only part of the disc (dim photo, partial response); its own diameter would then
# give a crop that cuts the disc. The crop diameter is therefore never smaller than min_disc_frac x the
# retina's width. Measured on the hand-drawn training masks (retina width = 1): disc diameter 0.12 - 0.41,
# median 0.22 (14 masks); 0.20 (just under the median) is generous: smaller detections are enlarged.
CROP_DEFAULTS = {'min_disc_frac': 0.20}
ISLAND_DEFAULTS = {'threshold': 0.5, 'open_frac': 0.03, 'min_area_frac': 0.0005, 'fallback_rel': 0.5,
                   'fallback_floor': 0.05}


class DefinitorRunner:
    def __init__(self, model_path, device='cpu', prepare: Optional[dict] = None):
        from .definitor_model import load_definitor_file
        self.device = torch.device(device)
        self.model = load_definitor_file(model_path).to(self.device).eval()
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        pp = ck.get('preprocess', {}) if isinstance(ck, dict) else {}
        self.size = int(pp.get('size', 288))
        self.grayscale = bool(pp.get('grayscale', True))
        self.crop2 = bool(pp.get('crop_to_retina', True))
        self.black2 = int(pp.get('black_level', 10))
        self.threshold = float(ck.get('decision_threshold', 0.5)) if isinstance(ck, dict) else 0.5
        if isinstance(prepare, (str, Path)):
            prepare = json.loads(Path(prepare).read_text(encoding='utf-8'))['params']
        self.prep = {**PREPARE_DEFAULTS, **(prepare or {})}

    @torch.no_grad()
    def probability(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 RGB [H, W, 3] original photo -> disc probability [H, W] float32 (0 outside the retina)."""
        p = self.prep
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        H, W = rgb.shape[:2]
        out = np.zeros((H, W), np.float32)
        box1 = E.retina_box(rgb, p['black_level']) if p['crop_to_retina'] else (0, 0, W, H)
        if box1 is None:
            return out
        x0, y0, x1, y1 = box1
        enh = E.enhance(rgb[y0:y1, x0:x1], p['hsv_s'], p['clahe'], p['clip_limit'], p['tile'], p['black_level'])
        S = int(p['size'])
        prep = Image.fromarray(enh).resize((S, S), Image.LANCZOS)            # = prepared_512 image
        box2 = (0, 0, S, S)
        if self.crop2:                                                         # = DefinitorTask.load
            g = np.asarray(prep.convert('L'))
            ys, xs = np.nonzero(g > self.black2)
            if len(xs):
                box2 = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                prep = prep.crop(box2)
        if self.grayscale:
            prep = prep.convert('L').convert('RGB')
        a = np.asarray(prep.resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)
        x = torch.from_numpy(a.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)[None].to(self.device)
        prob = torch.sigmoid(self.model(x).float())[0, 0].cpu().numpy()
        # back: model size -> box2 inside the S x S image -> box1 inside the photo
        bw, bh = box2[2] - box2[0], box2[3] - box2[1]
        p_s = np.zeros((S, S), np.float32)
        p_s[box2[1]:box2[3], box2[0]:box2[2]] = cv2.resize(prob, (bw, bh), interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] = cv2.resize(p_s, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        return out


@dataclass
class Island:
    index: int                     # 0 = largest
    mask: np.ndarray               # bool [H, W] at the photo's resolution, holes filled
    area_px: int
    mean_prob: float
    fallback: bool = False         # found only with the dim-photo threshold


def find_islands(prob: np.ndarray, threshold=None, open_frac=None, min_area_frac=None,
                 fallback_rel=None) -> List[Island]:
    """Every disc candidate: threshold -> opening (cuts thin bridges / specks) -> connected blobs of at
    least min_area_frac of the photo -> holes filled. Largest first.
    Fallback for dim photos: if nothing passes the threshold, the photo's own brightest response is used -
    threshold = fallback_rel x max probability (not below fallback_floor). The disc is usually there,
    just faint; the extra islands this may bring are rejected by the classifier (valid_image)."""
    d = ISLAND_DEFAULTS
    thr = d['threshold'] if threshold is None else threshold
    rel = d['fallback_rel'] if fallback_rel is None else fallback_rel
    out = _islands(prob, thr, open_frac, min_area_frac)
    if not out and rel and rel > 0:
        t2 = max(d['fallback_floor'], rel * float(prob.max()))
        if t2 < thr:
            out = _islands(prob, t2, open_frac, min_area_frac)
            for i in out:
                i.fallback = True
    return out


def _islands(prob, thr, open_frac, min_area_frac) -> List[Island]:
    d = ISLAND_DEFAULTS
    of = d['open_frac'] if open_frac is None else open_frac
    maf = d['min_area_frac'] if min_area_frac is None else min_area_frac
    m = (prob >= thr).astype(np.uint8)
    k = int(round(of * max(m.shape)))
    if k >= 3:
        k += 1 - k % 2
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_area = maf * m.shape[0] * m.shape[1]
    out = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = (lab == j).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(blob)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        fb = filled.astype(bool)
        out.append(Island(0, fb, int(fb.sum()), float(prob[fb].mean())))
    out.sort(key=lambda i: -i.area_px)
    for i, isl in enumerate(out):
        isl.index = i
    return out


def crop_island(rgb: np.ndarray, isl: Island, params: Optional[dict] = None):
    """(DiscInfo, uint8 RGB crop or None) for one island, cut from the original photo.
    info.diameter_px stays the detected diameter; info.crop_diameter_px is the one the crop used
    (at least min_disc_frac x the retina width), info.enlarged says whether it had to be raised."""
    from dataclasses import replace
    p = {**C.DEFAULTS, **CROP_DEFAULTS, **(params or {}), 'open_frac': 0.0}     # the island is already cleaned
    info = C.analyse(rgb, isl.mask.astype(np.uint8) * 255, p)
    info.crop_diameter_px, info.enlarged = info.diameter_px, False
    if info.status == 'no_disc':
        return info, None
    box = E.retina_box(rgb, int(p.get('black_level', 10)))
    if box is not None and p['min_disc_frac'] > 0:
        d_min = p['min_disc_frac'] * max(box[2] - box[0], box[3] - box[1])
        if not (info.diameter_px >= d_min):
            info.crop_diameter_px, info.enlarged = d_min, True
    img, _ = C.crop(rgb, replace(info, diameter_px=info.crop_diameter_px), p)
    return info, img
