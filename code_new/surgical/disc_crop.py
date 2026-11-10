"""Cut the optic disc out of a fundus photo, scale-normalised, and judge whether it is complete.

    mask (hand-drawn or from the definitor) -> largest blob -> circle fitted to the visible outline
    -> completeness = visible area / fitted circle area (a disc cut by the photo edge or the black
       border of the retina is < 1) -> square crop centred on the circle, side = margin x diameter,
       black padding where it runs past the photo -> resized to `size` x `size`.

Because the crop side follows the disc diameter, every crop shows the disc at the same scale
whatever the camera magnification (discs of 154-230 px all end up the same size).

Only numpy + OpenCV: reusable as-is in the ProcessorApplication script.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

DEFAULTS = {'size': 256, 'margin': 1.75, 'min_completeness': 0.90, 'min_solidity': 0.85,
            'black_level': 10, 'edge_px': 2, 'min_area_px': 200}


@dataclass
class DiscInfo:
    status: str                    # ok | partial | irregular | no_disc
    reason: str
    completeness: float = float('nan')
    diameter_px: float = float('nan')
    center_x: float = float('nan')
    center_y: float = float('nan')
    touches_edge: bool = False
    solidity: float = float('nan')
    area_px: int = 0

    def as_row(self):
        return asdict(self)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == k).astype(np.uint8)
    # fill holes (vessels drawn as gaps)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    return filled.astype(bool)


def _fit_circle(pts: np.ndarray):
    """least-squares circle (Kasa) -> (cx, cy, r)"""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    r = float(np.sqrt(max(1e-9, c[2] + cx * cx + cy * cy)))
    return float(cx), float(cy), r


def analyse(rgb: np.ndarray, mask: np.ndarray, p: Optional[dict] = None) -> DiscInfo:
    """rgb: uint8 [H,W,3] original photo; mask: [H,W] disc mask (any dtype, >0 / >127 = disc)."""
    p = {**DEFAULTS, **(p or {})}
    m = mask > (127 if mask.dtype == np.uint8 and mask.max() > 1 else 0)
    disc = _largest_blob(m)
    area = int(disc.sum())
    if area < p['min_area_px']:
        return DiscInfo('no_disc', 'optic disc not found in the photo')

    h, w = disc.shape
    retina = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > p['black_level']
    # "outside" = black border of the retina or beyond the photo; grown by edge_px
    outside = ~retina
    outside[:p['edge_px'], :] = outside[-p['edge_px']:, :] = True
    outside[:, :p['edge_px']] = outside[:, -p['edge_px']:] = True
    k = 2 * p['edge_px'] + 1
    near_out = cv2.dilate(outside.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    cnts, _ = cv2.findContours(disc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]                     # (N, 2) x, y
    on_edge = near_out[cnt[:, 1], cnt[:, 0]]
    touches = bool(on_edge.any())
    free = cnt[~on_edge] if (~on_edge).sum() >= 8 else cnt
    cx, cy, r = _fit_circle(free)
    hull = cv2.convexHull(cnt.reshape(-1, 1, 2))
    solidity = area / max(1.0, cv2.contourArea(hull))
    if touches:
        completeness = min(1.0, area / (np.pi * r * r))
    else:
        completeness = 1.0

    info = DiscInfo('ok', '', completeness, 2 * r, cx, cy, touches, float(solidity), area)
    if completeness < p['min_completeness']:
        info.status = 'partial'
        info.reason = (f"optic disc cut at the edge of the photo (about {completeness:.0%} visible) - "
                       f"retake with the disc centred")
    elif solidity < p['min_solidity']:
        info.status = 'irregular'
        info.reason = f"disc outline is irregular (solidity {solidity:.2f}) - check the photo or the mask"
    return info


def crop(rgb: np.ndarray, info: DiscInfo, p: Optional[dict] = None, mask: Optional[np.ndarray] = None):
    """square crop around the fitted circle, black padding, resized -> (rgb crop, mask crop or None)"""
    p = {**DEFAULTS, **(p or {})}
    if info.status == 'no_disc':
        return None, None
    side = max(8, int(round(p['margin'] * info.diameter_px)))
    x0, y0 = int(round(info.center_x - side / 2)), int(round(info.center_y - side / 2))
    h, w = rgb.shape[:2]

    def cut(img, fill):
        out = np.full((side, side) + img.shape[2:], fill, img.dtype)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        return out

    size = int(p['size'])
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    c = cv2.resize(cut(rgb, 0), (size, size), interpolation=interp)
    mc = None
    if mask is not None:
        mc = cv2.resize(cut(mask.astype(np.uint8), 0), (size, size), interpolation=cv2.INTER_NEAREST)
    return c, mc
