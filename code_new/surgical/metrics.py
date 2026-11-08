"""Multi-label metrics for the gate (accuracy / recall per label)."""
import math
from typing import Dict, List

import numpy as np


def compute_metrics(probs: np.ndarray, targets: np.ndarray, label_names: List[str],
                    pos_thr: float = 0.5, dec_thr: float = 0.5) -> Dict[str, float]:
    out = {'n': int(len(targets))}
    if len(targets) == 0:
        return out
    y = targets >= pos_thr
    p = probs >= dec_thr
    for j, name in enumerate(label_names):
        tp = int(np.sum(p[:, j] & y[:, j])); fn = int(np.sum(~p[:, j] & y[:, j]))
        fp = int(np.sum(p[:, j] & ~y[:, j])); tn = int(np.sum(~p[:, j] & ~y[:, j]))
        out[f'{name}_acc'] = (tp + tn) / max(1, tp + tn + fp + fn)
        out[f'{name}_recall'] = tp / (tp + fn) if tp + fn else float('nan')
        out[f'{name}_precision'] = tp / (tp + fp) if tp + fp else float('nan')
        out[f'{name}_pos'] = tp + fn
    out['exact_match'] = float(np.mean(np.all(p == y, axis=1)))
    return out


def gate_values(m: Dict[str, float], acc_labels: List[str], recall_labels: List[str]):
    """-> (accuracy, recall). Labels without positives in the data are ignored for recall."""
    accs = [m[f'{n}_acc'] for n in acc_labels if f'{n}_acc' in m]
    recs = [m[f'{n}_recall'] for n in recall_labels if not math.isnan(m.get(f'{n}_recall', float('nan')))]
    acc = float(np.mean(accs)) if accs else float('nan')
    rec = float(min(recs)) if recs else float('nan')
    return acc, rec


def gate_check(acc: float, rec: float, g) -> (bool, bool):
    """-> (passed, degraded). nan recall (no positives) does not block the gate."""
    rec_ok = math.isnan(rec) or rec > g['recall']
    passed = (not math.isnan(acc)) and acc >= g['accuracy'] and rec_ok
    tol = g.get('degrade_tolerance', 0.02)
    degraded = (math.isnan(acc) or acc < g['accuracy'] - tol
                or (not math.isnan(rec) and rec < g['recall'] - tol))
    return passed, degraded
