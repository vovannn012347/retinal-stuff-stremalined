"""Channel selection, penalty and the cut itself – generic over a model's layer graph.

A model describes itself with `graph()`: an ordered list of nodes
    {'name': 'decoder4.0', 'inputs': ['intermidiate.0', 'encoder3.4'], 'spatial': 1, 'cut': True}
  * `inputs` are the producing nodes in concatenation order ('input' = the image);
  * `spatial` > 1 for a Linear after Flatten (each input channel spans `spatial` features);
  * `cut` = whether the node's outputs may be removed (final outputs never are).
Supported layers: Conv2d, ConvTranspose2d, Linear.  The model also provides
`widths()` -> {node: outputs} and `with_widths(widths)` -> a new, empty model of that shape.
"""
from typing import Dict, List

import torch
import torch.nn as nn


def _mods(model):
    return dict(model.named_modules())


def out_axis(m):
    return 1 if isinstance(m, nn.ConvTranspose2d) else 0


def in_axis(m):
    return 0 if isinstance(m, nn.ConvTranspose2d) else 1


def n_out(m):
    return m.out_features if isinstance(m, nn.Linear) else m.out_channels


def _input_segments(model, node, widths):
    """[(source, start_channel)] of a node's input in concatenation order."""
    segs, off = [], 0
    for src in node['inputs']:
        segs.append((src, off))
        off += model.in_channels if src == 'input' else widths[src]
    return segs


def unit_names(model, exclude=()) -> List[str]:
    return [n['name'] for n in model.graph() if n['cut'] and n['name'] not in exclude]


def importance(model, name: str, mode: str = "l1") -> torch.Tensor:
    mods = _mods(model)
    m = mods[name]
    w = m.weight.detach().abs().transpose(0, out_axis(m))           # outputs first
    score = w.flatten(1).sum(1)
    if m.bias is not None:
        score = score + m.bias.detach().abs()
    if mode == "l1_inout":
        widths = model.widths()
        usage = torch.zeros_like(score)
        for node in model.graph():
            for src, start in _input_segments(model, node, widths):
                if src != name:
                    continue
                c = mods[node['name']]
                cw = c.weight.detach().abs().transpose(0, in_axis(c))   # inputs first
                S = node.get('spatial', 1)
                block = cw[start * S:(start + widths[name]) * S]
                usage += block.reshape(widths[name], S, -1).sum((1, 2))
        score = score * usage
    elif mode != "l1":
        raise ValueError(f"surgical.importance must be l1 | l1_inout, got {mode!r}")
    return score


def select_units(model, ratio: float, min_channels: int, exclude=(), mode: str = "l1") -> Dict[str, List[int]]:
    """Per layer: the `ratio` least important outputs (never below min_channels)."""
    mods = _mods(model)
    sel = {}
    for name in unit_names(model, exclude):
        n = n_out(mods[name])
        n_cut = min(int(n * ratio), max(0, n - min_channels))
        if n_cut > 0:
            sel[name] = sorted(torch.argsort(importance(model, name, mode))[:n_cut].tolist())
    return sel


def _rows(m, idx):
    """weight view with the selected outputs on dim 0"""
    return m.weight.transpose(0, out_axis(m))[idx]


@torch.no_grad()
def shrink(model, sel: Dict[str, List[int]], lam: float):
    """Penalty, applied once per pass: w_sel *= (1 - lambda)."""
    mods = _mods(model)
    f = 1.0 - float(lam)
    for name, idx in sel.items():
        m = mods[name]
        ix = torch.as_tensor(idx, device=m.weight.device)
        ax = out_axis(m)
        m.weight.index_copy_(ax, ix, m.weight.index_select(ax, ix) * f)
        if m.bias is not None:
            m.bias[ix] *= f


def l2_penalty(model, sel: Dict[str, List[int]]) -> torch.Tensor:
    mods = _mods(model)
    terms = []
    for name, idx in sel.items():
        m = mods[name]
        ix = torch.as_tensor(idx, device=m.weight.device)
        terms.append(m.weight.index_select(out_axis(m), ix).pow(2).mean())
        if m.bias is not None:
            terms.append(m.bias[ix].pow(2).mean())
    return torch.stack(terms).mean() if terms else torch.zeros((), device=next(model.parameters()).device)


@torch.no_grad()
def selected_norm_ratio(model, sel: Dict[str, List[int]]) -> float:
    """mean |w| of selected outputs / mean |w| of kept outputs (1.0 = untouched, ~0 = ready to cut)."""
    mods = _mods(model)
    s_sum = s_n = k_sum = k_n = 0.0
    for name, idx in sel.items():
        m = mods[name]
        w = m.weight.detach().abs().transpose(0, out_axis(m)).flatten(1).mean(1)
        mask = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
        mask[idx] = True
        s_sum += w[mask].sum().item(); s_n += mask.sum().item()
        k_sum += w[~mask].sum().item(); k_n += (~mask).sum().item()
    if s_n == 0 or k_n == 0 or k_sum == 0:
        return float('nan')
    return (s_sum / s_n) / (k_sum / k_n)


@torch.no_grad()
def cut(model, sel: Dict[str, List[int]]):
    """Returns a new, smaller model with the selected outputs removed (weights copied exactly)."""
    mods = _mods(model)
    graph = model.graph()
    old_w = model.widths()
    keep = {}
    for node in graph:
        n = n_out(mods[node['name']])
        drop = set(sel.get(node['name'], []))
        keep[node['name']] = [i for i in range(n) if i not in drop]
    new_w = {k: len(keep[k]) for k in old_w}
    new = model.with_widths(new_w).to(next(model.parameters()).device)
    new_mods = _mods(new)
    for node in graph:
        name = node['name']
        src_m, dst_m = mods[name], new_mods[name]
        dev = src_m.weight.device
        S = node.get('spatial', 1)
        in_idx = []
        for src, start in _input_segments(model, node, old_w):
            ks = range(model.in_channels) if src == 'input' else keep[src]
            in_idx += [(start + c) * S + p for c in ks for p in range(S)]
        out_idx = torch.as_tensor(keep[name], device=dev)
        in_idx = torch.as_tensor(in_idx, device=dev)
        w = src_m.weight.index_select(out_axis(src_m), out_idx).index_select(in_axis(src_m), in_idx)
        dst_m.weight.copy_(w)
        if src_m.bias is not None:
            dst_m.bias.copy_(src_m.bias[out_idx])
    for n, m in mods.items():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            new_mods[n].p = m.p
    new.train(model.training)
    return new
