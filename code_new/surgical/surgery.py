"""Channel selection, penalty and the cut itself (structured slicing of GlaucomaNet).

Units that can be cut:
  * output channels of every conv (the next conv loses the matching input channels;
    for the last conv the matching column blocks of fc.0 are removed);
  * hidden neurons of fc.0 (rows of fc.0, columns of fc.3).
The classifier outputs (fc.3 rows) are never cut.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .model import GlaucomaNet, build_from_arch

FC_HIDDEN = "fc.0"


def _mods(model):
    return dict(model.named_modules())


def unit_names(model: GlaucomaNet, cut_fc_hidden: bool) -> List[str]:
    names = model.conv_names()
    return names + [FC_HIDDEN] if cut_fc_hidden else names


def _consumer_weight(model: GlaucomaNet, name: str):
    """Weight slice-view function: returns per-unit L1 of the *next* layer's use of each unit."""
    mods = _mods(model)
    convs = model.conv_names()
    if name == FC_HIDDEN:
        w = mods["fc.3"].weight.detach().abs()                    # [classes, hidden]
        return w.sum(0)
    i = convs.index(name)
    if i + 1 < len(convs):
        w = mods[convs[i + 1]].weight.detach().abs()              # [out, in, k, k]
        return w.sum((0, 2, 3))
    w = mods["fc.0"].weight.detach().abs()                        # [hidden, C*S]
    c = mods[name].out_channels
    return w.view(w.shape[0], c, -1).sum((0, 2))


def importance(model: GlaucomaNet, name: str, mode: str = "l1") -> torch.Tensor:
    m = _mods(model)[name]
    w = m.weight.detach().abs()
    score = w.flatten(1).sum(1)
    if m.bias is not None:
        score = score + m.bias.detach().abs()
    if mode == "l1_inout":
        score = score * _consumer_weight(model, name)
    elif mode != "l1":
        raise ValueError(f"surgical.importance must be l1 | l1_inout, got {mode!r}")
    return score


def select_units(model: GlaucomaNet, ratio: float, min_channels: int, cut_fc_hidden: bool,
                 mode: str = "l1") -> Dict[str, List[int]]:
    """Per layer: the `ratio` least important units (never below min_channels)."""
    mods = _mods(model)
    sel = {}
    for name in unit_names(model, cut_fc_hidden):
        m = mods[name]
        n = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        n_cut = min(int(n * ratio), max(0, n - min_channels))
        if n_cut <= 0:
            continue
        order = torch.argsort(importance(model, name, mode))
        sel[name] = sorted(order[:n_cut].tolist())
    return sel


@torch.no_grad()
def shrink(model: GlaucomaNet, sel: Dict[str, List[int]], lam: float):
    """Penalty, applied once per pass: w_sel *= (1 - lambda)."""
    mods = _mods(model)
    f = 1.0 - float(lam)
    for name, idx in sel.items():
        m = mods[name]
        ix = torch.as_tensor(idx, device=m.weight.device)
        m.weight[ix] *= f
        if m.bias is not None:
            m.bias[ix] *= f


def l2_penalty(model: GlaucomaNet, sel: Dict[str, List[int]]) -> torch.Tensor:
    mods = _mods(model)
    terms = []
    for name, idx in sel.items():
        m = mods[name]
        ix = torch.as_tensor(idx, device=m.weight.device)
        terms.append(m.weight[ix].pow(2).mean())
        if m.bias is not None:
            terms.append(m.bias[ix].pow(2).mean())
    return torch.stack(terms).mean() if terms else torch.zeros((), device=next(model.parameters()).device)


@torch.no_grad()
def selected_norm_ratio(model: GlaucomaNet, sel: Dict[str, List[int]]) -> float:
    """mean |w| of selected units / mean |w| of the kept units (1.0 = untouched, ~0 = ready to cut)."""
    mods = _mods(model)
    s_sum = s_n = k_sum = k_n = 0.0
    for name, idx in sel.items():
        w = mods[name].weight.detach().abs().flatten(1).mean(1)
        mask = torch.zeros(w.shape[0], dtype=torch.bool, device=w.device)
        mask[idx] = True
        s_sum += w[mask].sum().item(); s_n += mask.sum().item()
        k_sum += w[~mask].sum().item(); k_n += (~mask).sum().item()
    if s_n == 0 or k_n == 0 or k_sum == 0:
        return float('nan')
    return (s_sum / s_n) / (k_sum / k_n)


@torch.no_grad()
def cut(model: GlaucomaNet, sel: Dict[str, List[int]]) -> GlaucomaNet:
    """Returns a new, smaller model with the selected units removed (weights copied exactly)."""
    mods = _mods(model)
    convs = model.conv_names()
    keep = {}
    for name in convs + [FC_HIDDEN]:
        m = mods[name]
        n = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
        drop = set(sel.get(name, []))
        keep[name] = [i for i in range(n) if i not in drop]

    arch = model.arch()
    arch['channels'] = [len(keep[n]) for n in convs]
    arch['hidden'] = len(keep[FC_HIDDEN])
    new = build_from_arch(arch).to(next(model.parameters()).device)
    new_mods = _mods(new)

    prev_keep = list(range(model.in_channels))
    for name in convs:
        src, dst = mods[name], new_mods[name]
        k_out = torch.as_tensor(keep[name], device=src.weight.device)
        k_in = torch.as_tensor(prev_keep, device=src.weight.device)
        dst.weight.copy_(src.weight[k_out][:, k_in])
        dst.bias.copy_(src.bias[k_out])
        prev_keep = keep[name]

    # fc.0: columns = (channel, spatial) blocks of the last conv, rows = hidden units
    S = model.spatial
    cols = torch.as_tensor([c * S + p for c in prev_keep for p in range(S)], device=src.weight.device)
    rows = torch.as_tensor(keep[FC_HIDDEN], device=src.weight.device)
    new_mods["fc.0"].weight.copy_(mods["fc.0"].weight[rows][:, cols])
    new_mods["fc.0"].bias.copy_(mods["fc.0"].bias[rows])
    new_mods["fc.3"].weight.copy_(mods["fc.3"].weight[:, rows])
    new_mods["fc.3"].bias.copy_(mods["fc.3"].bias)

    # keep dropout settings
    for n, m in mods.items():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            new_mods[n].p = m.p
    new.train(model.training)
    return new
