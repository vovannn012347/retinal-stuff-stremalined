"""Unit checks: slicing is exact, old checkpoints load, schedule timing, fold rotation, grouping.

    python code_new/tests/test_surgery.py
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgical import surgery as S                                  # noqa: E402
from surgical.data import FoldRotation, group_key, split_groups, Sample   # noqa: E402
from surgical.model import GlaucomaNet, default_channels, load_model_file, count_params  # noqa: E402
from surgical.phases import SurgicalSchedule                       # noqa: E402
from surgical.config import load_config                            # noqa: E402

import numpy as np                                                  # noqa: E402


def test_cut_is_exact():
    torch.manual_seed(0)
    m = GlaucomaNet(default_channels(8), hidden=16, input_size=32).eval()
    sel = S.select_units(m, 0.25, 2, True, 'l1_inout')
    assert sel and all(len(v) > 0 for v in sel.values())
    # zero the selected units completely -> removing them must not change the output
    S.shrink(m, sel, 1.0)
    x = torch.randn(4, 3, 32, 32)
    y0 = m(x)
    c = S.cut(m, sel).eval()
    y1 = c(x)
    assert torch.allclose(y0, y1, atol=1e-5), (y0 - y1).abs().max()
    assert count_params(c) < count_params(m)
    for name, idx in sel.items():
        mod = dict(m.named_modules())[name]
        n = mod.out_channels if isinstance(mod, nn.Conv2d) else mod.out_features
        nc = dict(c.named_modules())[name]
        assert (nc.out_channels if isinstance(nc, nn.Conv2d) else nc.out_features) == n - len(idx)
    print("cut exact: ok", count_params(m), '->', count_params(c))


def test_old_checkpoint_loads():
    # same state-dict layout as HandmadeGlaucomaClassifier (16_01_2026), incl. its Sigmoid at fc.4
    old = GlaucomaNet(default_channels(8), hidden=16, input_size=32)
    sd = old.state_dict()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'old.pth'
        torch.save({'state_dict': sd, 'optimizer': None,
                    'metadata': {'epoch': 5, 'base_channels': 8, 'num_classes': 3, 'input_size': 32}}, p)
        m = load_model_file(p)
        assert m.channels == default_channels(8) and m.hidden == 16
    print("old checkpoint format: ok")


def test_schedule_timing():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs' / 'surgical_base.yaml').surgical
    cfg['min_learn_passes'] = 1
    cfg['gate_consecutive_passes'] = 1
    s = SurgicalSchedule(cfg)
    s.begin_pass(); assert s.end_pass(True, False) == 'accept'
    s.start_ramp()
    trace, n = [], 0
    while True:
        info = s.begin_pass(); n += 1
        trace.append((info['dropout'], info['penalty'], info['select_now']))
        a = s.end_pass(True, False)
        if a == 'cut':
            break
        assert n < 100
    sel_at = [i for i, t in enumerate(trace) if t[2]]
    assert len(sel_at) == 1 and abs(trace[sel_at[0]][0] - 0.2) < 1e-9, trace[:12]
    assert abs(trace[-1][0] - 0.5) < 1e-9 and abs(trace[-1][1] - 0.5) < 1e-9
    pen_passes = sum(1 for t in trace if t[1] > 0)
    print(f"schedule: {n} ramp passes, select at dropout {trace[sel_at[0]][0]}, {pen_passes} penalty passes, "
          f"penalties {[t[1] for t in trace if t[1] > 0][:3]}...{trace[-1][1]}")
    assert n == 25 and pen_passes == 16, (n, pen_passes)


def test_hold():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs' / 'surgical_base.yaml').surgical
    s = SurgicalSchedule(cfg); s.start_ramp()
    s.begin_pass(); s.end_pass(False, True)          # degraded -> next pass holds
    d0 = s.dropout
    info = s.begin_pass()
    assert info['holding'] and info['dropout'] == d0 and not info['apply_penalty']
    print("hold: ok")


def test_rotation():
    r = FoldRotation(n_datasets=2, k=4, passes_per_dataset=3, seed=1)
    seen_switch = []
    for step in range(4 * 7):
        ds, tr, te = r.current()
        assert tr != te
        done, sw = r.advance()
        if sw:
            seen_switch.append(r.total_passes)
    assert seen_switch == [3, 6], seen_switch
    st = r.state_dict(); r2 = FoldRotation(2, 4, 3, 99); r2.load_state_dict(st)
    assert r2.current() == r.current()
    print("rotation: ok, dataset switches after passes", seen_switch)


def test_grouping_and_split():
    assert group_key('1138_left_0.jpg') == '1138'
    assert group_key('1138_right.jpg') == '1138'
    assert group_key('028ІГБ02_1_0.jpg') == '028ІГБ'
    assert group_key('037МГК01_0.jpg') == '037МГК'
    assert group_key('0_right_cropped_0.jpg') == '0'
    rng = np.random.default_rng(0)
    samples = []
    for pid in range(60):
        lab = rng.integers(0, 2, 3).astype(np.float32)
        for eye in ('left', 'right'):
            samples.append(Sample(f'{pid}_{eye}_0.jpg', lab, group_key(f'{pid}_{eye}_0.jpg')))
    test, folds = split_groups(samples, 5, 0.1, 0.5, 1)
    owner = {}
    for fi, f in enumerate(folds + [test]):
        for i in f:
            g = samples[i].group
            assert owner.setdefault(g, fi) == fi, "patient split across folds"
    assert sum(len(f) for f in folds) + len(test) == len(samples)
    print("grouping/split: ok, folds", [len(f) for f in folds], "test", len(test))


if __name__ == '__main__':
    test_cut_is_exact()
    test_old_checkpoint_loads()
    test_schedule_timing()
    test_hold()
    test_rotation()
    test_grouping_and_split()
    print("ALL OK")
