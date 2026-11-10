"""Unit checks: slicing is exact for both models, old checkpoints load, schedule timing, fold rotation,
grouping, mistake replay.

    python code_new/tests/test_surgery.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgical import surgery as S                                                     # noqa: E402
from surgical.config import load_config                                               # noqa: E402
from surgical.data import FoldRotation, MistakeTracker, Sample, group_key, split_groups  # noqa: E402
from surgical.definitor_model import ENC_CONVS, DefinitorNet, default_widths, load_definitor_file  # noqa: E402
from surgical.model import GlaucomaNet, default_channels, load_model_file             # noqa: E402
from surgical.phases import SurgicalSchedule                                          # noqa: E402

CFG = Path(__file__).resolve().parents[1] / 'configs'


def params(m):
    return sum(p.numel() for p in m.parameters())


def _check_cut(model, x, mode):
    model.eval()
    sel = S.select_units(model, 0.25, 2, (), mode)
    assert sel
    S.shrink(model, sel, 1.0)                    # selected outputs are now exactly zero
    y0 = model(x)
    c = S.cut(model, sel).eval()
    y1 = c(x)
    assert torch.allclose(y0, y1, atol=1e-4), (y0 - y1).abs().max()
    mods, cmods = dict(model.named_modules()), dict(c.named_modules())
    for name, idx in sel.items():
        assert S.n_out(cmods[name]) == S.n_out(mods[name]) - len(idx)
    return params(model), params(c)


def test_cut_exact_classifier():
    torch.manual_seed(0)
    a, b = _check_cut(GlaucomaNet(default_channels(8), hidden=16, input_size=32), torch.randn(4, 3, 32, 32),
                      'l1_inout')
    print(f"classifier cut exact: {a} -> {b}")


def test_cut_exact_definitor():
    torch.manual_seed(0)
    # LeakyReLU(0) = 0 and bias 0 after shrink, but the ConvTranspose bias of a removed channel is
    # also zeroed, so the skip/concat paths see exactly the same zeros -> outputs must match.
    a, b = _check_cut(DefinitorNet(default_widths(8)), torch.randn(2, 3, 64, 64), 'l1_inout')
    a2, b2 = _check_cut(DefinitorNet(default_widths(8)), torch.randn(2, 3, 64, 64), 'l1')
    print(f"definitor cut exact (skip connections, ConvTranspose): {a} -> {b}")


def test_cut_exact_classifier_flatten():
    torch.manual_seed(0)
    a, b = _check_cut(GlaucomaNet(default_channels(8), hidden=16, input_size=32, head='flatten'),
                      torch.randn(4, 3, 32, 32), 'l1_inout')
    print(f"classifier (old flatten head) cut exact: {a} -> {b}")


def test_sliced_vgg_init():
    import torchvision.models as tvm
    from surgical import vgg_init
    real = tvm.vgg16
    tvm.vgg16 = lambda weights=None: real(weights=None)        # offline: random VGG with the real shapes
    try:
        torch.manual_seed(0)
        m = DefinitorNet(default_widths(32))
        torch.manual_seed(5)
        vgg_init.copy_vgg16_into(m, ENC_CONVS)
        torch.manual_seed(5)
        v = [layer for layer in real(weights=None).features if isinstance(layer, nn.Conv2d)]
        e = dict(m.named_modules())
        k = torch.argsort(v[0].weight.abs().flatten(1).sum(1), descending=True)[:32].sort().values
        assert torch.equal(e['encoder1.0'].weight, v[0].weight[k])
        w1 = v[1].weight[:, k]
        k2 = torch.argsort(w1.abs().flatten(1).sum(1), descending=True)[:32].sort().values
        assert torch.equal(e['encoder1.2'].weight, w1[k2])
    finally:
        tvm.vgg16 = real
    print("sliced VGG16 init (base 32 takes the strongest, consistent filters): ok")


def test_boundary_tolerance():
    from surgical.tasks import _SegAcc
    y = torch.zeros(1, 1, 32, 32); y[..., 10:20, 10:20] = 1
    logits = torch.full_like(y, -10.0); logits[..., 11:19, 11:19] = 10      # 1 px short on every side
    strict = _SegAcc(0.5, 0); strict.add(logits, y); s = strict.result()
    tol = _SegAcc(0.5, 1); tol.add(logits, y); t = tol.result()
    assert s['recall'] < 0.7 and abs(t['recall'] - 1.0) < 1e-9 and abs(t['strict_recall'] - s['recall']) < 1e-9
    print(f"boundary tolerance: strict recall {s['recall']:.2f}, with 1 px tolerance {t['recall']:.2f}")


def test_old_classifier_checkpoint():
    old = GlaucomaNet(default_channels(8), hidden=16, input_size=32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'old.pth'
        torch.save({'state_dict': old.state_dict(), 'optimizer': None,
                    'metadata': {'epoch': 5, 'base_channels': 8, 'num_classes': 3, 'input_size': 32}}, p)
        m = load_model_file(p)
        assert m.channels == default_channels(8) and m.hidden == 16 and m.head == 'gap'
        old_flat = GlaucomaNet(default_channels(8), hidden=16, input_size=32, head='flatten')
        torch.save({'state_dict': old_flat.state_dict(), 'metadata': {'input_size': 32}}, p)
        assert load_model_file(p).head == 'flatten'
    print("old classifier checkpoint: ok")


def _old_definitor_sd(new: DefinitorNet, with_dropout: bool):
    """state dict laid out like FcnskipNerveDefinitor2 (use_dropout False/True)."""
    step = 3 if with_dropout else 2
    sd = new.state_dict()
    out = {}
    blocks = {'encoder1': 2, 'encoder2': 2, 'encoder3': 3, 'encoder4': 3, 'intermidiate': 1}
    for b, n in blocks.items():
        for i in range(n):
            for t in ('weight', 'bias'):
                out[f"{b}.{i * step}.{t}"] = sd[f"{b}.{2 * i}.{t}"]
    for b in ('decoder4', 'decoder3', 'decoder2'):
        for t in ('weight', 'bias'):
            out[f"{b}.0.{t}"] = sd[f"{b}.0.{t}"]
            out[f"{b}.{step}.{t}"] = sd[f"{b}.3.{t}"]
    for t in ('weight', 'bias'):
        out[f"decoder_1_final_conv.0.{t}"] = sd[f"decoder_1_final_conv.0.{t}"]
    return out


def test_old_definitor_checkpoint():
    torch.manual_seed(1)
    ref = DefinitorNet(default_widths(8)).eval()
    x = torch.randn(1, 3, 32, 32)
    with tempfile.TemporaryDirectory() as d:
        for drop in (False, True):
            p = Path(d) / f'old_{drop}.pth'
            torch.save({'nerve_definitor': _old_definitor_sd(ref, drop), 'optimizer': {}}, p)
            m = load_definitor_file(p).eval()
            assert torch.allclose(m(x), ref(x))
    print("old definitor checkpoints (with and without dropout layers): ok")


def test_schedule_timing():
    cfg = load_config(CFG / 'classifier_base.yaml').surgical
    cfg['min_learn_passes'] = 1
    cfg['gate_consecutive_passes'] = 1
    s = SurgicalSchedule(cfg)
    s.begin_pass(); assert s.end_pass(True, False) == 'accept'
    s.start_ramp()
    trace, n = [], 0
    while True:
        info = s.begin_pass(); n += 1
        trace.append((info['dropout'], info['penalty'], info['select_now']))
        if s.end_pass(True, False) == 'cut':
            break
        assert n < 100
    sel_at = [i for i, t in enumerate(trace) if t[2]]
    assert len(sel_at) == 1 and abs(trace[sel_at[0]][0] - 0.2) < 1e-9
    assert abs(trace[-1][0] - 0.5) < 1e-9 and abs(trace[-1][1] - 0.5) < 1e-9
    pen = sum(1 for t in trace if t[1] > 0)
    want = round((0.5 - float(cfg['learn_dropout'])) / 0.02)
    assert n == want and pen == 16, (n, want, pen)
    print(f"schedule: {n} ramp passes, select at dropout 0.2, {pen} penalty passes")


def test_hold():
    s = SurgicalSchedule(load_config(CFG / 'classifier_base.yaml').surgical); s.start_ramp()
    s.begin_pass(); s.end_pass(False, True)
    d0 = s.dropout
    info = s.begin_pass()
    assert info['holding'] and info['dropout'] == d0 and not info['apply_penalty']
    print("hold: ok")


def test_rotation():
    r = FoldRotation(n_datasets=2, k=4, passes_per_dataset=3, seed=1)
    sw = []
    for _ in range(4 * 7):
        ds, tr, te = r.current()
        assert tr != te
        if r.advance()[1]:
            sw.append(r.total_passes)
    assert sw == [3, 6], sw
    r2 = FoldRotation(2, 4, 3, 99); r2.load_state_dict(r.state_dict())
    assert r2.current() == r.current()
    print("rotation: ok, dataset switches after passes", sw)


def test_grouping_and_split():
    cases = {'1138_left_0.jpg': '1138', '1138_right.jpg': '1138', '0_right_cropped_0.jpg': '0',
             '028ІГБ02_1_0.jpg': 'ІГБ', '027ІГБ01.jpg': 'ІГБ', '001АЄЯ01.jpg': 'АЄЯ', '002АЄЯ02.jpg': 'АЄЯ',
             '015МЛФ01.mask.jpg': 'МЛФ', 'eye_case_1.png': 'eye_case_1', 'eye_case_10.png': 'eye_case_10'}
    for name, want in cases.items():
        assert group_key(name) == want, (name, group_key(name), want)
    rng = np.random.default_rng(0)
    samples = []
    for pid in range(60):
        sig = tuple(int(v) for v in rng.integers(0, 2, 3))
        for eye in ('left', 'right'):
            samples.append(Sample(f'{pid}_{eye}_0.jpg', group_key(f'{pid}_{eye}_0.jpg'), sig))
    test, folds = split_groups(samples, 5, 0.1, 1)
    owner = {}
    for fi, f in enumerate(folds + [test]):
        for i in f:
            assert owner.setdefault(samples[i].group, fi) == fi, "patient split across folds"
    assert sum(len(f) for f in folds) + len(test) == len(samples)
    print("grouping/split: ok, folds", [len(f) for f in folds], "test", len(test))


def test_mistake_replay():
    t = MistakeTracker({'gain': 1.0, 'min_error': 0.25, 'max_extra': 3, 'ema': 0.5})
    samples = [Sample(f's{i}', f'p{i}') for i in range(6)]
    t.update([s.path for s in samples], [0.0, 0.0, 0.1, 0.3, 0.6, 0.95])          # median 0.2
    idx, extra = t.expand(list(range(6)), samples)
    assert [idx.count(i) for i in range(6)] == [1, 1, 1, 2, 3, 4] and extra == 6, (idx, extra)
    t2 = MistakeTracker({'min_error': 0.25})
    t2.update([s.path for s in samples], [0.8, 0.85, 0.9, 0.9, 0.95, 0.99])        # all bad: only above median
    idx2, _ = t2.expand(list(range(6)), samples)
    assert [idx2.count(i) for i in range(6)] == [1, 1, 1, 1, 4, 4], idx2
    t.update(['s5'], [0.05])                                       # ema: 0.95 -> 0.5
    assert abs(t.rec['s5']['err'] - 0.5) < 1e-9
    # suspect: wrong while the rest of the fold is learned, N times in a row
    t3 = MistakeTracker({'min_error': 0.25, 'suspect_error': 0.5, 'suspect_after': 3})
    for _ in range(3):
        t3.update([s.path for s in samples], [0.9] * 6)            # all wrong: no suspicion yet
    assert t3.n_suspect() == 0
    for _ in range(3):
        t3.update([s.path for s in samples], [0.0] * 5 + [0.9])    # only s5 wrong
    assert t3.rec['s5']['suspect'] and t3.n_suspect() == 1
    idx3, _ = t3.expand(list(range(6)), samples)
    assert idx3.count(5) == 1                                      # suspect: not repeated
    t3.update(['s5', 's0'], [0.0, 0.0])
    assert not t3.rec['s5']['suspect']
    print("mistake replay: ok", idx)


def test_enhance():
    from surgical import enhance as E
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    img = np.zeros((96, 96, 3), np.uint8)
    inside = (yy - 48) ** 2 + (xx - 48) ** 2 < 40 ** 2
    img[inside] = np.clip(np.array([200, 90, 40]) + rng.normal(0, 15, (inside.sum(), 3)), 0, 255).astype(np.uint8)
    out = E.enhance(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[~inside].max() == 0                       # border stays black
    s_in = __import__('cv2').cvtColor(img, __import__('cv2').COLOR_RGB2HSV)[..., 1][inside].std()
    s_out = __import__('cv2').cvtColor(out, __import__('cv2').COLOR_RGB2HSV)[..., 1][inside].std()
    assert s_out > s_in                                  # saturation spread out
    assert E.retina_box(img) == (8, 9, 88, 88) or E.retina_box(img)[0] <= 9
    print("enhance (HSV-S + CLAHE, border kept black): ok")


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print("ALL OK")
