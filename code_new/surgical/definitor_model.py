"""Optic nerve disc definitor (segmentation) with an explicit, sliceable channel layout.

Same topology as `FcnskipNerveDefinitor2` (VGG16-like encoder, intermediate conv, three decoder
blocks with skip connections and ConvTranspose upsampling, 1x1 output conv), so its
checkpoints load (with or without its dropout layers). Differences:
  * every width is a constructor argument, so a sliced model can be rebuilt;
  * one Dropout2d at the end of every block (the places the dropout ramp acts on);
  * `forward` returns logits (no Sigmoid); `predict_proba` gives probabilities.

Shapes for input H (divisible by 16):
  enc1 H, pool1 H/2 | enc2 H/2, pool2 H/4 | enc3 H/4, pool3 H/8 | enc4 H/8, pool4 H/16 | interm H/16
  up(interm) H/8 ++ pool3 -> decoder4 -> H/4 ++ pool2 -> decoder3 -> H/2 ++ pool1 -> decoder2 -> H -> 1x1
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn

ENC_BLOCKS = [2, 2, 3, 3]

# node names in graph order
ENC_CONVS = ([f"encoder1.{i}" for i in (0, 2)] + [f"encoder2.{i}" for i in (0, 2)]
             + [f"encoder3.{i}" for i in (0, 2, 4)] + [f"encoder4.{i}" for i in (0, 2, 4)])
DEC = [("decoder4.0", "decoder4.3"), ("decoder3.0", "decoder3.3"), ("decoder2.0", "decoder2.3")]
FINAL = "decoder_1_final_conv.0"
UNITS = ENC_CONVS + ["intermidiate.0"] + [n for pair in DEC for n in pair]
DROPOUT_LAYERS = ["encoder1.4", "encoder2.4", "encoder3.6", "encoder4.6", "intermidiate.2",
                  "decoder4.2", "decoder3.2", "decoder2.2"]


def default_widths(base: int = 64) -> Dict[str, int]:
    w = {}
    for n, c in zip(ENC_CONVS, [base] * 2 + [base * 2] * 2 + [base * 4] * 3 + [base * 8] * 3):
        w[n] = c
    w["intermidiate.0"] = base * 8
    for (conv, convt), c in zip(DEC, [base * 4, base * 2, base]):
        w[conv] = c
        w[convt] = c
    return w


class DefinitorNet(nn.Module):
    def __init__(self, widths: Optional[Dict[str, int]] = None, num_classes: int = 1, in_channels: int = 3):
        super().__init__()
        w = dict(default_widths()) if widths is None else dict(widths)
        missing = [u for u in UNITS if u not in w]
        assert not missing, f"widths missing {missing}"
        self.w = {u: int(w[u]) for u in UNITS}
        self.num_classes, self.in_channels = num_classes, in_channels

        prev = in_channels
        names = iter(ENC_CONVS)
        for b, n in enumerate(ENC_BLOCKS, start=1):
            layers = []
            for _ in range(n):
                out = self.w[next(names)]
                layers += [nn.Conv2d(prev, out, 3, padding=1), nn.LeakyReLU(0.01, inplace=True)]
                prev = out
            layers.append(nn.Dropout2d(p=0.0))
            setattr(self, f"encoder{b}", nn.Sequential(*layers))
        wi = self.w["intermidiate.0"]
        self.intermidiate = nn.Sequential(nn.Conv2d(prev, wi, 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                          nn.Dropout2d(p=0.0))
        skips = [self.w["encoder3.4"], self.w["encoder2.2"], self.w["encoder1.2"]]
        up_in = wi
        for (conv, convt), skip in zip(DEC, skips):
            c, ct = self.w[conv], self.w[convt]
            block = nn.Sequential(nn.Conv2d(up_in + skip, c, 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                  nn.Dropout2d(p=0.0), nn.ConvTranspose2d(c, ct, 2, stride=2))
            setattr(self, conv.split('.')[0], block)
            up_in = ct
        self.decoder_1_final_conv = nn.Sequential(nn.Conv2d(up_in, num_classes, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # ── graph for surgery ─────────────────────────────────────────────────────────────
    def graph(self):
        g, prev = [], 'input'
        for n in ENC_CONVS:
            g.append({'name': n, 'inputs': [prev], 'cut': True})
            prev = n
        g.append({'name': 'intermidiate.0', 'inputs': ['encoder4.4'], 'cut': True})
        up = 'intermidiate.0'
        for (conv, convt), skip in zip(DEC, ['encoder3.4', 'encoder2.2', 'encoder1.2']):
            g.append({'name': conv, 'inputs': [up, skip], 'cut': True})
            g.append({'name': convt, 'inputs': [conv], 'cut': True})
            up = convt
        g.append({'name': FINAL, 'inputs': [up], 'cut': False})
        return g

    def widths(self):
        w = dict(self.w)
        w[FINAL] = self.num_classes
        return w

    def with_widths(self, widths):
        return DefinitorNet({u: widths[u] for u in UNITS}, self.num_classes, self.in_channels)

    def arch(self):
        return {'widths': dict(self.w), 'num_classes': self.num_classes, 'in_channels': self.in_channels}

    def set_dropout(self, p: float, layer_names: Optional[List[str]] = None):
        mods = dict(self.named_modules())
        for n in layer_names or DROPOUT_LAYERS:
            m = mods.get(n)
            if not isinstance(m, (nn.Dropout, nn.Dropout2d)):
                raise KeyError(f"'{n}' is not a dropout layer (dropout layers: {DROPOUT_LAYERS})")
            m.p = float(p)

    def forward(self, x):
        e1 = self.encoder1(x); p1 = self.pool(e1)
        e2 = self.encoder2(p1); p2 = self.pool(e2)
        e3 = self.encoder3(p2); p3 = self.pool(e3)
        e4 = self.encoder4(p3); p4 = self.pool(e4)
        i = self.intermidiate(p4)
        d4 = self.decoder4(torch.cat([self.upsample(i), p3], 1))
        d3 = self.decoder3(torch.cat([d4, p2], 1))
        d2 = self.decoder2(torch.cat([d3, p1], 1))
        return self.decoder_1_final_conv(d2)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# ── loading ───────────────────────────────────────────────────────────────────────────
def _param_layers_in_order(sd: dict, prefix: str):
    """indices of weight-carrying layers inside an old nn.Sequential, in order"""
    idx = sorted({int(k[len(prefix) + 1:].split('.')[0]) for k in sd
                  if k.startswith(prefix + '.') and k.endswith('.weight')})
    return [f"{prefix}.{i}" for i in idx]


def convert_old_state_dict(sd: dict) -> dict:
    """FcnskipNerveDefinitor2 state dict (any dropout layout) -> DefinitorNet names."""
    out = {}
    blocks = {'encoder1': ['encoder1.0', 'encoder1.2'], 'encoder2': ['encoder2.0', 'encoder2.2'],
              'encoder3': ['encoder3.0', 'encoder3.2', 'encoder3.4'],
              'encoder4': ['encoder4.0', 'encoder4.2', 'encoder4.4'], 'intermidiate': ['intermidiate.0'],
              'decoder4': ['decoder4.0', 'decoder4.3'], 'decoder3': ['decoder3.0', 'decoder3.3'],
              'decoder2': ['decoder2.0', 'decoder2.3'], 'decoder_1_final_conv': [FINAL]}
    for block, new_names in blocks.items():
        old_names = _param_layers_in_order(sd, block)
        if len(old_names) != len(new_names):
            raise RuntimeError(f"old checkpoint block {block}: {old_names} does not match {new_names}")
        for o, n in zip(old_names, new_names):
            out[f"{n}.weight"] = sd[f"{o}.weight"]
            if f"{o}.bias" in sd:
                out[f"{n}.bias"] = sd[f"{o}.bias"]
    return out


def widths_from_state_dict(sd: dict) -> Dict[str, int]:
    w = {}
    for u in UNITS:
        t = sd[f"{u}.weight"]
        w[u] = t.shape[1] if u in [ct for _, ct in DEC] else t.shape[0]
    return w


def load_definitor_file(path, map_location='cpu') -> DefinitorNet:
    """A model saved by this package, or an old definitor checkpoint ({'nerve_definitor': state_dict})."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'arch' in ckpt and 'model' in ckpt:
        m = DefinitorNet(**ckpt['arch'])
        m.load_state_dict(ckpt['model'])
        return m
    sd = ckpt
    if isinstance(ckpt, dict):
        for key in ('nerve_definitor', 'state_dict', 'model'):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
    sd = {k: v for k, v in sd.items() if not any(s in k for s in ('activation_post_process', 'quant', 'fake_quant'))}
    sd = convert_old_state_dict(sd)
    m = DefinitorNet(widths_from_state_dict(sd), num_classes=sd[f"{FINAL}.weight"].shape[0],
                     in_channels=sd["encoder1.0.weight"].shape[1])
    m.load_state_dict(sd)
    return m


def init_definitor_from_vgg16(model: DefinitorNet):
    from torchvision.models import vgg16, VGG16_Weights
    vgg_convs = [m for m in vgg16(weights=VGG16_Weights.DEFAULT).features if isinstance(m, nn.Conv2d)]
    mods = dict(model.named_modules())
    for name, src in zip(ENC_CONVS, vgg_convs):
        dst = mods[name]
        if dst.weight.shape != src.weight.shape:
            raise ValueError(f"VGG16 init needs model.base=64 ({name}: {tuple(dst.weight.shape)})")
        with torch.no_grad():
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
    return model
