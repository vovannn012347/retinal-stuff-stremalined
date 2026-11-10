"""Optic-disc classifier with an explicit, sliceable channel layout.

Same layout and module names as `HandmadeGlaucomaClassifier` from the 16_01_2026 trainer
(encoder1..4, intermediate, fc), so its checkpoints load directly. Differences:
  * every conv width is a constructor argument (`channels`), so a sliced model can be rebuilt;
  * `forward` returns logits (no Sigmoid); use `predict_proba` for probabilities;
  * head 'gap' (default for new models): global average pooling before fc.0, so fc.0 has
    C x hidden weights instead of C x 64 x hidden (2M of the 12M parameters at base 64).
    head 'flatten' = the old layout (old checkpoints are detected and loaded with it).
"""
from typing import List, Optional

import torch
import torch.nn as nn

# conv layers per encoder block (VGG16 layout) + 1 intermediate conv
BLOCKS = [2, 2, 3, 3]
N_CONV = sum(BLOCKS) + 1


def default_channels(base: int = 64) -> List[int]:
    return [base] * 2 + [base * 2] * 2 + [base * 4] * 3 + [base * 8] * 3 + [base * 8]


class GlaucomaNet(nn.Module):
    def __init__(self, channels: Optional[List[int]] = None, hidden: int = 64, num_classes: int = 3,
                 input_size: int = 128, in_channels: int = 3, head: str = 'gap'):
        super().__init__()
        channels = list(channels or default_channels())
        assert len(channels) == N_CONV, f"expected {N_CONV} conv widths, got {len(channels)}"
        self.channels, self.hidden = channels, hidden
        self.num_classes, self.input_size, self.in_channels = num_classes, input_size, in_channels
        if head not in ('gap', 'flatten'):
            raise ValueError(f"model.head must be gap | flatten, got {head!r}")
        self.head = head

        c = iter(channels)
        prev = in_channels
        for b, n in enumerate(BLOCKS, start=1):
            layers = []
            for _ in range(n):
                out = next(c)
                layers += [nn.Conv2d(prev, out, 3, padding=1), nn.LeakyReLU(0.01, inplace=True)]
                prev = out
            layers.append(nn.Dropout2d(p=0.0))
            setattr(self, f"encoder{b}", nn.Sequential(*layers))
        out = next(c)
        self.intermediate = nn.Sequential(nn.Conv2d(prev, out, 3, padding=1),
                                          nn.LeakyReLU(0.01, inplace=True), nn.Dropout2d(p=0.0))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.spatial = 1 if head == 'gap' else (input_size // 16) ** 2
        self.fc = nn.Sequential(nn.Linear(out * self.spatial, hidden), nn.LeakyReLU(0.01, inplace=True),
                                nn.Dropout(p=0.0), nn.Linear(hidden, num_classes))

    # ── layout helpers ────────────────────────────────────────────────────────────────
    def conv_names(self) -> List[str]:
        names = []
        for b, n in enumerate(BLOCKS, start=1):
            names += [f"encoder{b}.{2 * i}" for i in range(n)]
        return names + ["intermediate.0"]

    def arch(self) -> dict:
        return {'channels': list(self.channels), 'hidden': self.hidden, 'num_classes': self.num_classes,
                'input_size': self.input_size, 'in_channels': self.in_channels, 'head': self.head}

    # ── graph for surgery (see surgery.py) ──────────────────────────────────────────────
    def graph(self):
        convs = self.conv_names()
        nodes, prev = [], 'input'
        for n in convs:
            nodes.append({'name': n, 'inputs': [prev], 'cut': True})
            prev = n
        nodes.append({'name': 'fc.0', 'inputs': [prev], 'spatial': self.spatial, 'cut': True})
        nodes.append({'name': 'fc.3', 'inputs': ['fc.0'], 'cut': False})
        return nodes

    def widths(self):
        w = dict(zip(self.conv_names(), self.channels))
        w['fc.0'] = self.hidden
        w['fc.3'] = self.num_classes
        return w

    def with_widths(self, widths):
        arch = self.arch()
        arch['channels'] = [widths[n] for n in self.conv_names()]
        arch['hidden'] = widths['fc.0']
        return GlaucomaNet(**arch)

    def set_dropout(self, p: float, layer_names: Optional[List[str]] = None):
        mods = dict(self.named_modules())
        targets = layer_names or [n for n, m in mods.items() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
        for n in targets:
            m = mods.get(n)
            if not isinstance(m, (nn.Dropout, nn.Dropout2d)):
                raise KeyError(f"'{n}' is not a dropout layer (dropout layers: "
                               f"{[k for k, v in mods.items() if isinstance(v, (nn.Dropout, nn.Dropout2d))]})")
            m.p = float(p)

    def forward(self, x):
        x = self.pool(self.encoder1(x))
        x = self.pool(self.encoder2(x))
        x = self.pool(self.encoder3(x))
        x = self.pool(self.encoder4(x))
        x = self.intermediate(x)
        if self.head == 'gap':
            x = self.gpool(x)
        return self.fc(self.flatten(x))

    def param_groups(self):
        """(pretrained-encoder params, newly initialised params) for separate learning rates"""
        enc = [p for n, p in self.named_parameters() if n.startswith('encoder')]
        new = [p for n, p in self.named_parameters() if not n.startswith('encoder')]
        return enc, new

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── construction / loading ────────────────────────────────────────────────────────────
def arch_from_state_dict(sd: dict, input_size: int = 128) -> dict:
    tmp = GlaucomaNet(input_size=input_size)
    channels = [sd[f"{n}.weight"].shape[0] for n in tmp.conv_names()]
    head = 'gap' if sd['fc.0.weight'].shape[1] == channels[-1] and input_size > 16 else 'flatten'
    return {'channels': channels, 'hidden': sd['fc.0.weight'].shape[0], 'num_classes': sd['fc.3.weight'].shape[0],
            'input_size': input_size, 'in_channels': sd['encoder1.0.weight'].shape[1], 'head': head}


def build_from_arch(arch: dict) -> GlaucomaNet:
    return GlaucomaNet(**arch)


def load_model_file(path, map_location='cpu', input_size: int = 128) -> GlaucomaNet:
    """Loads a model saved by this package, or a checkpoint of the 16_01_2026 trainer."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'arch' in ckpt and 'model' in ckpt:
        model = build_from_arch(ckpt['arch'])
        model.load_state_dict(ckpt['model'])
        return model
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    meta = ckpt.get('metadata', {}) if isinstance(ckpt, dict) else {}
    arch = arch_from_state_dict(sd, input_size=meta.get('input_size', input_size))
    model = build_from_arch(arch)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint {path} does not match the model: missing={missing} unexpected={unexpected}")
    return model


def init_from_vgg16(model: GlaucomaNet):
    """VGG16 ImageNet conv weights into encoder1..4; narrower models get the strongest filters."""
    from .vgg_init import copy_vgg16_into
    copy_vgg16_into(model, model.conv_names()[:-1])
    return model
