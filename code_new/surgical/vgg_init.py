"""Copy VGG16 ImageNet conv weights into a VGG-shaped encoder of any width.

Full width (base 64): plain copy. Narrower (e.g. base 32): each layer keeps the VGG filters with
the largest L1 norm, and the next layer takes exactly the matching input channels, so the copied
encoder is a consistent sub-network of VGG16 rather than random slices.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def copy_vgg16_into(model: nn.Module, conv_names):
    from torchvision.models import vgg16, VGG16_Weights
    vgg_convs = [m for m in vgg16(weights=VGG16_Weights.DEFAULT).features if isinstance(m, nn.Conv2d)]
    mods = dict(model.named_modules())
    prev_keep = torch.arange(3)
    for name, src in zip(conv_names, vgg_convs):
        dst = mods[name]
        if dst.in_channels != len(prev_keep):
            raise ValueError(f"{name}: input width {dst.in_channels} does not follow the previous layer")
        if dst.out_channels > src.out_channels:
            raise ValueError(f"{name}: {dst.out_channels} outputs, VGG16 has only {src.out_channels} (base > 64?)")
        w = src.weight[:, prev_keep]
        keep = torch.argsort(w.abs().flatten(1).sum(1), descending=True)[:dst.out_channels].sort().values
        dst.weight.copy_(w[keep])
        dst.bias.copy_(src.bias[keep])
        prev_keep = keep
