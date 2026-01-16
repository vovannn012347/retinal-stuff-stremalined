import torch
import torch.nn as nn
import torchvision.models as models


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive, weight_negative, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.reduction = reduction

    def forward(self, input_sigmoid, target):
        # Calculate binary cross-entropy loss
        loss = - (self.weight_positive * target * torch.log(input_sigmoid + 1e-8) +
                  self.weight_negative * (1 - target) * torch.log(1 - input_sigmoid + 1e-8))

        if self.reduction == 'mean':
            return torch.mean(loss)  # Return scalar mean loss
        elif self.reduction == 'sum':
            return torch.sum(loss)  # Return scalar total loss
        elif self.reduction == 'none':
            return loss  # Return element-wise loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")

class HandmadeGlaucomaClassifier(nn.Module):
    """
    Main classifier - base can be any value, dropout can be changed
    """
    def __init__(self, input_size=64, input_channels=3, num_classes=3, base=64, dropout_p=0.0):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.base = base

        self.dropout = nn.Dropout(dropout_p)

        self.encoder1 = self._conv_block(input_channels,          base)
        self.encoder2 = self._conv_block(base,       base*2)
        self.encoder3 = self._conv_block(base*2,     base*4)
        self.encoder4 = self._conv_block(base*4,     base*8)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base*8, base*2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(base*2, num_classes),
            nn.Sigmoid()
        )

    @staticmethod
    def _conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder1(x); x = self.pool(x)
        x = self.encoder2(x); x = self.pool(x)
        x = self.encoder3(x); x = self.pool(x)
        x = self.encoder4(x); x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def set_dropout(self, p: float):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p


# ── Helper: extract current channel configuration ─────────────────────────────
def get_channel_config(model):
    """Returns dict of current out_channels for all Conv2d layers"""
    config = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            config[name] = m.out_channels
    return config


# ── Save checkpoint with pruning metadata ─────────────────────────────────────
def save_compressed_checkpoint(model, optimizer, epoch, path, extra_meta=None):
    """
    Save model + optimizer + metadata needed for exact reconstruction
    """
    metadata = {
        'epoch': epoch,
        'base_channels': model.base,
        'num_classes': model.num_classes,
        'input_size': model.input_size,
        'channel_config': get_channel_config(model),
        'pruning_stage': extra_meta.get('pruning_stage', 0) if extra_meta else 0,
    }

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'metadata': metadata
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint → {path}  (base={model.base}, pruning stage={metadata['pruning_stage']})")


# ── Load with reconstruction of pruned structure ──────────────────────────────
def load_compressed_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt['metadata']
    state_dict = ckpt['state_dict']

    model = HandmadeGlaucomaClassifier(
        input_size=meta['input_size'],
        num_classes=meta['num_classes'],
        base=meta['base_channels']
    ).to(device)

    # Dictionary to track which layers have been resized
    channel_cfg = meta['channel_config']

    with torch.no_grad():
        last_out_channels = 3  # Input RGB

        for name, m in model.named_modules():
            # Adjust Conv2d Layers
            if isinstance(m, nn.Conv2d):
                target_out = channel_cfg.get(name, m.out_channels)
                # Slice input weights to match previous layer's output
                m.weight.data = m.weight.data[:, :last_out_channels]
                # Slice output weights/bias to match this layer's pruned state
                m.weight.data = m.weight.data[:target_out]
                if m.bias is not None:
                    m.bias.data = m.bias.data[:target_out]

                m.in_channels = last_out_channels
                m.out_channels = target_out
                last_out_channels = target_out

            # Adjust Linear Layers (The Head)
            elif isinstance(m, nn.Linear):
                # Only the first Linear layer in the head needs input slicing
                if m.in_features != state_dict[f"{name}.weight"].shape[1]:
                    m.weight.data = m.weight.data[:, :state_dict[f"{name}.weight"].shape[1]]
                    m.in_features = state_dict[f"{name}.weight"].shape[1]

    model.load_state_dict(state_dict, strict=True)
    return model

def load_vgg16_weights(model):
    # Load pretrained VGG16 (features only)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

    # Map your encoder layers to VGG16 feature layers
    # VGG16: Conv(0, 2), Conv(5, 7), Conv(10, 12), etc.
    mapping = {
        'encoder1': [0, 2],
        'encoder2': [5, 7],
        'encoder3': [10, 12],
        'encoder4': [17, 19]  # Choosing deeper layers for deeper encoders
    }

    model_dict = model.state_dict()

    for my_layer, vgg_indices in mapping.items():
        for i, vgg_idx in enumerate(vgg_indices):
            # Map Conv2d in conv_block (indices 0 and 2)
            vgg_conv = vgg16[vgg_idx]
            my_conv_idx = 0 if i == 0 else 2

            # Extract weights and biases
            weight_key = f"{my_layer}.{my_conv_idx}.weight"
            bias_key = f"{my_layer}.{my_conv_idx}.bias"

            if weight_key in model_dict:
                # Resize if base channels don't match 64/128/256/512
                vgg_weight = vgg_conv.weight.data
                if vgg_weight.shape == model_dict[weight_key].shape:
                    model_dict[weight_key].copy_(vgg_weight)
                    model_dict[bias_key].copy_(vgg_conv.bias.data)
                    print(f"Successfully loaded VGG16 weights into {weight_key}")
                else:
                    print(f"Skipping {weight_key}: Shape mismatch")

    model.load_state_dict(model_dict)
