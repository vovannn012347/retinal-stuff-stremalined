import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights


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
    def __init__(self, num_classes=3, input_channels=3, base=64, input_size=128):
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.input_size = input_size

        # Encoder blocks with Dropout2d at the END of each block
        self.encoder1 = self._conv_block_2(input_channels, base)
        self.encoder2 = self._conv_block_2(base, base * 2)
        self.encoder3 = self._conv_block_3(base * 2, base * 4)
        self.encoder4 = self._conv_block_3(base * 4, base * 8)

        self.intermediate = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(p=0.0) # Standard placement for intermediate reg
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        flat_feat_size = base * 8 * (input_size // 16) * (input_size // 16)

        self.fc = nn.Sequential(
            nn.Linear(flat_feat_size, base),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(base, num_classes),
            nn.Sigmoid()
        )

    @staticmethod
    def _conv_block_2(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(p=0.0) # Per-block dropout
        )

    @staticmethod
    def _conv_block_3(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(p=0.0) # Per-block dropout
        )

    def set_dropout(self, p: float):
        """Safely updates both 1D and 2D dropout layers."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.p = p

    def forward(self, x):
        # Encoder passes with pooling
        x = self.pool(self.encoder1(x))
        x = self.pool(self.encoder2(x))
        x = self.pool(self.encoder3(x))
        x = self.pool(self.encoder4(x))

        # Bottleneck/Intermediate processing
        x = self.intermediate(x)

        # Flatten and Classify
        x = self.flatten(x)
        x = self.fc(x)
        return x



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
    # Ensure we capture the ACTUAL state of the layers after pruning
    metadata = {
        'epoch': epoch,
        'base_channels': getattr(model, 'base', 64),
        'num_classes': getattr(model, 'num_classes', 3),
        'input_size': getattr(model, 'input_size', 128),
        'channel_config': get_channel_config(model),  # Dynamically reads m.out_channels
        'pruning_stage': extra_meta.get('pruning_stage', 0) if extra_meta else 0,
    }

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'metadata': metadata
    }
    torch.save(checkpoint, path)
    print(f"Saved: {path} (Stage: {metadata['pruning_stage']})")


def load_compressed_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt['metadata']
    state_dict = ckpt['state_dict']
    channel_cfg = meta['channel_config']

    model = HandmadeGlaucomaClassifier(
        input_size=meta['input_size'],
        num_classes=meta['num_classes'],
        base=meta['base_channels']
    ).to(device)

    with torch.no_grad():
        last_out_channels = 3

        for name, m in model.named_modules():
            # Only resize layers that possess weight attributes
            if isinstance(m, nn.Conv2d):
                target_out = channel_cfg.get(name, m.out_channels)
                m.weight.data = m.weight.data[:, :last_out_channels]
                m.weight.data = m.weight.data[:target_out]
                if m.bias is not None:
                    m.bias.data = m.bias.data[:target_out]

                m.in_channels = last_out_channels
                m.out_channels = target_out
                last_out_channels = target_out

            elif isinstance(m, nn.Linear):
                weight_key = f"{name}.weight"
                if weight_key in state_dict:
                    # torch_pruning handles Linear input features in the state_dict
                    stored_in_features = state_dict[weight_key].shape[1]
                    if m.in_features != stored_in_features:
                        m.weight.data = m.weight.data[:, :stored_in_features]
                        m.in_features = stored_in_features

    model.load_state_dict(state_dict, strict=True)
    return model

def create_pretrained_model(num_classes=3, input_size=128):
    vgg = vgg16(weights=VGG16_Weights.DEFAULT)
    vgg_feats = list(vgg.features.children())
    model = HandmadeGlaucomaClassifier(num_classes=num_classes, input_size=input_size)

    vgg_layers = [model.encoder1, model.encoder2, model.encoder3, model.encoder4]
    vgg_ptr = 0
    for encoder in vgg_layers:
        for layer in encoder:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.copy_(vgg_feats[vgg_ptr].weight.data)
                layer.bias.data.copy_(vgg_feats[vgg_ptr].bias.data)
                vgg_ptr += 1
            # Skip LeakyReLU and Dropout2d during weight assignment
            elif isinstance(layer, (nn.LeakyReLU, nn.Dropout2d)):
                if isinstance(layer, nn.LeakyReLU):
                    vgg_ptr += 1 # VGG has a ReLU here
        vgg_ptr += 1 # Skip VGG MaxPool
    return model
