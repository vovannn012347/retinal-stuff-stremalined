import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune
from torchvision.models import VGG16_Weights


class FourMetrics(nn.Module):
    def __init__(self, threshold=0.75, correcteness_index=-1):
        super(FourMetrics, self).__init__()
        self.threshold = threshold
        self.correcteness_index = correcteness_index

    '''@staticmethod
    def relative_accuracy(pred, target):
        return 1 - torch.abs(pred - target)  # Relative score [0, 1]'''

    def forward(self, input_sigmoid, target):
        pred = torch.clamp(input_sigmoid, 0, 1)

        pred_binary = (pred >= self.threshold).float()
        target_binary = (target >= self.threshold).float()  # target is already 0 or 1

        # True positives: both prediction and target are 1
        tp = (pred_binary * target_binary).sum().item()

        # False positives: prediction is 1, but target is 0
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()

        # False negatives: prediction is 0, but target is 1
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()

        # True negatives: both prediction and target are 0
        tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()

        return tp, fp, fn, tn


class IoUWeightedMetrics(nn.Module):
    def __init__(self, weight_positive, weight_negative, reduction='mean'):
        super(IoUWeightedMetrics, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.reduction = reduction
        # self.bce_weight = bce_weight

    def forward(self, input_sigmoid, target):

        predicted = torch.clamp(input_sigmoid, 0, 1)
        weights = torch.where(target >= 0.95, self.weight_positive, self.weight_negative)

        intersection = torch.clamp(predicted * target, 0, 1)
        union = torch.clamp(predicted + target, 0, 1)

        intersection_weighted = (intersection * weights).sum(dim=(2, 3), keepdim=True)
        predicted_weighted = (predicted * weights).sum(dim=(2, 3), keepdim=True)
        target_weighted = (target * weights).sum(dim=(2, 3), keepdim=True)
        union_weighted = (union * weights).sum(dim=(2, 3), keepdim=True)

        # Compute IoU for each image in the batch
        iou = (intersection_weighted + 1e-8) / (union_weighted + 1e-8)
        precision = (intersection_weighted + 1e-8) / (predicted_weighted + 1e-8)
        recall = (intersection_weighted + 1e-8) / (target_weighted + 1e-8)
        f1_score = (2 * precision * recall + 1e-8) / (precision + recall + 1e-8)

        true_negative = (1 - predicted) * (1 - target) * weights
        tn_weighted = true_negative.sum(dim=(2, 3), keepdim=True)
        total_weighted = weights.sum(dim=(2, 3), keepdim=True)
        accuracy = (intersection_weighted + tn_weighted + 1e-8) / (total_weighted + 1e-8)

        # Apply reduction
        if self.reduction == 'mean':
            metrics = {
                "IoU": iou.mean().item(),
                "Accuracy": accuracy.mean().item(),
                "Precision": precision.mean().item(),
                "Recall": recall.mean().item(),
                "F1-Score": f1_score.mean().item()
            }
        elif self.reduction == 'sum':
            metrics = {
                "IoU": iou.sum().item(),
                "Accuracy": accuracy.sum().item(),
                "Precision": precision.sum().item(),
                "Recall": recall.sum().item(),
                "F1-Score": f1_score.sum().item()
            }
        elif self.reduction == 'none':
            metrics = {
                "IoU": iou,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score
            }
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")

        return metrics


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


# old definitor
class FcnskipNerveDefinitor(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=8):
        super(FcnskipNerveDefinitor, self).__init__()

        # Encoder: Convolution + Max Pooling layers
        self.conv1 = self._conv_block_1(input_channels, base)
        self.conv2 = self._conv_block_1(base, base * 2)
        self.conv3 = self._conv_block_1(base * 2, base * 4)
        # self.conv4 = self._conv_block_1(base * 4, base * 8)

        # Pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.intermidiate = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.activation = nn.LeakyReLU(inplace=True)

        # Decoder: Transpose Convolutions for Upsampling
        self.decoder1 = self._upconv_block(base * 4, base * 2)
        self.decoder2 = self._upconv_block(base * 2 + base * 4, base * 2)
        self.decoder3 = self._upconv_block(base * 2 + base * 2, base)

        # Final convolution for classification
        self.final_conv = nn.Conv2d(base, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_activation = nn.Sigmoid()  # For binary segmentation, use Sigmoid

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # [B, 32, 576, 576]
        x1_pooled = self.pool(x1)  # [B, 32, 288, 288]

        x2 = self.conv2(x1_pooled)  # [B, 64, 288, 288]
        x2_pooled = self.pool(x2)  # [B, 64, 144, 144]

        x3 = self.conv3(x2_pooled)  # [B, 256, 144, 144]
        x3_pooled = self.pool(x3)  # [B, 256, 72, 72]

        '''x4 = self.conv4(x3_pooled)  # [B, 512, 72, 72]
        x4_pooled = self.pool(x4)  # [B, 512, 36, 36]'''

        x = self.intermidiate(x3_pooled)
        x = self.upsample(x)

        # Decoder (Upsampling)
        x = self.decoder1(x)  # [B, 256, 72, 72]
        x = self.decoder2(torch.cat([x, x3_pooled], dim=1))  # Add skip connection from x3  [B, 128, 144, 144]
        x = self.decoder3(torch.cat([x, x2_pooled], dim=1))  # Add skip connection from x2  [B, 64, 288, 288]

        # Final 1x1 Convolution to reduce to num_classes output
        x = self.final_conv(x)  # [B, num_classes, 576, 576]
        # Final upsample to the original size
        return self.final_activation(x)

    @staticmethod
    def _conv_block_1(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def _upconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )


def prune_layer_weights(layer, pruning_ratio=0.5):
    """
    Prune weights of the layer, remove pruned biases, and reorder weights and biases.
    This function will also reorder the input weights of the next layer accordingly.

    Args:
        layer (nn.Module): The layer to be pruned.
        pruning_ratio (float): The percentage of weights to prune (0 to 1).

    Returns:
        mask (Tensor): The mask of pruned weights (non-zero weights remain).
    """
    # Step 1: Prune weights of the current layer (prune weights)
    # n=0 - prune across output
    # n=1 - prune across input
    prune.ln_structured(layer, name="weight", n=0, amount=pruning_ratio, dim=0)  # Prune based on output channels

    # Step 2: Get the mask of pruned weights
    # mask = layer.weight_mask  # This mask indicates which weights are zeroed out
    mask = layer.weight_mask.clone().sum(dim=(1, 2, 3)).bool()
    prune.remove(layer, "weight")

    # Step 4: Reorder weights and biases by moving non-zero weights to the beginning
    # Reorder the weights (flatten, compress, then reshape)
    non_zero_weights = layer.weight.data[mask.bool(), :, :, :]

    # Reshape and store the pruned weight
    layer.weight.data = non_zero_weights

    # Step 3: Remove corresponding biases
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[mask]

    return mask  # Returning the mask for later use


def prune_only_input_weights(next_layer, input_mask):
    input_mask = input_mask.bool()
    old_weights = next_layer.weight.data  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

    next_layer.weight.data = old_weights[:, input_mask, :, :]


def adjust_next_layer_input_weights(next_layer, mask):
    """
    Adjust the input weights of the next layer based on the pruning of the previous layer.

    Args:
        next_layer (nn.Module): The next layer to adjust input weights for.
        mask (Tensor): Mask of pruned weights (non-zero elements).
    """
    # Reorder input weights of the next layer to match the pruned outputs of the previous layer
    next_layer.weight.data = next_layer.weight.data[:, mask, :, :]  # Adjust next layer's input weights


'''def prune_conv(conv, pruning_ratio=0.5):
    """
    Prunes weights of current layer, returns prunned mask
    Also adjusts the input weights of the next layer accordingly.
    """

    prune.ln_structured(conv, name='weight', amount=pruning_ratio, dim=0)
    weight_mask = conv.weight_mask # Shape: (out_channels, in_channels, kernel_h, kernel_w)

    # Summing along all dimensions except the output channel (dim=0)
    bias_mask = weight_mask.sum(dim=(1, 2, 3)) > 0  # Shape: (out_channels,)
    # Apply the mask to the bias
    non_zero_indices = weight_mask.view(weight_mask.shape[0], -1).sum(dim=1) > 0

    return non_zero_indices

def remap_conv(conv1, non_zero_indices, conv2):

    compressed_weights = conv1.weight.data[non_zero_indices, :, :, :]
    compressed_biases = conv1.bias.data[non_zero_indices]
    conv1.weight = torch.nn.Parameter(compressed_weights)
    conv1.bias = torch.nn.Parameter(compressed_biases)

    if conv2:
        conv2.weight.data = conv2.weight.data[:, non_zero_indices, :, :]'''


def load_state_dict_custom(adjusted_model, pruned_state_dict):
    def replace_layer(model, layer_name, new_layer):
        """Recursively replace a layer in the model."""
        components = layer_name.split(".")
        parent = model
        for component in components[:-1]:
            parent = getattr(parent, component)
        if isinstance(parent, nn.Sequential):  # Handle Sequential container
            idx = int(components[-1])
            parent[idx] = new_layer
        else:
            setattr(parent, components[-1], new_layer)

    layers_to_update = []

    children = adjusted_model.named_modules()
    for name, layer in children:
        if f"{name}.weight" in pruned_state_dict:  # Check if the layer is in the pruned state dict
            if isinstance(layer, nn.Conv2d):
                # Get pruned weight and bias
                pruned_weight = pruned_state_dict[f"{name}.weight"]
                pruned_bias = pruned_state_dict[f"{name}.bias"]

                # Replace the layer with a new one matching pruned dimensions
                out_channels = pruned_weight.shape[0]
                in_channels = pruned_weight.shape[1]
                kernel_size = pruned_weight.shape[2:]
                padding = layer.padding
                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

                # Manually load the pruned weights and biases into the new layer
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(pruned_weight)
                    new_layer.bias = nn.Parameter(pruned_bias)

                # Update the model's layer
                layers_to_update.append((name, new_layer))
                #setattr(adjusted_model, name, new_layer)

            elif isinstance(layer, nn.Linear):
                # Get pruned weight and bias
                pruned_weight = pruned_state_dict[f"{name}.weight"]
                pruned_bias = pruned_state_dict[f"{name}.bias"]

                # Replace the layer with a new one matching pruned dimensions
                out_features = pruned_weight.shape[0]
                in_features = pruned_weight.shape[1]
                new_layer = nn.Linear(in_features, out_features)

                # Manually load the pruned weights and biases into the new layer
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(pruned_weight)
                    new_layer.bias = nn.Parameter(pruned_bias)

                # Update the model's layer
                layers_to_update.append((name, new_layer))
                #setattr(adjusted_model, name, new_layer)

            elif isinstance(layer, nn.ConvTranspose2d):
                # Get pruned weight and bias
                pruned_weight = pruned_state_dict[f"{name}.weight"]
                pruned_bias = pruned_state_dict[f"{name}.bias"]

                # Replace the layer with a new one matching pruned dimensions
                out_features = pruned_weight.shape[0]
                in_features = pruned_weight.shape[1]
                kernel_size = pruned_weight.shape[2:]
                stride = layer.stride
                new_layer = nn.ConvTranspose2d(in_features, out_features, kernel_size, stride=stride)

                # Manually load the pruned weights and biases into the new layer
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(pruned_weight)
                    new_layer.bias = nn.Parameter(pruned_bias)

                # Update the model's layer
                layers_to_update.append((name, new_layer))
                #setattr(adjusted_model, name, new_layer)

    for name, layer in layers_to_update:
        replace_layer(adjusted_model, name, layer)
        # setattr(adjusted_model, name, new_layer)


class FcnskipNerveDefinitor2(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=64,
                 dropout_probability=0.3,
                 use_dropout=False,
                 use_quantize=False,
                 use_inplace=True):
        super(FcnskipNerveDefinitor2, self).__init__()

        self.use_quantize = use_quantize

        if use_dropout:
            self.step = 3
        else:
            self.step = 2

        if use_quantize:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        self.encoder1 = self._conv_block_2(input_channels, base,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder2 = self._conv_block_2(base, base * 2,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder3 = self._conv_block_3(base * 2, base * 4,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder4 = self._conv_block_3(base * 4, base * 8,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)

        if use_dropout:
            self.intermidiate = nn.Sequential(
                nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=use_inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            self.intermidiate = nn.Sequential(
                nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=use_inplace)
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.decoder4 = self._upconv_block(base * 8 + base * 4, base * 4,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.decoder3 = self._upconv_block(base * 4 + base * 2, base * 2,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.decoder2 = self._upconv_block(base * 2 + base, base,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)

        self.decoder_1_final_conv = nn.Sequential(
            nn.Conv2d(base, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)



    @staticmethod
    def _conv_block_2(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace)
            )

    @staticmethod
    def _conv_block_3(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace)
            )

    @staticmethod
    def _upconv_block(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )

    def forward(self, x):

        if self.use_quantize:
            x = self.quant(x)

        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        interm = self.intermidiate(pool4)
        # Decoder
        up4 = self.upsample(interm)

        dec4 = self.decoder4(torch.cat([up4, pool3], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, pool2], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, pool1], dim=1))

        ret = self.decoder_1_final_conv(dec2)

        if self.use_quantize:
            ret = self.dequant(ret)

        return ret

    def prune_conv_block_2(self, encoder, pruning_ratio=0.5):
        i = 0
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
        adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
        i += self.step
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)

        return remap_mask

    def prune_conv_block_3(self, encoder, pruning_ratio=0.5):
        i = 0
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
        adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
        i += self.step
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
        adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)

        i += self.step
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)

        return remap_mask

    def prune_upconv_block(self, encoder, pruning_ratio=0.5):
        i = 0
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
        adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
        i += self.step
        remap_mask = prune_layer_weights(encoder[i], pruning_ratio)

        return remap_mask

    def add_dropout(self, dropout_prob=0.3):
        def dropout_block(seq_model_orig):
            new_layers = []
            for layer in seq_model_orig.children():
                new_layers.append(layer)  # Add existing layer
                if isinstance(layer, nn.LeakyReLU):  # Add dropout after every ReLU
                    new_layers.append(nn.Dropout(p=dropout_prob))

            return nn.Sequential(*new_layers)

        self.step = 3

        self.encoder1 = dropout_block(self.encoder1)
        self.encoder2 = dropout_block(self.encoder2)
        self.encoder3 = dropout_block(self.encoder3)
        self.encoder4 = dropout_block(self.encoder4)

        self.intermidiate = dropout_block(self.intermidiate)

        self.decoder4 = dropout_block(self.decoder4)
        self.decoder3 = dropout_block(self.decoder3)
        self.decoder2 = dropout_block(self.decoder2)

    def prune_layers(self, pruning_ratio=0.5):

        enc1_mask = self.prune_conv_block_2(self.encoder1, pruning_ratio)
        adjust_next_layer_input_weights(self.encoder2[0], enc1_mask)

        enc2_mask = self.prune_conv_block_2(self.encoder2, pruning_ratio)
        adjust_next_layer_input_weights(self.encoder3[0], enc2_mask)

        enc3_mask = self.prune_conv_block_3(self.encoder3, pruning_ratio)
        adjust_next_layer_input_weights(self.encoder4[0], enc3_mask)

        enc4_mask = self.prune_conv_block_3(self.encoder4, pruning_ratio)
        adjust_next_layer_input_weights(self.intermidiate[0], enc4_mask)

        intermidiate_mask = prune_layer_weights(self.intermidiate[0], pruning_ratio)
        adjust_next_layer_input_weights(self.decoder4[0], torch.cat([intermidiate_mask, enc3_mask]))

        dec4_mask = self.prune_upconv_block(self.decoder4, pruning_ratio)
        adjust_next_layer_input_weights(self.decoder3[0], torch.cat([dec4_mask, enc2_mask]))

        dec3_mask = self.prune_upconv_block(self.decoder3, pruning_ratio)
        adjust_next_layer_input_weights(self.decoder2[0], torch.cat([dec3_mask, enc1_mask]))

        dec2_mask = self.prune_upconv_block(self.decoder2, pruning_ratio)

        prune_only_input_weights(self.decoder_1_final_conv[0], dec2_mask)

    # this is used in conjunction with quantization chnages from outside the model
    def add_quantize(self):
        self.use_quantize = True
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    @staticmethod
    def create_model(num_classes=1,
                     input_channels=3,
                     use_dropout=False,
                     dropout_probability=0.3,
                     in_place=True):
        # Load VGG-16 pretrained model
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg_features = list(vgg16.features.children())  # Extract VGG-16 feature blocks

        # Create the custom model
        model = FcnskipNerveDefinitor2(num_classes=num_classes,
                                       input_channels=input_channels,
                                       use_inplace=in_place)

        # Map pretrained weights from VGG-16 to the custom model
        vgg_layers = [
            model.encoder1,  # Map first two VGG-16 blocks to encoder1
            model.encoder2,  # Map next two VGG-16 blocks to encoder2
            model.encoder3,  # Map next three VGG-16 blocks to encoder3
            model.encoder4  # Map next three VGG-16 blocks to encoder4
        ]

        start_idx = 0
        for i, encoder in enumerate(vgg_layers):
            num_layers = len(list(encoder.children()))
            for j in range(num_layers):
                if isinstance(encoder[j], nn.Conv2d):
                    encoder[j].weight.data = vgg_features[start_idx].weight.data.clone()
                    encoder[j].bias.data = vgg_features[start_idx].bias.data.clone()
                start_idx += 1
            # include pooling - it is not included in encoder layer for skip connections
            start_idx += 1

        if use_dropout:
            model.add_dropout(dropout_prob=dropout_probability)

        # Return the fully initialized model
        return model


class HandmadeGlaucomaClassifier(nn.Module):
    def __init__(self,
                 num_classes=3,
                 input_channels=3,
                 base=64,
                 input_size=128):
        super(HandmadeGlaucomaClassifier, self).__init__()

        # Load pre-trained VGG16
        # vgg = models.vgg16(pretrained=True)

        # Transfer weights from VGG16 for the first 4 blocks
        self.encoder1 = self._conv_block_2(input_channels, base)
        self.encoder2 = self._conv_block_2(base, base * 2)
        self.encoder3 = self._conv_block_3(base * 2, base * 4)
        self.encoder4 = self._conv_block_3(base * 4, base * 8)

        # Intermediate layer
        self.intermidiate = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Fully connected layers for 3 output channels
        self.flatten = nn.Flatten()

        lin1 = nn.Linear(base * 8 * int((input_size / 16) ** 2), base)
        lin2 = nn.Linear(base, num_classes)

        self.fc = nn.Sequential(
            lin1,  # nn.Linear(base*8 * 7 * 7, 256),  # Adjust based on output of encoder4
            nn.ReLU(),
            lin2,  # nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @staticmethod
    def _conv_block_2(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _conv_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _extract_vgg_weights(vgg, start, end):
        # Extracts a subset of VGG layers' weights as a state_dict
        sublayers = list(vgg.features.children())[start:end]
        state_dict = {f"{i}": layer.state_dict() for i, layer in enumerate(sublayers)}
        return {k: v for d in state_dict.values() for k, v in d.items()}

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        # Intermediate processing
        interm = self.intermidiate(pool4)

        # Flatten and FC output
        flattened = self.flatten(interm)
        output = self.fc(flattened)

        return output

    @staticmethod
    def create_model(num_classes=1, input_channels=3, input_size=288):
        # Load VGG-16 pretrained model
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg_features = list(vgg16.features.children())  # Extract VGG-16 feature blocks

        # Create the custom model
        model = HandmadeGlaucomaClassifier(num_classes=num_classes, input_channels=input_channels,
                                           input_size=input_size)

        # Map pretrained weights from VGG-16 to the custom model
        vgg_layers = [
            model.encoder1,  # Map first two VGG-16 blocks to encoder1
            model.encoder2,  # Map next two VGG-16 blocks to encoder2
            model.encoder3,  # Map next three VGG-16 blocks to encoder3
            model.encoder4  # Map next three VGG-16 blocks to encoder4
        ]

        start_idx = 0
        for i, encoder in enumerate(vgg_layers):
            num_layers = len(list(encoder.children()))
            for j in range(num_layers):
                if isinstance(encoder[j], nn.Conv2d):
                    encoder[j].weight.data = vgg_features[start_idx].weight.data.clone()
                    encoder[j].bias.data = vgg_features[start_idx].bias.data.clone()
                start_idx += 1
            # include pooling - it is not included in encoder layer for skip connections
            start_idx += 1

        # Return the fully initialized model
        return model
