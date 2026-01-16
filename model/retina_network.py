import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from torch.nn.utils.parametrizations import spectral_norm


def gaussian_2d(shape, center=None, sigma=1, min_value=0.0, max_value=1.0):
    if center is None:
        center = [shape[0] // 2 - 1, shape[1] // 2 - 1]  # Center of the array

    x = torch.arange(shape[0]).float()
    y = torch.arange(shape[1]).float()
    x, y = torch.meshgrid(x, y, indexing='ij')

    # Calculate distance from the center
    dist = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Calculate the Gaussian distribution
    gaussian = torch.exp(-dist ** 2 / (2 * sigma ** 2))

    upper_left = gaussian[:(center[0] + 1), :(center[1] + 1)]
    upper_right = torch.flip(upper_left, dims=[1])
    lower_left = torch.flip(upper_left, dims=[0])
    lower_right = torch.flip(lower_left, dims=[1])

    gauss_max = upper_left.max()
    gauss_min = upper_left.min()

    gaussian[:(center[0] + 1), -(center[1] + 1):] = upper_right
    gaussian[-(center[0] + 1):, :(center[1] + 1)] = lower_left
    gaussian[-(center[0] + 1):, -(center[1] + 1):] = lower_right

    gauss_multiplier = (max_value - min_value) / (gauss_max - gauss_min)
    gaussian = min_value + (gaussian - gauss_min) * gauss_multiplier

    return gaussian


# ----------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive, weight_negative):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative

    def forward(self, input_logits, target):
        # Calculate binary cross-entropy loss
        loss = - (self.weight_positive * target * torch.log(input_logits + 1e-8) +
                  self.weight_negative * (1 - target) * torch.log(1 - input_logits + 1e-8))

        # Compute the mean loss
        loss = torch.mean(loss)

        return loss


class WeightedBCELoss2(nn.Module):
    def __init__(self, weight_positive, weight_negative,
                 gauss_params=None,
                 blur_use=False,
                 blur_size=9):
        super(WeightedBCELoss2, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative

        self.use_blur = blur_use
        if self.use_blur is True:
            self.blur_size = blur_size
            self.blur_radius = int((blur_size - 1) / 2)
            self.blur_weight = torch.ones(1, 1, self.blur_size, self.blur_size) / (self.blur_size * self.blur_size)

        self.use_gauss = False
        if gauss_params is not None:
            self.use_gauss = True
            self.gauss = gaussian_2d([gauss_params[0], gauss_params[1]],
                                     min_value=gauss_params[2],
                                     max_value=gauss_params[3],
                                     sigma=gauss_params[4]).unsqueeze(0)

    def forward(self, input_logits, target):

        # Calculate binary cross-entropy loss
        loss = - (self.weight_positive * target * torch.log(input_logits + 1e-8) +
                  self.weight_negative * (1 - target) * torch.log(1 - input_logits + 1e-8))

        if self.use_blur is True:
            blur_pass = (1 + F.conv2d(target,
                                      weight=self.blur_weight,
                                      padding=self.blur_radius) *
                         ((target < 0.5).float()))
            loss = loss * blur_pass

        if self.use_gauss is True:
            loss = self.gauss * loss

        # Compute the mean loss
        loss = torch.mean(loss)

        return loss


class WeightedBCEPenalizeCornersLoss(nn.Module):
    def __init__(self, weight_positive, weight_negative, weight_corner_loss=20.0):
        super(WeightedBCEPenalizeCornersLoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.weight_corner_loss = weight_corner_loss

    def forward(self, input_logits, target):
        mask = torch.ones_like(target)

        mask[:, :, :, :1] = self.weight_corner_loss  # Top-left corner
        mask[:, :, :, -1:] = self.weight_corner_loss  # Top-right corner
        mask[:, :, -1:, :] = self.weight_corner_loss  # Bottom-left corner
        mask[:, :, 0:1, :] = self.weight_corner_loss  # Bottom-right corner

        # Calculate binary cross-entropy loss
        loss = - (self.weight_positive * target * mask * torch.log(input_logits + 1e-8) +

                  self.weight_negative * (1 - target) * torch.log(1 - input_logits + 1e-8))

        # Apply the mask to the standard loss
        # weighted_loss = loss * mask

        # Compute the mean loss
        loss = torch.mean(loss)

        return loss


class CustomActivation(nn.Module):
    def __init__(self, threshold=0.1):
        super(CustomActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp_min(x, self.threshold)


class CustomMultiply(nn.Module):
    def __init__(self, threshold):
        super(CustomMultiply, self).__init__()
        # self.theshold = threshold

    def forward(self, x):
        1 / (1 + torch.exp(-10 * (x - 0.5)))


class RetinalConvolutionNetwork(nn.Module):
    def __init__(self, cnum_in=1, cnum_out=1):
        super(RetinalConvolutionNetwork, self).__init__()

        self.conv = nn.ModuleList()
        self.activation = nn.ModuleList()

        # Initial 12 Convolutional Layers
        self.conv.append(nn.Conv2d(in_channels=cnum_in, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.activation.append(nn.LeakyReLU(inplace=True))

        for i in range(1, 12):
            self.conv.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.activation.append(nn.LeakyReLU(inplace=True))

        # Intermediate Convolutional Layers
        self.conv.append(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1))
        self.activation.append(nn.LeakyReLU(inplace=True))

        for i in range(14, 16):
            self.conv.append(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1))
            self.activation.append(nn.LeakyReLU(inplace=True))

        # Final Convolutional Layer
        self.final_conv = nn.Conv2d(in_channels=8, out_channels=cnum_out, kernel_size=3, stride=1, padding=1)

        # Sigmoid activation for binary classification
        self.final_activation = nn.Sigmoid()

    def forward(self, x):

        for i in range(len(self.activation)):
            x = self.activation[i](self.conv[i](x))

        x = self.final_activation(self.final_conv(x))

        return x
