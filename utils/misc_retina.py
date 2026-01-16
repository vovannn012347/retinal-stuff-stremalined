import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnnfunc
import torch.quantization as quant
import torchvision.transforms as tvtf
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import os


class DictConfig(object):
    """Creates a Config object from a dict
       such that object attributes correspond to dict keys.
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


def get_config(fname):
    print("Working directory:", os.getcwd())
    print("Given fname:", fname)

    resolved_path = os.path.abspath(fname)
    print("Resolved path:", resolved_path)

    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)


def save_nerve_definitor(fname, definitor, optimizer_pass, n_iter, config):
    state_dicts = {'nerve_definitor': definitor.state_dict(),
                   'adam_opt_nerve_definitor': optimizer_pass.state_dict(),
                   'n_iter': n_iter,
                   'quantized': definitor.quantized,
                   'n_learn_rate': optimizer_pass.param_groups[0]['lr']}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_2_convolution_states(fname, convolution_pass1, convolution_pass2, optimizer_pass1, optimizer_pass2, n_iter,
                              config):
    state_dicts = {'convolution_pass1': convolution_pass1.state_dict(),
                   'convolution_pass2': convolution_pass2.state_dict(),
                   'adam_opt_pass1': optimizer_pass1.state_dict(),
                   'adam_opt_pass2': optimizer_pass2.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_2_convolution_states_2(fname, convolution_pass, optimizer_pass, n_iter, config):
    state_dicts = {'convolution_pass': convolution_pass.state_dict(),
                   'adam_opt_pass': optimizer_pass.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def save_nerve_classifier(fname, nerve_classifier_pass, optimizer, n_iter, config):
    state_dicts = {'nerve_classifier': nerve_classifier_pass.state_dict(),
                   'adam_opt_nerve_classifier': optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


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


# requires image tensor with [-1, 1] values
def run_retina_cnn_2(image_tensor, retina_pass1, image_patch_size, image_patch_stride,
                     device=None):
    image_patches_unfolded_pass1 = image_tensor.unfold(1, image_patch_size[0], image_patch_stride[0])
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.unfold(2, image_patch_size[1], image_patch_stride[1])
    _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass1.size()
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.contiguous()
    image_patches_unfolded_pass1 = (image_patches_unfolded_pass1
                                    .view(image_tensor.size(0), -1,
                                          image_patch_size[0],
                                          image_patch_size[1]))
    image_patches_unfolded_pass1 = image_patches_unfolded_pass1.permute(1, 0, 2, 3)

    output_pass1 = retina_pass1(image_patches_unfolded_pass1)

    channels, h, w = image_tensor.shape
    image_stride_h = image_patch_stride[0]
    image_stride_w = image_patch_stride[1]

    output_mask = output_pass1.permute(1, 0, 2, 3)
    reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                 image_patch_size[0], image_patch_size[1])

    merged_tensor = torch.zeros([1, h, w]).to(device)
    gauss_count = torch.zeros_like(merged_tensor)
    gauss_patch = gaussian_2d([image_patch_size[0], image_patch_size[1]],
                              min_value=0.5, max_value=1.6, sigma=20)

    for ch in range(reshaped_refolded_patches.size(0)):
        for i in range(reshaped_refolded_patches.size(1)):
            for j in range(reshaped_refolded_patches.size(2)):
                x_start, y_start = i * image_stride_h, j * image_stride_w
                gauss_count[ch,
                            x_start:x_start + image_patch_size[0],
                            y_start:y_start + image_patch_size[1]] += gauss_patch

    for ch in range(reshaped_refolded_patches.size(0)):
        for i in range(reshaped_refolded_patches.size(1)):
            for j in range(reshaped_refolded_patches.size(2)):
                x_start, y_start = i * image_stride_h, j * image_stride_w

                # select all values
                # temp_mask = reshaped_refolded_patches[ch, i, j] > 0.05

                merged_tensor[ch,
                              x_start:x_start + image_patch_size[0],
                              y_start:y_start + image_patch_size[1]] += (
                        reshaped_refolded_patches[ch, i, j] * gauss_patch)
                # (reshaped_refolded_patches[ch, i, j] * temp_mask * gauss_patch)

    merged_tensor /= gauss_count
    merged_tensor = torch.clamp_max(merged_tensor, 1)

    merged_tensor = magic_wand_mask_selection(merged_tensor).to(torch.float32)

    """target_color = 0
    tolerance = 0.23

    # binary_mask = abs(merged_tensor - target_color) < tolerance
    merged_tensor = (abs(merged_tensor - target_color) > tolerance).to(torch.float32)"""

    return merged_tensor


def magic_wand_mask_selection_batch(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):  # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): batch of 1 channel tensors that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """

    output_mask = torch.zeros_like(image_tensor, dtype=torch.bool)
    for i in range(output_mask.size(0)):
        output_mask[i] = magic_wand_mask_selection(image_tensor[i], upper_multiplier, lower_multipleir)

    return output_mask


def magic_wand_mask_selection(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    # part 1: get above zero pixel values
    flat_image = image_tensor.flatten()

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()
    histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

    # part 2: get starting tolerance and starting pixel value
    bin_width = (max_pixel - min_pixel) / bin_count
    non_zero_indices = torch.nonzero(histogram, as_tuple=False)

    first_tolerance = upper_multiplier

    upper_bound_bin_index1 = non_zero_indices[-1].item()
    lower_bound_bin_index1 = int(upper_bound_bin_index1 * (1 - first_tolerance))

    # upper_bound = (upper_bound_bin_index + 1) * bin_width
    lower_bound = lower_bound_bin_index1 * bin_width

    # part 3: make starting global selection
    first_selection = image_tensor > lower_bound

    # result_image = first_selection.to(torch.float32)
    # result_image = TF.to_pil_image(result_image.squeeze().cpu(), mode="L")
    # result_image.save(f"{debug_dir}/test1.png")

    # part 4: replace selected pixel values with the lowest value from selected pixels
    image_tensor_editable = torch.clone(image_tensor)
    image_tensor_editable[first_selection] = lower_bound

    # result_image = TF.to_pil_image(image_tensor_editable.squeeze().cpu(), mode="L")
    # result_image.save(f"{debug_dir}/test2.png")

    # part 5: get second tolerance value
    # from the lowest color value in part 2 to total lowest value that is above 0 color
    lower_bound_bin_index2 = int(lower_bound_bin_index1 * lower_multipleir)
    if lower_bound_bin_index2 < 3:
        lower_bound_bin_index2 = 3

    lower_bound2 = lower_bound_bin_index2 * bin_width

    # part 6: make second selection starting from first selection locations using second tolerance
    output_mask = torch.zeros_like(first_selection, dtype=torch.bool)
    for i in range(first_selection.size(1)):
        for j in range(first_selection.size(2)):

            if first_selection[0, i, j].item() is True and output_mask[0, i, j].item() is False:
                to_test = [(i, j)]

                while to_test:
                    x, y = to_test.pop()

                    # Check if the pixel is already in the mask
                    if output_mask[0, x, y].item():
                        continue

                    # Get the value of the current pixel
                    pixel_value = image_tensor_editable[0, x, y]

                    # If the pixel value is within the tolerance range, include it in the mask
                    if lower_bound2 <= pixel_value:  # <= upper_bound2:
                        # actually we are not intrested in upper bound as we go top-down
                        output_mask[0, x, y] = True

                        # Explore neighboring pixels (4-connectivity: top, bottom, left, right)
                        if (x > 0
                                and output_mask[0, x - 1, y].item() is False):
                            to_test.append((x - 1, y))  # Top neighbor
                        if (x < image_tensor.shape[1] - 1
                                and output_mask[0, x + 1, y].item() is False):
                            to_test.append((x + 1, y))  # Bottom neighbor
                        if (y > 0
                                and output_mask[0, x, y - 1].item() is False):
                            to_test.append((x, y - 1))  # Left neighbor
                        if (y < image_tensor.shape[2] - 1
                                and output_mask[0, x, y + 1].item() is False):
                            to_test.append((x, y + 1))  # Right neighbor

    # for i, count in enumerate(histogram):
    #     lower_bound = min_pixel + i * bin_width
    #     upper_bound = lower_bound + bin_width
    #     print(f"Bin {i + 1}: Range [{lower_bound:.2f}, {upper_bound:.2f}], Count: {int(count)}")

    return output_mask


def magic_wand_mask_selection_batch_faster(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    if image_tensor.dim() != 4 or image_tensor.size(1) != 1:  # RGB
        raise Exception("invalid image_tensor dimensions")

    bounds = []
    masks = []
    image_tensor_wand = torch.clone(image_tensor)

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()

    for image_i in range(image_tensor.size(0)):
        # part 1: get above zero pixel values
        flat_image = image_tensor[image_i].flatten()

        histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

        # part 2: get starting tolerance and starting pixel value
        bin_width = (max_pixel - min_pixel) / bin_count
        non_zero_indices = torch.nonzero(histogram, as_tuple=False)

        first_tolerance = upper_multiplier

        first_bound_bin_index = int(non_zero_indices[-1].item() * (1 - first_tolerance))
        first_bound = first_bound_bin_index * bin_width

        # part 3: make starting global selection

        mask = image_tensor[image_i] > first_bound
        masks.append(mask)

        # part 4: replace selected pixel values with the lowest value from selected pixels
        image_tensor_wand[image_i][mask] = first_bound

        # part 5: get second tolerance value
        # from the lowest color value in part 2 to total lowest value that is above 0 color
        second_bound_bin_index = int(first_bound_bin_index * lower_multipleir)
        if second_bound_bin_index < 3:
            second_bound_bin_index = 3
            if second_bound_bin_index >= first_bound_bin_index:
                second_bound_bin_index = first_bound_bin_index - 1

        if second_bound_bin_index < 0:
            second_bound_bin_index = 0

        second_bound = second_bound_bin_index * bin_width
        bounds.append(second_bound)

    bounds = torch.tensor(bounds).view(-1, 1, 1, 1)

    masks = torch.stack(masks)
    diff_map = image_tensor_wand >= bounds

    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32,
                          device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    max_iters = int((masks.size(2) * masks.size(3)) / 4)

    for _ in range(max_iters):
        dilated_mask = tnnfunc.conv2d(masks.float(), kernel, padding=1).bool()

        # Mask update: keep pixels within threshold and add to current mask
        new_masks = dilated_mask & diff_map

        # Stop if no new pixels are added
        if torch.equal(new_masks, masks):
            break

        # Update mask with new selection
        masks = new_masks

    return masks


def magic_wand_mask_selection_faster(image_tensor, upper_multiplier=0.4, lower_multipleir=0.25):
    # , debug_dir):
    """ selects retilnal blood vessels from mask via magic wand

        Args:
            image_tensor (Tensor): 1 channel tensor that contains greyscale values for mask.
            upper_multiplier (float): tolerance multiplier that decides how far the first wand selection goes
            lower_multipleir (float): tolerance multiplier that decides how far the second wand selection goes

        Returns:
            Tensor: Tensor with boolean values that denote selected pixels.
        """
    if image_tensor.dim() != 3 or image_tensor.size(0) != 1:  # RGB
        raise Exception("invalid image_tensor dimensions")

    # part 1: get above zero pixel values
    flat_image = image_tensor.flatten()

    bin_count = 256
    min_pixel = 0.0  # torch.min(flat_image).item()
    max_pixel = 1.0  # torch.max(flat_image).item()
    histogram = torch.histc(flat_image.float(), bins=bin_count, min=0.0, max=1.0)

    # part 2: get starting tolerance and starting pixel value
    bin_width = (max_pixel - min_pixel) / bin_count
    non_zero_indices = torch.nonzero(histogram, as_tuple=False)

    first_tolerance = upper_multiplier

    first_bound_bin_index = int(non_zero_indices[-1].item() * (1 - first_tolerance))
    first_bound = first_bound_bin_index * bin_width

    if first_bound <= 1e-8:
        return torch.zeros_like(image_tensor)

    # part 3: make starting global selection
    mask = image_tensor > first_bound

    # part 4: replace selected pixel values with the lowest value from selected pixels
    image_tensor_wand = torch.clone(image_tensor)
    image_tensor_wand[mask] = first_bound

    # part 5: get second tolerance value
    # from the lowest color value in part 2 to total lowest value that is above 0 color
    lower_bound_bin_index = int(first_bound_bin_index * lower_multipleir)
    if lower_bound_bin_index < 3:
        lower_bound_bin_index = 3
        if lower_bound_bin_index >= first_bound_bin_index:
            lower_bound_bin_index = first_bound_bin_index - 1

    if lower_bound_bin_index < 0:
        lower_bound_bin_index = 0

    lower_bound = lower_bound_bin_index * bin_width

    diff_map = ((image_tensor_wand - lower_bound) >= 0).squeeze()

    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32,
                          device=image_tensor.device).unsqueeze(0).unsqueeze(0)

    max_iters = int((mask.size(1) * mask.size(2)) / 4)

    # mask = mask.squeeze(0)

    for _ in range(max_iters):
        dilated_mask = (tnnfunc.conv2d(mask.float().unsqueeze(0), kernel, padding=1)
                        .squeeze().bool())

        # Mask update: keep pixels within threshold and add to current mask
        new_mask = dilated_mask & diff_map

        # Stop if no new pixels are added
        if torch.equal(new_mask, mask):
            break

        # Update mask with new selection
        mask = new_mask

    return mask.unsqueeze(0)


class RandomGreyscale(torch.nn.Module):
    """Monochromes given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.transform = tvtf.Compose([
            tvtf.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
            tvtf.ToTensor()  # Convert PIL image to tensor
        ])

    def forward(self, img):

        """
        Args:
            img (PIL Image or Tensor): Image to be made monochrome.

        Returns:
            PIL Image or Tensor: image made monochrome or not.
        """
        if self.p >= 1 or torch.rand(1) < self.p:
            if isinstance(img, Image.Image):
                return img.convert("L")
            else:
                return self.transform(img)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class HistogramEqualizationHSV(torch.nn.Module):
    """
    Applies histogram equalization to the saturation channel (HSV) of a given image.
    Works for both PIL images and tensors.

    Args:None
    """

    def __init__(self):
        super().__init__()
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    @staticmethod
    def histogram_equalization_hsv_s(image):
        """
        Perform histogram equalization on the saturation channel of the HSV color space.
        """
        try:

            image_np = np.array(image)  # Convert PIL image to NumPy array

            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
            h, s, v = cv2.split(hsv)  # Split into H, S, V channels

            s_eq = cv2.equalizeHist(s)  # Equalize the S channel
            hsv_eq = cv2.merge((h, s_eq, v))  # Merge back into HSV

            image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)  # Convert back to BGR

            return Image.fromarray(image_equalized)  # Convert NumPy array to PIL image

        except Exception:
            return []

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image with histogram equalized saturation.
        """
        if isinstance(img, torch.Tensor):  # If input is a tensor
            img = self.to_pil(img)  # Convert to PIL for processing

        img_equalized = self.histogram_equalization_hsv_s(img)  # Apply histogram equalization

        return self.to_tensor(img_equalized) if isinstance(img, torch.Tensor) else img_equalized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CLAHETransformLAB(torch.nn.Module):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the luminance channel (LAB) of a given image.
    Works for both PIL images and tensors.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default is 2.0.
        tile_grid_size (tuple): Size of the grid for histogram equalization. Default is (16, 16).
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(16, 16)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    def apply_clahe_lab(self, image):
        """
        Perform CLAHE on the luminance channel of the LAB color space.
        """
        image_np = np.array(image)  # Convert PIL image to NumPy array

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)  # Split into L, A, B channels

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        res = clahe.apply(l)

        # Merge the channels back and convert to RGB
        lab_image = cv2.merge((res, a, b))
        image_clahe_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

        return Image.fromarray(image_clahe_rgb)  # Convert NumPy array to PIL image

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image with CLAHE applied to luminance.
        """
        if isinstance(img, torch.Tensor):  # If input is a tensor
            img = self.to_pil(img)  # Convert to PIL for processing

        img_clahe = self.apply_clahe_lab(img)  # Apply CLAHE transformation

        return self.to_tensor(img_clahe) if isinstance(img, torch.Tensor) else img_clahe

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


def histogram_equalization_lab(image):
    image_np = np.array(image)

    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2Lab)

    # Split into L, a, b channels
    l, a, b = cv2.split(lab)

    # Apply histogram equalization on the L channel
    l_eq = cv2.equalizeHist(l)

    # Merge the equalized L channel with a and b channels
    lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR (RGB) color space
    image_equalized = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    return Image.fromarray(image_equalized)


def histogram_equalization_hsv_s(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s_eq = cv2.equalizeHist(s)
    hsv_eq = cv2.merge((h, s_eq, v))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)


def histogram_equalization_hsv_v(image):
    image_np = np.array(image)

    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))

    image_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return Image.fromarray(image_equalized)


def apply_clahe_rgb(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Split the image into R, G, B channels
    channels = cv2.split(image_np)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = [clahe.apply(channel) for channel in channels]

    # Merge the channels back together
    image_clahe = cv2.merge(channels)

    # Convert back to PIL Image
    return Image.fromarray(image_clahe)


def apply_clahe_lab(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Convert RGB to LAB color space
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

    # Split into L, A, and B channels
    l, a, b = cv2.split(lab_image)

    # Apply CLAHE to the L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    res = clahe.apply(l)

    # Merge the channels back and convert to RGB
    lab_image = cv2.merge((res, a, b))
    image_clahe_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Convert back to PIL Image
    return Image.fromarray(image_clahe_rgb)


def quantize_model(model, quantization_type='dynamic', dtype=torch.qint8, layers_to_quantize=(nn.Linear,)):
    """
    Quantizes a given model.

    Parameters:
    - model: The pre-trained PyTorch model.
    - quantization_type: 'dynamic' or 'static'. Dynamic quantization requires less setup.
    - dtype: The target quantized dtype (e.g., torch.qint8).
    - layers_to_quantize: A tuple of layer types that you want to quantize.

    Returns:
    - quantized_model: The quantized model.
    """
    model.eval()  # Make sure the model is in evaluation mode

    if quantization_type == 'dynamic':
        # Dynamic quantization works well with Linear layers.
        quantized_model = quant.quantize_dynamic(model, layers_to_quantize, dtype=dtype)

    elif quantization_type == 'static':
        # Static quantization requires calibration and module fusion.
        # For demonstration, we'll assume the model is already prepared for static quantization.
        model.qconfig = quant.get_default_qconfig('fbgemm')
        # If needed, fuse modules (example for a simple CNN, adjust as needed)
        # torch.quantization.fuse_modules(model, [['conv', 'relu']])
        quant.prepare(model, inplace=True)

        # Calibration step: run a few batches through the model (replace with your calibration loader)
        # for inputs, _ in calibration_loader:
        #     model(inputs)

        quantized_model = quant.convert(model, inplace=True)

    else:
        raise ValueError("quantization_type must be either 'dynamic' or 'static'")

    return quantized_model

def k_means_clustering(weights, num_clusters=8, max_iters=100):
    """
    Perform k-means clustering on weight values using NumPy.
    :param weights: NumPy array of weights to be clustered
    :param num_clusters: Number of clusters (quantization levels)
    :param max_iters: Maximum iterations for convergence
    :return: Clustered weights and cluster centers
    """
    weights = weights.flatten()
    unique_weights = np.unique(weights)  # Remove duplicates for efficiency

    # Initialize cluster centers randomly
    centers = np.random.choice(unique_weights, num_clusters, replace=False)

    for _ in range(max_iters):
        # Assign each weight to the nearest cluster
        distances = np.abs(weights[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        # Update cluster centers
        new_centers = np.array([weights[labels == i].mean() if np.any(labels == i) else centers[i]
                                 for i in range(num_clusters)])

        # Check for convergence
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Replace weights with the nearest cluster center
    clustered_weights = centers[labels].reshape(weights.shape)

    return clustered_weights, centers


def apply_weight_sharing_numpy(model, num_clusters=8):
    """
    Apply weight sharing to a PyTorch model using NumPy-based k-means clustering.
    :param model: PyTorch model
    :param num_clusters: Number of weight clusters
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Convert weights to NumPy
                param_np = param.cpu().numpy()

                # Perform k-means clustering on weights
                clustered_weights, _ = k_means_clustering(param_np, num_clusters)

                # Convert back to PyTorch tensor and update model
                param.copy_(torch.tensor(clustered_weights, dtype=param.dtype, device=param.device))