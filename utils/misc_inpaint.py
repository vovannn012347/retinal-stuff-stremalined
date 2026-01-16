import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import torch.nn.functional as TF

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

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
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)


def save_testing_model_states(file_name, gen, conv1, config):
    state_dicts = {'G': gen.state_dict(),
                   'convolution_pass1': conv1.state_dict()
                   }
    torch.save(state_dicts, f"{config.checkpoint_dir}/{file_name}")
    print("Saved state dicts!")


def save_states(file_name, gen, dis, g_optimizer, d_optimizer, n_iter, config):
    state_dicts = {'G': gen.state_dict(),
                   'D': dis.state_dict(),
                   'G_optim': g_optimizer.state_dict(),
                   'D_optim': d_optimizer.state_dict(),
                   'gen_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{file_name}")
    print("Saved state dicts!")


def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


@torch.inference_mode()
def infer_deepfill(generator,
                   image,
                   mask,
                   return_vals=['inpainted', 'stage1']):

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image*2 - 1.)  # map image values to [-1, 1] range
    # 1.: masked 0.: unmasked
    mask = (mask > 0.).to(dtype=torch.float32)

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    x_stage1, x_stage2 = generator(x, mask)

    image_compl = image * (1.-mask) + x_stage2 * mask

    output = []
    for return_val in return_vals:
        if return_val.lower() == 'stage1':
            output.append(output_to_img(x_stage1))
        elif return_val.lower() == 'stage2':
            output.append(output_to_img(x_stage2))
        elif return_val.lower() == 'inpainted':
            output.append(output_to_img(image_compl))
        else:
            print(f'Invalid return value: {return_val}')

    return output


def random_bbox(config):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config.img_shapes
    maxt = img_height - config.vertical_margin - config.height
    maxl = img_width - config.horizontal_margin - config.width
    t = np.random.randint(config.vertical_margin, maxt)
    l = np.random.randint(config.horizontal_margin, maxl)

    return (t, l, config.height, config.width)


def bbox2mask(config, bbox):
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)

    Returns:
        torch.Tensor: output with shape [1, 1, H, W]

    """
    img_height, img_width, _ = config.img_shapes
    mask = torch.zeros((1, 1, img_height, img_width),
                       dtype=torch.float32)
    h = np.random.randint(config.max_delta_height // 2 + 1)
    w = np.random.randint(config.max_delta_width // 2 + 1)
    mask[:, :, bbox[0]+h: bbox[0]+bbox[2]-h,
         bbox[1]+w: bbox[1]+bbox[3]-w] = 1.
    return mask


def brush_stroke_mask(config):
    """Generate brush stroke mask \\
    (Algorithm 1) from `Generative Image Inpainting with Contextual Attention`(Yu et al., 2019) \\
    Returns:
        torch.Tensor: output with shape [1, 1, H, W]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    min_width = 12
    max_width = 40

    mean_angle = 2*np.pi / 5
    angle_range = 2*np.pi / 15

    H, W, _ = config.img_shapes

    average_radius = np.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(
                    2*np.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)),
                      int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 1, H, W))
    return torch.Tensor(mask)


def test_contextual_attention(imageA, imageB, contextual_attention):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    rate = 2
    stride = 1
    grid = rate*stride

    b = Image.open(imageA)
    b = b.resize((b.width//2, b.height//2), resample=Image.BICUBIC)
    b = T.ToTensor()(b)

    _, h, w = b.shape
    b = b[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageA: {}'.format(b.shape))

    f = T.ToTensor()(Image.open(imageB))
    _, h, w = f.shape
    f = f[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageB: {}'.format(f.shape))

    yt, flow = contextual_attention(f*255., b*255.)

    return yt, flow


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
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
            T.ToTensor()  # Convert PIL image to tensor
        ])

    def forward(self, img):

        """
        Args:
            img (PIL Image or Tensor): Image to be made monochrome.

        Returns:
            PIL Image or Tensor: image made monochrome or not.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, Image.Image):
                return img.convert("L")
            else:
                return self.transform(img)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def dilate_mask(masks, kernel_size):
    """
    Dilate a binary mask by a specified diameter.

    Args:
        mask (torch.Tensor): Binary mask with shape (B, 1, H, W).
        kernel_size (int): Size of the dilation kernel (must be an odd number).

    Returns:
        torch.Tensor: Dilated binary mask with the same shape as the input.
    """
    # Create a square structuring element
    padding = kernel_size // 2
    structuring_element = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(masks.device)

    # Apply dilation using 2D convolution
    dilated_mask = TF.conv2d(masks, structuring_element, padding=padding, groups=masks.shape[1])
    dilated_mask = dilated_mask.clamp(max=1.0)  # Ensure the output is binary

    return dilated_mask


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size).float()
    coords -= (size - 1) / 2.0

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, size, size)


def get_gaussian_conv2d(device, channels, kernel_size: int, sigma: float) -> torch.nn.Conv2d:
    """
    Apply Gaussian filter to a tensor.

    Args:
        image (torch.Tensor): Image tensor of shape (B, C, H, W).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.
    """
    kernel = gaussian_kernel(kernel_size, sigma).to(device)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    conv = torch.nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, padding=kernel_size // 2)
    conv.weight.data = kernel
    conv.weight.requires_grad = False

    return conv
