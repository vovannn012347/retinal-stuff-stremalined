import torch
import torchvision.transforms as tv_tf
import yaml
from PIL import Image

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
        self.transform = tv_tf.Compose([
            tv_tf.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
            tv_tf.ToTensor()  # Convert PIL image to tensor
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
