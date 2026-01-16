import argparse
from PIL import Image
import torch
import os
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import pandas as pd

from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from model.retina_classifier_networks import FcnskipNerveDefinitor2, HandmadeGlaucomaClassifier
from utils.retinaldata import get_image_bbox, is_image_file, get_bounding_box_fast, pil_loader

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-stuff-classifier/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image-dir", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output_1",
                    help="path to the image files")
parser.add_argument("--out-result-dir", type=str,
                    default="training-data/retina-stuff-classifier/nerves_classify_output",
                    help="path for the output result files")

img_shapes = [576, 576, 3]
load_mode = "RGB"
data_labels_ordered = ['glaucoma', 'atrophy', 'valid_image']


def open_image(image_path):
    pil_image = pil_loader(image_path, load_mode)
    img_bbox = get_image_bbox(pil_image)

    pil_image = pil_image.crop(img_bbox)
    return pil_image


def main():

    args = parser.parse_args()

    if not os.path.exists(args.out_result_dir):
        os.makedirs(args.out_result_dir)

    # set up network
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    classifier = HandmadeGlaucomaClassifier(
        input_size=img_shapes[0]/2,
        num_classes=data_labels_ordered.__len__()).to(device)

    classifier.load_state_dict(convolution_state_dict['nerve_classifier'])

    for sample in [entry for entry in os.scandir(args.image_dir) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        name, extension = os.path.splitext(sample.name)

        extension = ".csv"

        pil_image_origin = open_image(sample.path)

        tensor_image = T.ToTensor()(pil_image_origin)
        tensor_image = T.Resize([img_shapes[0] / 2, img_shapes[1] / 2],
                                antialias=True,
                                interpolation=InterpolationMode.BILINEAR)(tensor_image)

        tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
        if tensor_image.size(0) == 1:
            tensor_image = torch.cat([tensor_image] * 3, dim=0)

        tensor_image = tensor_image.unsqueeze(0)
        output = classifier(tensor_image).squeeze(0)

        data = pd.DataFrame(output.detach().numpy().reshape(1, -1), columns=data_labels_ordered)
        img_name_path = os.path.join(args.out_result_dir, name + extension)
        data.to_csv(img_name_path, index=False)

if __name__ == '__main__':
    main()
