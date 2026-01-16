import argparse
import os.path
import time

import torch
import torch.nn.functional as tochfunc
import torchvision.transforms as tvtransf
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import QConfig

from model.retina_classifier_networks import FcnskipNerveDefinitor2
from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from utils.retinaldata import get_bounding_box_fast, open_image, get_bounding_box_rectanglified, \
    extract_objects_with_contours_np_cv2

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="C:/Users/User/PycharmProjects/retinal-stuff/training-data/retina-stuff-definitor/for-display/x64/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image", type=str,
                    default="training-data/preprocess-output/local-images/043КМН02.jpg",
                    help="path to the image file")

parser.add_argument("--out-dir", type=str,
                    default="training-data/retina-stuff-classifier/nerves_defined_output",
                    help="path to the output mask file")

img_shapes = [576, 576, 3]
load_mode = "RGB"
quantized = False
model_base = 64


def split_by_position(string, position):
    return [string[:position], string[position:]]


def main():

    args = parser.parse_args()
    time0 = time.time()
    # set up network
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    definitor = FcnskipNerveDefinitor2(num_classes=1, use_dropout=False, base=model_base).to(device)

    if quantized:
        activation_dtype = 'quint8'
        weight_dtype = 'qint8'
        activation_qscheme = 'per_tensor_affine'
        weight_qscheme = 'per_tensor_affine'
        activation_observer = 'MovingAverageMinMaxObserver'
        weight_observer = 'MovingAverageMinMaxObserver'

        observer_map = {
            'MinMaxObserver': MinMaxObserver,
            'PerChannelMinMaxObserver': PerChannelMinMaxObserver,
            'MovingAverageMinMaxObserver': MovingAverageMinMaxObserver
        }

        activation_map = {
            'per_tensor_affine': torch.per_tensor_affine,
            'per_channel_affine': torch.per_channel_affine,
            'per_tensor_symmetric': torch.per_tensor_symmetric
        }

        activation_fake_quant = FakeQuantize.with_args(
            observer=observer_map[activation_observer],
            dtype=torch.__dict__[activation_dtype],
            qscheme=activation_map[activation_qscheme]
        )

        weight_fake_quant = FakeQuantize.with_args(
            observer=observer_map[weight_observer],
            dtype=torch.__dict__[weight_dtype],
            qscheme=activation_map[weight_qscheme]
        )

        # Set up the QConfig with the created FakeQuantize layers
        qconfig = QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
        definitor.qconfig = qconfig
        definitor = torch.quantization.prepare_qat(definitor)

        definitor = torch.quantization.convert(definitor, inplace=True)

    definitor.load_state_dict(convolution_state_dict['nerve_definitor'])
    definitor.eval()

    print(f"input file at: {args.image}")

    '''pil_image = open_image(sample.path)  # Image.open(sample.path).convert('RGB')  # 3 channel

    # pil_image_processed = histogram_equalization_hsv_s(pil_image)
    # pil_image_processed = apply_clahe_lab(pil_image_processed)
    # pil_image_processed = pil_image_processed.convert("L")
    pil_image_processed = pil_image.convert("L")

    tensor_image = T.ToTensor()(pil_image_processed)

    tensor_image = T.Resize(max(img_shapes[:2]), antialias=True)(tensor_image)
    pil_image = pil_image.resize(img_shapes[:2])
    channels, h, w = tensor_image.shape
    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]'''

    image_name = os.path.basename(args.image) # file name
    image_name, image_ext = os.path.splitext(image_name) # name and extension

    pil_image_origin = open_image(args.image)  # Image.open(args.image).convert('RGB')  # 3 channel

    pil_image_processed = histogram_equalization_hsv_s(pil_image_origin)
    pil_image_processed = apply_clahe_lab(pil_image_processed)
    pil_image_processed = pil_image_processed.convert("L")
    # pil_image_processed = pil_image_origin.convert("L")

    tensor_image = tvtransf.ToTensor()(pil_image_processed)
    tensor_image = tvtransf.Resize(img_shapes[:2], antialias=True)(tensor_image)
    pil_image_origin = pil_image_origin.resize(img_shapes[:2])
    channels, h, w = tensor_image.shape
    # training is done on smaller resolution image
    tensor_image = tochfunc.interpolate(tensor_image.unsqueeze(0),
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=False).squeeze(0)

    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    tensor_image = tensor_image.unsqueeze(0)
    output = definitor(tensor_image).squeeze(0)

    to_pil_transform = tvtransf.ToPILImage(mode='L')

    output[output < 0.09] = 0
    # output = torch.clamp(output, 0.09, 1)
    output_wand_selected = (magic_wand_mask_selection_faster(output, upper_multiplier=0.15, lower_multipleir=0.3)
                              .to(torch.float32))

    channels_bb, h_bb, w_bb = output_wand_selected.shape
    split_tensors = extract_objects_with_contours_np_cv2(output_wand_selected)

    out_filenames = []

    for split_idx, tensor in enumerate(split_tensors):

        '''output_selected = to_pil_transform(tensor)
        mask_path_out = os.path.join(args.out_dir, f"{image_name}_mask_{split_idx}{image_ext}")
        output_selected.save(mask_path_out)'''

        img_bbox = get_bounding_box_fast(tensor)  # left, top, right, bottom

        expand_constant = 0.2
        bb_w = img_bbox[2] - img_bbox[0]
        bb_h = img_bbox[3] - img_bbox[1]
        img_bbox2 = (img_bbox[0] - bb_w * expand_constant) / w_bb, (img_bbox[1] - bb_h * expand_constant) / h_bb, (
                    img_bbox[2] + bb_w * expand_constant) / w_bb, (img_bbox[3] + bb_h * expand_constant) / h_bb
        img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
        img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

        img_bbox4 = get_bounding_box_rectanglified(img_bbox4, h, w)

        if (img_bbox4[2] - img_bbox4[0]) > 1 and (img_bbox4[3] - img_bbox4[1]) > 1:
            pil_image_cropped = pil_image_origin.crop(img_bbox4)
            # pil_image_cropped.save(args.out_cropped)
            out_filename = f"{image_name}_cropped_{split_idx}{image_ext}"
            image_cropped_path_out = os.path.join(args.out_dir, out_filename)
            pil_image_cropped.save(image_cropped_path_out)
            out_filenames.append(out_filename)

    dt = time.time() - time0

    print(f"@timespan: {dt} s")


if __name__ == '__main__':
    main()

