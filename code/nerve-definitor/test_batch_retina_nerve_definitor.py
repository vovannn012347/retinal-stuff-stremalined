import argparse
from PIL import Image
import torch
import os
import time
import cv2
import numpy as np
import torchvision.transforms as tvtransf
import torch.nn.functional as tochfunc
from utils.misc_retina import (magic_wand_mask_selection_faster, apply_clahe_lab, histogram_equalization_hsv_s)
from model.retina_classifier_networks import FcnskipNerveDefinitor2
from utils.retinaldata import get_image_bbox, is_image_file, get_bounding_box_fast, pil_loader, open_image, \
    get_bounding_box_rectanglified, extract_objects_with_contours_np_cv2

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    # default="training-data/retina-stuff-definitor/for-display/x32_drop30/checkpoints/states.pth",
                    default="training-data/retina-stuff-definitor/for-display/x8/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image-dir", type=str,
                    default="training-data/preprocess-output/local_nd_kaggle_selection",
                    help="path to the image file")
parser.add_argument("--out-dir", type=str,
                    # default="training-data/retina-stuff-classifier/for-display/x64/nerves_defined_output",
                    # default="training-data/retina-stuff-definitor/for-display/x32_drop30/define",
                    default="training-data/retina-stuff-definitor/for-display/x8/define",
                    help="path for the output cropped files")

img_shapes = [576, 576, 3]
load_mode = "RGB"
use_dropout = False
base = 8


def main():

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # set up network
    to_pil_transform = tvtransf.ToPILImage(mode='L')
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    definitor = FcnskipNerveDefinitor2(num_classes=1, base=base, use_dropout=use_dropout).to(device)
    definitor.load_state_dict(convolution_state_dict['nerve_definitor'])
    definitor.eval()

    image_null = Image.new("RGB", (1, 1), color="white")
    time0 = time.time()
    time_total = 0
    time_samples = 0
    for sample in [entry for entry in os.scandir(args.image_dir) if is_image_file(entry.name)]:

        print(f"input file at: {sample.name}")
        image_name, image_ext = os.path.splitext(sample.name)

        '''if extension.endswith("jpg"):
            extension = ".png"'''
        image_ext = ".jpg"

        if os.path.exists(os.path.join(args.out_dir, f"{image_name}_0{image_ext}")):
            continue

        pil_image = open_image(sample.path) #Image.open(sample.path).convert('RGB')  # 3 channel

        pil_image_processed = histogram_equalization_hsv_s(pil_image)
        pil_image_processed = apply_clahe_lab(pil_image_processed)
        pil_image_processed = pil_image_processed.convert("L")
        time0 = time.time()
        #pil_image_processed = pil_image.convert("L")

        tensor_image = tvtransf.ToTensor()(pil_image_processed)

        tensor_image = tvtransf.Resize(img_shapes[:2], antialias=True)(tensor_image)
        pil_image = pil_image.resize(img_shapes[:2])

        channels, h, w = tensor_image.shape
        tensor_image = tochfunc.interpolate(tensor_image.unsqueeze(0),
                                            scale_factor=0.5,
                                            mode='bilinear',
                                            align_corners=False).squeeze(0)

        tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
        #tensor_image = F.interpolate(tensor_image, scale_factor=0.5, mode='bilinear', align_corners=False)

        if tensor_image.size(0) == 1:
            tensor_image = torch.cat([tensor_image] * 3, dim=0)

        tensor_image = tensor_image.unsqueeze(0)
        output = definitor(tensor_image).squeeze(0)

        output[output < 0.09] = 0
        wand_output = magic_wand_mask_selection_faster(output, upper_multiplier=0.15, lower_multipleir=0.3).to(torch.float32)

        channels_bb, h_bb, w_bb = wand_output.shape
        split_tensors = extract_objects_with_contours_np_cv2(wand_output)

        for split_idx, tensor in enumerate(split_tensors):

            img_bbox = get_bounding_box_fast(tensor)

            '''wand_output = to_pil_transform(wand_output)
            img_mask_name_path = os.path.join(args.out_dir, f"{image_name}_mask_{split_idx}{image_ext}")
            wand_output.save(img_mask_name_path)'''

            #output = to_pil_transform(wand_output)
            #img_bbox = get_bounding_box_fast(output) # left, top, right, bottom

            bb_w = img_bbox[2] - img_bbox[0]
            bb_h = img_bbox[3] - img_bbox[1]

            expand_constant = 0.2
            img_bbox2 = ((img_bbox[0] - bb_w * expand_constant) / w_bb, (img_bbox[1] - bb_h * expand_constant) / h_bb,
                         (img_bbox[2] + bb_w * expand_constant) / w_bb, (img_bbox[3] + bb_h * expand_constant) / h_bb)

            img_bbox3 = max(img_bbox2[0], 0), max(img_bbox2[1], 0), min(img_bbox2[2], 1.0), min(img_bbox2[3], 1.0),
            img_bbox4 = int(img_bbox3[0] * h), int(img_bbox3[1] * w), int(img_bbox3[2] * h), int(img_bbox3[3] * w)

            img_bbox4 = get_bounding_box_rectanglified(img_bbox4, h, w)

            if (img_bbox4[2] - img_bbox4[0]) > 1 and (img_bbox4[3] - img_bbox4[1]) > 1:
                pil_image_cropped = pil_image.crop(img_bbox4)
                img_name_path = os.path.join(args.out_dir, f"{image_name}_{split_idx}{image_ext}")
                # os.path.join(args.out_dir, image_name + image_ext)
                pil_image_cropped.save(img_name_path)

        if len(split_tensors) == 0:
            img_name_path = os.path.join(args.out_dir, f"{image_name}_0{image_ext}")
            image_null.save(img_name_path)

        time_samples += 1
        dt = time.time() - time0
        time_total += dt

        print(f"{time_samples} @avg time: {time_total/time_samples} s, @time: {dt}")


if __name__ == '__main__':
    main()
