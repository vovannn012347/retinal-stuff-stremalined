import argparse
import os

from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import pandas as pd

from model.retina_classifier_networks import HandmadeGlaucomaClassifier
from utils.retinaldata import get_image_bbox, pil_loader, open_image

parser = argparse.ArgumentParser(description='Test retina glaucoma detection')
parser.add_argument("--pretrained", type=str,
                    default="training-data/retina-stuff-classifier/checkpoints/states.pth",
                    help="path to the checkpoint file")

parser.add_argument("--image", type=str,
                    #default="training-data/retina-stuff-classifier/nerves_defined_output_1/1_left.jpg",
                    default="training-data/retina-stuff-classifier/nerves_defined_output/1171_left_cropped.jpg",
                    help="path to the image file")
parser.add_argument("--out-result", type=str,
                    #default="training-data/retina-stuff-classifier/nerves_classify_output/1_left.csv",
                    default="training-data/retina-stuff-classifier/nerves_classify_output/1171_left.csv",
                    help="path to the output result file")

img_shapes = [576, 576, 3]
load_mode = "RGB"
data_labels_ordered = ['glaucoma', 'atrophy', 'valid_image']


def main():

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.out_result)):
        os.makedirs(os.path.dirname(args.out_result))

    # set up network
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    convolution_state_dict = torch.load(args.pretrained,
                                        map_location=torch.device('cpu'))

    classifier = HandmadeGlaucomaClassifier(
        input_size=img_shapes[0]/2,
        num_classes=data_labels_ordered.__len__()).to(device)

    classifier.load_state_dict(convolution_state_dict['nerve_classifier'])

    print(f"input file at: {args.image}")

    pil_image_origin = open_image(args.image)

    tensor_image = T.ToTensor()(pil_image_origin)
    tensor_image = T.Resize(tuple([int(img_shapes[0]/2), int(img_shapes[1]/2)]),
                            antialias=True,
                            interpolation=InterpolationMode.BILINEAR)(tensor_image)

    tensor_image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
    if tensor_image.size(0) == 1:
        tensor_image = torch.cat([tensor_image] * 3, dim=0)

    tensor_image = tensor_image.unsqueeze(0)
    output = classifier(tensor_image).squeeze(0)

    data = pd.DataFrame(output.detach().numpy().reshape(1, -1), columns=data_labels_ordered)
    data = data.round(5)

    '''for col in data.columns:
        data[col] = data[col].apply(lambda x: f'{x:.40f}')'''

    # pd.options.display.float_format = '{:.40f}'.format
    data.to_csv(args.out_result, index=False)

if __name__ == '__main__':
    main()

