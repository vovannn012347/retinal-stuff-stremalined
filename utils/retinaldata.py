import os

from PIL import Image
from torch.utils.data import Dataset
from torch import manual_seed, Tensor
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import time
from collections.abc import Iterable
import pandas as pd
import random
from utils.misc_retina import run_retina_cnn_2
import cv2
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

IMG_MASK_EXTENSIONS = ('.mask.jpg', '.mask.jpeg', '.mask.png', '.mask.ppm', '.mask.bmp',
                       '.mask.pgm', '.mask.tif', '.mask.tiff', '.mask.webp')

IMG_DATA_EXTENSIONS = ('.csv', '.json')


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def open_image(image_path, load_mode='RGB'):
    pil_image = pil_loader(image_path, load_mode)
    img_bbox = get_image_bbox(pil_image)

    pil_image = pil_image.crop(img_bbox)
    return pil_image


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


def is_image_mask_file(fname):
    return fname.lower().endswith(IMG_MASK_EXTENSIONS)

#get bounds of non-0 regions
# def get_image_bbox(gray_image):
#     # convert image to grayscale,
#     # get the bounding box of non-black regions
#     width, height = gray_image.size
#     top, left, bottom, right = height, width, 0, 0
#     threshold = 10
#
#     # Iterate over each pixel
#     for y in range(height):
#         for x in range(width):
#             # Get pixel value
#             pixel = gray_image.getpixel((x, y))
#
#             # Check if pixel is almost black
#             if pixel > threshold:
#                 # Update bounding box coordinates
#                 top = min(top, y)
#                 left = min(left, x)
#                 bottom = max(bottom, y)
#                 right = max(right, x)
#
#     return left, top, right, bottom


def get_bounding_box_fast(image_tensor: Tensor):
    # Check for non-zero values along each axis to find the bbox
    img = image_tensor.squeeze(0)
    rows = torch.any(img > 0, dim=1)  # True for rows with non-zero pixels
    cols = torch.any(img > 0, dim=0)  # True for columns with non-zero pixels

    # Get min and max of non-zero rows and columns
    min_y, max_y = (0, 1)
    min_x, max_x = (0, 1)

    search_y = torch.where(rows)
    search_x = torch.where(cols)

    if search_y[0].size(0) > 0:
        min_y, max_y = torch.where(rows)[0][[0, -1]]
        min_y, max_y = (min_y.item(), max_y.item())

    if search_x[0].size(0) > 0:
        min_x, max_x = torch.where(cols)[0][[0, -1]]
        min_x, max_x = (min_x.item(), max_x.item())

    # left, top, right, bottom
    bbox = (min_x, min_y, max_x, max_y)
    return bbox


def get_bounding_box_rectanglified(bbox, img_width, img_height, rect_min_w=128, rect_min_h=128):
    # bbbox order: left, top, right, bottom

    x_min = bbox[0]
    y_min = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    if width < rect_min_w:
        width = rect_min_w

    if height < rect_min_h:
        height = rect_min_h

    if height > img_height:
        height = img_height

    if width > img_width:
        width = img_width

    side_length = max(width, height)
    x_new_min = x_center - side_length / 2
    y_new_min = y_center - side_length / 2

    x_new_min = max(0, min(x_new_min, img_width - side_length))
    y_new_min = max(0, min(y_new_min, img_height - side_length))

    return (int(x_new_min),
            int(y_new_min),
            min(int(x_new_min + side_length), img_width),
            min(int(y_new_min + side_length), img_height))


def extract_objects_with_contours_np_cv2(tensor):

    if tensor.ndim == 3:  # If shape is (1, H, W), squeeze to (H, W)
        tensor = tensor.squeeze(0)
    binary_mask = tensor.cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tensors = []

    for idx, contour in enumerate(contours):

        object_mask = np.zeros_like(binary_mask)
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)

        tensor = torch.from_numpy(object_mask).unsqueeze(0)
        tensors.append(tensor)

    return tensors


"""
Extract individual objects from a ground truth tensor and save them as separate files.

Args:
    tensor (torch.Tensor): Ground truth mask tensor of shape (H, W) or (1, H, W).
    output_dir (str): Directory to save the extracted object masks.
"""
'''import torch
import numpy as np
from scipy.ndimage import label
from PIL import Image

def extract_objects_from_tensor(tensor, output_dir):
    # Ensure the tensor is on the CPU and convert to NumPy
    if tensor.ndim == 3:  # If shape is (1, H, W), squeeze to (H, W)
        tensor = tensor.squeeze(0)
    binary_mask = tensor.cpu().numpy().astype(np.uint8)

    # Label connected components
    labeled_mask, num_objects = label(binary_mask > 0)

    print(f"Number of objects detected: {num_objects}")

    # Iterate through each object and save it
    for obj_id in range(1, num_objects + 1):
        # Create a binary mask for the current object
        object_mask = (labeled_mask == obj_id).astype(np.uint8) * 255

        # Save the object mask to a file
        output_path = f"{output_dir}/object_{obj_id}.png"
        Image.fromarray(object_mask).save(output_path)
        print(f"Saved object {obj_id} to {output_path}")

# Example usage
# Simulate a PyTorch tensor with two objects
ground_truth_tensor = torch.tensor([
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2]
], dtype=torch.uint8)

extract_objects_from_tensor(ground_truth_tensor, "./output_masks")'''


# gets bounds of non-black regions
def get_image_bbox(image_pil):
    # convert image to grayscale,
    # get the bounding box of non-black regions
    gray_image = image_pil.convert("L")

    width, height = gray_image.size
    top, left, bottom, right = height, width, 0, 0
    threshold = 10

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Get pixel value
            pixel = gray_image.getpixel((x, y))

            # Check if pixel is almost black
            if pixel > threshold:
                # Update bounding box coordinates
                top = min(top, y)
                left = min(left, x)
                bottom = max(bottom, y)
                right = max(right, x)

    return left, top, right, bottom


# for use in blood vessel detection training
class RetinalTrainingDataset(Dataset):
    def __init__(self,
                 folder_path,
                 mask_folder_path,
                 img_shape,  # [W, H, C]
                 image_patch_size,  # [W, H]
                 image_patch_stride,  # [Hor, Ver]
                 random_crop=False,
                 scan_subdirs=False,
                 transforms=None,
                 device=None
                 ):
        super().__init__()
        self.img_shape = img_shape
        self.image_patch_size = image_patch_size
        self.image_patch_stride = image_patch_stride
        self.random_crop = random_crop
        self.device = device

        """sample_unfolded = torch.zeros([img_shape[2], img_shape[0], img_shape[1]])
        sample_unfolded = sample_unfolded.unfold(1, self.image_patch_size[0], self.image_patch_stride[0])
        sample_unfolded = sample_unfolded.unfold(2, self.image_patch_size[1], self.image_patch_stride[1])
        self.unfolded_size = sample_unfolded.size()"""

        self.random_crop = random_crop

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L'  # convert to greyscale

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path, mask_folder_path)
        else:
            self.data = self.make_dataset_from_dir(folder_path, mask_folder_path)

        self.size_multiply = (((img_shape[0] - image_patch_size[0]) // image_patch_stride[0] + 1) *
                              ((img_shape[1] - image_patch_size[1]) // image_patch_stride[1] + 1))
        self.is_data_unfolded = np.full(len(self.data), False, dtype=bool)
        self.patch_data = [False] * self.size_multiply
        self.mask_data = [False] * self.size_multiply

        self.transforms = None
        if transforms is not None:
            self.transforms = []
            if isinstance(transforms, Iterable):
                for transform in transforms:
                    self.transforms.append(transform)
            self.transforms = T.Compose(self.transforms)

    # only samples that have mask files are returned
    def make_dataset_from_dir(self, folder_path, mask_folder_path):

        samples_to_return = []

        samples_dict = {}
        for sample in [entry for entry in os.scandir(folder_path) if is_image_file(entry.name)]:
            samples_dict[sample.name] = sample.path

        for mask_sample in [entry for entry in os.scandir(mask_folder_path) if is_image_mask_file(entry.name)]:
            name, extension = os.path.splitext(mask_sample.name)
            value = name[:-5] + extension
            value = samples_dict.get(value)
            if value is not None:
                samples_to_return.append([value, mask_sample.path])

        return samples_to_return

    # only samples that have mask files are returned
    def make_dataset_from_subdirs(self, folder_path, mask_folder_path):

        samples_to_return = []
        samples_dict = {}
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples_dict[os.path.join(root[len(folder_path):], fname)] = os.path.join(root, fname)

        for root, _, fnames in os.walk(mask_folder_path, followlinks=True):
            for fname in fnames:
                if is_image_mask_file(fname):
                    file_key = os.path.join(root[len(folder_path):], fname[:-5])
                    value = samples_dict.get(file_key)
                    if value is not None:
                        samples_to_return.append([value, os.path.join(root, fname)])

        return samples_to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = pil_loader(self.data[index][0], self.mode)
        img_bbox = get_image_bbox(img_data)
        img_data = img_data.crop(img_bbox)

        img_mask = pil_loader(self.data[index][1], "L")
        img_mask = img_mask.crop(img_bbox)


        # use random manipulation to repeat same random transforms
        # also 42 :)
        hand_seed = int(time.time()) ^ 42

        manual_seed(hand_seed)
        img_data = self.transforms(img_data)
        if self.random_crop:
            img_data = T.RandomCrop(self.image_patch_size)(img_data)
        img_data = T.ToTensor()(img_data)
        img_data = TF.adjust_sharpness(img_data, 2)
        img_data = img_data * 2 - 1

        manual_seed(hand_seed)
        img_mask = self.transforms(img_mask)
        if self.random_crop:
            img_mask = T.RandomCrop(self.image_patch_size)(img_mask)
        img_mask_tensor = T.ToTensor()(img_mask)
        img_mask = torch.gt(img_mask_tensor, 0.5).to(torch.float32)

        manual_seed(int(time.time()))

        if img_data.size(0) == 1:
            img_data = torch.cat([img_data] * 3, dim=0)

        img_data = T.Resize(self.img_shape[:2], antialias=True)(img_data)
        img_mask = T.Resize(self.img_shape[:2], antialias=True)(img_mask)

        return img_data, img_mask  # [-1, 1], [0, 1]


# for use in inpaint training
class RetinalFCNNMaskDataset(Dataset):
    def __init__(self,
                 folder_path,
                 retina_pass,  # retina first pass
                 img_shape,  # [W, H, C]
                 image_patch_size,  # [W, H]
                 image_patch_stride,  # [Hor, Ver]
                 random_crop=False,
                 # provide_greyscale=False,
                 scan_subdirs=False,
                 transforms=None,
                 device=None
                 ):
        super().__init__()
        self.img_shape = img_shape
        self.image_patch_size = image_patch_size
        self.image_patch_stride = image_patch_stride
        self.random_crop = random_crop
        # self.provide_greyscale = provide_greyscale
        self.retina_pass = retina_pass
        self.device = device

        self.mode = 'RGB'
        #if img_shape[2] == 1:
        #    self.mode = 'L'  # convert to greyscale ALWAYS

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = self.make_dataset_from_dir(folder_path)

        self.size_multiply = (((img_shape[0] - image_patch_size[0]) // image_patch_stride[0] + 1) *
                              ((img_shape[1] - image_patch_size[1]) // image_patch_stride[1] + 1))
        self.is_data_unfolded = np.full(len(self.data), False, dtype=bool)
        self.patch_data = [False] * self.size_multiply
        self.mask_data = [False] * self.size_multiply

        self.transforms = None
        if transforms is not None:
            self.transforms = []
            if isinstance(transforms, Iterable):
                for transform in transforms:
                    self.transforms.append(transform)
            self.transforms = T.Compose(self.transforms)

        # self.greyscale_transform = T.Compose([
        #         T.Grayscale(),
        #         T.ToTensor()
        #     ])

    def mask_from_image(self, image):

        return 0

    # only samples that have mask files are returned
    def make_dataset_from_dir(self, folder_path):

        samples_to_return = []
        for sample in [entry for entry in os.scandir(folder_path) if is_image_file(entry.name)]:
            samples_to_return.append(sample.path)

        return samples_to_return

    # only samples that have mask files are returned
    def make_dataset_from_subdirs(self, folder_path):

        samples_to_return = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples_to_return.append(os.path.join(root, fname))

        return samples_to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #img_greyscale = None
        pil_image = pil_loader(self.data[index], self.mode)
        img_bbox = get_image_bbox(pil_image)

        pil_image = pil_image.crop(img_bbox)
        pil_image = T.Resize(max(self.img_shape[:2]), antialias=True)(pil_image)

        # if self.provide_greyscale:
        #     img_greyscale = self.greyscale_transform(pil_image).unsqueeze(1)

        if self.random_crop:
            pil_image = T.RandomCrop(self.image_patch_size)(pil_image)

        if self.transforms is not None:
            pil_image = self.transforms(pil_image)
        pil_image = T.ToTensor()(pil_image)

        pil_image = pil_image * 2 - 1  # [0, 1] -> [-1, 1]

        if pil_image.size(0) == 1:
            pil_image = pil_image.repeat(3, 1, 1)

        img_data = T.Resize(self.img_shape[:2], antialias=True)(pil_image)

        img_mask = run_retina_cnn_2(TF.adjust_sharpness(img_data, 2),
                                    self.retina_pass,
                                    self.image_patch_size,
                                    self.image_patch_stride,
                                    self.device)

        img_mask = T.Resize(self.img_shape[:2], antialias=True)(img_mask)

        return img_data, img_mask #, img_greyscale


# for use in nerve definitor
class ImageMaskDataset(Dataset):
    def __init__(self,
                 folder_path,
                 mask_folder_path,
                 img_shape,  # [W, H, C]
                 scan_subdirs=False,
                 transforms=None,
                 device=None
                 ):
        super().__init__()
        self.img_shape = img_shape
        self.device = device

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L'  # convert to greyscale

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path, mask_folder_path)
        else:
            self.data = self.make_dataset_from_dir(folder_path, mask_folder_path)

        self.transforms = None
        if transforms is not None:
            self.transforms = []
            if isinstance(transforms, Iterable):
                for transform in transforms:
                    self.transforms.append(transform)
            self.transforms = T.Compose(self.transforms)

    # only samples that have mask files are returned
    def make_dataset_from_dir(self, folder_path, mask_folder_path):

        samples_to_return = []

        samples_filenames = {}
        samples_dict = {}
        for sample in [entry for entry in os.scandir(folder_path) if is_image_file(entry.name)]:
            name, extension = os.path.splitext(sample.name)
            samples_dict[name] = sample.path
            samples_filenames[name] = sample.name

        for mask_sample in [entry for entry in os.scandir(mask_folder_path) if is_image_mask_file(entry.name)]:
            name, extension = os.path.splitext(mask_sample.name)
            file_name = name[:-5]
            value_stored = samples_filenames.get(file_name)
            if value_stored is not None:
                samples_to_return.append([samples_dict.get(file_name), mask_sample.path])

        return samples_to_return

    # only samples that have mask files are returned
    def make_dataset_from_subdirs(self, folder_path, mask_folder_path):

        samples_to_return = []
        samples_dict = {}
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples_dict[os.path.join(root[len(folder_path):], fname)] = os.path.join(root, fname)

        for root, _, fnames in os.walk(mask_folder_path, followlinks=True):
            for fname in fnames:
                if is_image_mask_file(fname):
                    file_key = os.path.join(root[len(folder_path):], fname[:-5])
                    value = samples_dict.get(file_key)
                    if value is not None:
                        samples_to_return.append([value, os.path.join(root, fname)])

        return samples_to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = pil_loader(self.data[index][0], self.mode)
        img_bbox = get_image_bbox(img_data)
        img_data = img_data.crop(img_bbox)
        #img_data = apply_clahe_lab(img_data)

        img_mask = pil_loader(self.data[index][1], "L")
        img_mask = img_mask.crop(img_bbox)

        # use random manipulation to repeat same random transforms
        # also 42
        hand_seed = int(time.time()) ^ 42

        manual_seed(hand_seed)
        img_data = self.transforms(img_data)

        img_data = T.ToTensor()(img_data)
        img_data = img_data * 2 - 1

        manual_seed(hand_seed)
        img_mask = self.transforms(img_mask)
        img_mask_tensor = T.ToTensor()(img_mask)
        img_mask = torch.gt(img_mask_tensor, 0.5).to(torch.float32)

        manual_seed(int(time.time()))

        if img_data.size(0) == 1:
            img_data = torch.cat([img_data] * 3, dim=0)

        img_data = T.Resize(self.img_shape[:2], antialias=True)(img_data)
        img_mask = T.Resize(self.img_shape[:2], antialias=True)(img_mask)

        return img_data, img_mask, os.path.basename(self.data[index][0])


def is_image_data_file(fname):
    return fname.lower().endswith(IMG_DATA_EXTENSIONS)


# for use in classifier
class ImageResultsDataset(Dataset):
    def __init__(self,
                 folder_path,
                 folder_path_iccorect,
                 data_folder_path,
                 data_label_ordering,
                 img_shape,  # [W, H, C]
                 label_correct='',
                 scan_subdirs=False,
                 transforms=None,
                 device=None
                 ):
        super().__init__()
        self.img_shape = img_shape
        self.device = device

        self.mode = 'RGB'
        if img_shape[2] == 1:
            self.mode = 'L'  # convert to greyscale

        self.data_label_ordering = data_label_ordering
        self.label_correct = label_correct
        if self.label_correct:
            self.label_correct_index = self.data_label_ordering.index(self.label_correct)
        else:
            self.label_correct_index = -1

        if scan_subdirs:
            data1 = [(x, 1) for x in self.make_dataset_from_subdirs(folder_path, data_folder_path)]
            data2 = [(x, 0) for x in self.make_dataset_from_subdirs(folder_path_iccorect, data_folder_path)]
            self.data = data1 + data2
        else:
            data1 = [(x, 1) for x in self.make_dataset_from_dir(folder_path, data_folder_path)]
            data2 = [(x, 0) for x in self.make_dataset_from_dir(folder_path_iccorect, data_folder_path)]
            self.data = data1 + data2

        self.transforms = None
        if transforms is not None:
            self.transforms = []
            if isinstance(transforms, Iterable):
                for transform in transforms:
                    self.transforms.append(transform)
            self.transforms = T.Compose(self.transforms)

    # only samples that have data files files are returned
    def make_dataset_from_dir(self, folder_path, mask_folder_path):

        samples_to_return = []
        samples_filenames = {}
        samples_dict = {}
        for sample in [entry for entry in os.scandir(folder_path) if is_image_file(entry.name)]:
            name, extension = os.path.splitext(sample.name)
            samples_dict[name] = sample.path
            samples_filenames[name] = sample.name

        for mask_sample in [entry for entry in os.scandir(mask_folder_path) if is_image_data_file(entry.name)]:
            name, extension = os.path.splitext(mask_sample.name)
            file_name, extension2 = os.path.splitext(name)
            #file_name = name[:-4]
            value_stored = samples_filenames.get(file_name)
            if value_stored is not None:
                samples_to_return.append([samples_dict.get(file_name), mask_sample.path])

        return samples_to_return

    # only samples that have mask files are returned
    def make_dataset_from_subdirs(self, folder_path, mask_folder_path):

        samples_to_return = []
        samples_dict = {}
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples_dict[os.path.join(root[len(folder_path):], fname)] = os.path.join(root, fname)

        for root, _, fnames in os.walk(mask_folder_path, followlinks=True):
            for fname in fnames:
                if is_image_data_file(fname):
                    name, extension = os.path.splitext(fname)
                    file_key = os.path.join(root[len(folder_path):], name)
                    value = samples_dict.get(file_key)
                    if value is not None:
                        samples_to_return.append([value, os.path.join(root, fname)])

        return samples_to_return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_path_data, correct = self.data[index]
        img_data = pil_loader(img_path_data[0], self.mode)
        img_bbox = get_image_bbox(img_data)

        if img_bbox[2] != 0 and img_bbox[3] != 0:
            img_data = img_data.crop(img_bbox)

        img_labels = pd.read_csv(img_path_data[1])
        img_labels = [float(img_labels[label].values[0]) for label in self.data_label_ordering]

        if float(correct) <= 0.5:
            # image is iccorect, we do not see anything
            for i in range(0, len(img_labels)):
                img_labels[i] = 0

        if self.label_correct_index != -1:
            img_labels[self.label_correct_index] = float(correct)


        # use random manipulation to repeat same random transforms
        # also 42
        hand_seed = int(time.time()) ^ 42

        manual_seed(hand_seed)
        img_data = self.transforms(img_data)

        img_data = T.ToTensor()(img_data)
        img_data = img_data * 2 - 1

        manual_seed(int(time.time()))

        if img_data.size(0) == 1:
            img_data = torch.cat([img_data] * 3, dim=0)

        img_data = T.Resize(self.img_shape[:2], antialias=True)(img_data)

        return img_data, torch.tensor(img_labels), os.path.basename(img_path_data[0])