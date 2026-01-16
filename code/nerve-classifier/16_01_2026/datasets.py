import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as tv_tfs

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


class ImageResultsDataset(Dataset):
    """
    Dataset that loads retina images + per-image label .csv files.

    Labels are read from .csv files in data_folder_path and ordered
    exactly according to data_label_ordering list.
    """

    def __init__(self,
                 folder_path: str,  # path with images
                 data_folder_path: str,  # path with per-image .csv labels
                 data_label_ordering: list,  # ['glaucoma', 'atrophy', 'valid_image']
                 img_shape: list,  # [W, H, C] e.g. [128, 128, 3]
                 label_correct: str = '',  # optional: name of correctness label to override
                 scan_subdirs: bool = False,
                 transforms=None,
                 device=None):
        super().__init__()

        self.folder_path = folder_path
        self.data_folder_path = data_folder_path
        self.data_label_ordering = data_label_ordering
        self.img_shape = img_shape
        self.label_correct = label_correct
        self.scan_subdirs = scan_subdirs
        self.transforms = transforms
        self.device = device

        # If label_correct is given, remember its index in ordering
        self.label_correct_index = -1
        if label_correct and label_correct in data_label_ordering:
            self.label_correct_index = data_label_ordering.index(label_correct)

        # Collect image paths + corresponding label csv paths
        self.samples = self._collect_samples()

        print(f"Dataset loaded: {len(self.samples)} samples found")

    def _collect_samples(self):
        """Find all valid image + label pairs"""
        samples = []

        if self.scan_subdirs:
            # Recursive scan
            for root, _, files in os.walk(self.folder_path):
                for fname in files:
                    if not is_image_file(fname):
                        continue
                    img_path = os.path.join(root, fname)
                    label_path = self._find_label_path(fname)
                    if label_path:
                        samples.append((img_path, label_path))
        else:
            # Flat folder
            for fname in os.listdir(self.folder_path):
                if not is_image_file(fname):
                    continue
                img_path = os.path.join(self.folder_path, fname)
                label_path = self._find_label_path(fname)
                if label_path:
                    samples.append((img_path, label_path))

        return samples

    def _find_label_path(self, image_fname):
        """Find corresponding .csv label file (same name without extension + .csv)"""
        base_name, _ = os.path.splitext(image_fname)
        csv_name = base_name + ".csv"
        csv_path = os.path.join(self.data_folder_path, csv_name)

        if os.path.isfile(csv_path):
            return csv_path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms if provided
        if self.transforms is not None:
            img = self.transforms(img)

        # To tensor + normalize [-1..1]
        if not isinstance(img, torch.Tensor):
            img = tv_tfs.ToTensor()(img)
        img = img * 2.0 - 1.0

        # If single channel → repeat to 3
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)

        # Resize to target shape (if needed)
        if img.shape[1:] != tuple(self.img_shape[:2]):
            img = tv_tfs.Resize(self.img_shape[:2], antialias=True)(img)

        # Load and order labels
        labels_df = pd.read_csv(label_path)
        # Take values in the exact order from data_label_ordering
        label_values = []
        for label_name in self.data_label_ordering:
            if label_name in labels_df.columns:
                val = float(labels_df[label_name].iloc[0])
            else:
                val = 0.0  # fallback if column missing
            label_values.append(val)

        labels = torch.tensor(label_values, dtype=torch.float32)

        # Optional: override correctness label
        if self.label_correct_index >= 0:
            # You can set logic here if needed (e.g. always 1 for correct crops)
            # Currently left as-is from file
            pass

        return img, labels, os.path.basename(img_path)
