"""
Dataset for soccer-field segmentations.
This dataset is used to train the NimbroNet model for image segmentation
"""

import os
import json
import cv2
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from data.data_utils import pad_img_target
from lib.augmentations import Augmentator


class SegmentationDataset(Dataset):
    """
    Dataset for soccer-field segmentations.
    The targets have the size of the input divided by 4: (H, W) --> (H / 4, W / 4)
    
    Args:
    -----
    path: string
        Path to where the data is stored
    resources_path: string
        Path to the resources directory. The train/test data split should be stored in that directory
    split: string
        Dataset split to instanciate. Must be one of ['train', 'valid', 'test']
    img_size: tuple/list
        Size to which we will resize the images. Format is (height, width)
    split_file: string
        Name of the json file containing the images to process and the dataset splits
    augmentations: dict
        Dictionary with the augmentations and augmentation parameters to apply.
    use_augmentations: bool
        If True, augmentations are applied. Otherwise, image from db is directly used.
    """

    CLASSES = ["Background", "Field", "Line"]
    NUM_CLASSES = len(CLASSES)
    # SPLITS_FILE = "new_seg_db_splits.json"
    # SPLITS_FILE = "seg_db_splits.json"

    def __init__(self, path, resources_path, split, img_size, split_file, augmentations, use_augmentations=True):
        """ Dataset initializer """
        assert os.path.exists(path), f"Given dataset path {path} does not exist"
        assert split in ["train", "valid", "test"], f"Unknown split {split}. Use one of ['train', 'valid', 'test']"
        self.path = path
        self.resources_path = resources_path
        self.split = split
        self.split_file = split_file
        self.img_size = img_size
        self.use_augmentations = use_augmentations
        self.target_size = [i // 4 for i in img_size]

        # augmentations
        self.augmentator = Augmentator(augment_params=augmentations)
        self.tranform_input = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # loading splot files
        split_file_path = os.path.join(resources_path, self.split_file)
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file {split_file_path} does not exist...")
        with open(split_file_path) as file:
            split_info = json.load(file)
        self.split_files = split_info[self.split] if split != "all" else []

        # getting image/annotation paths
        image_files = []
        for ext in ('**/image/*.jpeg', '**/image/*.png', '**/image/*.jpg'):
            image_files.extend(glob(os.path.join(path, ext), recursive=True))
        self.image_files = [f for f in image_files if os.path.basename(f) in self.split_files or split == "all"]
        return

    def __len__(self):
        """ """
        return len(self.image_files)

    def __getitem__(self, index):
        """ """
        image_path = self.image_files[index]
        target_path = image_path.replace("/image/", "/target/")
        ext = image_path.split(".")[-1]
        target_path = target_path.replace("." + ext, ".png")

        image = torch.FloatTensor(self.load_image(image_path))
        target = torch.FloatTensor(self.load_mask(target_path))

        meta = {
            "fname": image_path
        }

        if self.use_augmentations:
            image, target = self.augmentator(img=image, label=target)
        image = self.tranform_input(image)
        image, target = pad_img_target(img=image, target=target)
        return image, target, meta

    def load_image(self, path):
        """ Loading an image given path, resizing, and converting to PyTorch """
        raw_image = Image.open(path).convert('RGB')
        raw_image = F.resize(raw_image, self.img_size)
        norm_image = np.array(raw_image, dtype=np.float32) / 255.0
        img_rearrange = torch.from_numpy(np.transpose(norm_image, (2, 0, 1)))
        return img_rearrange

    def load_mask(self, path):
        """ Loading a segmentation mask, postprocessing, resizing, and converting to desired data-format """
        raw_image = Image.open(path)
        raw_image = np.array(raw_image)

        raw_image[raw_image == 1] = 3  # change ball to ground
        raw_image[raw_image == 2] = 2
        raw_image[raw_image == 3] = 1

        img_resized = cv2.resize(
                raw_image,
                dsize=(self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
        img_array = np.array(img_resized)
        img_array = np.transpose(img_array, (0, 1))
        return img_array

#
