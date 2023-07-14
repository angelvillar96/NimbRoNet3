"""
Blob dataset for Blob-Based Object Detection.
This is the actual dataset used to train our NimbroNet for Object-Detection
"""

import os
import json
from glob import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from lib.augmentations import Augmentator
from data.data_utils import get_heatmap, pad_img_target


class BlobDataset(Dataset):
    """
    Blob dataset for Blob-Based Object Detection
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
    max_kpts: int
        Maximum number of detections to consider. If the actual number of objects is smaller than this, 
        we pad the center locations with [-1, -1] to allow for batching,
    """

    CLASSES = ["Ball", "Goal Post", "Robot"]
    NUM_CLASSES = len(CLASSES)
    # SPLITS_FILE = "new_blod_db_splits.json"
    # SPLITS_FILE = "blod_db_splits.json"

    def __init__(self, path, resources_path, split, img_size, split_file, augmentations,
                 use_augmentations=True, max_kpts=8):
        """ 
        Dataset initializer
        """
        assert os.path.exists(path), f"Given dataset path {path} does not exist"
        assert split in ["train", "valid", "test"], f"Unknown split {split}. Use one of ['train', 'valid', 'test']"
        self.path = path
        self.resources_path = resources_path
        self.split = split
        self.split_file = split_file
        self.use_augmentations = use_augmentations
        self.img_size = img_size
        self.sigma_factor = ((img_size[0] / 480) + (img_size[1] / 640)) / 2
        self.max_kpts = max_kpts
        self.target_size = [i // 4 for i in img_size]

        # augmentations
        if self.use_augmentations:
            self.augmentator = Augmentator(augment_params=augmentations)
        self.tranform_input = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_heatmap = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

        # loading splot files
        split_file_path = os.path.join(resources_path, self.split_file)
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file {split_file_path} does not exist in resources path...")
        with open(split_file_path) as file:
            split_info = json.load(file)
        self.split_files = split_info[self.split] if split != "all" else []
        print(f"  --> {len(self.split_files)} {self.split} files found in split json")

        # getting images and annotations
        files = []
        for ext in ('**/*.jpeg', '**/*.png', '**/*.jpg'):
            files.extend(glob(os.path.join(path, ext), recursive=True))
        self.filtered_files = []
        for f in files:
            if os.path.basename(f) not in self.split_files and split != "all":
                # print(f"WARNING! File {os.path.basename(f)} not found...")
                continue
            ext = f.split(".")[-1]
            self.xml_path = f.replace("." + ext, ".xml")
            if os.path.isfile(self.xml_path):
                self.filtered_files.append(f)
            else:
                print(f"WARNING! XML File {self.xml_path} not found...")
        return

    def __len__(self):
        """ """
        return len(self.filtered_files)

    def __getitem__(self, index):
        """ """
        # loading image and annotations
        image_path = self.filtered_files[index]
        ext = image_path.split(".")[-1]
        xml_path = image_path.replace("." + ext, ".xml")
        annotation = ET.parse(xml_path).getroot()
        image = F.to_tensor(Image.open(image_path))

        # initializing annotations
        H, W = image.shape[-2:]
        y_mult, x_mult = self.img_size[0] / H / 4, self.img_size[1] / W / 4
        heatmap_placeholder = torch.zeros([3, int(self.target_size[0]), int(self.target_size[1])])
        kpts = [[], [], []]

        # filling heatmaps and keypoint list
        class_flags = [False, False, False]
        for obj in annotation.findall("object"):
            label = obj.find("name").text
            anno_bound_box = obj.find("bndbox")
            bound_box_center = self.get_center(anno_bound_box, label=label)
            kpt = torch.tensor([bound_box_center[0], bound_box_center[1]])
            kpt[0], kpt[1] = int((kpt[0] * x_mult).round()), int((kpt[1] * y_mult).round())

            if(label == "ball"):
                class_flags[0] = True
                heatmap_placeholder[0] = np.maximum(
                        heatmap_placeholder[0],
                        get_heatmap(self.target_size, kpt, 8 * self.sigma_factor)  # 15
                    )
                kpts[0].append(kpt)
            elif (label == "goalpost"):
                class_flags[1] = True
                heatmap_placeholder[1] = np.maximum(
                        heatmap_placeholder[1],
                        get_heatmap(self.target_size, kpt, 8 * self.sigma_factor)  # 15
                    )
                kpts[1].append(kpt)
            elif (label == "robot"):
                class_flags[2] = True
                heatmap_placeholder[2] = np.maximum(
                        heatmap_placeholder[2],
                        get_heatmap(self.target_size, kpt, 30 * self.sigma_factor)  # 50
                    )
                kpts[2].append(kpt)

        # normalizing heatmaps to enforce [0, 1]
        for i in range(heatmap_placeholder.shape[0]):
            heatmap_placeholder[i] = self.get_normalized_heatmap(heatmap_placeholder[i], class_flags[i])
        heatmap = heatmap_placeholder

        # applying augmentations
        if self.use_augmentations:
            image, heatmap = self.augmentator(img=image, label=heatmap)
        image, heatmap = self.tranform_input(image), self.transform_heatmap(heatmap)
        image, heatmap = pad_img_target(img=image, target=heatmap)

        # stacking keypoing centers, and filling with [-1, 1] to stack into batches
        for i in range(3):
            kpts[i] = kpts[i] + [torch.tensor([-1, -1])] * (self.max_kpts - len(kpts[i]))
            kpts[i] = torch.stack(kpts[i])
        kpts = torch.stack(kpts)
        kpts[kpts < 0] = -1

        meta = {
                "fname": image_path,
                "kpts": kpts.int()
            }
        return image, heatmap, meta

    def get_normalized_heatmap(self, heatmap_single_channel, class_flag):
        """ Normalizing the magnitude of a heatmap """
        # n = (heatmap_single_channel - torch.min(heatmap_single_channel))
        # d = (torch.max(heatmap_single_channel) - torch.min(heatmap_single_channel))
        # norm_hmap = n / d
        norm_hmap = heatmap_single_channel / heatmap_single_channel.max()
        return norm_hmap

    def get_center(self, bound_box, label):
        """
        Obtaining the center of a bounding box for placing Blob
        NOTE!: For robots and goal-posts, the bottom-center (y=y_max) of the bounding box is taken instead
        """
        xmin = int(bound_box.find('xmin').text)
        ymin = int(bound_box.find('ymin').text)
        xmax = int(bound_box.find('xmax').text)
        ymax = int(bound_box.find('ymax').text)
        x = int((xmin + xmax) / 2)
        if label == "ball":
            y = int((ymin + ymax) / 2)
        else:
            y = int(ymax)
        return (x, y)

#
