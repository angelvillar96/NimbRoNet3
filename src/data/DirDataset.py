"""
Dummy datasets for loading the images from a BAG-file or from a directory
"""

import os
import json
from glob import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from data.data_utils import get_heatmap, pad_img_target
from lib.augmentations import Augmentator

PATH = "/home/user/villar/datasets/Robocup_Hafez_Data/test/forceTrainDet"


class DirDataset:
    """ Dummy dataset for loading the images from the Test Bag """

    def __init__(self, img_size, augmentations, path=None, use_augmentations=True):
        """ Module initializer """
        self.path = PATH if path is None else path
        self.img_size = img_size
        self.use_augmentations = use_augmentations
        self.target_size = [i // 4 for i in img_size]

        if self.use_augmentations:
            self.augmentator = Augmentator(augment_params=augmentations)
        else:
            self.augmentator = None
        self.tranform_input = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # getting image/annotation paths
        self.get_data()
        return

    def __len__(self):
        """ """
        return len(self.image_files)

    def __getitem__(self, i):
        """ """
        raise NotImplementedError("Base class does not implement __getitem__()...")

    def load_image(self, path):
        """ """
        raw_image = Image.open(path).convert('RGB')
        raw_image = F.resize(raw_image, self.img_size)
        norm_image = np.array(raw_image, dtype=np.float32) / 255.0
        img_rearrange = torch.from_numpy(np.transpose(norm_image, (2, 0, 1)))
        return img_rearrange


class ConfigDatasetUnlabelled(DirDataset):
    """
    Simple dataset to load unlabelled images given a config file with the names of the images.
    """

    def __init__(self, config, img_size, augmentations, use_augmentations, path=None):
        """ Dataset initialization """
        assert os.path.exists(config), f"Confif {config} does not exist..."
        self.config = config
        super().__init__(
            img_size=img_size,
            augmentations=augmentations,
            use_augmentations=use_augmentations,
            path=path
        )
        
    def __getitem__(self, index):
        """ Sampling and loading an image"""
        image_path = self.image_files[index]

        image = torch.FloatTensor(self.load_image(image_path))
        if self.use_augmentations:
            image, _ = self.augmentator(img=image)
        meta = {
            "name": image_path.split("/")[-1],
            "path": image_path
        }
        image = self.tranform_input(image)
        image, _ = pad_img_target(img=image, target=None)
        return image, meta

    def get_data(self):
        """ Getting images"""
        with open(self.config) as f:
            img_names = json.load(f)["imgs"]
        image_files = [os.path.join(self.path, img) for img in img_names]
        self.image_files = sorted(image_files)
        return


class DirDatasetUnlabelled(DirDataset):
    """
    Simple dataset to load unlabelled images from a directory.
    This can be used, for instance, for the image auto labelling
    """

    def __init__(self, img_size, augmentations, use_augmentations, path=None, max_kpts=8):
        """ Dataset initialization """
        self.sigma_factor = (img_size[0] / img_size[1])
        self.max_kpts = max_kpts
        super().__init__(
            img_size=img_size,
            augmentations=augmentations,
            use_augmentations=use_augmentations,
            path=path
        )
        
    def __getitem__(self, index):
        """ Sampling and loading an image"""
        image_path = self.image_files[index]

        image = torch.FloatTensor(self.load_image(image_path))
        if self.use_augmentations:
            image, _ = self.augmentator(img=image)
        meta = {
            "name": image_path.split("/")[-1],
            "path": image_path
        }
        image = self.tranform_input(image)
        # image, _ = pad_img_target(img=image, target=None)
        return image, meta

    def get_data(self):
        """ Getting images"""
        image_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".png")]
        self.image_files = sorted(image_files)
        return


class DirDatasetDet(DirDataset):
    """ Simple dataset for loading detection data from directory """

    CLASSES = ["Ball", "Goal Post", "Robot"]
    CLASS_TO_LBL = {"ball": 0, "goalpost": 1, "robot": 2}
    LBL_TO_CLASS = {0: "ball", 1: "goalpost", 2: "robot"}
    LBL_TO_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    NUM_CLASSES = len(CLASSES)

    def __init__(self, img_size, augmentations, use_augmentations, path=None, max_kpts=8):
        self.sigma_factor = (img_size[0] / 480)
        self.max_kpts = max_kpts
        super().__init__(img_size, augmentations, use_augmentations, path)
        self.transform_heatmap = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        return

    def __getitem__(self, index):
        """ """
        # loading image and annotations
        image_path = self.image_files[index]
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

        # augmentations
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

    def get_data(self):
        """ Getting images and annotations """
        files = []
        for ext in ('*.jpeg', '*.png', '*.jpg'):
            files.extend(glob(os.path.join(self.path, ext), recursive=True))
        self.image_files = []
        for file in files:
            ext = file.split(".")[-1]
            self.xml_path = file.replace("." + ext, ".xml")
            if os.path.isfile(self.xml_path):
                self.image_files.append(file)
        return

    def get_normalized_heatmap(self, heatmap_single_channel, class_flag):
        """ Normalizing the magnitude of a heatmap """
        norm_hmap = heatmap_single_channel / heatmap_single_channel.max()
        return norm_hmap

    def get_center(self, bound_box, label):
        """
        Obtaining the center of a bounding box for placing Blob
        NOTE!: For robots and goal-posts, the bottom-center of the bounding box is taken instead
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


class DirDatasetSeg(DirDataset):
    """ Simple dataset for loading segmentation data from directory """

    def __init__(self, img_size, augmentations, use_augmentations, path=None):
        super().__init__(img_size, augmentations, use_augmentations, path)

    def __getitem__(self, index):
        """ Sampling image and corresponding mask """
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

    def get_data(self):
        """ Getting images"""
        image_files = []
        for ext in ('**/image/*.jpeg', '**/image/*.png', '**/image/*.jpg'):
            image_files.extend(glob(os.path.join(self.path, ext), recursive=True))
        self.image_files = image_files
        return

    def load_mask(self, path):
        """ """
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
