"""
Dataset module used for training a human pose estimation model.
Code is inspired by:
    - https://github.com/AIS-Bonn/HumanoidRobotPoseEstimation
"""

import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pycocotools
from pycocotools.coco import COCO

from data.data_utils import generate_pose_heatmaps, generate_limb_heatmaps, fliplr_kpts, \
                            get_affine_transform, affine_transform


class PoseDataset(Dataset):
    """
    Dataset module used for training a human pose estimation model.
    The targets have the size of the input divided by 4: (H, W) --> (H / 4, W / 4)
    
    Args:
    -----
    path: string
        Path to where the data is stored
    split: string
        Dataset split to instanciate. Must be one of ["train", "valid", "test"]
    img_size: tuple/list
        Size to which we will resize the images. Format is (height, width)
    split_file: string
        Name of the json file containing the images to process and the dataset splits
    augmentations: dict
        Dictionary with the augmentations and augmentation parameters to apply.
    use_augmentations: bool
        If True, augmentations are applied. Otherwise, image from db is directly used.
    keep_ratio: bool
        TODO
    """

    NUM_KEYPOINTS = 6
    MAX_NUM_DETECTIONS = 10
    # SIGMA = 2.0
    SIGMA = 1.4
    NUM_SCALES = 2
    KPT_FLIP_ORDER = [0, 1, 3, 2, 5, 4]
    LIMBS = np.array([[0, 1], [1, 2], [1, 3], [1, 4], [1, 5]])


    def __init__(self, path, split, img_size, augmentations, use_augmentations, keep_ratio=False):
        """ Dataset initializer """
        assert split in ["train", "valid", "test"], f"Unknown split {split}. Use one of ['train', 'valid', 'test']"
        assert isinstance(img_size, (tuple, list)), f"img_size must be tuple or list, not {type(img_size)}"
        assert len(img_size) == 2, f"img_size must have two elements, not {len(img_size)}"
        assert isinstance(img_size[0], int) and isinstance(img_size[1], int), f"{img_size = } must contain ints"
        self.path = path
        self.split = split
        self.path = path
        self.input_size = np.array((img_size[1], img_size[0]))
        self.output_size = np.array((img_size[1] // 4, img_size[0] // 4))
        # self.output_size = np.array((img_size[1] // 2, img_size[0] // 2))
        self.stride = np.int64(self.input_size / self.output_size)
        self.use_augmentations = use_augmentations
        self.keep_ratio = keep_ratio
        self.sigma = self.SIGMA * ((img_size[0] / 384) + (img_size[1] / 384)) / 2

        # augmentations
        if self.use_augmentations:
            self.flip_prob = augmentations.get("mirror_prob", 0.5)
            self.rotation = augmentations.get("degrees", 40)
            self.translation = augmentations.get("translate", 0.4)
            self.scale = augmentations.get("scale", [0.5, 1.5])
        else:
            self.flip_prob = 0.
            self.rotation = 0.
            self.translation = 0.
            self.scale = [1., 1.]
        print("Using Augmentations:")
        print(f"  --> Flip Prob. : {self.flip_prob}")
        print(f"  --> Rotation   : {self.rotation}")
        print(f"  --> Translation: {self.translation}")
        print(f"  --> Scale      : {self.scale}")
        self.tranform_input = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # COCO data loader
        self.annotation_file = self._get_annotation_file()
        self.coco = COCO(os.path.join(self.path, self.annotation_file))
        cat_ids = self.coco.getCatIds()
        cat = self.coco.loadCats(cat_ids)[0]
        self.limbs = np.array(cat["skeleton"], dtype=np.int64) - 1

        self.data = self._load_images(cat_ids)
        return

    def _get_annotation_file(self):
        """
        Getting the annotation file
        """
        if self.split == "train":
            ann = "robot_keypoints_train.json"
        elif self.split == "valid":
            ann = "robot_keypoints_val.json"
        elif self.split == "test":
            ann = "robot_keypoints_test.json"
        else:
            raise ValueError(f"No annotation file for dataset split {self.split}")
        return ann

    def _load_images(self, cat_ids):
        """
        Loading images corresponding to the current dataset split
        """
        data = []
        print("Loading pose estimation images...")
        for img_id in tqdm(self.coco.getImgIds(catIds=cat_ids)):
            image = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            mask = None
            keypoints = []
            for ann in self.coco.loadAnns(ann_ids):
                if ann["num_keypoints"] > 0:
                    keypoints.append(np.array(ann["keypoints"], dtype=np.int64).reshape(-1, 3))
                if ann["iscrowd"]:
                    mask = pycocotools.mask.decode(ann["segmentation"])

            if len(keypoints) == 0:
                keypoints.append([[0, 0, 0]] * self.num_keypoints)

            filename = image["file_name"]
            image_data = cv2.imread(os.path.join(self.path, "images", filename), cv2.IMREAD_COLOR)
            if image_data is None:
                raise ValueError("Fail to read {}".format(filename))

            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            data.append({
                "id": image["id"],
                "image": image_data,
                "mask": mask,
                "keypoints": np.stack(keypoints),                    
            })
        return data
    
    def __len__(self):
        """ Number of images """
        return len(self.data)

    def __getitem__(self, index):
        """
        Sampling image and keypoints from the dataset, generating the targets (e.g. heatmaps and limbs),
        and preprocessing and augmenting the inputs and targets.
        """
        cur_data = self.data[index]

        image = cur_data["image"].copy()
        H, W = image.shape[0], image.shape[1]        
        size = np.array([W, H], dtype=np.float64)
        input_size = self.input_size
        if self.keep_ratio:
            input_size = size.astype(np.uint32, copy=True)
            min_size = (self.input_size.min() + 63) // 64 * 64
            input_size[np.argsort(input_size)] = min_size, (min_size / size.min() * size.max() + 63) // 64 * 64

        # augmenting image
        center = size / 2
        in_translate = np.zeros_like(size)
        in_scale = input_size / size
        rotate = 0
        if self.use_augmentations:
            # sampling augmentation values
            flip, translate, scale, rotate = self._get_augmentation_params()
            in_scale = in_scale * scale
            in_translate = in_translate + translate * size * in_scale
            image = image[:, ::-1] if flip else image
        
        # transforming the image
        in_translate = in_translate + size * ((input_size / size) - in_scale) / 2
        in_trans = get_affine_transform(
            center=center,
            translate=in_translate,
            scale=in_scale,
            rotate=rotate
        )
        image = cv2.warpAffine(
            image,
            in_trans,
            (input_size[0], input_size[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.
        image = self.tranform_input(image)

    
        # generating targets, i.e., heatmaps and affinity fields        
        target_list, target_weight_list, keypoints_list = [], [], []
        for stride_factor in [2 ** i for i in range(int(self.NUM_SCALES - 1), -1, -1)]:
            mask = np.zeros(image.shape[:2], dtype=np.uint8) if cur_data["mask"] is None else cur_data["mask"].copy()
            kpts = cur_data["keypoints"].copy().swapaxes(0, 1)
            num_robots = kpts.shape[1]
            output_size = np.uint32(input_size / (self.stride * stride_factor))

            # flipping keypoints and switching left-right pairs
            if self.use_augmentations and flip:
                mask = mask[:, ::-1]
                kpts = fliplr_kpts(
                    keypoints=kpts,
                    width=W,
                    parts_order=self.KPT_FLIP_ORDER
                )

            # TODO: Why is this really needed?
            in_target_weight = cv2.warpAffine(
                mask,
                in_trans,
                (input_size[0], input_size[1]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            out_target_weight = cv2.resize(
                in_target_weight,
                (output_size[0], output_size[1]),
                interpolation=cv2.INTER_LINEAR
            )
            target_weight = np.float32(out_target_weight == 0)  # binary
            target_weight_list.append(torch.tensor(target_weight))

            # transforming keypoints given transform
            kpts_in = kpts.copy()
            kpts_in[..., :2] = np.int64(affine_transform(kpts_in[..., :2], in_trans))
            visibility = np.all((kpts_in[..., :2] >= 0) & (kpts_in[..., :2] <= (input_size - 1)), axis=2) & (kpts_in[..., 2] > 0)
            kpts_in = kpts_in * visibility[..., None]
            kpts[..., :2] = np.int64(affine_transform(kpts[..., :2], in_trans) / (self.stride * stride_factor))
            visibility = np.all((kpts[..., :2] >= 0) & (kpts[..., :2] <= (output_size - 1)), axis=2) & (kpts[..., 2] > 0)
            kpts *= visibility[..., None]

            # sort by scores
            if num_robots > self.MAX_NUM_DETECTIONS:
                scores = np.sum(kpts[:, :, 2], axis=0)
                kpts_order = np.argsort(-scores)
                kpts = kpts[:, kpts_order]
                kpts_in = kpts_in[:, kpts_order]
                if num_robots > self.MAX_NUM_DETECTIONS:
                    num_robots = self.MAX_NUM_DETECTIONS

            # generating targets
            heatmaps = generate_pose_heatmaps(
                keypoints=kpts,
                heatmap_size=output_size,
                sigma=self.sigma
            )
            background = np.maximum(1 - np.max(heatmaps, axis=0), 0.)
            heatmaps = np.vstack((heatmaps, background[None, ...]))
            limbs = []
            if self.NUM_SCALES == 1 or self.stride[0] * stride_factor == 8:
                limbs = generate_limb_heatmaps(
                    keypoints=kpts_in,
                    limbs=self.limbs,
                    input_size=input_size,
                    stride=self.stride * stride_factor,
                    sigma=4 * self.sigma,
                    thr=1.0,
                    dist_thr=0.015
                )
            target_list.append((torch.tensor(heatmaps), torch.tensor(limbs)))

            # padding/limiting
            keypoints = np.zeros((self.MAX_NUM_DETECTIONS, self.NUM_KEYPOINTS, 3), dtype=np.int64)
            if num_robots > 0:
                keypoints[:num_robots] = kpts[:, :num_robots].swapaxes(0, 1)
            keypoints_list.append(keypoints)
     
        meta = {
            "id": cur_data["id"],
            "keypoints": keypoints_list,
            "center": center,
            "translate": in_translate,
            "scale": in_scale
        }
        targets = {}
        targets["heatmaps_lr"] = target_list[0][0]
        targets["limbs_lr"] = target_list[0][1]
        if len(target_list) > 1:
            targets["heatmaps_hr"] = target_list[1][0]
        return image, targets, target_weight_list, meta
    
    
    def _get_augmentation_params(self):
        """" """
        flip = np.random.random() < self.flip_prob
        translate = (np.random.random(2) * 2 - 1) * self.translation
        scale = self.scale[0] + np.random.random() * (self.scale[1] - self.scale[0])
        rotate = (np.random.random() * 2 - 1) * self.rotation
        return flip, translate, scale, rotate
    