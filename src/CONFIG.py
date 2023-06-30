"""
Global configurations
"""

import os

########################################################
# High level configurations, such as paths or random seed
########################################################
CONFIG = {
    "random_seed": 13,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "num_workers": 8,
    "paths": {
        "data_path": os.path.join("/home/nfs/inf6/data/datasets/soccer_data", "Robocup_Hafez_Data","data"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
        "test_path": os.path.join(os.getcwd(), "test"),
        "resources_path": os.path.join(os.getcwd(), "resources"),
        "coco_path": "/home/nfs/inf6/data/datasets/coco"
    }
}


#################################################
# Supported datasets, models, metrics, and so on
################################################
# datasets
DATASETS = ["BlobDataset", "SegmentationDataset", "PoseDataset"]
AUGMENTATIONS = ["Horizontal flip", "Rotation", "Brightness", "Contrast", "Saturation", "Blur"]
SPATIAL_AUGMENTATIONS = ["Horizontal flip", "Rotation"]

# models
MODELS = [
    "NimbroNetV2", "NimbroNetV2plus",  # detection + segmentation
    "NimbroNetV3"                      # detection + segmentation + robot-pose-estimation
]
ENCODERS = ["ResNet18", "ResNet34", "ResNet50", "EfficientNetB0", "MobileNetv3"]

# losses and metrics
LOSSES = [
    "mse", "l2", "mae", "l1",
    "total_variation", "tv", "segmentation total_variation", "seg tv",
    "cross_entropy", "ce",
    "pose_loss"
]
METRICS = [
        "detection F1",                  # detection
        "segmentation accuracy", "IOU",  # segmentation
        "pose_ap"                        # pose estimation
    ]
METRIC_SETS = {
    "detection": ["detection F1"],
    "segmentation": ["segmentation accuracy", "IOU"],
    "pose_estimation": ["pose_ap"],
}


############################################
# Specific configurations and default values
############################################
DEFAULTS = {
    "dataset": {
        "dataset_name": "BlobDataset",
        "shuffle_train": True,
        "shuffle_eval": False,
        "augment_eval": False,
        "img_size": (540, 960),
        "augmentations": {
            "augmentations": ["Horizontal flip", "Brightness", "Contrast", "Saturation", "Blur"],
            "mirror_prob": 0.5,
            "degrees": 20,
            "mean_brightness": 1,
            "std_brightness": 0.6,
            "mean_contrast": 1,
            "std_contrast": 0.3,
            "mean_saturation": 1,
            "std_saturation": 0.4,
            "blur_prob": 0.5,
            "blur_kernel": (3, 7),
            "blur_sigma": (1, 10),
        },
        "det_files": "blod_db_splits.json",
        "seg_files": "seg_db_splits.json"
    },
    "model": {
        "model_name": "NimbroNetV3",
        "backbone": "ResNet18",
        "pretrained": True,
        "NimbroNetV2": {},
        "NimbroNetV2plus": {
            "base_channels": 128    
        },
        "NimbroNetV3": {
            "base_channels": 128
        },
    },
    "loss": {
        "segmentation": [
            {
                "type": "cross_entropy",
                "weight": 1,
                "ce_weight": [1, 1, 1]
            },
            {
                "type": "segmentation total_variation",
                "weight": 2e-4
            }
        ],
        "detection": [
            {
                "type": "mse",
                "weight": 50
            },
            {
                "type": "total_variation",
                "weight": 2e-3
            }
        ],
        "pose": [
            {
                "type": "pose",
                "weight": 1
            }
        ]
    },
    "training": {  # training related parameters
        "num_epochs": 100,      # number of epochs to train for
        "frozen_epochs": 12,    # number of epochs to train first with the frozen encoder
        "save_frequency": 5,    # saving a checkpoint after these eoochs ()
        "log_frequency": 10,    # logging stats after this amount of updates
        "batch_size": 32,
        "lr": 1e-3,
        "lr_encoder": 1e-3,     # Special learning rate for the pretrained encoder
        "optimizer": "adam",    # optimizer parameters: name, L2-reg, momentum
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "plateau",        # learning rate scheduler parameters
        "lr_factor": 0.5,       # Meaning depends on scheduler. See lib/model_setup.py
        "patience": 6,
        "lr_warmup": False,     # learning rate warmup parameters (2 epochs or 200 iters default)
        "warmup_steps": 2000,
        "warmup_epochs": 2,
        "early_stopping": True,
        "early_stopping_patience": 10,
    },
    "evaluation": {  # hyper-params for evaluating detection
        "match_thr": 10,     # Target and detected kpts are matched if they are closer than ths thr
        "filter_thr": 12,    # Distance threshold (in pixels) for NMS.
        "kernel_size": 5,    # Kernel size for morphological operations (erosion & dilation)
        "heatmap_thr": 0.1,  # Pixels with magnitude smaller than this thr are masked out
        "peak_thr": 0.3      # Minimum magnitude for a peak
    },
}


