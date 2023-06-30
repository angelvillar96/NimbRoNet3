"""
Methods for loading specific datasets, fitting data loaders and other
"""

import os
from torch.utils.data import DataLoader, random_split
from CONFIG import DATASETS, CONFIG
from data import BlobDataset, SegmentationDataset, PoseDataset, DetectionDataset, \
                 DenoisingCOCO, DirDatasetDet, DirDatasetSeg
from lib.utils import set_random_seed


def load_data(exp_params, split, datapath=None, resources_path=None, **kwargs):
    """
    Loading a dataset given the parameters

    Args:
    -----
    exp_params: dictionary
        dict with the experiment specific parameters
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')
    transform: Torch Transforms
        Compose of torchvision transforms to apply to the data

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    in_channels: integer
        number of channels in the dataset samples (e.g. 1 for B&W, 3 for RGB)
    """
    assert split in ["train", "valid", "test", "all"], f"Unknow dataset split {split}..."

    datapath = CONFIG["paths"]["data_path"] if datapath is None else datapath
    resources_path = CONFIG["paths"]["resources_path"] if resources_path is None else resources_path
    dataset_name = exp_params["dataset"]["dataset_name"]
    img_size = exp_params["dataset"]["img_size"]
    augmentations = exp_params["dataset"]["augmentations"]
    augment_eval = exp_params["dataset"].get("augment_eval", False)
    use_augmentations = True if augment_eval or split == "train" else False

    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset'{dataset_name}' is not available. Use one of {DATASETS}...")

    if(dataset_name == "BlobDataset"):
        path = os.path.join(datapath, "blob", "dataset")
        dataset = BlobDataset(
                path=path,
                resources_path=resources_path,
                split=split,
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations,
                split_file=exp_params["dataset"].get("det_files", "blod_db_splits.json")
            )
    elif(dataset_name == "SegmentationDataset"):
        path = os.path.join(datapath, "segmentation", "dataset")
        dataset = SegmentationDataset(
                path=path,
                resources_path=resources_path,
                split=split,
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations,
                split_file=exp_params["dataset"].get("seg_files", "seg_db_splits.json")
            )
    elif(dataset_name == "PoseDataset"):
        path = os.path.join(datapath, "poses", "hrp")
        dataset = PoseDataset(
                path=path,
                split=split,
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations
            )
    elif(dataset_name == "DetectionDataset"):
        path = os.path.join(datapath, "blob", "dataset")
        dataset = DetectionDataset(
                path=path,
                resources_path=resources_path,
                split=split,
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations
            )
    elif(dataset_name == "DenoisingCOCO"):
        dataset = DenoisingCOCO(
                resources_path=resources_path,
                split=split,
                noise_strength=kwargs.get("noise_strength", 0.2)
            )
    elif(dataset_name == "DirDatasetDet"):
        dataset = DirDatasetDet(
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations,
                path=datapath
            )
    elif(dataset_name == "DirDatasetSeg"):
        dataset = DirDatasetSeg(
                img_size=img_size,
                augmentations=augmentations,
                use_augmentations=use_augmentations,
                path=datapath
            )
    else:
        raise NotImplementedError(f"Dataset'{dataset_name}' is not available. Use one of {DATASETS}...")

    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )

    return data_loader


def manual_data_split(dataset, train_size=0.7, valid_size=0.15, test_size=0.15, fix_seed=True):
    """ Splitting a dataset into train-valid-test """
    assert train_size + valid_size + test_size == 1
    train_size = int(train_size * len(dataset))
    val_size = int(valid_size * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # fixing random seed for reproducibility
    if fix_seed:
        set_random_seed()

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

#
