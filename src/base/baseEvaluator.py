"""
Evaluating a model checkpoint
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class BaseEvaluator:
    """ Class for evaluating a model """

    def __init__(self, exp_path, checkpoint):
        """ Initializing the evaluator object """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file(exp_path)
        self.checkpoint = checkpoint

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        return

    def load_data(self):
        """
        Loading the detection and segmentation datasets and fitting data-loader for
        iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        # detection
        print_("Loading Detection Dataset...")
        self.exp_params["dataset"]["dataset_name"] = "BlobDataset"
        det_test_set = data.load_data(exp_params=self.exp_params, split="test")
        self.det_dataset = det_test_set
        self.det_test_loader = data.build_data_loader(
                dataset=det_test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Num. Test Samples: {len(det_test_set)}")

        # segmentation
        print_("Loading Segmentation Dataset...")
        self.exp_params["dataset"]["dataset_name"] = "SegmentationDataset"
        seg_test_set = data.load_data(exp_params=self.exp_params, split="test")
        self.seg_dataset = seg_test_set
        self.seg_test_loader = data.build_data_loader(
                dataset=seg_test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Num. Test Samples: {len(seg_test_set)}")
        return

    def setup_model(self):
        """ Initializing model and loading pretrained paramters """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading metrick trackers for all three tasks
        eval_params = self.exp_params["evaluation"]
        self.det_metric_tracker = MetricTracker(
                metrics="detection",
                num_classes=self.det_dataset.NUM_CLASSES,
                **eval_params
            )
        self.seg_metric_tracker = MetricTracker(
                metrics="segmentation",
                num_classes=self.seg_dataset.NUM_CLASSES,
                **eval_params
            )
        
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(self.model, exp_path=self.exp_path)
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        self.model = self.model.eval()
        return

    @torch.no_grad()
    def evaluate_detection(self):
        """
        Evaluating for the task of detection
        """
        progress_bar = tqdm(enumerate(self.det_test_loader), total=len(self.det_test_loader))
        for i, (det_imgs, det_targets, _) in progress_bar:
            det_imgs, det_targets = det_imgs.to(self.device), det_targets.to(self.device)
            model_out = self.model(det_imgs)
            detections = model_out["detections"]
            self.det_metric_tracker.accumulate(preds=detections.cpu(), targets=det_targets.cpu())
            progress_bar.set_description(f"Iter {i}/{len(self.det_test_loader)}")
        self.det_metric_tracker.aggregate()
        return

    @torch.no_grad()
    def evaluate_segmentation(self):
        """
        Evaluating the model for the task of segmentation
        """
        progress_bar = tqdm(enumerate(self.seg_test_loader), total=len(self.seg_test_loader))
        for i, (seg_imgs, seg_targets, _) in progress_bar:
            seg_imgs, seg_targets = seg_imgs.to(self.device), seg_targets.to(self.device)
            model_out = self.model(seg_imgs)
            segmentation = model_out["segmentation"]
            seg_preds = segmentation.argmax(dim=1)
            self.seg_metric_tracker.accumulate(preds=seg_preds, targets=seg_targets)
            progress_bar.set_description(f"Iter {i}/{len(self.seg_test_loader)}")
        self.seg_metric_tracker.aggregate()
        return

    @torch.no_grad()
    def evaluate_pose_estimation(self):
        """
        Evaluating for the task of robot pose estimation
        """
        progress_bar = tqdm(enumerate(self.pose_test_loader), total=len(self.pose_test_loader))
        for i, (imgs, targets, _, metas) in progress_bar:
            imgs = imgs.to(self.device)
            model_out = self.model(imgs)
            pose_preds = {k: v.cpu() for k, v in model_out.items() if k in ["heatmaps_hr", "heatmaps_lr", "limbs_lr"]}
            self.pose_metric_tracker.accumulate(preds=pose_preds, targets=targets, metas=metas)
            progress_bar.set_description(f"Iter {i}/{len(self.pose_test_loader)}")
        self.pose_metric_tracker.aggregate()
        return

#