"""
Evaluating a NimbRoNet-v2plus model checkpoint for the following vision tasks:
  - Object detection
  - Field segmentation
"""

import os
from tqdm import tqdm
import torch

from lib.arguments import evaluation_arguments
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
from lib.visualizations import plot_confusion_matrix
import models.model_utils as model_utils
import data


@for_all_methods(log_function)
class Evaluator:
    """
    Class for evaluating a NimbRoNet-v2plus model checkpoint for the following vision tasks:
        - Object detection
        - Field segmentation
        - Pose Estimation
    """

    @torch.no_grad()
    def evaluate(self):
        """ Evaluating model epoch loop """
        ######################
        # evaluating detection
        ######################
        print_("  --> Evaluating detection performance")
        self.evaluate_detection()
        _ = self.det_metric_tracker.summary()
        self.det_metric_tracker.save_results(
                exp_path=self.exp_path,
                checkpoint_name=self.checkpoint,
                name="detection"
            )

        #########################
        # evaluating segmentation
        #########################
        print_("  --> Evaluating segmentation performance")
        self.evaluate_segmentation()
        _ = self.seg_metric_tracker.summary()
        self.seg_metric_tracker.save_results(
                exp_path=self.exp_path,
                checkpoint_name=self.checkpoint,
                name="segmentation"
            )

        # confusion matrix
        if "segmentation accuracy" in self.seg_metric_tracker.metric_computers.keys():
            accuracy_computer = self.seg_metric_tracker.metric_computers["segmentation accuracy"]
            cm = accuracy_computer.cm

            # saving confusion matrix as image
            savepath = os.path.join(self.plots_path, "segmentation_confusion_matrix.png")
            plot_confusion_matrix(cm, classes=self.seg_dataset.CLASSES, normalize=False, savepath=savepath)
            savepath = os.path.join(self.plots_path, "segmentation_confusion_matrix_norm.png")
            plot_confusion_matrix(cm, classes=self.seg_dataset.CLASSES, normalize=True, savepath=savepath)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = evaluation_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting evaluation procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Evaluator...")
    evaluator = Evaluator(
            exp_path=exp_path,
            checkpoint=args.checkpoint,
            quantize=args.quantize,
            half_precision=args.half_precision
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and loading pretrained parameters")
    evaluator.setup_model()
    print_("Starting evaluation")
    evaluator.evaluate()


#
