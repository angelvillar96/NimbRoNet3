"""
Base trainer from which all trainer classes inherit.
Basically it removes the scaffolding that is repeat across all training modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
from lib.metrics import MetricTracker
from lib.schedulers import EarlyStop, Freezer
from lib.setup_model import emergency_save
from models.model_utils import GradientInspector
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class BaseTrainer:
    """
    Base Class for training and validating a model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the experiment directory.
        If given, the model is initialized with the parameters of such checkpoint.
        This can be used to continue training or for transfer learning.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ... are restored
        in order to continue training from the checkpoint
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config()
        self.exp_params = self.cfg.load_exp_config_file(exp_path)
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        # setting paths and creating subdirectories
        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        """
        Loading detection and segmentation dataset and fitting data-loader for iterating
        in a batch-like fashion.
        If pose dataset is required, it should be loaded in the specific trainer.
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        print_("Loading Blob-based Detection Dataset...")
        self.exp_params["dataset"]["dataset_name"] = "BlobDataset"
        det_train_set = data.load_data(exp_params=self.exp_params, split="train")
        det_valid_set = data.load_data(exp_params=self.exp_params, split="valid")
        self.det_train_loader = data.build_data_loader(
                dataset=det_train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.det_valid_loader = data.build_data_loader(
                dataset=det_valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Num. Train Samples: {len(det_train_set)}")
        print_(f"  --> Num. Valid Samples: {len(det_valid_set)}")

        print_("Loading Segmentation Dataset...")
        self.exp_params["dataset"]["dataset_name"] = "SegmentationDataset"
        seg_train_set = data.load_data(exp_params=self.exp_params, split="train")
        seg_valid_set = data.load_data(exp_params=self.exp_params, split="valid")
        self.seg_train_loader = data.build_data_loader(
                dataset=seg_train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.seg_valid_loader = data.build_data_loader(
                dataset=seg_valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Num. Train Samples: {len(seg_train_set)}")
        print_(f"  --> Num. Valid Samples: {len(seg_valid_set)}")
        return


    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler = setup_model.setup_optimization(
                exp_params=self.exp_params["training"],
                model=model
            )
        seg_loss_tracker = LossTracker(loss_params=self.exp_params["loss"]["segmentation"])
        det_loss_tracker = LossTracker(loss_params=self.exp_params["loss"]["detection"])
        pose_loss_tracker = LossTracker(loss_params=self.exp_params["loss"]["pose"])
        epoch = 0

        # loading pretrained model and other necessary objects for resuming training or fine-tuning
        if self.checkpoint is not None:
            print_(f"  --> Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                model, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"  --> Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        # other optimization objects
        self.model = model
        self.freezer = Freezer(
                module=self.model.encode,
                frozen_epochs=self.exp_params["training"]["frozen_epochs"]
            )
        self.early_stopping = EarlyStop(
                mode="min",
                use_early_stop=self.exp_params["training"]["early_stopping"],
                patience=self.exp_params["training"]["early_stopping_patience"]
            )
        self.optimizer, self.scheduler, self.epoch = optimizer, scheduler, epoch
        self.seg_loss_tracker = seg_loss_tracker
        self.det_loss_tracker = det_loss_tracker
        self.pose_loss_tracker = pose_loss_tracker
        return

    @emergency_save
    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.freezer(epoch)
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)
            stop_training = self.early_stopping(
                    value=self.validation_losses[-1],
                    writer=self.writer,
                    epoch=epoch
                )
            if stop_training:
                break

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss/CE_comb_loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler if loss increases or plateaus
            setup_model.update_scheduler(
                scheduler=self.scheduler,
                exp_params=self.exp_params,
                control_metric=self.validation_losses[-1],
                end_epoch=True
            )

            # saving backup model checkpoint and (if reached saving frequency) epoch checkpoint
            setup_model.save_checkpoint(  # Gets overriden every epoch: checkpoint_last_saved.pth
                    trainer=self,
                    savedir="models",
                    savename="checkpoint_last_saved.pth"
                )
            if(epoch % save_frequency == 0 and epoch != 0):  # checkpoint_epoch_xx.pth
                print_("Saving model checkpoint")
                setup_model.save_checkpoint(
                        trainer=self,
                        savedir="models"
                    )

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        setup_model.save_checkpoint(
                trainer=self,
                savedir="models",
                finished=not stop_training
            )
        return

    def train(self):
        """ Setting all the components to training mode """
        self.model.train()

    def eval(self):
        """ Setting all the components to evaluation mode """
        self.model.eval()

#