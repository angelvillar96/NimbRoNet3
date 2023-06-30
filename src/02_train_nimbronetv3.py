"""
Training the complete NimbRoNet3 model for real-time detection, segmentation
and robot pose estimation on the soccer field.
"""

import os
from tqdm import tqdm
import torch

from base.baseTrainer import BaseTrainer
from lib.arguments import get_directory_argument
from lib.logger import Logger, log_function, for_all_methods
import lib.utils as utils
from lib.visualizations import visualize_pose_predictions, visualize_img_target_pred


@for_all_methods(log_function)
class Trainer(BaseTrainer):
    """
    Class for training the complete NimbRoNet3 model for real-time detection, segmentation
    and robot pose estimation on the soccer field.
    """

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.det_loss_tracker.reset()
        self.seg_loss_tracker.reset()
        self.pose_loss_tracker.reset()

        n_batch_lim = min(len(self.det_train_loader), len(self.seg_train_loader), len(self.pose_train_loader))
        progress_bar = tqdm(enumerate(self.det_train_loader), total=n_batch_lim)
        seg_iterator = iter(self.seg_train_loader)
        pose_iterator = iter(self.pose_train_loader)

        for i, (det_imgs, det_targets, _) in progress_bar:
            iter_ = n_batch_lim * epoch + i
            if i >= n_batch_lim:
                break
            # setting all the inputs and outputs to the GPU
            seg_imgs, seg_targets, _ = next(seg_iterator)
            pose_imgs, pose_targets, pose_target_weights, _ = next(pose_iterator)
            det_imgs, det_targets = det_imgs.to(self.device), det_targets.to(self.device)
            seg_imgs, seg_targets = seg_imgs.to(self.device), seg_targets.to(self.device)
            pose_imgs = pose_imgs.to(self.device)
            pose_targets = {k: v.to(self.device) for k, v in pose_targets.items()}
            pose_target_weights = [t.to(self.device) for t in pose_target_weights]

            # forward pass
            detections = self.model(det_imgs)["detections"]
            segmentation = self.model(seg_imgs)["segmentation"]
            pose_preds = self.model(pose_imgs)

            # losses
            self.det_loss_tracker(preds=detections, targets=det_targets)
            self.seg_loss_tracker(preds=segmentation, targets=seg_targets)
            self.pose_loss_tracker(preds=pose_preds, targets=pose_targets, target_weights=pose_target_weights)
            det_loss = self.det_loss_tracker.get_last_losses(total_only=True)
            seg_loss = self.seg_loss_tracker.get_last_losses(total_only=True)
            pose_loss = self.pose_loss_tracker.get_last_losses(total_only=True)
            loss = det_loss + seg_loss + pose_loss
            
            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # logging
            if iter_ % self.exp_params["training"]["log_frequency"] == 0:
                self.writer.log_full_dictionary(
                    dict=self.pose_loss_tracker.get_last_losses(),
                    step=iter_,
                    plot_name="Train Pose Loss",
                    dir="Train Pose Loss Iter",
                )
                self.writer.log_full_dictionary(
                    dict=self.det_loss_tracker.get_last_losses(),
                    step=iter_,
                    plot_name="Train Detection Loss",
                    dir="Train Detection Loss Iter",
                )
                self.writer.log_full_dictionary(
                    dict=self.seg_loss_tracker.get_last_losses(),
                    step=iter_,
                    plot_name="Train Segmentation Loss",
                    dir="Train Segmentation Loss Iter",
                )
                self.writer.add_scalars(
                    plot_name="Learning/Learning",
                    val_names=["Encoder LR", "Decoder LR"],
                    vals=[self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[-1]["lr"]],
                    step=epoch + 1,
                )

            # update progress bar
            progress_bar.set_description(
                f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. "
            )

        # aggreating mean loss over the whole epoch and logging
        self.det_loss_tracker.aggregate()
        self.seg_loss_tracker.aggregate()
        self.pose_loss_tracker.aggregate()
        average_det_loss_vals = self.det_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_det_loss_vals,
            step=epoch + 1,
            plot_name="Train Detection Loss",
            dir="Train Detection Loss",
        )
        average_seg_loss_vals = self.seg_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_seg_loss_vals,
            step=epoch + 1,
            plot_name="Train Segmentation Loss",
            dir="Train Segmentation Loss",
        )
        average_pose_loss_vals = self.pose_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_pose_loss_vals,
            step=epoch + 1,
            plot_name="Train Pose Loss",
            dir="Train Pose Loss",
        )
        total = average_det_loss_vals["_total"] + average_seg_loss_vals["_total"] + average_pose_loss_vals["_total"]
        self.training_losses.append(total.item())
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.det_loss_tracker.reset()
        self.seg_loss_tracker.reset()
        self.pose_loss_tracker.reset()

        n_batch_lim = min(len(self.det_valid_loader), len(self.seg_valid_loader), len(self.pose_valid_loader))
        progress_bar = tqdm(enumerate(self.det_valid_loader), total=n_batch_lim)
        seg_iterator = iter(self.seg_valid_loader)
        pose_iterator = iter(self.pose_valid_loader)

        for i, (det_imgs, det_targets, _) in progress_bar:
            if i >= n_batch_lim:
                break
            # setting all the inputs and outputs to the GPU
            seg_imgs, seg_targets, _ = next(seg_iterator)
            pose_imgs, pose_targets, pose_target_weights, _ = next(pose_iterator)
            det_imgs, det_targets = det_imgs.to(self.device), det_targets.to(self.device)
            seg_imgs, seg_targets = seg_imgs.to(self.device), seg_targets.to(self.device)
            pose_imgs = pose_imgs.to(self.device)
            pose_targets = {k: v.to(self.device) for k, v in pose_targets.items()}
            pose_target_weights = [t.to(self.device) for t in pose_target_weights]

            # forward pass
            detections = self.model(det_imgs)["detections"]
            segmentation = self.model(seg_imgs)["segmentation"]
            pose_preds = self.model(pose_imgs)

            # losses
            self.det_loss_tracker(preds=detections, targets=det_targets)
            self.seg_loss_tracker(preds=segmentation, targets=seg_targets)
            self.pose_loss_tracker(preds=pose_preds, targets=pose_targets, target_weights=pose_target_weights)
            det_loss = self.det_loss_tracker.get_last_losses(total_only=True)
            seg_loss = self.seg_loss_tracker.get_last_losses(total_only=True)
            pose_loss = self.pose_loss_tracker.get_last_losses(total_only=True)
            loss = det_loss + seg_loss + pose_loss
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

            # visualizing some examples
            if i < 1 and self.savefigs is True:
                # segmentation visualization
                seg_preds = segmentation.argmax(dim=1)
                fig, _, _ = visualize_img_target_pred(
                        imgs=seg_imgs[:8].cpu().detach(),
                        targets=(seg_targets[:8].unsqueeze(1).cpu().detach() / 3).clamp(0, 1),
                        preds=(seg_preds[:8].unsqueeze(1).cpu().detach() / 3).clamp(0, 1),
                        n_cols=8,
                    )
                self.writer.add_figure(tag=f"Segmentation Eval {i+1}", figure=fig, step=epoch + 1)

                # detection visualization
                fig, _, _ = visualize_img_target_pred(
                        imgs=det_imgs[:8].cpu().detach(),
                        targets=det_targets[:8].cpu().detach().clamp(0, 1),
                        preds=detections[:8].cpu().detach().clamp(0, 1),
                        n_cols=8,
                    )
                self.writer.add_figure(tag=f"Detection Eval {i+1}", figure=fig, step=epoch + 1)
                
                # pose visualizations
                fig, _, _ = visualize_pose_predictions(
                        imgs=pose_imgs,
                        targets=pose_targets,
                        preds=pose_preds,
                        n_cols=6,
                    )
                self.writer.add_figure(tag=f"Pose Predictions Eval {i+1}", figure=fig, step=epoch + 1)

        # logging of validation loss
        self.det_loss_tracker.aggregate()
        self.seg_loss_tracker.aggregate()
        self.pose_loss_tracker.aggregate()
        average_det_loss_vals = self.det_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_det_loss_vals,
            step=epoch + 1,
            plot_name="Valid Detection Loss",
            dir="Valid Detection Loss",
        )
        average_seg_loss_vals = self.seg_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_seg_loss_vals,
            step=epoch + 1,
            plot_name="Valid Segmentation Loss",
            dir="Valid Segmentation Loss",
        )
        average_pose_loss_vals = self.seg_loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
            dict=average_pose_loss_vals,
            step=epoch + 1,
            plot_name="Valid Pose Loss",
            dir="Valid Pose Loss",
        )
        total = average_det_loss_vals["_total"] + average_seg_loss_vals["_total"] + average_pose_loss_vals["_total"]
        self.validation_losses.append(total.item())
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting training procedure", message_type="new_exp")
    logger.log_git_hash()

    print("\nInitializing Trainer...")
    trainer = Trainer(
        exp_path=exp_path,
        checkpoint=args.checkpoint,
        resume_training=args.resume_training,
    )
    print("\nLoading dataset...")
    trainer.load_data()
    print("\nSetting up model and optimizer")
    trainer.setup_model()
    print("\nStarting to train")
    trainer.training_loop()


#
