"""
Loss functions and loss-related utils
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from lib.logger import log_info, print_
from CONFIG import LOSSES


class LossTracker:
    """
    Class for computing, weighting and tracking several loss functions

    Args:
    -----
    loss_params: dict
        Loss section of the experiment paramteres JSON file
    """

    def __init__(self, loss_params):
        """ Loss tracker initializer """
        assert isinstance(loss_params, list), f"Loss_params must be a list, and not {type(loss_params)}"
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(f"Loss {loss['type']} not implemented. Use one of {LOSSES}")

        print_("Setting up loss functions...")
        self.loss_computers = {}
        for loss in loss_params:
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
        self.reset()
        return

    def reset(self):
        """ Reseting loss tracker """
        self.loss_values = {loss: [] for loss in self.loss_computers.keys()}
        self.loss_values["_total"] = []
        return

    def __call__(self, **kwargs):
        """ Wrapper for calling accumulate """
        self.accumulate(**kwargs)

    def accumulate(self, **kwargs):
        """ Computing the different metrics and adding them to the results list """
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            self.loss_values[loss].append(loss_val)
            total_loss = total_loss + loss_val * self.loss_computers[loss]["weight"]
        self.loss_values["_total"].append(total_loss)
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_computers:
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        """ Fetching the last computed loss value for each loss function """
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        """ Printing and fetching the results """
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val


def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    if loss_type not in LOSSES:
        raise NotImplementedError(f"Loss {loss_type} not available. Use one of {LOSSES}")

    if loss_type in ["mse", "l2"]:
        loss = MSELoss()
    elif loss_type in ["mae", "l1"]:
        loss = L1Loss()
    elif loss_type in ["total_variation", "tv"]:
        loss = TotalVariationLoss()
    elif loss_type in ["segmentation total_variation", "seg tv"]:
        loss = SegmentationTotalVariationLoss()
    elif loss_type in ["cross_entropy", "ce"]:
        loss = CrossEntropyLoss(**kwargs)
    elif loss_type in ["pose_loss"]:
        loss = CustomPoseLoss(**kwargs)

    return loss


class Loss(nn.Module):
    """ Base class for losses """

    def __init__(self):
        """ """
        super().__init__()
        print_(f" --> Using {self.NAME} loss function")

    def forward(self, **kwargs):
        """ Computing loss """
        raise NotImplementedError("Base class does not implement forward...")


class MSELoss(Loss):
    """ Overriding MSE Loss"""

    NAME = "L2-Loss"

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = kwargs.get("preds"), kwargs.get("targets")
        loss = self.mse(preds, targets)
        return loss


class L1Loss(Loss):
    """ Overriding L1 Loss"""

    NAME = "L1-Loss"

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = kwargs.get("preds"), kwargs.get("targets")
        loss = self.mae(preds, targets)
        return loss


class TotalVariationLoss(Loss):
    """ Total Variation Loss"""

    NAME = "TotalVariation"

    def __init__(self):
        """ Module initializer """
        super().__init__()

    def forward(self, **kwargs):
        """ Computing loss """
        preds = kwargs.get("preds")
        B, C, H, W = preds.shape
        preds = F.pad(preds, (2, 2, 2, 2), mode="constant", value=0)
        # horizontal and vertical gradients
        pixel_dif_x = preds[..., :-1] - preds[..., 1:]
        pixel_dif_y = preds[..., :-1, :] - preds[..., 1:, :]
        # regularization computation and normalization
        loss = pixel_dif_x.abs().sum() + pixel_dif_y.abs().sum()
        loss = loss / (B * C * H * W)
        return loss


class SegmentationTotalVariationLoss(Loss):
    """
    Total Variation Loss for Segmentation maps.
    Regularizer is only computed for the background and field segmentation maps, i.e.
    it is not applied to the lines.
    """

    NAME = "SegmentationTotalVariation"

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.total_variation = TotalVariationLoss()

    def forward(self, **kwargs):
        """ Computing total variation loss for the field and background segmentation masks """
        preds = kwargs.get("preds")
        B, C, H, W = preds.shape
        bkgr, field = preds[:, :1], preds[:, 1:2]
        loss_bkgr, loss_field = self.total_variation(preds=bkgr), self.total_variation(preds=field)
        loss = loss_bkgr + loss_field
        return loss


class CrossEntropyLoss(Loss):
    """ Overriding Cross Entropy Loss"""

    NAME = "CrossEntropyLoss"

    def __init__(self, **kwargs):
        """ Module initializer """
        super().__init__()
        weight = kwargs.get("ce_weight", [1, 1, 1])
        print_(f"   --> Weighting classes by {weight}")
        self.weight = torch.Tensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = kwargs.get("preds"), kwargs.get("targets")
        if self.ce.weight.device != preds.device:
            self.ce.to(preds.device)
        loss = self.ce(preds, targets.long())
        return loss


class CustomPoseLoss(Loss):
    """ Custom Loss for the pose estimation model """
    
    NAME = "PoseLoss"
    
    def __init__(self, **kwargs):
        """ Module initialzer """
        super().__init__()
        
    def forward(self, **kwargs):
        """ Computing Loss """
        preds = kwargs.get("preds", None)
        targets = kwargs.get("targets", None)
        target_weights = kwargs.get("target_weights", None)
        assert preds is not None, "'preds' cannot be None!"
        assert targets is not None, "'targets' cannot be None!"

        # computing each loss element        
        loss_kpts_lr = self.weighted_mse(
            pred=preds["heatmaps_lr"],
            target=targets["heatmaps_lr"],
            weights=target_weights[0] if target_weights is not None else None
        ).mean(dim=(1, 2, 3))
        loss_kpts_hr = self.weighted_mse(
            pred=preds["heatmaps_hr"],
            target=targets["heatmaps_hr"],
            weights=target_weights[1] if target_weights is not None else None
        ).mean(dim=(1, 2, 3))
        loss_limbs_lr = self.weighted_mse(
            pred=preds["limbs_lr"],
            target=targets["limbs_lr"],
            weights=target_weights[0] if target_weights is not None else None
        ).mean(dim=(1, 2, 3))
        
        # total loss
        loss = (loss_kpts_lr + loss_kpts_hr).mean(dim=0).sum() + loss_limbs_lr.mean(dim=0).sum()
        return loss

    def weighted_mse(self, pred, target, weights=None):
        """ """
        loss = torch.pow(pred - target, 2)
        if weights is not None:
            loss = loss * weights.unsqueeze(1)
        return loss

    

class _FocalLoss(nn.Module):
    """
    Focal Loss for object detection
        - Adapted from https://github.com/clcarwin/focal_loss_pytorch
    EXPERIMENTAL!
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        """ Module initializer """
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha, 1-alpha]) if isinstance(alpha, (float, int)) else torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """ Forward pass """
        if input.dim() > 2:
            input = input.view(input.shape[0], input.shape[1], -1)  # (B, C, H, W) -> (B, C, H*W)
            input = input.transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
            input = input.contiguous().view(-1, input.shape[-1])  # (B, H*W, C) => (B*H*W, C)
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean() if self.size_average else loss.sum()
        return loss


#
