"""
Implementation of the NimbroNetV3 model for real-time perception on the soccer field.
The model simultaneously predicts:
  - Detections of ball, obstacle and goal posts
  - Segmentation of the field
  - Joints and Pose estimates of robots
"""

import torch
import torch.nn as nn

from models.NimbroNetBase import NimbroNetBase
from models.Blocks import ConvBilinearBlock, conv1x1, conv3x3
from models.Encoders import ResNet18


class NimbroNetV3(NimbroNetBase):
    """
    Implementation of the NimbroNetV3 model for real-time perception on the soccer field.
    The model simultaneously predicts:
        - Detections of ball, obstacle and goal posts
        - Segmentation of the field
        - Joints and Pose estimates of robots

    Args:
    -----
    pose_outputs: list/tuple
        List containing the number of limbs and the number of keypoints, respectively.
    encoder: torch.nn.Module
        Module used as (possibly pretrained) encoder in our network
    """
    
    IMG_SIZE = (540, 960)
    OUTPUTS = [5, 7]

    
    def __init__(self, det_classes=3, seg_classes=3, pose_outputs=None, base_channels=128,
                 encoder=ResNet18(pretrained=True), **kwargs):
        """ Module initializer """
        pose_outputs = pose_outputs if pose_outputs is not None else NimbroNetV3.OUTPUTS
        assert isinstance(pose_outputs, (list, tuple)), f"Pose-Outputs must be list or tuple, not {type(pose_outputs)}"
        assert len(pose_outputs) == 2, f"Pose-Outputs must be a list of length 2, not {len(pose_outputs)}"
        super().__init__()
        self.det_classes = det_classes
        self.seg_classes = seg_classes
        self.pose_outputs = pose_outputs
        self.base_channels = base_channels
        self.img_size = kwargs.get("img_size", NimbroNetV3.IMG_SIZE)
        self.encoder = encoder

        # getting spatial sizes and channels in the lateral connections
        self.img_size = kwargs.get("img_size") if "img_size" in kwargs else NimbroNetV3.IMG_SIZE
        num_channels, sp_dims = encoder.get_feat_size(img_size=self.img_size)
    
        # Lateral Connections
        self.lateral_block_3 = conv1x1(num_channels[-2], base_channels)
        self.lateral_block_2 = conv1x1(num_channels[-3], base_channels)
        self.lateral_block_1 = conv1x1(num_channels[-4], base_channels)

        # Decoder blocks
        scale_factor = (int(round(sp_dims[-2][0] / sp_dims[-1][0])), int(round(sp_dims[-2][1] / sp_dims[-1][1])))
        self.decoder_block_3 = ConvBilinearBlock(
            in_channels=num_channels[-1],
            out_channels=base_channels,
            act=True,
            norm=True,
            scale_factor=scale_factor
        )
        scale_factor = (int(round(sp_dims[-3][0] / sp_dims[-2][0])), int(round(sp_dims[-3][1] / sp_dims[-2][1])))
        self.decoder_block_2 = ConvBilinearBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            act=True,
            norm=True,
            scale_factor=scale_factor
        )
        scale_factor = (int(round(sp_dims[-4][0] / sp_dims[-3][0])), int(round(sp_dims[-4][1] / sp_dims[-3][1])))
        self.decoder_block_1 = ConvBilinearBlock(
            in_channels=base_channels * 2 + pose_outputs[0] + pose_outputs[1],
            out_channels=base_channels,
            act=True,
            norm=True,
            scale_factor=scale_factor
        )

        # output heads
        self.pose_lr_head = conv3x3(base_channels * 2, pose_outputs[0] + pose_outputs[1], stride=1, bias=True)
        self.pose_hr_head = conv3x3(base_channels * 2, pose_outputs[1], stride=1, bias=True)
        self.segmentation_head = conv3x3(base_channels * 2, seg_classes, stride=1, bias=True)
        self.detection_head = conv3x3(base_channels * 2, det_classes, stride=1, bias=True)

        self._init_parameters()
        return
    
    def forward(self, x):
        """ Forward pass """
        H = x.size(-2)
        x = self._add_padding(x=x, H=H)
        
        # encoder and computing intermediate features
        feats = self.encoder(x)
        
        # first decoder block and lateral connection
        out_dec_3 = self.decoder_block_3(feats[-1])
        out_lateral_3 = self.lateral_block_3(feats[-2])
        out_cat_3 = torch.cat([out_dec_3, out_lateral_3], dim=1)
        
        # second decoder block, lateral connection, and low-resolution output
        out_dec_2 = self.decoder_block_2(out_cat_3)
        out_lateral_2 = self.lateral_block_2(feats[-3])
        out_cat_2 = torch.cat([out_dec_2, out_lateral_2], dim=1)
        pose_out_lr = self.pose_lr_head(out_cat_2)
        out_cat_2 = torch.cat([out_cat_2, pose_out_lr], dim=1)
        
        # third (and final) decoder block, lateral connection, and high-resolution output
        out_dec_1 = self.decoder_block_1(out_cat_2)
        out_lateral_1 = self.lateral_block_1(feats[-4])
        out_cat_1 = torch.cat([out_dec_1, out_lateral_1], dim=1)

        # outputs heads        
        pose_out_hr = self.pose_hr_head(out_cat_1)
        detections = self.detection_head(out_cat_1)
        segmentation = self.segmentation_head(out_cat_1)
        
        # removing extra padding from all model outputs
        lr_outs, hr_outs = self._remove_padding(
            hr_outs=[pose_out_hr, detections, segmentation],
            lr_outs=[pose_out_lr],
            H=H
        )
        pose_out_hr, detections, segmentation = hr_outs[0], hr_outs[1], hr_outs[2]
        pose_out_lr = lr_outs[0] 
        
        # making output dictionary
        limbs_lr, heatmaps_lr = pose_out_lr[:, :self.pose_outputs[0]], pose_out_lr[:, self.pose_outputs[0]:]
        heatmaps_hr = pose_out_hr
        out_model = {
            "detections": detections,
            "segmentation": segmentation,
            "heatmaps_hr": heatmaps_hr,
            "heatmaps_lr": heatmaps_lr,
            "limbs_lr": limbs_lr
        }
        return out_model

    def _init_parameters(self):
        """ Initializing model parameters """
        for m in [self.lateral_block_1, self.lateral_block_2, self.lateral_block_3]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in [
            self.decoder_block_1, self.decoder_block_2, self.decoder_block_3,
            self.pose_lr_head, self.pose_hr_head, self.segmentation_head, self.detection_head
        ]:
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return
    
    def _remove_padding(self, lr_outs, hr_outs, H):
        """
        Removing the extra padding added to handle sizes not divisible by 32
        """
        pad_size = int(H / 32. + .5) * 32 - H
        pad_size_lr, pad_size_hr = int(pad_size / 4.), int(pad_size / 2.)
        if pad_size_lr > 0:
            lr_outs = [lr_out[:, :, :-pad_size_lr] for lr_out in lr_outs]
        if pad_size_hr > 0:    
            hr_outs = [hr_out[:, :, :-pad_size_lr] for hr_out in hr_outs]
        return lr_outs, hr_outs
