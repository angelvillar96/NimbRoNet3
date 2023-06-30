"""
Extended version of the NimbRo-v2 Unet-based Architecture.
It is based on the architecture from in:
    Rodriguez et al. "RoboCup 2019 AdultSize Winner NimbRo: Deep Learning Perception, In-Walk
                      Kick, Push Recovery, and Team Play Capabilities", Robot World Cup 2019
"""

import torch.nn as nn

from models.NimbroNetBase import NimbroNetBase
from models.Blocks import UnetBlock
from models.Encoders import ResNet18


class NimbroNetV2(NimbroNetBase):
    """
    Generalized and Extended version of the NimbRo-v2 from 2019 RoboCup paper

    Args:
    -----
    det_classes, seg_classes: integers
        Number of classes to detect and segment, respectively.
        For our RoboCup, det_classes=3 and seg_classes=3
    backbone: torch.nn.Module
        Module used as (possibly pretrained) encoder in our network
    """

    IMG_SIZE = (480, 640)

    def __init__(self, det_classes=3, seg_classes=3, encoder=ResNet18(pretrained=True), **kwargs):
        """ Module initializer """
        super().__init__()
        # backbone
        self.det_classes = det_classes
        self.seg_classes = seg_classes
        self.encoder = encoder
        num_channels, sp_dims = encoder.get_feat_size()

        # UNet blocks
        downsample = (int(sp_dims[2][0] / sp_dims[3][0]), int(sp_dims[2][1] / sp_dims[3][1]))
        self.unet_block_3 = UnetBlock(
            in_channels_dec=num_channels[3],   # 512 in ResNet
            in_channels_skip=num_channels[2],  # 256 in ResNet
            out_channels=512,                  # 512 in ResNet
            act=True,
            norm=True,
            downsample=downsample
        )
        downsample = (int(sp_dims[1][0] / sp_dims[2][0]), int(sp_dims[1][1] / sp_dims[2][1]))
        self.unet_block_2 = UnetBlock(
            in_channels_dec=512,                # 512 in ResNet
            in_channels_skip=num_channels[1],   # 128 in ResNet
            out_channels=512,                   # 512 in ResNet
            act=True,
            norm=True,
            downsample=downsample
        )
        downsample = (int(sp_dims[0][0] / sp_dims[1][0]), int(sp_dims[0][1] / sp_dims[1][1]))
        self.unet_block_1 = UnetBlock(
            in_channels_dec=512,                # 512 in ResNet
            in_channels_skip=num_channels[0],   # 64 in ResNet
            out_channels=256,                   # 256 in ResNet
            act=True,
            norm=True,
            downsample=downsample
        )

        # head blocks
        self.detection_head = nn.ConvTranspose2d(
                in_channels=256,
                out_channels=det_classes,
                kernel_size=1
            )
        self.segmentation_head = nn.ConvTranspose2d(
                in_channels=256,
                out_channels=seg_classes,
                kernel_size=1
            )

        self.initialize_params()
        return

    def forward(self, x):
        """ Forward pass """
        # forwad pass through encoder
        feats = self.encoder(x)
        x_skip_1, x_skip_2, x_skip_3, x_skip_4 = feats

        # UNet blocks
        out_block_3 = self.unet_block_3(x_dec=x_skip_4, x_skip=x_skip_3)
        out_block_2 = self.unet_block_2(x_dec=out_block_3, x_skip=x_skip_2)
        out_block_1 = self.unet_block_1(x_dec=out_block_2, x_skip=x_skip_1)

        # detection/segmentatio head
        detections = self.detection_head(out_block_1)
        segmentation = self.segmentation_head(out_block_1)
        return (detections, segmentation)

    def initialize_params(self):
        """ Calling the initialization of model parameters for the decoder and heads """
        self.unet_block_3.apply(self.initialize_weights)
        self.unet_block_2.apply(self.initialize_weights)
        self.unet_block_1.apply(self.initialize_weights)
        self.detection_head.apply(self.initialize_weights)
        self.segmentation_head.apply(self.initialize_weights)
        return

    def initialize_weights(self, m):
        """ Initializing model parameters """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        return


#