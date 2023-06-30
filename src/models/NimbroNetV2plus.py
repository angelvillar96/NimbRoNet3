"""
Implementation of the NimbroNetV3 model for real-time perception on the soccer field.
The model simultaneously predicts:
  - Detections of ball, obstacle and goal posts
  - Segmentation of the field
"""

import torch
import torch.nn as nn

from models.NimbroNetBase import NimbroNetBase
from models.Blocks import ConvBilinearBlock, conv1x1, conv3x3
from models.Encoders import ResNet18


class NimbroNetV2plus(NimbroNetBase):
    """
    Implementation of the NimbroNetV3 model for real-time perception on the soccer field.
    The model simultaneously predicts:
        - Detections of ball, obstacle and goal posts
        - Segmentation of the field

    Args:
    -----
    encoder: torch.nn.Module
        Module used as (possibly pretrained) encoder in our network
    """
    
    IMG_SIZE = (540, 960)

    
    def __init__(self, det_classes=3, seg_classes=3, base_channels=128,
                 encoder=ResNet18(pretrained=True), **kwargs):
        """ Module initializer """
        super().__init__()
        self.det_classes = det_classes
        self.seg_classes = seg_classes
        self.base_channels = base_channels
        self.img_size = kwargs.get("img_size", NimbroNetV2plus.IMG_SIZE)
        self.encoder = encoder

        # getting spatial sizes and channels in the lateral connections
        self.img_size = kwargs.get("img_size") if "img_size" in kwargs else NimbroNetV2plus.IMG_SIZE
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
            in_channels=base_channels * 2,
            out_channels=base_channels,
            act=True,
            norm=True,
            scale_factor=scale_factor
        )

        # output heads
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
        
        # third (and final) decoder block, lateral connection, and high-resolution output
        out_dec_1 = self.decoder_block_1(out_cat_2)
        out_lateral_1 = self.lateral_block_1(feats[-4])
        out_cat_1 = torch.cat([out_dec_1, out_lateral_1], dim=1)

        # outputs heads        
        detections = self.detection_head(out_cat_1)
        segmentation = self.segmentation_head(out_cat_1)
        
        # removing extra padding from all model outputs
        outs = self._remove_padding(
            outs=[detections, segmentation],
            H=H
        )
        detections, segmentation = outs[0], outs[1]
        
        # making output dictionary
        out_model = {
            "detections": detections,
            "segmentation": segmentation,
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
            self.segmentation_head, self.detection_head
        ]:
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return
    
