"""
Different encoders modules to use

TODO: Use better backbones, such as EdgeNeXt or MobileViT
"""

import torch
import torch.nn as nn
from torchvision import models


class EncoderBackbone(nn.Module):
    """ Abstract class for all Encoder Backbones"""

    def __init__(self, name="", pretrained=True, **kwargs):
        """ Module initializer """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.model = self._build(**kwargs)
        self.check_blocks()

    def forward(self, x):
        """
        Forward pass. We store all output features of the Encoder stages.
          - Note that we do not store the output of the first block
        """
        feats = []
        for i, block in enumerate(self.model):
            x = block(x)
            if i > 0:
                feats.append(x)
        return feats

    def _build(self):
        """ Building encoder backbone pass """
        raise NotImplementedError("Abstract class does not implement _build method")

    def check_blocks(self):
        if self.model is None:
            raise ValueError("Model is not yet initialized...")
        n_blocks = len(self.model)
        if n_blocks != 5:
            raise ValueError(f"Number of blocks is {n_blocks}, but it must be 5...")
        return

    @torch.no_grad()
    def get_feat_size(self, img_size=(480, 640)):
        """ Getting the number of channels at the output of each block """
        img = torch.rand(1, 3, *img_size)
        feats = self.forward(img)
        channels = [f.shape[1] for f in feats]
        spatial_dims = [(f.shape[2], f.shape[3]) for f in feats]
        return channels, spatial_dims


class ResNet18(EncoderBackbone):
    """ Pretrained ResNet18-based encoder backbone """

    NAME = "ResNet18"

    def __init__(self, pretrained=True, **kwargs):
        """ Module initialization """
        super().__init__(name=self.NAME, pretrained=pretrained, **kwargs)

    def _build(self, **kwargs):
        """ Loading pretrained backbone for the encoder part of the model """
        model = models.resnet18(pretrained=self.pretrained, **kwargs)
        init_block = nn.Sequential(*[c for i, c in enumerate(model.children()) if i < 4])
        blocks = [getattr(model, f"layer{i}") for i in range(1, 5)]
        backbone = nn.ModuleList([init_block, *blocks])
        return backbone
    
    
class ResNet34(EncoderBackbone):
    """ Pretrained ResNet34-based encoder backbone """

    NAME = "ResNet34"

    def __init__(self, pretrained=True, **kwargs):
        """ Module initialization """
        super().__init__(name=self.NAME, pretrained=pretrained, **kwargs)

    def _build(self, **kwargs):
        """ Loading pretrained backbone for the encoder part of the model """
        model = models.resnet34(pretrained=self.pretrained, **kwargs)
        init_block = nn.Sequential(*[c for i, c in enumerate(model.children()) if i < 4])
        blocks = [getattr(model, f"layer{i}") for i in range(1, 5)]
        backbone = nn.ModuleList([init_block, *blocks])
        return backbone
    

class ResNet50(EncoderBackbone):
    """ Pretrained ResNet50-based encoder backbone """

    NAME = "ResNet50"

    def __init__(self, pretrained=True, **kwargs):
        """ Module initialization """
        super().__init__(name=self.NAME, pretrained=pretrained, **kwargs)

    def _build(self, **kwargs):
        """ Loading pretrained backbone for the encoder part of the model """
        model = models.resnet50(pretrained=self.pretrained, **kwargs)
        init_block = nn.Sequential(*[c for i, c in enumerate(model.children()) if i < 4])
        blocks = [getattr(model, f"layer{i}") for i in range(1, 5)]
        backbone = nn.ModuleList([init_block, *blocks])
        return backbone


class EfficientnetB0(EncoderBackbone):
    """ Pretrained EfficientNet-B0 backbone """

    NAME = "EfficientNet-B0"

    def __init__(self, pretrained=True, **kwargs):
        """ Module initialization """
        super().__init__(name=self.NAME, pretrained=pretrained, **kwargs)

    def _build(self, **kwargs):
        """ Loading pretrained backbone for the encoder part of the mode"""
        model = models.efficientnet_b0(pretrained=False, **kwargs)
        init_block = model.features[0]
        blocks = [nn.Sequential(model.features[2*i + 1], model.features[2*i + 2]) for i in range(4)]
        backbone = nn.ModuleList([init_block, *blocks])
        return backbone


class MobileNetV3(EncoderBackbone):
    """ Pretrained MobileNetV3-Large backbone """

    NAME = "MobileNetV3"

    def __init__(self, pretrained=True, **kwargs):
        """ Module initialization """
        super().__init__(name=self.NAME, pretrained=pretrained, **kwargs)

    def _build(self, **kwargs):
        """ Loading pretrained backbone for the encoder part of the mode"""
        model = models.mobilenet_v3_large(pretrained=True, **kwargs)
        init_block = model.features[0:2]
        block1 = model.features[2:4]
        block2 = model.features[4:7]
        block3 = model.features[7:13]
        block4 = model.features[13:]
        backbone = nn.ModuleList([init_block, block1, block2, block3, block4])
        return backbone



#
