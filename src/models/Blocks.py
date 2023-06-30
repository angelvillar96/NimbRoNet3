"""
Basic building blocks for convolutional neural networks

TODO:
 - Try additional UnetBlock that first upsamples and concats and then apply a single convolution.
"""

from math import ceil, floor
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetBlock(nn.Module):
    """
    Basic UNet decoder block containing two convolutions.
      - 1. First ConvLayer processes the channels from the skip connections 
      - 2. Second ConvLayer proecesses and upsamples features maps via a transposed convolution.
    Outputs of both layers are concatenated
    
    Args:
    -----
    in_channels_dec: int
        Number of convolutional channels coming from the decoder path.
    in_channels_skip: int
        Number of convolutional channels coming from the skip connection
    out_channels: int
        Total number of ouput feature maps after the concatenation of both conv layers.
    act: bool
        If True, a ReLU activation function is applied after convolutions
    norm: bool
        If True, batch norm is applied after convolutions
    downsample: tuple/list
        Amout of downsampling to use in the block.
        Default (2, 2) downsamples the feature maps by a factor of 2 both in height and width
    """

    def __init__(self, in_channels_dec, in_channels_skip, out_channels, act=True,
                 norm=True, downsample=(2, 2)):
        """ """
        super().__init__()
        conv_channels = out_channels // 2
        self.conv = nn.Conv2d(
                in_channels=in_channels_skip,
                out_channels=conv_channels,
                kernel_size=1,
                padding=0
            )
        if downsample != (1, 1):
            self.conv_tr = nn.ConvTranspose2d(
                    in_channels=in_channels_dec,
                    out_channels=conv_channels,
                    kernel_size=downsample,
                    stride=downsample
                )
        else:
            self.conv_tr = nn.Conv2d(
                    in_channels=in_channels_dec,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1
                )
        self.bn = nn.BatchNorm2d(out_channels) if norm else None
        self.act = nn.ReLU() if act else None
        return

    def forward(self, x_dec, x_skip):
        """ Forward pass through block """
        y_dec = self.conv_tr(x_dec)
        y_skip = self.conv(x_skip)
        y = torch.cat([y_dec, y_skip], dim=1)

        if self.act is not None:
            y = self.act(y)
        if self.bn is not None:
            y = self.bn(y)
        return y


class TransposedConvBlock(nn.Module):
    """
    Basic transposed-convolutional block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=True, act=True):
        """ Basic transposed convolutional block """
        super().__init__()
        self.modules = nn.Sequential()
        self.modules.add_module("ConvTranspose", nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            )
        )
        if(norm):
            self.block.add_module("Norm", nn.BatchNorm2d(num_features=out_channels))
        if act != "":
            self.block.add_module("Act", nn.ReLU())
        return

    def forward(self, x):
        """ Forward pass through block """
        y = self.module(x)
        return y


class ConvBlock(nn.Module):
    """
    Basic convolutional block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 stride=1, norm=True, act=True, bias=True):
        """ Block initializer """
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv", nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        )
        if(norm):
            self.block.add_module("Norm", nn.BatchNorm2d(num_features=out_channels))
        if act != "":
            self.block.add_module("Act", nn.ReLU())
        return

    def forward(self, x):
        """ Forward pass through block """
        y = self.block(x)
        return y


class DoubleConv(nn.Module):
    """
    Double convolutional block
    """

    def __init__(self, in_ch, out_ch):
        """ Module initializer """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return

    def forward(self, x):
        y = self.conv(x)
        return y


class UpsampleDoubleConv(nn.Module):
    """
    Double convolutional block
    """

    def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=True):
        """ Module initializer """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=scale_factor)
        self.double_conv = DoubleConv(in_channels, out_channels)
        return

    def forward(self, x_dec, x_skip):
        """ Forward pass: Upsampling feature maps, concatenating, and joined processing """
        x_dec = self.up(x_dec)
        diffX = x_dec.size()[2] - x_skip.size()[2]
        diffY = x_dec.size()[3] - x_skip.size()[3]
        x_skip = F.pad(x_skip, (int(diffX / 2), int(diffX / 2), int(diffY / 2), int(diffY / 2)))

        x = torch.cat([x_dec, x_skip], dim=1)
        y = self.double_conv(x)
        return y


class BilinearUpsampleConv(nn.Module):
    """
    Double convolutional block
    """

    def __init__(self, in_channels, out_channels, act=True, norm=True, scale_factor=2):
        """ Module initializer """
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        self.conv_block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                act=act,
                bias=False
            )
        return

    def forward(self, x_dec, x_skip):
        """
        Forward pass: Upsampling feature maps, concatenating, and joined processing
        """
        x_dec = self.up(x_dec)

        # for weird odd sizes we might need to pad the tensors
        # diffX = x_dec.size()[2] - x_skip.size()[2]
        # diffY = x_dec.size()[3] - x_skip.size()[3]
        # x_pad = (diffX // 2, diffX // 2) if diffX % 2 == 0 else (int(ceil(diffX / 2)), int(floor(diffX / 2)))
        # y_pad = (diffY // 2, diffY // 2) if diffY % 2 == 0 else (int(ceil(diffY / 2)), int(floor(diffY / 2)))
        # x_skip = F.pad(x_skip, (*y_pad, *x_pad))
        # x_skip = F.pad(x_skip, (int(diffX / 2), int(diffX / 2), int(diffY / 2), int(diffY / 2)))

        x = torch.cat([x_dec, x_skip], dim=1)
        y = self.conv_block(x)
        return y


class ConvBilinearBlock(nn.Module):
    """
    Convolutional block that processes feature maps with a convolution and upsamples them
    with bilinear interpolation
    """

    def __init__(self, in_channels, out_channels, act=True, norm=True, scale_factor=2):
        """ Module initializer """
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        self.conv_block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                act=act,
                bias=False
            )
        return

    def forward(self, x):
        """
        Forward pass: Processing features with a convolution and upsampling via bilinear interpolation.
        """
        x = self.conv_block(x)
        y = self.up(x)
        return y


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """ Default 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """ Default 3x3 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


#
