"""
Base class from which all NimbRoNet models inherit
"""

import sys
import torch
import torch.nn as nn

from models.Blocks import ConvBilinearBlock, conv1x1, conv3x3
from models.Encoders import ResNet18


class NimbroNetBase(nn.Module):
    """
    Base class from which all NimbRoNet models inherit
    """
    
    IMG_SIZE = (540, 960)

    
    def __init__(self, ):
        """ Module initializer """
        super().__init__()
        return

    
    def forward(self, x):
        """ Forward pass """
        raise NotImplementedError("Base class does not implement 'forward' method'")


    def _init_parameters(self):
        """ Initializing model parameters """
        raise NotImplementedError("Base class does not implement '_init_parameters' method'")
    
    
    def _add_padding(self, x, H):
        """
        Adding padding to handle sizes not divisible by 32
        """
        pad_size = int(H / 32. + .5) * 32 - H
        x = nn.functional.pad(x, (0, 0, 0, pad_size))
        return x
    
    def _remove_padding(self, outs, H):
        """
        Removing the extra padding added to handle sizes not divisible by 32
        """
        pad_size = int(H / 32. + .5) * 32 - H
        pad_size_lr, pad_size_hr = int(pad_size / 4.), int(pad_size / 2.)
        if pad_size_hr > 0:    
            outs = [out[:, :, :-pad_size_lr] for out in outs]
        return outs


