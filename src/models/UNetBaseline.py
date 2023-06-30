"""
Baseline Nimbrov2 Unet-based Architecture
   from: https://git.ais.uni-bonn.de/nimbro/nimbro_op/-/blob/master/cv/vision_module/scripts/deep.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

img_size = (480, 640)
cut = 8
lr_cut = 6


def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
            raise "KeyError(f'Idx {idx} is out-of-bounds')"
        yield seq[last:idx]
        last = idx
    yield seq[last:]


def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]


def get_base(f):
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


class SaveFeatures():
    features = None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(UnetBlock, self).__init__()
        up_out = x_out = n_out//2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet34(nn.Module):
    def __init__(self, rn=models.resnet18):
        super(UNet34, self).__init__()
        self.rn = get_base(rn)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 512)
        self.up2 = UnetBlock(512, 128, 512)
        self.up3 = UnetBlock(512, 64, 256)
        self.up6 = nn.ConvTranspose2d(256, 6, 1)
        self.lastActivation = None

    def forward(self, x):
        inp = x
        bx = F.relu(self.rn(x))
        bx = self.up1(bx, self.sfs[3].features)
        bx = self.up2(bx, self.sfs[2].features)
        bx = self.up3(bx, self.sfs[1].features)

        self.lastActivation = x.clone()

        return self.up6(bx)

    def close(self):
        for sf in self.sfs:
            sf.remove()


class UNetModel():
    def __init__(self, model, name='unet'):
        self.model = model
        self.name = name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


#
