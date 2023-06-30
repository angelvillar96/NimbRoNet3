"""
Some model utils, mainly to benchmark models in terms of performance
or to add/remove/exchange model heads.
"""

from time import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

from lib.logger import print_, log_info


def custom_init_weights(model):
    """ Initializing model parameters """
    for m in model.decoder.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 1e-8)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 1e-8)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return


def count_model_params(model, verbose=False):
    """Counting number of learnable parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print_(f"  --> Number of learnable parameters: {num_params}")
    return num_params


def compute_flops(model, dummy_input, verbose=True, detailed=False):
    """ Computing the number of activations and flops in a forward pass """
    func = print_ if verbose else log_info

    # benchmarking
    fca = FlopCountAnalysis(model, dummy_input)
    act = ActivationCountAnalysis(model, dummy_input)
    if detailed:
        fcs = flop_count_str(fca)
        func(fcs)
    total_flops = fca.total()
    total_act = act.total()

    # logging
    func("  --> Number of FLOPS in a forward pass:")
    func(f"   --> FLOPS = {total_flops}")
    func(f"    --> FLOPS = {round(total_flops / 1e9, 3)}G")
    func("  --> Number of activations in a forward pass:")
    func(f"    --> Activations = {total_act}")
    func(f"    --> Activations = {round(total_act / 1e6, 3)}M")
    return total_flops, total_act


def compute_throughput(model, dataset, device, half_precission=False, num_imgs=500,
                       use_tqdm=True, verbose=True):
    """ Computing the throughput of a model in imgs/s """
    times = []
    N = min(num_imgs, len(dataset))
    iterator = tqdm(range(N)) if use_tqdm else range(N)
    model = model.to(device)
    if half_precission:
        model = model.half()

    # benchmarking by averaging over N images
    for i in iterator:
        img = dataset[i][0].unsqueeze(0).to(device)
        if half_precission:
            img = img.half()
        torch.cuda.synchronize()
        start = time()
        _ = model(img)
        torch.cuda.synchronize()
        times.append(time() - start)
    avg_time_per_img = np.mean(times)
    throughput = 1 / avg_time_per_img

    # logging
    func = print_ if verbose else log_info
    func(f"  --> Average time per image: {round(avg_time_per_img, 3)}s")
    func(f"  --> Throughput: {round(throughput)} imgs/s")
    return throughput, avg_time_per_img


def freeze_params(model):
    """Freezing model params to avoid updates in backward pass"""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_params(model):
    """Unfreezing model params to allow for updates during backward pass"""
    for param in model.parameters():
        param.requires_grad = True
    return model


def quantize_model(model, dtype=torch.qint8):
    """ Quantizing the parameters of the model """
    qlayers = {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d}
    model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=qlayers,
            dtype=dtype
        )
    return model


def split_model_parameters(model):
    """ Splitting model paramters into encoder and rest """
    encoder_params = list(filter(lambda kv: "rn." in kv[0], model.named_parameters()))
    model_params = list(filter(lambda kv: "rn." not in kv[0], model.named_parameters()))
    encoder_params = [p[1] for p in encoder_params]
    model_params = [p[1] for p in model_params]
    return model_params, encoder_params


def size_checker(img_size, divisor=2, times=2):
    """ Checks that the spatial dimensions can be divided by 2 """
    H, W = img_size
    h_is_good = H / divisor ** times == int(H / divisor ** times)
    w_is_good = W / divisor ** times == int(W / divisor ** times)
    if h_is_good is not True:
        raise Exception(f"Image height {H} is not divisable by {divisor} {times} times")
    if w_is_good is not True:
        raise Exception(f"Image width {W} is not divisable by {divisor} {times} times")
    return


class BBoxTransform(nn.Module):
    """
    Adapted from https://github.com/yhenon/pytorch-retinanet
    """

    def __init__(self, mean=None, std=None):
        """ """
        super().__init__()
        self.mean = torch.tensor([0, 0, 0, 0]) if mean is None else mean
        self.std = torch.tensor([0.1, 0.1, 0.2, 0.2]) if std is None else std
        return

    def forward(self, boxes, deltas):
        """ Unclear what this is doing """
        # computing height. width and center from bbox coords
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
        return pred_boxes


def clip_boxes(boxes, img):
    """ Enforcing BBox coordinates do not overflow the image boundaries """
    B, C, H, W = img.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=W)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=H)
    return boxes


#
