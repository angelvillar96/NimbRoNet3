"""
Data Utils
"""

from math import ceil, floor
from scipy.stats import multivariate_normal
import numpy as np
import torch
import torch.nn.functional as F


def unnorm_img(img):
    """ Removing the normalization of the image for visualization purposes """
    numel = len(img.shape)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    if numel == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    imgs = std * img + mean
    imgs = imgs.clamp(0, 1)
    return imgs


def pad_img_target(img, target=None, divisor=32):
    """
    Padding an image and the corresponding target so that the spatial dimensions
    are divisible by a certain integer 'divisor'
    """
    img_size = img.shape[-2:]

    H = ceil(img_size[0] / divisor) * divisor
    pad_H = (ceil((H - img_size[0]) / 2), floor((H - img_size[0]) / 2))

    W = ceil(img_size[1] / divisor) * divisor
    pad_W = (ceil((W - img_size[1]) / 2), floor((W - img_size[1]) / 2))

    pad_img = F.pad(
        input=img,
        pad=(*pad_W, *pad_H),
    )
    if target is not None:
        pad_target = F.pad(
            input=target,
            pad=(int(floor(pad_W[0] / 4)), int(ceil(pad_W[1] / 4)), int(floor(pad_H[0] / 4)), int(ceil(pad_H[1] / 4))),
        )
    else:
        pad_target = None
    return pad_img, pad_target


def get_heatmap(image_size, center_point, variance):
    """
    Generating a heatmap centered at the given point

    Args
    ----
    image_size: tuple/list
        Shape of the annotations. For instance, (H, W)
    center_point: list/tuple
        Center coordinate where to place the heatmap. Given as (x, y)
    variance: integer/float
        Variance of the Gaussian blob1

    Returns:
    --------
    hmap: numpy array
        Array of shape 'image_size' with a Gaussian blob of variance 'variance'
        centered at 'center_point'
    """
    new_height, new_width = int(image_size[0]), int(image_size[1])
    mean = [center_point[1], center_point[0]]
    pos = np.dstack(np.mgrid[0:new_height:1, 0:new_width:1])
    rv = multivariate_normal(mean, cov=variance)
    hmap = variance * rv.pdf(pos).reshape(new_height, new_width)
    return hmap


def fliplr_kpts(keypoints, width, parts_order=None):
    """
    Horizontal flipping keypoints.
    Some of them change from being left to right joints, or vice-versa
    """
    keypoints[:, :, 0] = width - keypoints[:, :, 0] - 1
    if parts_order is not None:
        keypoints = keypoints[parts_order]
    return keypoints


def generate_pose_heatmaps(keypoints, heatmap_size, sigma, heatmaps=None):
    """ 
    Generating a numpy array with heatmaps centered at the keypoint locations
         - https://github.com/AIS-Bonn/HumanoidRobotPoseEstimation/blob/main/src/utils/ops.py

    Args:
    -----
    keypoints: np array
        Array with the keypoints to convert to heatmaps. Shape is (num_kpts, num_robots, 3)
    heatmap_size: tuple/list/array
        Width (W) and height (H) of the output
    sigma: float
        Standard deviation of the gaussian used to generate the heatmaps
        
    Returns:
    --------
    heatmaps: np array
        Heatmaps correcsponding to the keypojnts. Shape is (num_kpts, H, W)
    """
    # generate canvas
    num_keypoints = keypoints.shape[0]
    if heatmaps is None:
        heatmaps = np.zeros((num_keypoints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)

    mu = keypoints[..., :2]
    s = 3 * sigma + 1
    e = 3 * sigma + 2
    # check that each part of the Gaussian is in-bounds
    ul = np.int64(np.floor(mu - s))
    br = np.int64(np.ceil(mu + e))

    # generate 2D Gaussian
    x = np.arange(np.floor(-s), np.ceil(e), dtype=np.float64)
    y = x[:, np.newaxis]
    x0 = y0 = 0.0
    gaussian = 1.0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # usable Gaussian range
    g_lt = np.maximum(0, -ul)
    g_rb = np.minimum(br, heatmap_size) - ul
    # image range
    img_lt = np.maximum(0, ul)
    img_rb = np.minimum(br, heatmap_size)

    # generate heatmaps
    for i in range(num_keypoints):
        for j in range(keypoints.shape[1]):  # number of keypoints
            if keypoints[i, j, 2] == 0 or np.any(ul[i, j] >= heatmap_size) or np.any(br[i, j] < 0):
                continue
            heatmaps[i, img_lt[i, j, 1]:img_rb[i, j, 1], img_lt[i, j, 0]:img_rb[i, j, 0]] = np.maximum(
                heatmaps[i, img_lt[i, j, 1]:img_rb[i, j, 1], img_lt[i, j, 0]:img_rb[i, j, 0]],
                gaussian[g_lt[i, j, 1]:g_rb[i, j, 1], g_lt[i, j, 0]:g_rb[i, j, 0]]
            )
    return heatmaps


def generate_limb_heatmaps(keypoints, limbs, input_size, stride, sigma, thr=1., dist_thr=.015):
    """
    Generating limb heatmaps that connect two related heatmaps.
    This is almost equivalent to a part affinity field.
        - https://github.com/AIS-Bonn/HumanoidRobotPoseEstimation/blob/main/src/utils/ops.py
        
    Args:
    -----
    """
    num_limbs = limbs.shape[0]
    heatmap_size = input_size // stride
    heatmaps = np.zeros((num_limbs, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    grid = np.mgrid[0:input_size[0]:stride[0], 0:input_size[1]:stride[1]] + stride[:, None, None] / 2 - 0.5
    if grid.shape[-1] == 68:  # HACK
       grid = grid[..., :-1] 

    for i in range(num_limbs):
        parts = keypoints[limbs[i]]
        count = np.zeros((heatmap_size[1], heatmap_size[0]), dtype=np.uint32)
        for j in range(keypoints.shape[1]):  # number of keypoints
            src, dst = parts[:, j]
            limb_vec = dst[:2] - src[:2]
            norm = np.linalg.norm(limb_vec)

            if src[2] == 0 or dst[2] == 0 or norm == 0:
                continue

            min_p = np.maximum(np.round((np.minimum(src[:2], dst[:2]) - (thr * stride)) / stride), 0)
            max_p = np.minimum(np.round((np.maximum(src[:2], dst[:2]) + (thr * stride)) / stride), input_size - 1)
            range_x = slice(int(min_p[0]), int(max_p[0]) + 1)
            range_y = slice(int(min_p[1]), int(max_p[1]) + 1)

            min_x = max(int(round((min(src[0], dst[0]) - (thr * stride[0])) / stride[0])), 0)
            max_x = min(int(round((max(src[0], dst[0]) + (thr * stride[0])) / stride[0])), input_size[0] - 1)
            min_y = max(int(round((min(src[1], dst[1]) - (thr * stride[1])) / stride[1])), 0)
            max_y = min(int(round((max(src[1], dst[1]) + (thr * stride[1])) / stride[1])), input_size[1] - 1)
            slice_x = slice(min_x, max_x + 1)
            slice_y = slice(min_y, max_y + 1)

            assert np.array_equal(grid[:, range_x, range_y], grid[:, slice_x, slice_y])

            deta = src[:2][:, None, None] - grid[:, range_x, range_y]
            dist = (limb_vec[0] * deta[1] - deta[0] * limb_vec[1]) / (norm + 1e-6)
            dist = np.abs(dist)
            gauss_dist = np.exp(-(dist - 0.0) ** 2 / (2 * sigma ** 2)).T
            # gauss_dist[gauss_dist <= dist_thr] = 0.01

            mask = gauss_dist > 0
            heatmaps[i, slice_y, slice_x][mask] += gauss_dist[mask]
            count[slice_y, slice_x][mask] += 1

        mask = count > 0
        heatmaps[i][mask] /= count[mask]

    return heatmaps


def get_affine_transform(center, translate, scale, rotate=0, inv=False):
    """
    Computting an affine transfom matrix that implements translation, rotation and scaling
    """
    rotate_rad = np.pi * rotate / 180
    cs, sn = np.cos(rotate_rad), np.sin(rotate_rad)

    # M = T * C * RS * C^-1
    transform = np.zeros((3, 3), dtype=np.float32)
    transform[0, 0] = cs
    transform[1, 1] = cs
    transform[0, 1] = -sn
    transform[1, 0] = sn
    transform[:2, :2] *= scale
    if rotate != 0:
        transform[0, 2] = np.sum(transform[0, :2] * -center)
        transform[1, 2] = np.sum(transform[1, :2] * -center)
        transform[:2, 2] += center * scale
    transform[:2, 2] += translate
    transform[2, 2] = 1

    if inv:
        transform = np.linalg.pinv(np.vstack([transform[:2], [0, 0, 1]]))

    return transform[:2]


def affine_transform(src_pts, trans):
    """ 
    Applying an affine transform 
    """
    src_pts_reshaped = src_pts.reshape(-1, 2)
    dst_pts = np.dot(np.concatenate((src_pts_reshaped, np.ones((src_pts_reshaped.shape[0], 1))), axis=1), trans.T)

    return dst_pts.reshape(src_pts.shape)

#
