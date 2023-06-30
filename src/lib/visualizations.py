"""
Data visualization
"""

import itertools
from math import ceil
import numpy as np
from PIL import Image, ImageDraw
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import cv2

import data.data_utils as data_utils
from lib.postprocessing import filter_heatmaps, get_kpt_from_heatmaps


KPT_COLORS = ["purple", "white", "cyan", "blue", "green", "orange"]
LIMB_COLORS =["purple", "cyan", "blue", "green", "orange"]


def visualize_sequence(sequence, savepath=None, add_title=True, add_axis=False, n_cols=8,
                       size=3, n_channels=3, titles=None, **kwargs):
    """ Visualizing a sequence of images/frames """
    # initializing grid
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    if "fig" in kwargs and "ax" in kwargs:
        fig, ax = kwargs.pop("fig"), kwargs.pop("ax")
    else:
        fig, ax = plt.subplots(n_rows, n_cols)
        # adding super-title and resizing
        figsize = kwargs.pop("figsize", (size*n_cols, size*n_rows))
        fig.set_size_inches(*figsize)
        fig.suptitle(kwargs.pop("suptitle", ""))

    # plotting all frames from the sequence
    ims = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if len(ax.shape) == 2 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach()
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title)
            else:
                a.set_title(f"Frame {i}")

    # removing axis
    if(not add_axis):
        for i in range(n_cols * n_rows):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if len(ax.shape) == 2 else ax[col]
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax, ims


def visualize_img_targets(imgs, targets, n_cols=8, savepath=None, unnorm_img=True, title1=None, title2=None):
    """ Visualizing an image with orig. imgs in one row, and corresponding blobs on the row below """
    n_imgs = imgs.shape[0]
    n_rows = ceil(n_imgs / n_cols) * 2
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(h=n_rows*4, w=n_cols*4)

    titles = [f"Image {i+1}" for i in range(n_imgs)] if title1 is None else title1
    if unnorm_img:
        imgs = data_utils.unnorm_img(imgs)
    _ = visualize_sequence(
            sequence=imgs,
            savepath=None,
            add_title=True,
            titles=titles,
            n_cols=n_cols,
            fig=fig,
            ax=ax[:n_rows//2],
            add_axis=True
        )
    titles = [f"Target {i+1}" for i in range(n_imgs)] if title2 is None else title2
    _ = visualize_sequence(
            sequence=targets,
            savepath=None,
            add_title=True,
            titles=titles,
            n_cols=n_cols,
            fig=fig,
            ax=ax[n_rows//2:],
            add_axis=True
        )

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax, imgs


def visualize_img_target_pred(imgs, targets, preds, n_cols=8, savepath=None, unnorm_img=True, **kwargs):
    """ Visualizing an image with orig. imgs in one row, and corresponding blobs on the row below """
    n_imgs = imgs.shape[0]
    n_rows = ceil(n_imgs / n_cols)
    fig, ax = plt.subplots(n_rows * 3, n_cols)
    fig.set_size_inches(h=n_rows * 3 * 4, w=n_cols * 4)

    titles = kwargs.get("titles1", [f"Image {i+1}" for i in range(n_imgs)])
    if unnorm_img:
        imgs = data_utils.unnorm_img(imgs)

    _ = visualize_sequence(
            sequence=imgs,
            savepath=None,
            add_title=True,
            titles=titles,
            n_cols=n_cols,
            fig=fig,
            ax=ax[:n_rows],
            add_axis=True
        )
    titles = kwargs.get("titles2", [f"Target {i+1}" for i in range(n_imgs)])
    _ = visualize_sequence(
            sequence=targets,
            savepath=None,
            add_title=True,
            titles=titles,
            n_cols=n_cols,
            fig=fig,
            ax=ax[n_rows:n_rows*2],
            add_axis=True,
        )
    titles = kwargs.get("titles3", [f"Pred {i+1}" for i in range(n_imgs)])
    _ = visualize_sequence(
            sequence=preds,
            savepath=None,
            add_title=True,
            titles=titles,
            n_cols=n_cols,
            fig=fig,
            ax=ax[n_rows*2:],
            add_axis=True,
        )

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax, imgs


def filter_curve(f, k=5):
    """ Using a 1D low-pass filter to smooth a loss-curve """
    kernel = np.ones(k)/k
    f = np.concatenate([f[:k//2], f, f[-k//2:]])
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[k//2:-k//2]
    return smooth_f


def visualize_img_detections(imgs, targets, preds, n_cols=8, max_peaks=8, peak_thr=0.8, savepath=None):
    """ Visualizing peak-picked detections overlayed on image """
    imgs_disp = data_utils.unnorm_img(imgs)
    filtered_preds = filter_heatmaps(preds)
    filtered_targets = filter_heatmaps(targets)
    gt_peaks, gt_peak_vals = get_kpt_from_heatmaps(
            filtered_targets,
            kernel_size=5,
            max_peaks=max_peaks,
            peak_thr=peak_thr
        )
    pred_peaks, pred_peak_vals = get_kpt_from_heatmaps(
            filtered_preds,
            kernel_size=5,
            max_peaks=max_peaks,
            peak_thr=peak_thr
        )

    gt_disp = overlay_dets(img=imgs_disp, peaks=gt_peaks, vals=gt_peak_vals, thr=peak_thr)
    pred_disp = overlay_dets(img=imgs_disp, peaks=pred_peaks, vals=pred_peak_vals, thr=peak_thr)

    fig, ax, im = visualize_img_target_pred(
            imgs=imgs_disp,
            targets=gt_disp,
            preds=pred_disp,
            savepath=savepath,
            n_cols=n_cols,
            unnorm_img=False
        )
    return fig, ax, im


def overlay_dets(img, peaks, vals, thr=0.8):
    """ Overlaying circles depicting predictions upon image """
    H, W = img.shape[-2], img.shape[-1]
    overlay_img = img.clone()
    overlay_img = F.resize(overlay_img, (H // 4, W // 4))
    overlay_img = (overlay_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    imgs = []
    colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255)]

    for b in range(peaks.shape[0]):
        img = overlay_img[b]
        for c in range(peaks.shape[1]):
            for d in range(peaks.shape[2]):
                if vals[b, c, d] < 0.8:
                    continue
                x, y = peaks[b, c, d].cpu().numpy()
                cv2.circle(img, center=(x, y), radius=3, color=colors[c], thickness=1)
        imgs.append(img)

    imgs = torch.stack([torch.from_numpy(img) for img in imgs]).permute(0, 3, 1, 2)
    return imgs


def visualize_bboxes(imgs, targets, preds, dataset, n_cols=8, savepath=None):
    """ Visualizing peak-picked detections overlayed on image """
    imgs_disp = (data_utils.unnorm_img(imgs) * 255).to(torch.uint8)
    all_targets, all_preds = [], []
    for ID in range(imgs_disp.shape[0]):
        target_labels = [dataset.LBL_TO_CLASS[lbl] for lbl in targets[ID, :, -1].long().tolist()]
        target_colors = [dataset.LBL_TO_COLOR[lbl] for lbl in targets[ID, :, -1].long().tolist()]
        cur_target = draw_bounding_boxes(imgs_disp[ID], boxes=targets[ID, :, :4].long(),
                                         labels=target_labels, width=5, colors=target_colors)
        all_targets.append(cur_target)
        pred_labels = [dataset.LBL_TO_CLASS[lbl] for lbl in preds[ID, :, -1].long().tolist()]
        pred_colors = [dataset.LBL_TO_COLOR[lbl] for lbl in preds[ID, :, -1].long().tolist()]
        cur_pred = draw_bounding_boxes(imgs_disp[ID], boxes=targets[ID, :, :4].long(),
                                       labels=pred_labels, width=5, colors=pred_colors)
        all_preds.append(cur_pred)
    all_targets, all_preds = torch.stack(all_targets), torch.stack(all_preds)

    fig, ax, im = visualize_img_targets(
            imgs=all_targets,
            targets=all_preds,
            savepath=savepath,
            n_cols=n_cols,
            unnorm_img=False,
            title1=[f"GT Det {i+1}" for i in range(all_targets.shape[0])],
            title2=[f"Pred Det {i+1}" for i in range(all_preds.shape[0])]
        )

    return fig, ax, im



def plot_confusion_matrix(confusion_matrix, classes, normalize=False, cmap=plt.cm.Blues, savepath=None):
    """ Plottign and saving a confusion matrix """
    FONTSIZE_LARGE = 30
    FONTSIZE_SMALL = 18
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('classic')

    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=1, keepdims=True)
    else:
        confusion_matrix = confusion_matrix.int()

    plt.figure(figsize=(len(classes) * 3, len(classes) * 3))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix", fontsize=FONTSIZE_LARGE)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=FONTSIZE_SMALL)
    plt.yticks(tick_marks, classes, fontsize=FONTSIZE_SMALL)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    H, W = confusion_matrix.shape
    for i, j in itertools.product(range(H), range(W)):
        plt.text(
            x=j,
            y=i,
            s=format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black",
            fontsize=FONTSIZE_LARGE
        )
    plt.xlabel('Predicted Labes', fontsize=FONTSIZE_LARGE)
    plt.ylabel('Ground-Truth label', fontsize=FONTSIZE_LARGE)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return


def overlay_pose_heatmaps(img, heatmaps):
    """
    Overlaying pose heatmpas on top of an image
    
    Args:
    -----
    img: torch Tensor
        Image to overlay heatmaps upon. Shape is (3, H, W)
    heatmaps: torch Tensor
        Heatmaps to overlay on top of the image. Shape is (num_keypoints, H, W)
        
    Returns:
    --------
    out_img: torch tensor
        Image with the heatmaps overlayed upon. Shape is (3, H, W)
    single_heat_img: torch tensor
        Image with only the heatmaps of a single keypoint overlayed upon. Shape is (num_keypoints, 3, H, W)
    """
    
    h_factor = img.shape[-2] / heatmaps.shape[-2]
    w_factor = img.shape[-1] / heatmaps.shape[-1]
    heatmaps = nn.functional.interpolate(heatmaps.unsqueeze(1), scale_factor=(h_factor, w_factor))[:, 0]
    num_kpts, H, W = heatmaps.shape
    
    heatmaps = (heatmaps * 255).clamp(0, 255).byte().cpu().numpy()
    img = (img * 255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    
    single_heat_img = np.zeros((num_kpts, H, W, 3), dtype=np.uint8)
    for j in range(num_kpts):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = (colored_heatmap * 0.8 + img * 0.4).clip(0, 255)
        single_heat_img[j] = image_fused[:, :, np.array([2, 1, 0])]

    single_heat_img = torch.from_numpy(single_heat_img).permute(0, 3, 1, 2)
    out_img = torch.max(single_heat_img, dim=0)[0]
    return out_img, single_heat_img


def visualize_pose_predictions(imgs, targets, preds, n_cols=6):
    """
    Visualizing the pose predictions, including the high- and low-resolution heatmaps,
    as well as the limbs in low resolution.
    """
    n_rows = 4
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(h=n_rows * 3, w=n_cols * 3)

    imgs = data_utils.unnorm_img(imgs.cpu().detach())
    pose_heatmaps_lr = torch.stack(
        [overlay_pose_heatmaps(imgs[i], preds["heatmaps_lr"][i, :-1].cpu().detach())[0] for i in range(n_cols)]
    )
    pose_heatmaps_hr = torch.stack(
        [overlay_pose_heatmaps(imgs[i], preds["heatmaps_hr"][i, :-1].cpu().detach())[0] for i in range(n_cols)]
    )
    pose_limbs_lr = torch.stack(
        [overlay_pose_heatmaps(imgs[i], preds["limbs_lr"][i].cpu().detach())[0] for i in range(n_cols)]
    )
    titles = [f"Img. {i+1}" for i in range(n_cols)]
    _ = visualize_sequence(sequence=imgs[:n_cols], n_cols=n_cols, fig=fig, ax=ax[0], titles=titles)
    titles = [f"Hmap. LR {i+1}" for i in range(n_cols)]
    _ = visualize_sequence(sequence=pose_heatmaps_lr, n_cols=n_cols, fig=fig,ax=ax[1], titles=titles)
    titles = [f"Hmap. HR {i+1}" for i in range(n_cols)]
    _ = visualize_sequence(sequence=pose_heatmaps_hr, n_cols=n_cols, fig=fig,ax=ax[2], titles=titles)
    titles = [f"Limb. LR {i+1}" for i in range(n_cols)]
    _ = visualize_sequence(sequence=pose_limbs_lr, n_cols=n_cols, fig=fig,ax=ax[3], titles=titles)
    return fig, ax, None



def visualize_pose(image, keypoints, connectivity, kpt_colors=None, limb_colors=None, radius=10, width=8):
    """
    Displaying a pose on top of an image
    
    Args:
    -----
    image: torch tensor
        Image in uInt8 format. Shape is (3, H, W)
    keypoints: torch tensor
        Poses to display. Shape is (num_instances, num_kpts, 3)
    connectivity: list
        Limb connectivity between keypoints
    """
    # processing keypoint colors
    num_keypoints = keypoints.shape[1]
    if kpt_colors is None:
        kpt_colors = KPT_COLORS
    elif isinstance(kpt_colors, (str)):
        kpt_colors = [kpt_colors] * num_keypoints
    elif len(kpt_colors) == 1:
        kpt_colors = kpt_colors * num_keypoints
    else:
        raise ValueError(f"Wrong format for keypoint colors {kpt_colors}...")
    
    # processing limb colors
    num_limbs = len(connectivity)
    if limb_colors is None:
        limb_colors = LIMB_COLORS
    elif isinstance(limb_colors, (str)):
        limb_colors = [limb_colors] * num_limbs
    elif len(kpt_colors) == 1:
        limb_colors = limb_colors * num_limbs
    else:
        raise ValueError(f"Wrong format for limb colors {limb_colors}...")
    
    
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            if kpt[0] < 1 and kpt[1] < 1:
                continue
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=kpt_colors[inst_id], outline=None, width=0)

        if connectivity:
            for limb_id, connection in enumerate(connectivity):
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]
                if (start_pt_x < 1 and start_pt_y < 1) or (end_pt_x < 1 and end_pt_y < 1):
                    continue
                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                    fill=limb_colors[limb_id]
                )

    out_img = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
    return out_img


def visualize_poses(images, keypoints, connectivity):
    """ """
    images = (images* 255).to(torch.uint8)
    poses = torch.from_numpy(keypoints[:, :, :, :2])
    connectivity = connectivity.tolist()
    
    out_imgs = []
    for img, pose in zip(images, poses):
        out_img = visualize_pose(img, pose, connectivity)
        out_imgs.append(out_img)
    out_imgs = torch.stack(out_imgs)
    return out_imgs

#
