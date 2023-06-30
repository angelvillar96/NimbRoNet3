"""
Postprocessing operations used to process the detection and 
segmentation outputs of the model.
These operations include:
  - filter_heatmaps: Removing too small heatmaps with very low amplitude
  - get_kpt_from_heatmaps: Using NMS to pick the maximum values from a heatmap
  - kpt_distance_nms: Supressing keypoints that are too close
  - threshold_heatmaps: Binary thresholding of map
  
  - TODO: Pose stuff
"""

import numpy as np
import torch
import torch.nn.functional as F


def filter_heatmaps(heatmaps, kernel_size=5, heatmap_thr=0.1):
    """
    Filtering detected hearmaps with morphological operations

    Args:
    -----
    kernel_size: integer
        Size of the window used for morphological operations: erosion and dilation
    heatmap_thr: float
        Magnitude filter. Elements below this magnitude are removed
    """
    B, C, H, W = heatmaps.shape

    # morphological operations for removing small blobs that do not fulfill minimum support
    heatmaps_pad = F.pad(
            input=heatmaps,
            pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2),
            mode="constant"
        )
    eroded_hmaps = (-1 * F.max_pool2d(
        input=heatmaps_pad * -1,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1)
    )).clamp(0, 1)
    eroded_hmaps_pad = F.pad(
            input=eroded_hmaps,
            pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2),
            mode="reflect"
        )
    dilated_hmaps = F.max_pool2d(
            input=eroded_hmaps_pad,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1)
        ).clamp(0, 1)
    mask = dilated_hmaps > heatmap_thr
    heatmaps = heatmaps * mask

    return heatmaps


def get_kpt_from_heatmaps(heatmaps, kernel_size=5, max_peaks=8, peak_thr=0.8):
    """
    Getting peak coordinates from heatmaps
        Code inspired by: https://github.com/princeton-vl/pose-ae-train/blob/master/utils/group.py

    Args:
    -----
    heatmaps: torch Tensor
        Heatmaps detected by the model. Shape is (B, 3, H, W)
    kernel_size: int
        Size of the non-maximum suppresion kernel
    max_peaks: int
        Max. number of detections in the image
    peak_thr: float
        Minimum confidence for the detections

    Returns:
    --------
    peaks: torch.Tensor
        Detected peaks. Shape is (B, C, num_dets, 2)
    peak_vals: torch.Tensor
        Magnitued of the detected peaks. Shape is (B, C, num_dets)
    """
    B, C, H, W = heatmaps.shape

    # non-maximum supression
    pad_size = (kernel_size - 1) // 2
    max_vals = F.max_pool2d(heatmaps, kernel_size, stride=1, padding=pad_size)
    max_vals = torch.eq(max_vals, heatmaps).float()
    max_vals = max_vals * heatmaps

    # peak-picking and thresholding
    max_vals = max_vals.view(B, C, -1)
    peak_vals, peak_idx = torch.topk(max_vals, max_peaks, dim=-1, sorted=True)
    peak_idx = peak_idx * (peak_vals >= peak_thr)

    # back to image-coords
    x, y = peak_idx % W, torch.div(peak_idx, W, rounding_mode='trunc')
    peaks = torch.stack((x, y), dim=-1)
    peaks[peak_vals < peak_thr, :] = torch.tensor([-1, -1])

    return peaks, peak_vals


def kpt_distance_nms(kpts, dist_thr=9):
    """
    Supressing maxima within a certain distance

    Args:
    -----
    kpts: torch.Tensor
        Detected peaks sorted by score (max. score first). Shape is (B, C, num_dets, 2)
    thr: float
        Distance threshold for NMS. Analogous to IoU threshold for bounding-boxes
    """
    B, C, N, _ = kpts.shape
    kpts = kpts.view(B * C, N, -1).float()

    # filtering keypoints that do not fulfill the distance condition
    for i in range(N):
        pairwise_distances = torch.cdist(kpts, kpts).triu()
        ids = (0 < pairwise_distances) * (pairwise_distances < dist_thr)
        kpts[ids[:, i]] = torch.ones((1, 2)) * -1

    kpts = kpts.view(B, C, N, -1)
    return kpts


def threshold_heatmaps(heatmaps, filter=False, heatmap_thr=0.1, kernel_size=5):
    """
    Binary thresholding of heatmaps

    Args:
    -----
    heatmaps: torch Tensor
        Detection heatmaps predicted by the model
    filter: bool
        If True, filter_heatmap is applied
    heatmap_thr: float
        Magnitude filter. Elements below this magnitude are set to zero, rest to one
    """
    if filter:
        heatmaps = filter_heatmaps(heatmaps, kernel_size=kernel_size, heatmap_thr=heatmap_thr)
    thr_hmaps = heatmaps.clone()
    thr_hmaps[heatmaps >= heatmap_thr] = 1.
    thr_hmaps[heatmaps < heatmap_thr] = 0.
    return thr_hmaps


@torch.no_grad()
def group_poses(heatmaps, limb_heatmaps, limbs, cfg=None, nms_kernel=3, max_num_dets=8, det_thr=.1):
    """ 
    Grouping heatmaps into robot poses by solving a greedy matching algorithm that integrates 
    over the predicted libs.
    Keypoints get assigned to the same pose if the integration yields a high value.
    
    Args:
    -----
    heatmaps: torch Tensor
        Heatmaps depicting the location of robot joints. Shape is (B, N_kpts, H, W)
    limb_heatmaps: torch Tensor
        Limbs depicting the connectionsn between related robot joints. Shape is (B, N_limbs, H, W)
    limbs: np array
        Array containing the indices of the keypoints that are connected by a limb.
        Shape is (N_limbs, 2), where thr 2 correspods to src_kpt_id, and dst_kpt_id .
    cfg: dict or None
        Configurations used for the matching. Can be left as None
    nms_kernel: int
        Size of the kernel used for non-maximum suppresion when extracting the peak pixel
        coordinates from the heatmaps.
    max_num_dets: int
        Maximum number of poses (robots) in an image
    det_thr: float
        Values below this value wont be considered as peaks.
    
    Returns:
    --------
    results: np array 
        Robot pose results after association of keypoints to a robot.
        Shape is (B, max_n_robots, n_kpts, 3), where 3 is (x, y, value)
    dets: np array
        Keypoint detections obtained from the heatmaps. They are not associated to robots, but#
        are the raw peaks instead.
        Shape is (B, n_kpts, max_n_robots, 2)
    """
    if cfg is None:
        cfg = {}
    num_limb_sampled_points = cfg.get("NUM_MIDPOINTS", 20)
    paf_thr = cfg.get("THRESHOLD", 0.05)
    ignore_few_parts = cfg.get("IGNORE_FEW_PARTS", False)
    connection_ratio = cfg.get("CONNECTION_RATIO", 0.8)
    len_rate = cfg.get("LENGTH_RATE", 16)
    connection_tol = cfg.get("CONNECTION_TOLERANCE", 0.7)
    delete_shared_parts = cfg.get("DELETE_SHARED_PARTS", False)
    min_num_connected_parts = cfg.get("MIN_NUM_CONNECTED_PARTS", 3)
    min_mean_score = cfg.get("MIN_MEAN_SCORE", 0.2)

    # extracting keypoint locations from heatmaps    
    dets, vals = get_kpt_from_heatmaps(
        heatmaps=heatmaps,
        kernel_size=nms_kernel,
        max_peaks=max_num_dets,
        peak_thr=det_thr
    )
    
    limb_heatmaps = limb_heatmaps.cpu().numpy()
    H, W = limb_heatmaps.shape[-2], limb_heatmaps.shape[-2]
    N_imgs, N_kpts = dets.shape[0], dets.shape[1]
    
    outputs = np.concatenate((dets, vals[..., np.newaxis]), axis=3)  # (N_imgs, N_kpts, (H, W, val))
    results = np.zeros((N_imgs, max_num_dets, N_kpts, 3), dtype=np.float64)

    # iterating over all images
    for i in range(N_imgs):
        connected_limbs = []
        robot_to_part_assoc = []

        # iterating over all limbs
        for j in range(limbs.shape[0]):
            parts_src, parts_dst = outputs[i, limbs[j]]
            part_src_type, part_dst_type = limbs[j]
            vis_parts_src, vis_parts_dst = np.any(parts_src, axis=1), np.any(parts_dst, axis=1)
            num_parts_src, num_parts_dst = vis_parts_src.sum(), vis_parts_dst.sum()

            # iterating over pairs of joints belonging to the start or end of the particular limb in order 
            # to find candidate pairs of joints 
            if num_parts_src > 0 and num_parts_dst > 0:
                candidates = []
                for k, src in enumerate(parts_src):
                    if not vis_parts_src[k]:
                        continue
                    for l, dst in enumerate(parts_dst):
                        if not vis_parts_dst[l]:
                            continue
                        
                        # creating limb line, and sampling intermediate points for integration
                        limb_dir = dst[:2] - src[:2]
                        limb_dist = np.sqrt(np.sum(limb_dir ** 2))
                        if limb_dist == 0:
                            continue
                        num_midpts = min(int(np.round(limb_dist + 1)), num_limb_sampled_points)

                        # sampling points on the limb and computing the integration score
                        limb_midpts_coords = np.empty((2, num_midpts), dtype=np.int64)
                        limb_midpts_coords[0] = np.round(np.linspace(src[0], dst[0], num=num_midpts))
                        limb_midpts_coords[1] = np.round(np.linspace(src[1], dst[1], num=num_midpts))
                        score_midpts = limb_heatmaps[i, j, limb_midpts_coords[1], limb_midpts_coords[0]]
                        long_dist_penalty = min(0.5 * H / limb_dist - 1, 0)
                        connection_score = score_midpts.mean() + long_dist_penalty

                        # if the score is high fir many points, we keep this pairing as a candiate
                        criterion1 = np.count_nonzero(score_midpts > paf_thr) >= (connection_ratio * num_midpts)
                        criterion2 = (connection_score > 0)
                        if criterion1 and criterion2:
                            match_score = 0.5 * connection_score + 0.25 * src[2] + 0.25 * dst[2]
                            # candidates.append([k, l, connection_score, limb_dist, match_score])
                            candidates.append({
                                "src": k,
                                "dst": l,
                                "connection_score": connection_score,
                                "limb_dist": limb_dist,
                                "match_score": match_score
                            })

                candidates = sorted(candidates, key=lambda x: x["match_score"], reverse=True)
                max_connections = min(num_parts_src, num_parts_dst)
                connections = np.empty((0, 4), dtype=np.float64)

                for can in candidates:
                    if can["src"] not in connections[:, 0] and can["dst"] not in connections[:, 1]:
                        connections = np.vstack(
                            (connections, [can["src"], can["dst"], can["connection_score"], can["limb_dist"]])
                        )
                        if len(connections) >= max_connections:
                            break
                connected_limbs.append(connections)
            else:
                connected_limbs.append([])

            # given the candiate connections, associating them to a robot
            for limb_info in connected_limbs[j]:
                robot_assoc_idx = []
                # finding if joints from the limb is already assigned to one or more robots robot
                for robot_id, robot_limbs in enumerate(robot_to_part_assoc):
                    if robot_limbs[part_src_type, 0] == limb_info[0] or robot_limbs[part_dst_type, 0] == limb_info[1]:
                        robot_assoc_idx.append(robot_id)

                # only one robot assigned, so we know this new limb belongs to him
                if len(robot_assoc_idx) == 1:
                    robot_limbs = robot_to_part_assoc[robot_assoc_idx[0]]
                    if robot_limbs[part_dst_type, 0] != limb_info[1]:
                        robot_limbs[part_dst_type] = limb_info[[1, 2]]
                        robot_limbs[-2, 0] += vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                        robot_limbs[-1, 0] += 1
                        robot_limbs[-1, 1] = max(limb_info[-1], robot_limbs[-1, 1])
                    
                # each joint from the new limb belongs to a new robot.
                elif len(robot_assoc_idx) == 2:
                    robot1_limbs = robot_to_part_assoc[robot_assoc_idx[0]]
                    robot2_limbs = robot_to_part_assoc[robot_assoc_idx[1]]
                    membership1 = (robot1_limbs[:-2, 0] >= 0)
                    membership2 = (robot2_limbs[:-2, 0] >= 0)
                    membership = (membership1 & membership2)
                    if not np.any(membership):
                        min_limb1 = np.min(robot1_limbs[:-2, 1][membership1])
                        min_limb2 = np.min(robot2_limbs[:-2, 1][membership2])
                        min_tol = min(min_limb1, min_limb2)  # min confidence
                        if limb_info[2] >= (connection_tol * min_tol) and limb_info[-1] < (robot1_limbs[-1, 1] * len_rate):
                            robot1_limbs[:-2] += robot2_limbs[:-2] + 1
                            robot1_limbs[-2, 0] += robot2_limbs[-2, 0] + limb_info[2]
                            robot1_limbs[-1, 0] += robot2_limbs[-1, 0]
                            robot1_limbs[-1, 1] = max(limb_info[-1], robot1_limbs[-1, 1])
                            robot_to_part_assoc.pop(robot_assoc_idx[1])
                    else:
                        if delete_shared_parts:
                            if limb_info[0] in robot1_limbs[:-2, 0]:
                                conn1_idx = int(np.where(robot1_limbs[:-2, 0] == limb_info[0])[0])
                                conn2_idx = int(np.where(robot2_limbs[:-2, 0] == limb_info[1])[0])
                            else:
                                conn1_idx = int(np.where(robot1_limbs[:-2, 0] == limb_info[1])[0])
                                conn2_idx = int(np.where(robot2_limbs[:-2, 0] == limb_info[0])[0])
                            assert conn1_idx != conn2_idx, "an candidate keypoint is used twice, shared by two object"
                            if limb_info[2] >= robot1_limbs[conn1_idx, 1] and limb_info[2] >= robot2_limbs[conn2_idx, 1]:
                                if robot1_limbs[conn1_idx, 1] > robot2_limbs[conn2_idx, 1]:
                                    low_conf_idx = robot_assoc_idx[1]
                                    delete_conn_idx = conn2_idx
                                else:
                                    low_conf_idx = robot_assoc_idx[0]
                                    delete_conn_idx = conn1_idx

                                robot_to_part_assoc[low_conf_idx][-2, 0] -= vals[i, int(robot_to_part_assoc[low_conf_idx][delete_conn_idx, 0]), int(limb_info[1])]
                                robot_to_part_assoc[low_conf_idx][-2, 0] -= robot_to_part_assoc[low_conf_idx][delete_conn_idx, 1]
                                robot_to_part_assoc[low_conf_idx][delete_conn_idx, 0] = -1
                                robot_to_part_assoc[low_conf_idx][delete_conn_idx, 1] = -1
                                robot_to_part_assoc[low_conf_idx][-1, 0] -= 1

                # new limb not associated to an existing robot, so we create a new robot
                elif len(robot_assoc_idx) == 0:
                    row = np.ones((N_kpts + 2, 2), dtype=np.float64) * -1
                    row[part_src_type] = limb_info[[0, 2]]
                    row[part_dst_type] = limb_info[[1, 2]]
                    row[-2, 0] = vals[i, part_src_type, int(limb_info[0])] + \
                                 vals[i, part_dst_type, int(limb_info[1])] + limb_info[2]
                    row[-1] = [2, limb_info[-1]]
                    robot_to_part_assoc.append(row)

        # removing robots that have too few limbs associated
        if ignore_few_parts:
            robots_to_delete = []
            for robot_id, robot_info in enumerate(robot_to_part_assoc):
                if robot_info[-1, 0] < min_num_connected_parts or robot_info[-2, 0] / robot_info[-1, 0] < min_mean_score:
                    robots_to_delete.append(robot_id)
            for index in robots_to_delete[::-1]:
                robot_to_part_assoc.pop(index)

        # shaping the results
        det_idx = 0
        for robot in sorted(robot_to_part_assoc, key=lambda x: x[-1, 0], reverse=True):
            for k in range(N_kpts):
                idx = robot[k, 0]
                if idx > -1:
                    results[i, det_idx, k] = np.append(dets[i, k, int(idx)], robot[k, 1])
            det_idx += 1
            if det_idx >= max_num_dets:
                break

    return results, dets


@torch.no_grad()
def aggregate_multi_scale_hmaps(lr_hmaps, hr_hmaps=None, hr_size=None, num_kpts=None):
    """
    Fusing the information of heatmaps from low- and high-resolution outputs
    
    Args:
    -----
    lr_hmaps: torch Tensor
        Heatmaps predicted at low resolution. Shape is (B, N_kpts, H_lr, W_lr)
    hr_hmaps: torch Tensor
        Heatmaps predicted at high resolution. Shape is (B, N_kpts, H_hr, W_hr)
    num_kpts: int
        Number of keypoints to consider
        
    Returns:
    ---------
    fused_hmaps: torch Tensor
        Fusion of low- and high-resolution heatmaps. Shape is (B, N_kpts, H_lr, W_lr)
    """    
    if hr_hmaps is None and hr_size is None:
        raise ValueError(f"'hr_maps' and 'hr_size' cannot both be None...")
    if hr_hmaps is not None and hr_size is not None:
        raise ValueError(f"'hr_maps' and 'hr_size' cannot both be different than None...")
    if num_kpts is not None:
        lr_hmaps = lr_hmaps[:, :num_kpts]
        if hr_hmaps is not None:
            hr_hmaps = hr_hmaps[:, :num_kpts]
        
    # getting size from HR-HMaps and fusing
    if hr_hmaps is not None:
        given_hr_size = (hr_hmaps.shape[-2], hr_hmaps.shape[-1])
        lr_hmaps_upsampled = F.interpolate(lr_hmaps, size=given_hr_size, mode='bilinear', align_corners=False)
        fused_hmaps = (hr_hmaps + lr_hmaps_upsampled) / 2
        
    # no HR-HMaps, so directly using the given size for upsampling and thats it
    if hr_size is not None:
        lr_hmaps_upsampled = F.interpolate(lr_hmaps, size=hr_size, mode='bilinear', align_corners=False)
        fused_hmaps = lr_hmaps_upsampled
    return fused_hmaps



#
