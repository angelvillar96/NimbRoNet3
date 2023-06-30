"""
Computation of different metrics
"""

import os
import copy
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.PoseDataset import PoseDataset
from data.data_utils import get_affine_transform, affine_transform
from lib.utils import create_directory
from lib.postprocessing import filter_heatmaps, get_kpt_from_heatmaps, kpt_distance_nms, \
                               aggregate_multi_scale_hmaps, group_poses
from CONFIG import METRICS, METRIC_SETS


class MetricTracker:
    """ Class for computing several evaluation metrics """

    def __init__(self, metrics="detection", **kwargs):
        """ Module initializer """
        assert isinstance(metrics, (list, str)), f"Metrics must be string/list, not {type(metrics)}"
        if isinstance(metrics, str):
            if metrics not in METRIC_SETS.keys():
                raise ValueError(f"If str, metrics must be one of {METRIC_SETS.keys()}, not {metrics}")
        else:
            for metric in metrics:
                if metric not in METRICS:
                    raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")

        metrics = METRIC_SETS[metrics] if isinstance(metrics, str) else metrics
        self.metric_computers = {}
        for metric in metrics:
            self.metric_computers[metric] = self._get_metric(metric, **kwargs)
        self.reset_results()
        self.params = kwargs
        return

    def reset_results(self):
        """ Reseting results and metric computers """
        self.results = {}
        for m in self.metric_computers.values():
            m.reset()
        return

    def accumulate(self, preds, targets, **kwargs):
        """ Computing the different metrics and adding them to the results list """
        for _, metric_computer in self.metric_computers.items():
            metric_computer.accumulate(preds=preds, targets=targets, **kwargs)
        return

    def aggregate(self, get_results=False):
        """ Aggregating the results for each metric """
        for metric, metric_computer in self.metric_computers.items():
            mean_metric = metric_computer.aggregate()
            if isinstance(mean_metric, dict):
                for key, val in mean_metric.items():
                    self.results[key] = val
            else:
                self.results[f"{metric}"] = mean_metric

        results = self.results if get_results else None
        return results

    def summary(self, get_results=True):
        """ Printing and fetching the results """
        print("RESULTS:")
        print("--------")
        for metric, val in self.results.items():
            if isinstance(val, (int, float)):
                print(f"  {metric}:  {round(val, 3)}")
            elif isinstance(val, list):
                val = [round(v, 3) for v in val]
                print(f"  {metric}:  {val}")
        return self.results

    def save_results(self, exp_path, checkpoint_name, name="",for_quantization=False):
        """ Storing results into JSON file """
        if for_quantization:
            fname = f"{name}_quantization.json"
        else:
            checkpoint_name = checkpoint_name.split(".")[0]
            fname = f"{name}_{checkpoint_name}.json"
        dirname = ""
        for name, val in self.params.items():
            if name == "coco_gt":
                continue
            dirname = dirname + f"{name}_{val}_"
        results_path = os.path.join(exp_path, "results", dirname[:-1])
        results_file = os.path.join(results_path, fname)

        create_directory(dir_path=results_path)

        with open(results_file, "w") as file:
            json.dump(self.results, file)
        return

    def _get_metric(self, metric, **kwargs):
        """ """
        # detection
        if metric == "detection F1":
            metric_computer = DetectionF1(**kwargs)
        # segmentation
        elif metric == "segmentation accuracy":
            metric_computer = SegmentationAccuracy(**kwargs)
        elif metric == "IOU":
            metric_computer = IOU(**kwargs)
        elif metric == "pose_ap":
            metric_computer = PoseEstimationEval(**kwargs)
        else:
            raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS} ...")
        return metric_computer


class Metric:
    """
    Base class for metrics
    """

    def __init__(self, **kwargs):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")


class DetectionF1(Metric):
    """
    Metric Computer for Detection F1 score and related metrics: accuracy, recall, precision, and FDR

    Args:
    -----
    num_classes: integer
        Total number of classes in the dataset
    match_thr: float
        A target kpt and a detection are matched if they are closer (in pixels) than ths threshold
    filter_thr: integer
        Distance threshold (in pixels) for NMS. Analogous to IoU threshold for bounding-boxes
        If two peaks are closer than 'filter_thr', the one with the lowest magnitude is removed.
    kernel_size: integer
        Size of the window used for morphological operations: erosion and dilation
    heatmap_thr: float
        Magnitude filter. Elements below this magnitude are removed
    """

    def __init__(self, num_classes, match_thr=1, filter_thr=9, kernel_size=5, heatmap_thr=0.5, **kwargs):
        """ """
        self.num_classes = num_classes
        self.match_thr = match_thr
        self.filter_thr = filter_thr
        self.kernel_size = kernel_size
        self.heatmap_thr = heatmap_thr
        self.peak_thr = kwargs.get("peak_thr", 0.1)
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.true_positives = {c: 0 for c in range(self.num_classes)}
        self.true_negatives = {c: 0 for c in range(self.num_classes)}
        self.false_positives = {c: 0 for c in range(self.num_classes)}
        self.false_negatives = {c: 0 for c in range(self.num_classes)}

    def accumulate(self, preds, targets=None, target_centers=None, **kwargs):
        """ Computing metric """
        verbose = kwargs.get("verbose", False)
        if targets is None and target_centers is None:
            raise ValueError("'Targets' and 'Target centers' cannot both be None")

        # filtering predicted heatmaps
        filtered_preds = filter_heatmaps(
                heatmaps=preds,
                kernel_size=self.kernel_size,
                heatmap_thr=self.heatmap_thr
            )
        pred_kpts, _ = get_kpt_from_heatmaps(
                heatmaps=filtered_preds,
                kernel_size=self.kernel_size,
                peak_thr=self.peak_thr
            )
        filtered_pred_kpts = kpt_distance_nms(pred_kpts, dist_thr=self.filter_thr)

        if target_centers is not None:
            filtered_target_kpts = target_centers
        else:
            filtered_targets = filter_heatmaps(
                    heatmaps=targets,
                    kernel_size=self.kernel_size,
                    heatmap_thr=self.heatmap_thr
                )
            target_kpts, _ = get_kpt_from_heatmaps(
                    heatmaps=filtered_targets,
                    kernel_size=self.kernel_size,
                    peak_thr=self.peak_thr
                )
            filtered_target_kpts = kpt_distance_nms(target_kpts, dist_thr=self.filter_thr)

        # finding matches and computing metrics
        B, C, N, _ = filtered_target_kpts.shape
        for b in range(B):
            for c in range(C):
                # removing [-1, -1] placeholders for current image and detection class
                cur_pred_kpts = [f for f in filtered_pred_kpts[b, c].tolist() if f != [-1, -1]]
                cur_target_kpts = [f for f in filtered_target_kpts[b, c].tolist() if f != [-1, -1]]

                # increase true negatives if there are no detections
                if len(cur_pred_kpts) == 0 and len(cur_target_kpts) == 0:
                    self.true_negatives[c] += 1

                # find pairwise matches and update metrics accordingly
                self.false_positives[c] += len(cur_pred_kpts)
                for target_kpt in cur_target_kpts:
                    match_found = False
                    for pred_kpt in cur_pred_kpts:
                        # there is a match if distance between gt and pred is smaller than threshold
                        dist = ((target_kpt[0] - pred_kpt[0])**2 + (target_kpt[1] - pred_kpt[1])**2) ** 0.5
                        if dist <= self.match_thr:
                            target_kpt[0], target_kpt[1] = -100, -100  # removing from list
                            match_found = True
                            self.true_positives[c] += 1
                            self.false_positives[c] -= 1
                            break
                    if not match_found:
                        self.false_negatives[c] += 1

        if verbose:
            print(f"{self.true_positives = }")
            print(f"{self.false_positives = }")
            print(f"{self.true_negatives = }")
            print(f"{self.false_negatives = }")
        return

    def aggregate(self):
        """ Aggregating the results for each metric """

        # overall metrics
        tp = np.sum([tp for tp in self.true_positives.values()])
        tn = np.sum([tp for tp in self.true_negatives.values()])
        fp = np.sum([tp for tp in self.false_positives.values()])
        fn = np.sum([tp for tp in self.false_negatives.values()])
        full_precision = tp / (tp + fp + 1e-12)
        full_recall = tp / (tp + fn + 1e-12)
        full_f1 = 2 * full_precision * full_recall / (full_precision + full_recall + 1e-12)
        full_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        full_fdr = 1 - full_precision

        # classwise metrics
        recall, precision, accuracy, f1, fdr = {}, {}, {}, {}, {}
        for c in range(self.num_classes):
            sum = self.true_positives[c] + self.true_negatives[c] + \
                  self.false_positives[c] + self.false_negatives[c]
            precision[c] = self._div(self.true_positives[c], self.true_positives[c] + self.false_positives[c])
            recall[c] = self._div(self.true_positives[c], self.true_positives[c] + self.false_negatives[c])
            f1[c] = self._div(2 * precision[c] * recall[c], precision[c] + recall[c])
            accuracy[c] = self._div((self.true_positives[c] + self.true_negatives[c]), sum)
            fdr[c] = 1 - precision[c]

        # final
        results = {
                "Precision": full_precision,
                "Recall": full_recall,
                "Accuracy": full_accuracy,
                "F1": full_f1,
                "FDR": full_fdr,
                "Class Precision": [precision[c] for c in range(self.num_classes)],
                "Class Recall": [recall[c] for c in range(self.num_classes)],
                "Class Accuracy": [accuracy[c] for c in range(self.num_classes)],
                "Class F1": [f1[c] for c in range(self.num_classes)],
                "Class FDR": [fdr[c] for c in range(self.num_classes)]
            }
        return results

    def _div(self, a, b):
        """ Safe division """
        if a < 1e-5 and b < 1e-5:
            result = -1.
        else:
            result = a / (b + 1e-8)
        return result


class SegmentationAccuracy(Metric):
    """ Accuracy and ClassWise accuracy computer """

    def __init__(self, num_classes, **kwargs):
        """ """
        self.num_classes = num_classes
        self.all_preds = []
        self.all_targets = []
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.all_preds = []
        self.all_targets = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self.all_preds = self.all_preds + preds.tolist()
        self.all_targets = self.all_targets + targets.tolist()
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        all_preds = torch.tensor(self.all_preds).flatten()
        all_targets = torch.tensor(self.all_targets).flatten()
        cm = compute_confusion_matrix(
                targets=all_targets,
                preds=all_preds,
                num_classes=self.num_classes
            )
        self.cm = cm
        per_cls_correct_preds = cm.diag()
        per_cls_targets = cm.sum(dim=-1)

        per_cls_acc = per_cls_correct_preds / per_cls_targets
        acc = per_cls_correct_preds.sum() / per_cls_targets.sum()
        return {"Segm. Acc.": acc.item(), "Class Segm. Acc.": per_cls_acc.tolist()}


class IOU(Metric):
    """ Intersection-over-Union and Classwise IoU computer """

    def __init__(self, num_classes, **kwargs):
        """ """
        self.all_preds = []
        self.all_targets = []
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.all_preds = []
        self.all_targets = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        B = preds.shape[0]
        preds, targets = preds.view(B, -1), targets.view(B, -1)
        self.all_preds.append(preds)
        self.all_targets.append(targets)

    def aggregate(self):
        """ Computing average accuracy """
        all_preds = torch.cat(self.all_preds, dim=0).flatten()
        all_targets = torch.cat(self.all_targets, dim=0).flatten()
        cm = compute_confusion_matrix(
                targets=all_targets,
                preds=all_preds,
                num_classes=self.num_classes
            )
        per_cls_correct_preds = cm.diag()
        per_cls_targets = cm.sum(dim=-1)
        per_cls_preds = cm.sum(dim=0)

        # per-class IoU & mean IoU
        union_preds_targets = per_cls_targets + per_cls_preds - per_cls_correct_preds
        class_iou = per_cls_correct_preds / union_preds_targets
        iou = class_iou.mean()

        return {"IoU": iou.item(), "Class IoU": class_iou.tolist()}


class PoseEstimationEval(Metric):
    """
    Module employed for computing the pose estimation metrics
    """

    def __init__(self, coco_gt, **kwargs):
        """ Metric computer initializer """
        self.coco_gt = coco_gt
        self.pose_evaluator = PoseEval(coco_gt=coco_gt)
        super().__init__()

    def reset(self):
        """ Resetting the evaluator """
        self.pose_evaluator = PoseEval(coco_gt=self.coco_gt)
        return

    def accumulate(self, preds, targets, metas):
        """ """
        heatmaps_lr = preds["heatmaps_lr"]
        heatmaps_hr = preds["heatmaps_hr"]
        limbs_lr = preds["limbs_lr"]
        n_kpts = heatmaps_hr.shape[1] - 1
        
        fused_hmaps = aggregate_multi_scale_hmaps(lr_hmaps=heatmaps_lr, hr_hmaps=heatmaps_hr, num_kpts=n_kpts)
        hr_size = (heatmaps_hr.shape[-2], heatmaps_hr.shape[-1])
        fused_limbs = aggregate_multi_scale_hmaps(lr_hmaps=limbs_lr, hr_size=hr_size)
        
        poses, _ = group_poses(
            heatmaps=fused_hmaps,
            limb_heatmaps=fused_limbs,
            limbs=np.array(PoseDataset.LIMBS),
            cfg=None,
            nms_kernel=3,
            max_num_dets=8,
            det_thr=0.1
        )
        poses[..., :2] = poses[..., :2] * 4  # because our outputs have 1/4 of img_size
        
        self.pose_evaluator.collect(poses, metas)
        return

    def aggregate(self):
        """ """
        results = self.pose_evaluator.evaluate()
        return results



#############
## HELPERS ##
#############


def compute_confusion_matrix(preds, targets, num_classes=3):
    """ Computing confusion matrix """
    if not torch.is_tensor(preds) or not torch.is_tensor(targets):
        preds, targets = torch.tensor(preds).flatten(), torch.tensor(targets).flatten()
    preds, targets = preds.cpu(), targets.cpu()
    cm = confusion_matrix(targets, preds)
    cm = torch.from_numpy(cm)
    return cm


class PoseEval:
    """
    Wrapper of the COCO pose evaluation module
    """
    
    METRICS = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]", 
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]",
        "Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]",
        "Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]"
    ]

    def __init__(self, coco_gt):
        """ Module initializer """
        self.coco_gt = copy.deepcopy(coco_gt)
        cat_ids = self.coco_gt.getCatIds()
        cat = self.coco_gt.loadCats(cat_ids)[0]

        self.num_keypoints = len(cat['keypoints'])

        coco_eval = COCOeval(self.coco_gt, iouType='keypoints')
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = self.coco_gt.getImgIds()
        coco_eval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        coco_eval.params.kpt_oks_sigmas = np.ones(self.num_keypoints) * .4
        self.coco_eval = coco_eval

        self.detections = []
        return

    def collect(self, dts, metas, adjust=.5):
        """ Computing intermediate values """
        dts = dts.copy()
        ids = metas['id'].cpu().numpy()
        centers = metas['center'].cpu().numpy()
        translates = metas['translate'].cpu().numpy()
        scales = metas['scale'].cpu().numpy()

        for img_dts, img_id, center, translate, scale in zip(dts, ids, centers, translates, scales):
            trans = get_affine_transform(center, translate, scale, inv=True)
            img_dts[..., :2] = np.int64(affine_transform(img_dts[..., :2] + adjust, trans))
            mask = img_dts[..., 2] > 0
            img_dts *= mask[..., None]

            for keypoints in img_dts:
                if np.any(keypoints[..., 2]):
                    self.detections.append({
                        'image_id': int(img_id),
                        'category_id': 1,
                        'keypoints': keypoints.flatten().tolist(),
                        'score': keypoints[:, 2].mean()
                    })
        return

    def evaluate(self):
        """ Using the COCO Eval module to  """
        ap = 0.0
        if len(self.detections) > 0:
            self.coco_eval.cocoDt = COCO.loadRes(self.coco_gt, self.detections)
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            results = {m: self.coco_eval.stats[i] for i, m in enumerate(PoseEval.METRICS)}
            self.detections = []
        return results


#

