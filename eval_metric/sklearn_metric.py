'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-09-02 22:39:56
 # @ Modified by: Bin-Bin Gao
 # @ Modified time: 2025-09-02 22:40:23
 # @ Description: some classes and functions for MetaUAS
 '''

import glob
import logging
import os
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from skimage import measure
import time

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]


    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = metrics.auc(fprs, pros[idxes])
    return pro_auc


def _connected_regions(binary_mask):
    """Return 8-connected components as boolean masks."""
    labels = measure.label(binary_mask.astype(np.uint8), connectivity=2)
    return [labels == region.label for region in measure.regionprops(labels)]


def _matched_gt_regions(gt_regions, pred_regions, min_gt_coverage):
    """Maximum one-to-one matching of predictions to sufficiently covered GTs."""
    eligible = {index: [] for index in range(len(gt_regions))}
    for gt_index, gt_region in enumerate(gt_regions):
        gt_area = int(gt_region.sum())
        for pred_index, pred_region in enumerate(pred_regions):
            intersection = int(np.logical_and(gt_region, pred_region).sum())
            if intersection == 0:
                continue
            pred_area = int(pred_region.sum())
            union = gt_area + pred_area - intersection
            coverage = intersection / gt_area
            if coverage >= min_gt_coverage:
                eligible[gt_index].append((pred_index, intersection / union, coverage))

    for options in eligible.values():
        options.sort(key=lambda item: (item[1], item[2]), reverse=True)

    pred_owner = {}

    def assign(gt_index, visited):
        for pred_index, _, _ in eligible[gt_index]:
            if pred_index in visited:
                continue
            visited.add(pred_index)
            previous_gt = pred_owner.get(pred_index)
            if previous_gt is None or assign(previous_gt, visited):
                pred_owner[pred_index] = gt_index
                return True
        return False

    for gt_index in range(len(gt_regions)):
        assign(gt_index, set())
    return len(set(pred_owner.values()))


def region_metrics(gt_masks, pred_masks, pred_threshold=0.5, min_gt_coverage=0.9):
    """Compute all-regions image accuracy, region accuracy, and region FPR."""
    anomaly_images = 0
    successful_images = 0
    total_gt_regions = 0
    detected_gt_regions = 0
    total_pred_regions = 0
    false_positive_regions = 0

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        gt_binary = gt_mask > 0
        pred_binary = pred_mask >= pred_threshold
        gt_regions = _connected_regions(gt_binary)
        pred_regions = _connected_regions(pred_binary)

        total_gt_regions += len(gt_regions)
        detected_in_image = _matched_gt_regions(
            gt_regions, pred_regions, min_gt_coverage
        )
        detected_gt_regions += detected_in_image
        if gt_regions:
            anomaly_images += 1
            successful_images += int(detected_in_image == len(gt_regions))
        total_pred_regions += len(pred_regions)
        false_positive_regions += sum(
            1
            for pred_region in pred_regions
            if not any(np.logical_and(pred_region, gt_region).any() for gt_region in gt_regions)
        )

    return {
        'I-Accuracy': (
            successful_images / anomaly_images if anomaly_images else float('nan')
        ),
        'R-Accuracy': (
            detected_gt_regions / total_gt_regions
            if total_gt_regions else float('nan')
        ),
        'R-FPR': (
            false_positive_regions / total_pred_regions if total_pred_regions else 0.0
        ),
    }

class Evaluator(object):
    def __init__(self, device, metrics=[], pred_threshold=0.5, min_gt_coverage=0.9):
        if len(metrics) == 0:
            self.metrics = [
                'I-AUROC', 'I-AP', 'I-F1max', 
                'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'
            ]
        else:
            self.metrics = metrics
        if not 0.0 <= pred_threshold <= 1.0:
            raise ValueError('pred_threshold must be in [0, 1]')
        if not 0.0 <= min_gt_coverage <= 1.0:
            raise ValueError('min_gt_coverage must be in [0, 1]')
        self.pred_threshold = pred_threshold
        self.min_gt_coverage = min_gt_coverage

    def run(self, results, cls_name):
        idxes = results['cls_names'] == cls_name
        gt_px = results['gt_masks'][idxes] 
        pr_px = results['pr_masks'][idxes]

        gt_sp = results['gt_anomalys'][idxes]
        #pr_sp = results['pr_anomalys'][idxes]
        
        raw_pr_px = pr_px
        value_range = pr_px.max() - pr_px.min()
        pr_px = (pr_px - pr_px.min()) / value_range if value_range > 0 else torch.zeros_like(pr_px)
        pr_sp = pr_px.max(dim=-1)[0].max(dim=-1)[0]
        
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
            raw_pr_px = raw_pr_px.squeeze(1)

        pr_px = pr_px.numpy()
        raw_pr_px = raw_pr_px.numpy()
        gt_px = gt_px.numpy()

        pr_sp = pr_sp.numpy()
        gt_sp = gt_sp.numpy()

    
        eval_results = {}
        requested_region_metrics = {'I-Accuracy', 'R-Accuracy', 'R-FPR'}.intersection(self.metrics)
        if requested_region_metrics:
            region_results = region_metrics(
                gt_px,
                raw_pr_px,
                pred_threshold=self.pred_threshold,
                min_gt_coverage=self.min_gt_coverage,
            )
            eval_results.update(
                {name: region_results[name] for name in requested_region_metrics}
            )
        for metric in self.metrics:
            if metric.startswith('I-AUROC'):
                fpr, tpr, thresholds = metrics.roc_curve(gt_sp.ravel(), pr_sp.ravel(), pos_label=1)
                i_auc = metrics.auc(fpr, tpr)
                if i_auc < 0.5:
                    i_auc = 1 - i_auc
                eval_results[metric] = i_auc
                
            elif metric.startswith('P-AUROC'):
                fpr, tpr, thresholds = metrics.roc_curve(gt_px.ravel(), pr_px.ravel(), pos_label=1)
                p_auc = metrics.auc(fpr, tpr)
                if p_auc < 0.5:
                    p_auc = 1 - p_auc
                eval_results[metric] = p_auc
                    
            elif metric.startswith('I-AP'):
                eval_results[metric] = metrics.average_precision_score(gt_sp.ravel(), pr_sp.ravel(), pos_label=1, average=None)
                
            elif metric.startswith('P-AP'):
                eval_results[metric] = metrics.average_precision_score(gt_px.ravel(), pr_px.ravel(), pos_label=1, average=None)
                    
            elif metric.startswith('I-F1max'):
                precisions, recalls, thresholds = metrics.precision_recall_curve(gt_sp.ravel(), pr_sp.ravel())
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                eval_results[metric] = np.max(f1_scores[np.isfinite(f1_scores)])
            
            elif metric.startswith('P-F1max'):
                precisions, recalls, thresholds = metrics.precision_recall_curve(gt_px.ravel(), pr_px.ravel())
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                eval_results[metric] = np.max(f1_scores[np.isfinite(f1_scores)])
            
            elif metric.startswith('P-AUPRO'):
                eval_results[metric] = cal_pro_score(gt_px, pr_px)
            
        return eval_results
