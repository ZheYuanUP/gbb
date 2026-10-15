'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-09-02 16:08:23
 # @ Modified by: Bin-Bin Gao
 # @ Modified time: 2025-09-03 22:15:52
 # @ Description: test script for MetaUAS
 '''


import os 
import sys
import shutil
import cv2
import json
import random
import numpy as np
from tqdm import tqdm 

from argparse import ArgumentParser

import kornia as K
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import shapely.geometry
from datetime import datetime
from tabulate import tabulate

import torch
import torch.nn.functional as F
import torchmetrics


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torch.utils.data import DataLoader
from dataset import ADDataset
from datasets.pair_masks import PairMaskTestDataset
from metauas import MetaUAS, set_random_seed, safely_load_state_dict
#from eval_metric.torch_metric import Evaluator
cpu_eva = False
from eval_metric.sklearn_metric import Evaluator, _connected_regions, _matched_gt_regions
cpu_eva = True 


def prediction_failed(gt_mask, pred_mask, min_gt_coverage):
    """Return whether a sample fails the project's connected-region criteria."""
    gt_regions = _connected_regions(gt_mask > 0)
    pred_regions = _connected_regions(pred_mask > 0)
    matched_gt_regions = _matched_gt_regions(
        gt_regions,
        pred_regions,
        min_gt_coverage,
    )
    has_false_positive_region = any(
        not any(np.logical_and(pred_region, gt_region).any() for gt_region in gt_regions)
        for pred_region in pred_regions
    )
    return matched_gt_regions != len(gt_regions) or has_false_positive_region


def save_test_visualizations(
    paths,
    gt_masks,
    pred_masks,
    image_size,
    save_dir,
    pred_threshold,
    min_gt_coverage,
    start_index=0,
):
    """Save every sample under all/ and failed samples under failure/."""
    all_dir = os.path.join(save_dir, 'all')
    failure_dir = os.path.join(save_dir, 'failure')
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    gt_masks = gt_masks.detach().cpu().numpy()
    pred_masks = pred_masks.detach().cpu().numpy()
    failure_count = 0

    for batch_index, path in enumerate(paths):
        sample_index = start_index + batch_index
        stem = os.path.splitext(os.path.basename(path))[0]
        filename_prefix = f'{sample_index:06d}_{stem}'

        gt_mask = np.squeeze(gt_masks[batch_index])
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        pred_score = np.squeeze(pred_masks[batch_index])
        pred_score = np.nan_to_num(pred_score, nan=0.0, posinf=1.0, neginf=0.0)
        pred_score = np.clip(pred_score, 0.0, 1.0)
        # Preserve the model's continuous 0-1 prediction as an 8-bit grayscale
        # image. Unlike pred_mask, this output is not thresholded.
        pred_gray = np.round(pred_score * 255).astype(np.uint8)
        pred_mask = (pred_score >= pred_threshold).astype(np.uint8) * 255

        query_image = cv2.imread(path, cv2.IMREAD_COLOR)
        if query_image is None:
            raise FileNotFoundError(f'Unable to read query image: {path}')
        query_image = cv2.resize(query_image, (image_size, image_size))
        color_map = cv2.applyColorMap(
            pred_gray,
            cv2.COLORMAP_JET,
        )
        heatmap = cv2.addWeighted(query_image, 0.5, color_map, 0.5, 0)

        outputs = {
            f'{filename_prefix}_gt_mask.png': gt_mask,
            f'{filename_prefix}_pred_mask.png': pred_mask,
            f'{filename_prefix}_pred_gray.png': pred_gray,
            f'{filename_prefix}_heatmap.png': heatmap,
        }
        for filename, image in outputs.items():
            output_path = os.path.join(all_dir, filename)
            if not cv2.imwrite(output_path, image):
                raise OSError(f'Unable to save visualization: {output_path}')

        if prediction_failed(gt_mask, pred_mask, min_gt_coverage):
            failure_count += 1
            for filename in outputs:
                shutil.copy2(
                    os.path.join(all_dir, filename),
                    os.path.join(failure_dir, filename),
                )

    return failure_count


def save_pixel_score_distribution(score_maps, save_dir, bins=100, log_y=False):
    """Save predicted pixel anomaly scores and plot their histogram.

    Every pixel is treated as one detection sample.  Scores are saved without
    per-image normalization so that values remain comparable across images.
    Ground-truth masks and prediction thresholds are not used here.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(score_maps, torch.Tensor):
        score_maps = score_maps.detach().cpu().numpy()

    score_maps = np.asarray(score_maps, dtype=np.float32)

    # Keep the complete spatial score maps so downstream analyses can recover
    # which image and location each pixel came from.
    score_path = os.path.join(save_dir, 'pixel_score_maps.npy')
    np.save(score_path, score_maps, allow_pickle=False)

    flat_scores = score_maps.reshape(-1)
    finite = np.isfinite(flat_scores)
    if not finite.any():
        raise ValueError('Cannot plot pixel scores because all values are NaN or Inf.')

    flat_scores = flat_scores[finite]

    score_min = float(flat_scores.min())
    score_max = float(flat_scores.max())
    if score_min == score_max:
        # matplotlib requires an interval with non-zero width.
        margin = max(abs(score_min) * 0.01, 1e-6)
        hist_range = (score_min - margin, score_max + margin)
    else:
        hist_range = (score_min, score_max)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        flat_scores,
        bins=bins,
        range=hist_range,
        color='steelblue',
        alpha=0.8,
        label=f'All pixels (n={flat_scores.size:,})',
    )
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel('Pixel anomaly score')
    ax.set_ylabel('Pixel count')
    ax.set_title('Pixel anomaly score distribution')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    histogram_path = os.path.join(save_dir, 'pixel_score_histogram.png')
    fig.savefig(histogram_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return score_path, histogram_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="") 
    parser.add_argument("--seed", type=int, default="1") 
    parser.add_argument("--image_dir", type=str, default="") 
    parser.add_argument("--img_size", type=int, default="256") 
    parser.add_argument("--test_json", type=str, default="") 
    parser.add_argument("--prompt_json", type=str, default="") 
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO', 'I-Accuracy', 'R-Accuracy', 'R-FPR'], help='evaluation metrics')
    parser.add_argument("--pred_threshold", type=float, default=0.5, help='threshold for image/region prediction')
    parser.add_argument("--min_gt_coverage", type=float, default=0.6, help='minimum intersection / GT region area for a correct region')
    parser.add_argument("--score_hist_bins", type=int, default=100, help='number of bins in pixel-score histograms')
    parser.add_argument("--score_hist_log_y", action='store_true', help='use a logarithmic y-axis in pixel-score histograms')
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join('result', f"hedian_{datetime.now().strftime('%m%d_%H%M')}"),
        help='flat visualization output directory',
    )
    args = parser.parse_args()        

    random_seed = args.seed
    set_random_seed(random_seed)
    img_size = args.img_size
    ckt_path = args.checkpoint

    # init model
    encoder = 'efficientnet-b4' 
    decoder = 'unet' 
    encoder_depth = 5
    decoder_depth = 5
    num_crossfa_layers = 3
    alignment_type =  'sa' 
    fusion_policy = 'cat'

    model = MetaUAS(encoder, 
                decoder, 
                encoder_depth, 
                decoder_depth, 
                num_crossfa_layers, 
                alignment_type, 
                fusion_policy
                )       

    # load pre-trained model
    model = safely_load_state_dict(model, ckt_path)
    model.cuda()
    model.eval()

    device = torch.device("cuda")
    method = 'metauas'
    image_size = img_size
    
    # load testing dataset
    if args.test_csv:
        dataset = PairMaskTestDataset(args.image_dir, args.test_csv, args.img_size)
    else:
        dataset = ADDataset(args.image_dir, args.test_json, args.img_size, prompt_meta_file=args.prompt_json)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    
    # one-shot testing
    if cpu_eva:
        evaluator = Evaluator(
            'cpu', metrics=args.eval_metrics,
            pred_threshold=args.pred_threshold,
            min_gt_coverage=args.min_gt_coverage,
        )
    else:
        evaluator = Evaluator(device, metrics=args.eval_metrics)

    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, img_paths = [], [], [], [], [], []
    saved_sample_count = 0
    failed_sample_count = 0
    for batch_data in tqdm(dataloader):
        test_data = {
            "query_image": batch_data["query_image"].cuda(),
            "prompt_image": batch_data["prompt_image"].cuda()

        }

        cls_name = batch_data['cls_name']
        gt_label = batch_data['query_label'].cuda()
        gt_mask = batch_data['query_mask'].cuda()
        gt_mask[gt_mask > 0] = 1
        query_path = batch_data['query_filename']

        with torch.no_grad():
            pr_mask = model(test_data)
           
        cls_names.append(np.array(cls_name))
        img_paths.append(np.array(query_path))

        if cpu_eva:
            gt_masks.append(gt_mask.cpu())
            gt_anomalys.append(gt_label.cpu())
            pr_masks.append(pr_mask.cpu())  
        else:
            gt_masks.append(gt_mask)
            gt_anomalys.append(gt_label)
            pr_masks.append(pr_mask)

        # Save GT, thresholded prediction, continuous grayscale score, and heatmap.
        failed_sample_count += save_test_visualizations(
            query_path,
            gt_mask,
            pr_mask[:, 0],
            args.img_size,
            args.save_path,
            args.pred_threshold,
            args.min_gt_coverage,
            start_index=saved_sample_count,
        )
        saved_sample_count += len(query_path)

        
            
    print(
        f'Saved {saved_sample_count} samples to {os.path.join(args.save_path, "all")}; '
        f'{failed_sample_count} failures to {os.path.join(args.save_path, "failure")}.'
    )

    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_eval = dict(
                        gt_masks=gt_masks, 
                        pr_masks=pr_masks, 
                        cls_names=cls_names, 
                        gt_anomalys=gt_anomalys, 
                        img_paths=img_paths
                        )
    results_eval = {k: np.concatenate(v, axis=0) if k in ['cls_names', 'img_paths'] else torch.cat(v, dim=0) for k, v in results_eval.items()}

    score_path, histogram_path = save_pixel_score_distribution(
        results_eval['pr_masks'],
        args.save_path,
        bins=args.score_hist_bins,
        log_y=args.score_hist_log_y,
    )
    print(f'Saved raw pixel score maps to {score_path}.')
    print(f'Saved pixel score histogram to {histogram_path}.')
    
   
    # print results
    obj_list = list(sorted(set(results_eval['cls_names'])))
    msg = {}
    for idx, cls_name in enumerate(tqdm(obj_list)): 
        metric_results = evaluator.run(results_eval, cls_name)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
        msg['Name'].append('Avg') if avg_act else None
    
        for metric in args.eval_metrics:
            metric_result = metric_results[metric] #* 100

            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)
            
            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                msg[metric].append(metric_result_avg)
                
    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", ) 
    print(tab)
