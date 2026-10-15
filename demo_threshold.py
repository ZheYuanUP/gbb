# -*- coding: utf-8 -*-
'''
@File    :   demo_metauas.py
@Time    :   2025/03/26 23:49:14
@Author  :   Bin-Bin Gao
@Email   :   csgaobb@gmail.com
@Homepage:   https://csgaobb.github.io/
@Version :   1.0
@Desc    :   MetaUAS Demo
'''


import os
import cv2
import torch
import json
import shutil
import kornia as K
import numpy as np
import time

from easydict import EasyDict
from argparse import ArgumentParser
from metauas import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, read_image_as_tensor, safely_load_state_dict

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_initial_mask(img):
    """用SAM分割或基于黑色背景生成初始掩码"""
    if len(img.shape) == 3:
        # mask外已处理为黑色，三通道均为0的区域为背景
        return np.logical_or.reduce((img[...,0] > 0, img[...,1] > 0, img[...,2] > 0)).astype(np.uint8)
    else:
        return (img > 0).astype(np.uint8)   

def process_pred_map(pred_map, mask):
    """
    处理预测的异常分数图：屏蔽掩码外区域，并对掩码内区域重新归一化
    Args:
        pred_map: 模型输出的异常分数图（NumPy数组，形状[H, W]或[H, W, 1]）
        mask: 二值掩码（NumPy数组，形状[H, W]，1表示有效区域，0表示背景）
    Returns:
        processed_map: 处理后的异常分数图（掩码外为0，掩码内归一化到0~1）
    """
    # 确保pred_map是单通道（[H, W]）
    if pred_map.ndim == 3 and pred_map.shape[-1] == 1:
        pred_map = pred_map.squeeze(-1)  # 转为[H, W]

    # 1. 屏蔽掩码外区域
    masked_pred = pred_map * mask  # 背景区域直接归零

    # 2. 仅对掩码内的有效区域重新归一化
    valid_values = masked_pred[mask == 1]  # 提取掩码内的所有值
    if len(valid_values) == 0:
        # 掩码内无有效区域则返回全0
        return masked_pred
    # 归一化到0-1
    min_val = valid_values.min()
    max_val = valid_values.max()
    if max_val - min_val < 1e-6:
        # 若所有有效值相同，避免除零错误
        normalized_valid = np.zeros_like(valid_values)
    else:
        normalized_valid = (valid_values - min_val) / (max_val - min_val)

    # 将归一化后的值回填到掩码内区域
    processed_map = np.zeros_like(masked_pred)
    processed_map[mask == 1] = normalized_valid

    return processed_map
def tensor_to_numpy(tensor):
    """
    将图像张量（可能含批次维度）转为numpy数组（H, W, C）
    """
    # 若含批次维度（4维），则去除第一个维度（取第0个样本）
    if tensor.dim() == 4:
        tensor = tensor[0]  # 从 [B, C, H, W] 变为 [C, H, W]
    # 调整通道顺序并转为numpy
    return tensor.permute(1, 2, 0).cpu().numpy()

def set_outer_and_inner_edge_to_one(pred_single, mask, pixels):
    """
    将掩码外区域 + 掩码内边缘向内5像素区域全部设为1
    仅保留掩码最核心的内部区域（非边缘部分）的原始值
    Args:
        pred_single: 单通道预测图（H, W）
        mask: 二值掩码（H, W），1=目标区域，0=背景
        inner_pixels: 掩码内边缘向内的像素数（固定为5）
    Returns:
        processed_pred: 处理后的预测图
    """
    inner_pixels = pixels
    # 1. 确保掩码为二值图（0/1）
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape

    # 2. 定位“掩码外区域”：mask=0的区域
    outer_mask = 1 - mask  # 1=掩码外，0=掩码内

    # 3. 定位“掩码内边缘向内5像素区域”：
    # 3.1 腐蚀掩码，得到“核心区域”（边缘向内收缩5像素后的区域）
    kernel = np.ones((2 * inner_pixels + 1, 2 * inner_pixels + 1), np.uint8)
    core_mask = cv2.erode(mask, kernel, iterations=1)  # 核心区域（非边缘）
    # 3.2 掩码内的边缘区域 = 原始掩码 - 核心区域（即边缘向内5像素）
    inner_edge_mask = mask - core_mask  # 1=掩码内边缘区，0=其他

    # 4. 合并“掩码外区域”和“掩码内边缘区”，两个区域设为1
    regions_to_set_one = outer_mask + inner_edge_mask  # 1=需要设为1的区域
    regions_to_set_one = (regions_to_set_one > 0).astype(np.uint8)  # 确保二值化

    # 5. 处理预测图：
    # - 核心区域（core_mask=1）→ 保留原始值
    processed_pred = pred_single * core_mask + regions_to_set_one * 1

    return processed_pred, core_mask  # 返回处理结果和核心区域掩码（用于后续归一化）


def calculate_i_auc_metrics(anomaly_map, ground_truth_path):
    """
    仅计算 I-AUROC（Inverse AUROC）和 I-AUCPR（Inverse AUPR）核心指标
    若原始AUC < 0.5，取1-AUC（修正置信值与标签反向的情况）

    Args:
        anomaly_map: 预测的异常分数图 (H, W)，分数越高代表异常概率越大
        ground_truth_path: 真实标注掩码路径（二值图，缺陷区域为白色/非零值）
    """
    gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"未找到真实标注文件：{ground_truth_path}")
    
    # 调整标注尺寸与异常图一致（双线性插值，保证对齐）
    gt_mask = cv2.resize(gt_mask, (anomaly_map.shape[1], anomaly_map.shape[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    # 转为二值标签（0=正常，1=异常）并展平
    gt_labels = (gt_mask > 0).astype(np.uint8).flatten()
    cv2.imwrite('./gt_mask.jpg', gt_mask)
    # ---------------------- 2. 预处理异常分数图 ----------------------
    # 校验异常图维度（必须是2D单通道）
    if len(anomaly_map.shape) != 2:
        raise ValueError(f"异常分数图必须是2D数组，当前维度：{anomaly_map.shape}")
    
    # 归一化到 [0,1] 并展平（确保分数范围统一）
    anomaly_scores = normalize(anomaly_map).flatten()

    # ---------------------- 3. 计算核心指标 ----------------------
    # 3.1 计算 I-AUROC
    # 先计算原始 AUROC，处理标签与分数反向的情况
    original_auroc = roc_auc_score(gt_labels, anomaly_scores)
    i_auroc = original_auroc if original_auroc >= 0.5 else 1 - original_auroc

    # 3.2 计算 I-AUCPR（先计算原始 AUPR，再修正）
    precisions, recalls, _ = precision_recall_curve(gt_labels, anomaly_scores)
    original_aupr = auc(recalls, precisions)
    # PR曲线特殊情况：若原始AUPR < 0.5，需用 1 - original_aupr 修正（对应置信值反向）
    i_aucpr = original_aupr if original_aupr >= 0.5 else 1 - original_aupr

    print("=" * 50)
    print("核心指标计算结果")
    print("=" * 50)
    print(f"I-AUROC (修正后AUROC): {i_auroc:.4f}")
    print(f"I-AUCPR (修正后AUPR): {i_aucpr:.4f}")
    print("=" * 50)

    return {
        'i_auroc': i_auroc,
        'i_aucpr': i_aucpr
    }

if __name__ == "__main__":
    random_seed = 1

    set_random_seed(random_seed)

    # ckt_path = 'weights/metauas-256.ckpt'
    # img_size = 256
    ckt_path = "weights/metauas-512.ckpt"
    img_size = 512
    

    # load test images
    path_root = "./images/20251110/"
    path_to_prompt = "./images/1013.png"
    path_to_query = "./images/10133.png"
    # path_to_prompt = path_root + "2-2.png"
    # path_to_query = path_root + "2-1.png"
    path_to_gt = path_root + "2-2gt.png"

    query = read_image_as_tensor(path_to_query)
    prompt = read_image_as_tensor(path_to_prompt)
    mask_query = get_initial_mask(tensor_to_numpy(query))


    if query.shape[1] != img_size:
        resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
        query = resize_trans(query)[0]
        prompt = resize_trans(prompt)[0]
    

    test_data = {
            "query_image": query.cuda(),
            "prompt_image": prompt.cuda(),
        }
    # load model
    encoder = 'efficientnet-b4'
    decoder = 'unet'
    encoder_depth = 5
    decoder_depth = 5
    num_crossfa_layers = 3
    alignment_type =  'sa'
    fusion_policy = 'cat'
    mask = 'mask'



    model = MetaUAS(encoder,
                decoder,
                encoder_depth,
                decoder_depth,
                num_crossfa_layers,
                alignment_type,
                fusion_policy
                )


    model = safely_load_state_dict(model, ckt_path)
    model.cuda()
    model.eval()

    start_time = time.time()
    # forward
    predicted_masks = model(test_data)
    # visualization
    query_img = test_data["query_image"][0] * 255
    query_img = query_img.permute(1,2,0)
    # 从模型输出得到初始预测图
    # 先获取单通道预测图，后续处理完再转3通道，避免通道维度干扰掩码操作
    # 去除冗余维度+脱离计算图+反转分数
    pred_single = (1 - predicted_masks.squeeze().detach()).cpu().numpy()  # 形状：(H, W)
    mask = cv2.resize(mask_query, (pred_single.shape[1], pred_single.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('./mask_query_raw.jpg', (mask * 255).astype(np.uint8))

    pixels = 2
    pred_processed, core_mask = set_outer_and_inner_edge_to_one(pred_single, mask, pixels)

    cv2.imwrite('./core_mask.jpg', (core_mask * 255).astype(np.uint8))

    # 恢复原代码的3通道格式（用于后续可视化，如与RGB图叠加）
    pred = pred_processed[:, :, None].repeat(3, axis=2)  # 形状：(H, W, 3)
    # normalize just for analysis
    scoremap_self = apply_ad_scoremap(query_img.cpu(), normalize(pred))
    end_time = time.time()
    cv2.imwrite('./anomaly_map.jpg', scoremap_self)
    cv2.imwrite('./pred_map.jpg', (1-normalize(pred))*255)
    run_time = end_time - start_time
    print(f"运行时间: {run_time:.4f} 秒")
    anomaly_map = 1 - normalize(pred)[:, :, 0]
    if os.path.exists(path_to_gt):
        roc_results = calculate_i_auc_metrics(anomaly_map, path_to_gt)
    else:
        print(f"警告: 真实标注文件不存在: {path_to_gt}")
        print("跳过ROC计算")