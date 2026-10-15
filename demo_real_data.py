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
import torch.nn.functional as F
import json
import shutil
import kornia as K
import numpy as np
import time

from easydict import EasyDict
from argparse import ArgumentParser
from metauas import (
    MetaUAS,
    set_random_seed,
    read_image_as_tensor,
    safely_load_state_dict
)


if __name__ == "__main__":
    random_seed = 1

    set_random_seed(random_seed)

    # ckt_path = "weights/metauas-512.ckpt"
    ckt_path = "checkpoints/best_val_dice.pth"
    img_size = 512

    # load model
    encoder = 'efficientnet-b4'
    decoder = 'unet'
    encoder_depth = 5
    decoder_depth = 5
    num_crossfa_layers = 3
    alignment_type = 'sa'
    fusion_policy = 'cat'

    model = MetaUAS(
        encoder,
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

    # load test images
    # path_to_prompt = "./real_data/device1/render/query_010.jpg"
    # path_to_query = "./real_data/device1/query/query_010.jpg"

    path_to_prompt = './kf_0008_f000510_right.jpg'
    path_to_query = './kf_0008_f000510_left.jpg'

    # 读取原始图片。最终保存尺寸以 OpenCV 读取到的 Query 原图为准。
    query_img = cv2.imread(path_to_query, cv2.IMREAD_COLOR)
    if query_img is None:
        raise FileNotFoundError(f"无法读取 Query 图片: {path_to_query}")
    original_height, original_width = query_img.shape[:2]

    original_query = read_image_as_tensor(path_to_query)
    prompt = read_image_as_tensor(path_to_prompt)

    if original_query.shape[-2:] != (original_height, original_width):
        raise ValueError(
            "PIL 与 OpenCV 读取到的 Query 尺寸不一致: "
            f"{tuple(original_query.shape[-2:])} != "
            f"{(original_height, original_width)}"
        )

    # query 用于模型输入，original_query 用于最终可视化
    query = original_query

    print(
        f"原始 Query 尺寸: "
        f"{original_width} × {original_height}"
    )
    print(
        f"原始 Prompt 尺寸: "
        f"{prompt.shape[-1]} × {prompt.shape[-2]}"
    )

    # Query 缩放到模型要求的输入尺寸
    if query.shape[-2:] != (img_size, img_size):
        resize_query = K.augmentation.Resize(
            [img_size, img_size],
            return_transform=True
        )
        query = resize_query(query)[0]

    # Prompt 缩放到模型要求的输入尺寸
    if prompt.shape[-2:] != (img_size, img_size):
        resize_prompt = K.augmentation.Resize(
            [img_size, img_size],
            return_transform=True
        )
        prompt = resize_prompt(prompt)[0]

    test_data = {
        "query_image": query.cuda(),
        "prompt_image": prompt.cuda(),
    }

    print("模型输入 Query 尺寸:", test_data["query_image"].shape)
    print("模型输入 Prompt 尺寸:", test_data["prompt_image"].shape)

    start_time = time.time()

    # forward
    with torch.no_grad():
        predicted_masks = model(test_data)

    print("模型输出掩码尺寸:", predicted_masks.shape)
    print(predicted_masks)

    # ---------------------------------------------------------
    # 将模型输出掩码恢复到原始 Query 分辨率
    # ---------------------------------------------------------

    # 去除 batch、channel 等长度为 1 的维度，得到 [H, W]
    predicted_mask = predicted_masks.detach().squeeze()

    # 转换为 F.interpolate 需要的 [N, C, H, W]
    predicted_mask = predicted_mask[None, None, :, :]

    # 从模型输出尺寸恢复为 Query 原始尺寸
    restored_mask = F.interpolate(
        predicted_mask,
        size=(original_height, original_width),
        mode='bilinear',
        align_corners=False
    )[0, 0]

    print("恢复后的掩码尺寸:", restored_mask.shape)

    # ---------------------------------------------------------
    # visualization
    # ---------------------------------------------------------

    # 与 test/test.py 的保存逻辑保持一致：模型输出越大，异常程度越高。
    # 不取反，也不对单张图片做 min-max normalize，否则会颠倒/扭曲
    # 模型输出的异常分数含义。
    anomaly_score = restored_mask.detach().cpu().numpy()
    anomaly_score = np.nan_to_num(
        anomaly_score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
    pred_gray = np.round(anomaly_score * 255).astype(np.uint8)

    # OpenCV 的读取、JET 着色、叠加和保存均使用 BGR，避免 RGB/BGR
    # 通道在流程中来回转换。JET 中低分为蓝色，高分为红色。
    color_map = cv2.applyColorMap(pred_gray, cv2.COLORMAP_JET)
    scoremap_self = cv2.addWeighted(query_img, 0.5, color_map, 0.5, 0)

    end_time = time.time()

    output_path = './anomaly/result/blue_craw.png'

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---------------------------------------------------------
    # 保存模型预测掩码
    # ---------------------------------------------------------

    # 保存灰度分数图：0 表示正常，255 表示异常
    gray_mask = pred_gray

    mask_output_path = './anomaly/gray_mask/blue_craw.png'

    # 保存二值掩码
    # 阈值可根据实际检测效果调整，例如 0.3、0.5、0.7
    threshold = 0.5

    binary_mask = (anomaly_score >= threshold).astype(np.uint8) * 255

    binary_mask_output_path = './anomaly/mask/1.png'

    expected_size = (original_height, original_width)
    for output_name, output_image in (
        ("灰度掩码", gray_mask),
        ("二值掩码", binary_mask),
        ("热力图", scoremap_self),
    ):
        if output_image.shape[:2] != expected_size:
            raise ValueError(
                f"{output_name}尺寸 {output_image.shape[:2]} "
                f"与 Query 原图尺寸 {expected_size} 不一致"
            )

    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(binary_mask_output_path), exist_ok=True)

    if not cv2.imwrite(mask_output_path, gray_mask):
        raise OSError(f"灰度掩码保存失败: {mask_output_path}")
    if not cv2.imwrite(binary_mask_output_path, binary_mask):
        raise OSError(f"二值掩码保存失败: {binary_mask_output_path}")
    if not cv2.imwrite(output_path, scoremap_self):
        raise OSError(f"热力图保存失败: {output_path}")

    run_time = end_time - start_time

    print("最终热力图尺寸:", scoremap_self.shape)
    print(f"所有结果均按原图尺寸保存: {original_width} × {original_height}")
    print(f"热力图已保存至: {output_path}")
    print(f"运行时间: {run_time:.4f} 秒")
