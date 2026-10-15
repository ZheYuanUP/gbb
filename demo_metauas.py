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
from metauas import MetaUAS, set_random_seed, read_image_as_tensor, safely_load_state_dict

if __name__ == "__main__":
    random_seed = 1
    
    set_random_seed(random_seed)

    # ckt_path = 'weights/metauas-256.ckpt'
    # img_size = 256
    # ckt_path = "weights/metauas-512.ckpt"
    ckt_path = "checkpoints/best_val_dice.pth"
    img_size = 512

    # load model
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


    model = safely_load_state_dict(model, ckt_path)
    model.cuda()
    model.eval()
    

    # load test images
    # path_root = "./images/"
    # path_to_prompt = path_root + "光强0参考.png"
    # path_to_query = path_root + "光强0缺陷.png"

    path_to_prompt = "./prompt1.png"
    path_to_query = "./query2.png"

    # path_to_prompt = '/home_ssd/lwj/batch1_results/prompt/14_1.jpg'
    # path_to_query = '/home_ssd/lwj/batch1_results/query/14_1.png'


    # Keep the original query for visualization. The model still receives a
    # resized copy, while the predicted mask is restored to this resolution.
    original_query = read_image_as_tensor(path_to_query)
    original_height, original_width = original_query.shape[-2:]
    query = original_query
    prompt = read_image_as_tensor(path_to_prompt)

    if query.shape[-2:] != (img_size, img_size):
        resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
        query = resize_trans(query)[0]
    if prompt.shape[-2:] != (img_size, img_size):
        resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
        prompt = resize_trans(prompt)[0]
    

    test_data = {
            "query_image": query.cuda(),
            "prompt_image": prompt.cuda(),
        }
    start_time = time.time()
    # forward
    predicted_masks = model(test_data)

    # visualization
    query_img = (original_query * 255).permute(1, 2, 0).cpu().numpy()

    restored_mask = F.interpolate(
        predicted_masks.detach().squeeze()[None, None],
        size=(original_height, original_width),
        mode='bilinear',
        align_corners=False,
    )[0, 0]
    anomaly_score = restored_mask.cpu().numpy()
    visualization_threshold = 0.5

    # Color only predicted anomaly pixels. Normal pixels retain the original
    # query color instead of receiving the blue end of the JET color map.
    heatmap = cv2.applyColorMap(
        (np.clip(anomaly_score, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha_map = (anomaly_score >= visualization_threshold).astype(np.float32)
    alpha_map = (0.5 * alpha_map)[:, :, None]
    scoremap_self = (
        query_img.astype(np.float32) * (1.0 - alpha_map)
        + heatmap * alpha_map
    ).astype(np.uint8)
    end_time = time.time()
    scoremap_self = cv2.cvtColor(scoremap_self, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./anomaly_map7.jpg', scoremap_self)
    run_time = end_time - start_time
    print(f"运行时间: {run_time:.4f} 秒")
   
