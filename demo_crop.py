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
from metauas_copy import MetaUAS, set_random_seed, normalize, apply_ad_scoremap, read_image_as_tensor, safely_load_state_dict, crop_tensor_pair, infer_tensor_pairs, stitch_masks


if __name__ == "__main__":
    random_seed = 1
    
    set_random_seed(random_seed)

    # ckt_path = 'weights/metauas-256.ckpt'
    # img_size = 256
    ckt_path = "weights/metauas-512.ckpt"
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
    
    start_time = time.time()
    # load test images
    path_to_prompt = "./images/mask_rgb_4/mask_rgb_4_0/ref.png"
    path_to_query = "./images/mask_rgb_4/mask_rgb_4_0/rgb8.png"

    query = read_image_as_tensor(path_to_query)
    prompt = read_image_as_tensor(path_to_prompt)

    if query.shape[1] != prompt.shape[1] or query.shape[2] != prompt.shape[2]:
        resize_trans = K.augmentation.Resize([prompt.shape[1], prompt.shape[2]], return_transform=True)
        query = resize_trans(query)[0].squeeze(0)

    # if query.shape[1] != img_size:
    #     resize_trans = K.augmentation.Resize([img_size, img_size], return_transform=True)
    #     query = resize_trans(query)[0]
    #     prompt = resize_trans(prompt)[0]
    

    # test_data = {
    #         "query_image": query.cuda(),
    #         "prompt_image": prompt.cuda(),
    #     }

    tensor_pairs, query_padded, cols, rows = crop_tensor_pair(
        query_tensor = query,
        prompt_tensor = prompt,
        crop_size= img_size
    )

    all_pred_masks = infer_tensor_pairs(
        tensor_pairs=tensor_pairs,
        model=model
    )

    full_mask = stitch_masks(
    pred_masks_list = all_pred_masks,
    crop_size = 512,
    cols = cols,
    rows = rows
    # original_h = query_padded.shape[2],
    # original_w = query_padded.shape[3]
    )
    pred = (1 - full_mask.squeeze())[:, :, None].repeat(3, axis=2)

    query_img = query_padded.squeeze(0)* 255
    query_img = query_img.permute(1,2,0)

    # if query_img.shape[0] != pred.shape[0] or query_img.shape[1] != pred.shape[1]:
    #     resize_trans = K.augmentation.Resize([query_img.shape[0], query_img.shape[1]], return_transform=True)
    #     pred = resize_trans(pred)[0].squeeze(0)

    scoremap_self = apply_ad_scoremap(query_img.cpu(), normalize(pred))
    cv2.imwrite('./anomaly_map.jpg', scoremap_self)
    pred_save = (1-normalize(pred))*255
    cv2.imwrite('./mask_rgb.jpg', pred_save)
    # forward
    # predicted_masks = model(test_data)

    # # visualization
    # query_img = test_data["query_image"][0] * 255
    # query_img = query_img.permute(1,2,0)

    # pred = (1-predicted_masks.squeeze().detach())[:, :, None].cpu().numpy().repeat(3, 2)
    # # normalize just for analysis
    # scoremap_self = apply_ad_scoremap(query_img.cpu(), normalize(pred))
    end_time = time.time()
    # cv2.imwrite('./anomaly_map.jpg', scoremap_self)
    run_time = end_time - start_time
    print(f"运行时间: {run_time:.4f} 秒")
   
