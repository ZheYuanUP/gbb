#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   metauas.py
@Time    :   2025/03/26 23:46:12
@Author  :   Bin-Bin Gao
@Email   :   csgaobb@gmail.com
@Homepage:   https://csgaobb.github.io/
@Version :   1.0
@Desc    :   some classes and functions for MetaUAS
'''


import os
import random
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
import time
import cv2

from PIL import Image
from einops import rearrange
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from segmentation_models_pytorch.unet.model import UnetDecoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders import get_encoder, get_preprocessing_params

# -------------------------- 核心工具函数 --------------------------
import torch
import numpy as np

def stitch_masks(pred_masks_list, crop_size=512, cols=None, rows=None, original_h=None, original_w=None):
    """
    将裁剪推理后的mask列表拼接为完整mask
    Args:
        pred_masks_list: 所有子mask列表（infer_tensor_pairs返回的all_predicted_masks）
        crop_size: 裁剪尺寸（默认512）
        original_h: 原始图片高度（可选，用于去除填充部分）
        original_w: 原始图片宽度（可选，用于去除填充部分）
    Returns:
        full_mask: 拼接后的完整mask（numpy数组，shape=(H,W) 或 (H,W,3)）
    """
    # 1. 确定拼接的行列数（按裁剪逻辑：先按行后按列）
    # 先获取子mask数量，计算行列数（假设裁剪时是rows行×cols列）
    num_masks = len(pred_masks_list)
    # 从第一个子mask获取单mask尺寸
    single_mask = pred_masks_list[0]
    # 统一转为numpy并去除batch/通道维（适配 (1,1,H,W) 或 (1,H,W,3) 等格式）
    if isinstance(single_mask, torch.Tensor):
        single_mask = single_mask.squeeze().detach().cpu().numpy()
    else:
        single_mask = single_mask.squeeze()
    # # 计算行列数（cols=总宽度/crop_size，rows=总高度/crop_size）
    # cols = int(np.ceil(np.sqrt(num_masks)))  # 先估算列数，或按原始尺寸计算
    # rows = int(np.ceil(num_masks / cols))
    
    # 2. 初始化拼接画布
    full_h = rows * crop_size
    full_w = cols * crop_size
    # 匹配子mask的通道数（单通道/三通道）
    if len(single_mask.shape) == 2:  # 单通道 (H,W)
        full_mask = np.zeros((full_h, full_w), dtype=single_mask.dtype)
    else:  # 三通道 (H,W,3)
        full_mask = np.zeros((full_h, full_w, 3), dtype=single_mask.dtype)
    
    # 3. 逐个子mask拼接到对应位置
    for idx, mask in enumerate(pred_masks_list):
        # 转为numpy并去除多余维度
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().detach().cpu().numpy()
        else:
            mask_np = mask.squeeze()
        # 计算当前mask的行列位置
        row = idx // cols
        col = idx % cols
        # 计算拼接坐标
        h_start = row * crop_size
        h_end = h_start + crop_size
        w_start = col * crop_size
        w_end = w_start + crop_size
        # 拼接到画布
        full_mask[h_start:h_end, w_start:w_end] = mask_np
    
    # 4. 去除填充部分（还原原始尺寸）
    if original_h and original_w:
        full_mask = full_mask[:original_h, :original_w]
    
    return full_mask

def tensor_to_padded_tensor(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    将输入tensor填充至crop_size的整数倍
    输入tensor格式：[1, C, H, W]（batch=1，通道在前）
    返回填充后的tensor：[1, C, H_padded, W_padded]
    """
    _, h, w = tensor.shape
    # 计算需要填充的高度和宽度
    pad_h = (crop_size - (h % crop_size)) % crop_size
    pad_w = (crop_size - (w % crop_size)) % crop_size
    # 填充（上下左右：上0，下pad_h；左0，右pad_w）
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=1.0)  # 白色填充（归一化后1.0）
    return tensor

def crop_tensor_pair(query_tensor: torch.Tensor, prompt_tensor: torch.Tensor, crop_size: int, save_dir: str = "cropped_images") -> list:
    """
    对已转换为tensor的query和prompt进行成对裁剪，返回子tensor列表
    输入tensor格式：[1, C, H, W]
    返回格式：[(query_sub1, prompt_sub1), (query_sub2, prompt_sub2), ...]
    """
    # 新增创建保存目录
    query_save_dir = os.path.join(save_dir, "query")
    prompt_save_dir = os.path.join(save_dir, "prompt")
    os.makedirs(query_save_dir, exist_ok=True)
    os.makedirs(prompt_save_dir, exist_ok=True)

    # 统一填充至crop_size整数倍
    query_padded = tensor_to_padded_tensor(query_tensor, crop_size)
    prompt_padded = tensor_to_padded_tensor(prompt_tensor, crop_size)
    assert query_padded.shape == prompt_padded.shape, "Query和Prompt tensor尺寸必须一致！"
    
    _, h, w = query_padded.shape
    cols = w // crop_size
    rows = h // crop_size
    tensor_pairs = []
    query_padded = query_padded[None,:,:,:]  # 添加batch维度
    prompt_padded = prompt_padded[None,:,:,:]
    # 逐行逐列裁剪tensor
    for row in range(rows):
        for col in range(cols):
            # 计算裁剪坐标（H方向：row*crop_size → (row+1)*crop_size）
            h_start = row * crop_size
            h_end = (row + 1) * crop_size
            w_start = col * crop_size
            w_end = (col + 1) * crop_size
            
            # 裁剪子tensor（保持[1, C, crop_size, crop_size]格式）
            query_sub = query_padded[:, :, h_start:h_end, w_start:w_end]
            prompt_sub = prompt_padded[:, :, h_start:h_end, w_start:w_end]
            
            tensor_pairs.append((query_sub, prompt_sub))
                        # -------------------------- 保存裁剪后的图片 --------------------------
            def save_tensor_as_img(tensor, save_path):
                """将 [1, C, H, W] 的tensor转为PIL图片并保存"""
                # 1. 去除batch维度 + 移至CPU + 转为numpy
                img_np = tensor.squeeze(0).detach().cpu().numpy()  # [C, H, W]
                # 2. 调整维度：(C, H, W) → (H, W, C)
                img_np = np.transpose(img_np, (1, 2, 0))
                # 3. 转为[0,255]
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                # 4. 转为PIL图片并保存
                img = Image.fromarray(img_np)
                img.save(save_path)
            # 生成保存路径
            img_name = f"row{row}_col{col}.jpg"
            query_save_path = os.path.join(query_save_dir, img_name)
            prompt_save_path = os.path.join(prompt_save_dir, img_name)
            
            # 保存query和prompt子图
            save_tensor_as_img(query_sub, query_save_path)
            save_tensor_as_img(prompt_sub, prompt_save_path)
    
    print(f"Tensor裁剪完成：生成 {len(tensor_pairs)} 对{crop_size}×{crop_size}子tensor")
    print(f"裁剪图片已保存至：{save_dir}")
    return tensor_pairs, query_padded, cols, rows

def infer_tensor_pairs(tensor_pairs: list, model):
    """
    遍历裁剪后的成对子tensor，依次构造test_data并送入模型推理
    复用原始推理逻辑（计时、cuda、model forward）
    """
    all_predicted_masks = []  # 保存所有子对推理结果
    total_time = 0.0
    
    for idx, (query_sub, prompt_sub) in enumerate(tensor_pairs):
        print(f"\n===== 推理第 {idx+1} 对子tensor =====")
        
        # 1. 构造测试数据
        test_data = {
            "query_image": query_sub.cuda(),
            "prompt_image": prompt_sub.cuda(),
        }
        
        # 2. 模型推理
        start_time = time.time()
        with torch.no_grad():
            predicted_masks = model(test_data)
        infer_time = time.time() - start_time
        total_time += infer_time
        
        # 3. 打印结果
        all_predicted_masks.append(predicted_masks)
        print(f"第 {idx+1} 对推理耗时：{infer_time:.2f}s")
        print(f"预测mask形状：{predicted_masks.shape if hasattr(predicted_masks, 'shape') else '未知'}")
    
    print(f"\n推理完成：共{len(tensor_pairs)}对，总耗时{total_time:.2f}s，平均每对{total_time/len(tensor_pairs):.2f}s")
    return all_predicted_masks


def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True 

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
    
def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def tensor_pairs(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def read_image_as_tensor(path_to_image):
    pil_image = Image.open(path_to_image).convert("RGB")        # (H, W, 3)
    image_as_tensor = pil_to_tensor(pil_image).float() / 255.0  # [C, H, W]
    return image_as_tensor

def safely_load_state_dict(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint), strict=True)
    return model

def visualizer(paths, anomaly_map, img_size, save_path, cls_name, shot=0):
    for idx, path in enumerate(paths):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        #mask = normalize(anomaly_map[idx])
        mask = anomaly_map[idx]
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'imgs-shot' + str(shot), cls_name[idx], cls)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)

def few_shot(memory, query):
    simscore = torch.einsum("bcij,bckl->bijkl", F.normalize(query, dim=1),  F.normalize(memory, dim=1))
    simscore = rearrange(simscore, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")

    M = 1/2 * torch.min(1.0 - simscore, dim = -1)[0]
    return M

class AlignmentModule(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, alignment_type="sa", fusion_policy='cat'):
        super().__init__()
        self.fusion_policy = fusion_policy
        self.alignment_layer = AlignmentLayer(input_channels, hidden_channels, alignment_type=alignment_type)
    
    def forward(self, query_features, prompt_features):
            if isinstance(prompt_features, list):
                aligned_prompt = []
                for i in range(len(prompt_features)):
                    weighted_prompt.append(self.alignment_layer(query_features, prompt_features[i]))
                aligned_prompt = torch.mean(torch.stack(aligned_prompt),0)
                
            else:
                aligned_prompt = self.alignment_layer(query_features, prompt_features)
           
            if self.fusion_policy == 'cat':
                query_features = rearrange(
                    [query_features, aligned_prompt], "two b c h w -> b (two c) h w"
                )
            elif self.fusion_policy == 'add':
                query_features = query_features + aligned_prompt
                
            elif self.fusion_policy == 'absdiff':
                query_features = (query_features - aligned_prompt).abs()
        
            return query_features
        
class AlignmentLayer(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256, alignment_type="sa"):
        super().__init__()
        self.alignment_type = alignment_type
        if alignment_type != "na":
            self.dimensionality_reduction = nn.Conv2d(
                input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, query_features, prompt_features):
        # no-alignment
        if self.alignment_type == 'na':
            return prompt_features
        else:
            Q = self.dimensionality_reduction(query_features)
            K = self.dimensionality_reduction(prompt_features)
            V = rearrange(prompt_features, "b c h w -> b c (h w)")

            soft_attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
            soft_attention_map = rearrange(soft_attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
            soft_attention_map = nn.Softmax(dim=3)(soft_attention_map)

            # soft-alignment
            if self.alignment_type == 'sa':
                aligned_features = torch.einsum("bijp,bcp->bcij", soft_attention_map, V)
            # hard-alignment
            if self.alignment_type == 'ha':
                max_v, max_index = attention_map.max(dim=-1, keepdim=True)
                hard_attention_map = (attention_map == max_v).float()
                aligned_features = torch.einsum("bijp,bcp->bcij", hard_attention_map, V)

            return aligned_features


class MetaUAS(pl.LightningModule):
    def __init__(self, encoder_name, decoder_name, encoder_depth, decoder_depth, num_alignment_layers, alignment_type, fusion_policy):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name 
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.num_alignment_layers = num_alignment_layers
        self.alignment_type = alignment_type
        self.fusion_policy = fusion_policy
        
      
        align_input_channels = [448, 160, 56]
        align_hidden_channels = [224, 80, 28]
        encoder_channels = [3, 48, 32, 56, 160, 448] 
        decoder_channels = [256, 128, 64, 64, 48]
        
        self.encoder = get_encoder(
            self.encoder_name,
            in_channels=3,
            depth=self.encoder_depth,
            weights="imagenet",)

        preparams = get_preprocessing_params(
            self.encoder_name, 
            pretrained="imagenet"
            )
        
        self.preprocess = transforms.Normalize(preparams['mean'], preparams['std'])

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if self.decoder_name == "unet":
            encoder_out_channels = encoder_channels[self.encoder_depth-self.decoder_depth:]
            if self.fusion_policy == 'cat':
                num_alignment_layers = self.num_alignment_layers
            elif self.fusion_policy == 'add' or self.fusion_policy == 'absdiff':
                num_alignment_layers = 0

            self.decoder = UnetDecoder(
                encoder_channels=encoder_out_channels,
                decoder_channels=decoder_channels,
                n_blocks= self.decoder_depth,
                attention_type="scse",
                num_coam_layers= num_alignment_layers,
            )
             
        elif self.decoder_name == "fpn":
            encoder_out_channels = encoder_channels
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=decoder_channels[-1],
                dropout=0.2,
                merge_policy="add",
            )

        elif self.decoder_name == "fpnadd":
            segmentation_channels = 256 #128
            encoder_out_channels = encoder_channels
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="add",
            )
        elif self.decoder_name == "fpncat":
            encoder_out_channels = encoder_channels
            segmentation_channels = 256 #128
            if self.fusion_policy == 'cat':
                for i in range(self.num_alignment_layers):
                    encoder_out_channels[-(i+1)] = 2 * encoder_out_channels[-(i+1)]
            
            self.decoder = FPNDecoder(
                encoder_channels= encoder_out_channels,
                encoder_depth=self.encoder_depth,
                pyramid_channels=256,
                segmentation_channels=segmentation_channels,
                dropout=0.2,
                merge_policy="cat",
            )
             
         
        if self.alignment_type == "sa" or self.alignment_type == "na" or  self.alignment_type == "ha" :
            self.alignment = nn.ModuleList(
                [
                    AlignmentModule(
                        input_channels=align_input_channels[i],
                        hidden_channels=align_hidden_channels[i],
                        alignment_type=self.alignment_type,
                        fusion_policy=self.fusion_policy,
                    )
                    for i in range(self.num_alignment_layers)
                ]
            )
       
        if self.decoder_name == "fpncat":
            self.mask_head = nn.Conv2d(
                segmentation_channels*4,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        elif self.decoder_name == "fpnadd":
            self.mask_head = nn.Conv2d(
                segmentation_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.mask_head = nn.Conv2d(
                decoder_channels[-1],
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
           
    def forward(self, batch):
        query_input = self.preprocess(batch["query_image"])
        prompt_input = self.preprocess(batch["prompt_image"]) 

        with torch.no_grad():
            query_encoded_features = self.encoder(query_input)
            prompt_encoded_features = self.encoder(prompt_input)
                
        for i in range(len(self.alignment)):
            query_encoded_features[-(i + 1)] = self.alignment[i](query_encoded_features[-(i + 1)], prompt_encoded_features[-(i + 1)])
        
        query_decoded_features = self.decoder(*query_encoded_features[self.encoder_depth-self.decoder_depth:])
        
        if self.decoder_name == "fpn" or self.decoder_name == "fpncat" or self.decoder_name == "fpnadd":
            output = F.interpolate(self.mask_head(query_decoded_features), scale_factor=4, mode="bilinear", align_corners=False) 
            
        elif self.decoder_name == "unet":
            if self.decoder_depth == 4:
                output = F.interpolate(self.mask_head(query_decoded_features), scale_factor=2, mode="bilinear", align_corners=False) 
            if self.decoder_depth == 5:
                if not self.training:
                    output = self.mask_head(query_decoded_features)

        return output.sigmoid()
