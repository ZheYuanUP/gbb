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
from metauas_copy import crop_pair_images

# 示例调用
if __name__ == "__main__":
    # 配置成对图像的路径
    IMG1_PATH = "./images/光强0参考.png"
    IMG2_PATH = "./images/光强0缺陷.png"
    
    # 执行成对裁剪
    crop_pair_images(
        img1_path=IMG1_PATH,
        img2_path=IMG2_PATH,
        root_output_dir="cropped_pair_images", 
        fill_mode=False, 
        fill_color=(255, 255, 255)
    )