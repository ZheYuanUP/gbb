# -*- coding: utf-8 -*-
"""Batch inference for paired images in ``prompt`` and ``query`` folders.

Expected directory layout::

    data_path/
        prompt/
            image_001.jpg
        query/
            image_001.jpg

The generated anomaly heatmaps are written to ``data_path/result``. Binary
masks are written to ``data_path/result/mask`` as lossless PNG files.

Edit ``DATA_PATH`` in the configuration section below, then run this script
directly with ``python batch_real_data.py``.
"""

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import kornia as K
import numpy as np
import torch
import torch.nn.functional as F

from metauas import MetaUAS, read_image_as_tensor, safely_load_state_dict, set_random_seed


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 配置区：运行前只需修改这里，不需要传递命令行参数。
# DATA_PATH 下应包含 prompt 和 query 两个文件夹。
# ---------------------------------------------------------------------------
DATA_PATH = Path(r"C:/Users/Jack/Desktop/核电/0完整流程测试结果")
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_val_dice_test_device1.pth"
IMG_SIZE = 512
DEVICE = "cuda"  # 可填写 "cuda"、"cuda:0" 或 "cpu"
MASK_THRESHOLD = 0.6


def read_bgr(path: Path) -> np.ndarray:
    """Read an image with OpenCV, including paths containing Chinese text."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image with OpenCV, including paths containing Chinese text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def find_query_files(query_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in query_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_model(checkpoint: Path, device: torch.device) -> MetaUAS:
    # Model configuration is kept identical to demo_real_data.py.
    model = MetaUAS(
        "efficientnet-b4",
        "unet",
        5,
        5,
        3,
        "sa",
        "cat",
    )

    if device.type == "cpu":
        # safely_load_state_dict uses torch.load without map_location. Loading
        # explicitly here also permits CUDA-produced checkpoints on CPU hosts.
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        model = safely_load_state_dict(model, str(checkpoint))

    model.to(device)
    model.eval()
    return model


def resize_for_model(image: torch.Tensor, img_size: int) -> torch.Tensor:
    if image.shape[-2:] == (img_size, img_size):
        return image
    resize = K.augmentation.Resize([img_size, img_size], return_transform=True)
    return resize(image)[0]


def infer_one(
    model: MetaUAS,
    prompt_path: Path,
    query_path: Path,
    img_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    query_bgr = read_bgr(query_path)
    original_height, original_width = query_bgr.shape[:2]

    query = resize_for_model(read_image_as_tensor(str(query_path)), img_size)
    prompt = resize_for_model(read_image_as_tensor(str(prompt_path)), img_size)
    test_data = {
        "query_image": query.to(device),
        "prompt_image": prompt.to(device),
    }

    with torch.no_grad():
        predicted_masks = model(test_data)

    predicted_mask = predicted_masks.detach().squeeze()[None, None, :, :]
    restored_mask = F.interpolate(
        predicted_mask,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # This visualization follows demo_real_data.py: a larger model output is a
    # higher anomaly score, without per-image min-max normalization.
    anomaly_score = restored_mask.cpu().numpy()
    anomaly_score = np.nan_to_num(anomaly_score, nan=0.0, posinf=1.0, neginf=0.0)
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
    pred_gray = np.round(anomaly_score * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(pred_gray, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(query_bgr, 0.5, color_map, 0.5, 0)
    binary_mask = (anomaly_score >= MASK_THRESHOLD).astype(np.uint8) * 255
    return heatmap, binary_mask


def main() -> None:
    data_path = DATA_PATH.expanduser().resolve()
    prompt_dir = data_path / "prompt"
    query_dir = data_path / "query"
    result_dir = data_path / "result"
    mask_dir = result_dir / "mask"
    checkpoint = CHECKPOINT_PATH.expanduser().resolve()

    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"prompt 目录不存在：{prompt_dir}")
    if not query_dir.is_dir():
        raise FileNotFoundError(f"query 目录不存在：{query_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint}")
    if IMG_SIZE <= 0:
        raise ValueError("IMG_SIZE 必须大于 0")
    if not 0.0 <= MASK_THRESHOLD <= 1.0:
        raise ValueError("MASK_THRESHOLD 必须在 0 到 1 之间")

    query_files = list(find_query_files(query_dir))
    if not query_files:
        raise FileNotFoundError(f"query 目录中没有支持的图片：{query_dir}")

    pairs = []
    missing_prompts = []
    for query_path in query_files:
        relative_path = query_path.relative_to(query_dir)
        prompt_path = prompt_dir / relative_path
        if prompt_path.is_file():
            pairs.append((prompt_path, query_path, relative_path))
        else:
            missing_prompts.append(relative_path)

    if missing_prompts:
        missing_text = "\n".join(f"  - {path}" for path in missing_prompts)
        raise FileNotFoundError(f"以下 query 图片没有同名 prompt 图片：\n{missing_text}")

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；如需使用 CPU，请将 DEVICE 改为 "cpu"')

    set_random_seed(1)
    print(f"正在加载模型：{checkpoint}")
    model = build_model(checkpoint, device)
    result_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"共找到 {len(pairs)} 对图片")
    print(f"热力图保存到：{result_dir}")
    print(f"二值 mask 保存到：{mask_dir}（阈值：{MASK_THRESHOLD}）")
    for index, (prompt_path, query_path, relative_path) in enumerate(pairs, start=1):
        output_path = result_dir / relative_path
        mask_output_path = mask_dir / relative_path.with_suffix(".png")
        heatmap, binary_mask = infer_one(
            model, prompt_path, query_path, IMG_SIZE, device
        )
        write_image(output_path, heatmap)
        write_image(mask_output_path, binary_mask)
        print(
            f"[{index}/{len(pairs)}] {relative_path} -> "
            f"{output_path}；mask -> {mask_output_path}"
        )

    print(f"处理完成，共生成 {len(pairs)} 个结果。")


if __name__ == "__main__":
    main()
