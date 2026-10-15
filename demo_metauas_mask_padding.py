# -*- coding: utf-8 -*-
"""MetaUAS inference without aspect-ratio distortion.

The subject bounding box is obtained from one shared mask.  Its longest edge
is rounded up to ``SIZE_MULTIPLE`` (512 by default), then the same
square crop/padding transform is applied to both prompt and query images.
Pixels outside the source image are filled with black.  The complete expanded
pair is sent to the model without tiling or resizing. Predictions are saved
both on the padded canvas and restored to the original query size.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 配置区：运行前直接修改下列变量，不需要传递命令行参数。
# prompt、query 和主体 mask 必须具有相同的原始尺寸。
# ---------------------------------------------------------------------------
PROMPT_PATH = PROJECT_ROOT / "prompt1.png"
QUERY_PATH = PROJECT_ROOT / "query1.png"
SUBJECT_MASK_PATH = PROJECT_ROOT / "subject_mask.png"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_val_dice.pth"
OUTPUT_DIR = PROJECT_ROOT / "mask_padding_results"
SIZE_MULTIPLE = 512
MASK_THRESHOLD = 0
PREDICTION_THRESHOLD = 0.5
DEVICE = "cuda"  # 可填写 "cuda"、"cuda:0" 或 "cpu"


@dataclass(frozen=True)
class SquareTransform:
    """A square window expressed in coordinates of the original image."""

    x0: int
    y0: int
    side: int
    original_width: int
    original_height: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int


def read_image(path: Path, flags: int) -> np.ndarray:
    """Read via imdecode so paths containing Chinese characters also work."""
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def make_transform(
    mask: np.ndarray, multiple: int, original_size: Tuple[int, int], threshold: int
) -> SquareTransform:
    """Build a mask-centred square whose side is a multiple of ``multiple``."""
    original_height, original_width = original_size
    if mask.shape != (original_height, original_width):
        raise ValueError(
            "主体 mask 尺寸必须与 prompt/query 一致："
            f"mask={mask.shape[::-1]}，图像={(original_width, original_height)}"
        )

    ys, xs = np.nonzero(mask > threshold)
    if xs.size == 0:
        raise ValueError("主体 mask 中没有前景像素，请检查 mask 或 MASK_THRESHOLD。")

    bbox_x0 = int(xs.min())
    bbox_y0 = int(ys.min())
    bbox_x1 = int(xs.max()) + 1
    bbox_y1 = int(ys.max()) + 1
    bbox_width = bbox_x1 - bbox_x0
    bbox_height = bbox_y1 - bbox_y0
    longest_edge = max(bbox_width, bbox_height)
    side = int(math.ceil(longest_edge / multiple) * multiple)

    # Integer arithmetic keeps the subject centre as close as possible to the
    # canvas centre. x0/y0 may be negative; that area becomes black padding.
    x0 = (bbox_x0 + bbox_x1 - side) // 2
    y0 = (bbox_y0 + bbox_y1 - side) // 2
    return SquareTransform(
        x0=x0,
        y0=y0,
        side=side,
        original_width=original_width,
        original_height=original_height,
        bbox_x=bbox_x0,
        bbox_y=bbox_y0,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )


def apply_transform(image: np.ndarray, transform: SquareTransform) -> np.ndarray:
    """Crop into a square canvas, using zeros where the window is out of bounds."""
    if image.shape[:2] != (transform.original_height, transform.original_width):
        raise ValueError("待处理图像尺寸与变换记录的原图尺寸不一致。")

    canvas_shape = (transform.side, transform.side) + image.shape[2:]
    canvas = np.zeros(canvas_shape, dtype=image.dtype)
    src_x0 = max(transform.x0, 0)
    src_y0 = max(transform.y0, 0)
    src_x1 = min(transform.x0 + transform.side, transform.original_width)
    src_y1 = min(transform.y0 + transform.side, transform.original_height)
    dst_x0 = src_x0 - transform.x0
    dst_y0 = src_y0 - transform.y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return canvas


def restore_to_original(canvas: np.ndarray, transform: SquareTransform) -> np.ndarray:
    """Paste the valid canvas area back without resizing or interpolation."""
    if canvas.shape[:2] != (transform.side, transform.side):
        raise ValueError("检测结果尺寸与扩充画布尺寸不一致。")

    restored_shape = (
        transform.original_height,
        transform.original_width,
    ) + canvas.shape[2:]
    restored = np.zeros(restored_shape, dtype=canvas.dtype)
    dst_x0 = max(transform.x0, 0)
    dst_y0 = max(transform.y0, 0)
    dst_x1 = min(transform.x0 + transform.side, transform.original_width)
    dst_y1 = min(transform.y0 + transform.side, transform.original_height)
    src_x0 = dst_x0 - transform.x0
    src_y0 = dst_y0 - transform.y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    restored[dst_y0:dst_y1, dst_x0:dst_x1] = canvas[src_y0:src_y1, src_x0:src_x1]
    return restored


def bgr_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return tensor.float().div_(255.0).unsqueeze(0).to(device)


def build_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    # Delay the project import so geometry utilities remain usable in a
    # lightweight environment without all model packages.
    from metauas import MetaUAS, safely_load_state_dict

    if not checkpoint.is_file():
        raise FileNotFoundError(f"模型权重不存在：{checkpoint}")
    model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat")
    if device.type == "cpu":
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        model = safely_load_state_dict(model, str(checkpoint))
    model.to(device)
    model.eval()
    return model


def infer_full_image(
    model: torch.nn.Module,
    padded_prompt: np.ndarray,
    padded_query: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Infer the complete expanded pair without tiling or resizing."""
    if padded_prompt.shape != padded_query.shape:
        raise ValueError("扩充后的 prompt/query 尺寸必须一致。")
    test_data = {
        "query_image": bgr_to_tensor(padded_query, device),
        "prompt_image": bgr_to_tensor(padded_prompt, device),
    }
    predicted = model(test_data).detach().squeeze().cpu().numpy()
    expected_shape = padded_query.shape[:2]
    if predicted.shape != expected_shape:
        raise RuntimeError(
            "模型输出尺寸与扩充图不一致，不能无失真还原："
            f"output={predicted.shape[::-1]}，canvas={expected_shape[::-1]}"
        )
    return predicted


def score_to_outputs(query_bgr: np.ndarray, score: np.ndarray, threshold: float):
    clipped = np.clip(np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    gray = np.round(clipped * 255).astype(np.uint8)
    binary = (clipped >= threshold).astype(np.uint8) * 255
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    alpha = (binary.astype(np.float32) / 255.0 * 0.5)[..., None]
    heatmap = (query_bgr.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(
        np.uint8
    )
    return gray, binary, heatmap


def validate_config() -> None:
    if SIZE_MULTIPLE <= 0:
        raise ValueError("SIZE_MULTIPLE 必须大于 0。")
    if not 0 <= MASK_THRESHOLD <= 254:
        raise ValueError("MASK_THRESHOLD 必须在 0~254 之间。")
    if not 0.0 <= PREDICTION_THRESHOLD <= 1.0:
        raise ValueError("PREDICTION_THRESHOLD 必须在 0~1 之间。")


def main() -> None:
    validate_config()
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；如需使用 CPU，请将 DEVICE 改为 "cpu"。')

    prompt_bgr = read_image(PROMPT_PATH, cv2.IMREAD_COLOR)
    query_bgr = read_image(QUERY_PATH, cv2.IMREAD_COLOR)
    subject_mask = read_image(SUBJECT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if prompt_bgr.shape[:2] != query_bgr.shape[:2]:
        raise ValueError(
            "为保证 prompt/query 使用完全相同的几何变换，两者原始尺寸必须一致："
            f"prompt={prompt_bgr.shape[1]}x{prompt_bgr.shape[0]}，"
            f"query={query_bgr.shape[1]}x{query_bgr.shape[0]}"
        )

    transform = make_transform(
        subject_mask, SIZE_MULTIPLE, query_bgr.shape[:2], MASK_THRESHOLD
    )
    padded_prompt = apply_transform(prompt_bgr, transform)
    padded_query = apply_transform(query_bgr, transform)
    padded_subject_mask = apply_transform(subject_mask, transform)

    output_dir = OUTPUT_DIR.resolve()
    write_image(output_dir / "padded_prompt.png", padded_prompt)
    write_image(output_dir / "padded_query.png", padded_query)
    write_image(output_dir / "padded_subject_mask.png", padded_subject_mask)

    # Import only when inference is actually requested; metauas.py pulls in
    # the complete training/inference dependency stack.
    from metauas import set_random_seed

    set_random_seed(1)
    model = build_model(CHECKPOINT_PATH, device)
    start_time = time.time()
    with torch.no_grad():
        predicted = infer_full_image(model, padded_prompt, padded_query, device)
    elapsed = time.time() - start_time
    if predicted.ndim != 2:
        raise RuntimeError(f"模型输出应为二维分数图，实际 shape={predicted.shape}")
    if predicted.shape != (transform.side, transform.side):
        raise RuntimeError(
            "模型输出尺寸与扩充图不一致，不能无失真还原："
            f"output={predicted.shape[::-1]}，canvas={(transform.side, transform.side)}"
        )

    padded_gray, padded_binary, padded_heatmap = score_to_outputs(
        padded_query, predicted, PREDICTION_THRESHOLD
    )
    restored_score = restore_to_original(predicted, transform)
    restored_gray, restored_binary, restored_heatmap = score_to_outputs(
        query_bgr, restored_score, PREDICTION_THRESHOLD
    )

    write_image(output_dir / "padded_score.png", padded_gray)
    write_image(output_dir / "padded_binary_mask.png", padded_binary)
    write_image(output_dir / "padded_detection.png", padded_heatmap)
    write_image(output_dir / "restored_score.png", restored_gray)
    write_image(output_dir / "restored_binary_mask.png", restored_binary)
    write_image(output_dir / "restored_detection.png", restored_heatmap)

    metadata = {
        "prompt": str(PROMPT_PATH.resolve()),
        "query": str(QUERY_PATH.resolve()),
        "subject_mask": str(SUBJECT_MASK_PATH.resolve()),
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "size_multiple": SIZE_MULTIPLE,
        "inference_mode": "full_image_without_resize",
        "mask_threshold": MASK_THRESHOLD,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "transform": asdict(transform),
        "inference_seconds": elapsed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transform.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"主体 bbox: {transform.bbox_width}x{transform.bbox_height}，"
        f"扩充画布: {transform.side}x{transform.side}"
    )
    print(f"共享窗口左上角（原图坐标）: ({transform.x0}, {transform.y0})")
    print(
        f"还原结果尺寸: {restored_heatmap.shape[1]}x{restored_heatmap.shape[0]}"
    )
    print(f"推理耗时: {elapsed:.4f} 秒")
    print(f"全部结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
