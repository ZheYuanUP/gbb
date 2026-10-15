#!/usr/bin/env python3
"""展示训练时 Prompt 轻微错位增强前后的对比。"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw
from torchvision.transforms.functional import resize

from datasets.pair_masks import affine_image_without_black_border


def parse_args():
    parser = argparse.ArgumentParser(description='生成 Prompt 几何变换前后的对比图。')
    parser.add_argument('--image', default='1.jpg')
    parser.add_argument('--output-dir', default='.', help='输出目录，默认为当前工作目录。')
    parser.add_argument('--size', type=int, default=512, help='训练输入尺寸。')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--degrees', type=float, default=0.0, help='最大旋转角度。')
    parser.add_argument(
        '--translate',
        type=float,
        default=0.004,
        help='最大平移比例，例如 0.02 表示宽高的 2%%。',
    )
    parser.add_argument('--scale-min', type=float, default=1.0)
    parser.add_argument('--scale-max', type=float, default=1.0)
    parser.add_argument('--padding', type=int, default=10, help='最小反射扩边像素数。')
    return parser.parse_args()


def validate_args(args):
    if args.degrees < 0:
        raise ValueError('--degrees 不能小于 0。')
    if args.translate < 0:
        raise ValueError('--translate 不能小于 0。')
    if not 0 < args.scale_min <= args.scale_max:
        raise ValueError('缩放范围必须满足 0 < scale-min <= scale-max。')
    if args.padding < 0:
        raise ValueError('--padding 不能小于 0。')
    if args.size <= 0:
        raise ValueError('--size 必须大于 0。')


def make_comparison(original, transformed):
    width, height = original.size
    title_height = 32
    canvas = Image.new('RGB', (width * 2, height + title_height), color='white')
    canvas.paste(original, (0, title_height))
    canvas.paste(transformed, (width, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), 'Before', fill='black')
    draw.text((width + 10, 9), 'After', fill='black')
    return canvas


def main():
    args = parse_args()
    validate_args(args)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f'找不到输入图片：{image_path}')

    with Image.open(image_path) as image:
        original = resize(image.convert('RGB'), [args.size, args.size])

    rng = random.Random(args.seed)
    width, height = original.size
    max_dx = int(round(args.translate * width))
    max_dy = int(round(args.translate * height))
    angle = rng.uniform(-args.degrees, args.degrees)
    translation = (
        rng.randint(-max_dx, max_dx) if max_dx else 0,
        rng.randint(-max_dy, max_dy) if max_dy else 0,
    )
    scale = rng.uniform(args.scale_min, args.scale_max)

    transformed, valid, actual_padding = affine_image_without_black_border(
        original,
        angle=angle,
        translate=translation,
        scale=scale,
        minimum_padding=args.padding,
    )
    if not valid:
        raise RuntimeError('扩边后仍存在无效区域，请增大 --padding。')
    if transformed.size != original.size:
        raise RuntimeError('变换前后的图片分辨率不一致。')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original.save(output_dir / 'prompt_before.png')
    transformed.save(output_dir / 'prompt_after.png')
    make_comparison(original, transformed).save(output_dir / 'prompt_comparison.png')

    print(f'输入图片：{image_path.resolve()}')
    print(f'图片尺寸：{width} x {height}')
    print(f'旋转角度：{angle:.4f}°')
    print(f'平移像素：x={translation[0]}, y={translation[1]}')
    print(f'缩放比例：{scale:.4f}')
    print(f'实际扩边：{actual_padding} 像素/边')
    print(f'对比图片：{(output_dir / "prompt_comparison.png").resolve()}')


if __name__ == '__main__':
    main()
