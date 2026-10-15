#!/usr/bin/env python3
'''
Build original/changed/mask training triples from MVTec-AD.

MVTec-AD expected layout:
  mvtec_root/
    bottle/
      train/good/*.png
      test/good/*.png
      test/broken_large/*.png
      ground_truth/broken_large/*_mask.png
    cable/
      ...

Output CSV columns:
  original,changed,mask,clsname,label,label_name
'''

import argparse
import csv
import random
from pathlib import Path

from PIL import Image


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def make_empty_mask(mask_path, image_path):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.exists():
        return
    with Image.open(image_path) as img:
        Image.new('L', img.size, 0).save(mask_path)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def find_mask(mask_dir, image_path):
    candidates = [
        mask_dir / f'{image_path.stem}_mask.png',
        mask_dir / f'{image_path.stem}_mask.jpg',
        mask_dir / f'{image_path.stem}.png',
        mask_dir / f'{image_path.stem}.jpg',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(mask_dir.glob(f'{image_path.stem}*'))
    for match in matches:
        if is_image(match):
            return match.resolve()
    raise FileNotFoundError(f'Cannot find mask for image: {image_path} in {mask_dir}')


def scan_mvtec(mvtec_root):
    mvtec_root = Path(mvtec_root).resolve()
    train_good_by_cls = {}
    samples = []

    for cls_dir in sorted(p for p in mvtec_root.iterdir() if p.is_dir()):
        clsname = cls_dir.name
        train_good_dir = cls_dir / 'train' / 'good'
        test_dir = cls_dir / 'test'
        if not train_good_dir.is_dir() or not test_dir.is_dir():
            continue

        train_good = sorted(p.resolve() for p in train_good_dir.iterdir() if is_image(p))
        if not train_good:
            continue
        train_good_by_cls[clsname] = train_good

        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            for image_path in sorted(p.resolve() for p in defect_dir.iterdir() if is_image(p)):
                if defect_type == 'good':
                    samples.append(
                        {
                            'filename': image_path,
                            'mask': None,
                            'clsname': clsname,
                            'label': 0,
                            'label_name': 'good',
                        }
                    )
                else:
                    mask_dir = cls_dir / 'ground_truth' / defect_type
                    mask_path = find_mask(mask_dir, image_path)
                    samples.append(
                        {
                            'filename': image_path,
                            'mask': mask_path,
                            'clsname': clsname,
                            'label': 1,
                            'label_name': defect_type,
                        }
                    )

    if not train_good_by_cls:
        raise ValueError(f'No MVTec classes found under: {mvtec_root}')
    if not samples:
        raise ValueError(f'No MVTec test images found under: {mvtec_root}')
    return train_good_by_cls, samples


def choose_originals(train_good_by_cls, clsname, mode, rng):
    originals = train_good_by_cls.get(clsname, [])
    if not originals:
        raise ValueError(f'No train/good images found for class: {clsname}')
    if mode == 'all_train':
        return originals
    if mode == 'random_per_sample':
        return [rng.choice(originals)]
    return [originals[0]]


def build_rows(args):
    rng = random.Random(args.seed)
    mvtec_root = Path(args.mvtec_root).resolve()
    output_root = Path(args.output_root).resolve()

    train_good_by_cls, samples = scan_mvtec(mvtec_root)
    for clsname in train_good_by_cls:
        train_good_by_cls[clsname] = sorted(train_good_by_cls[clsname])
        rng.shuffle(train_good_by_cls[clsname])

    rows = []
    for item in samples:
        label = int(item.get('label', 0))
        if label == 0 and args.exclude_good:
            continue

        clsname = item['clsname']
        changed_path = Path(item['filename']).resolve()
        if label == 1:
            mask_path = Path(item['mask']).resolve()
        else:
            rel = changed_path.relative_to(mvtec_root)
            mask_path = output_root / 'empty_masks' / rel.with_suffix('.png')
            make_empty_mask(mask_path, changed_path)
            mask_path = mask_path.resolve()

        if args.strict:
            for path in [changed_path, mask_path]:
                if not path.exists():
                    raise FileNotFoundError(path)

        originals = choose_originals(train_good_by_cls, clsname, args.prompt_mode, rng)
        for original_path in originals:
            if args.strict and not original_path.exists():
                raise FileNotFoundError(original_path)
            rows.append(
                {
                    'original': str(original_path),
                    'changed': str(changed_path),
                    'mask': str(mask_path),
                    'clsname': clsname,
                    'label': label,
                    'label_name': item.get('label_name', 'good' if label == 0 else 'defective'),
                }
            )

    rng.shuffle(rows)
    return rows


def write_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['original', 'changed', 'mask', 'clsname', 'label', 'label_name']
    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description='Build MetaUAS pair-mask manifest from MVTec-AD.')
    parser.add_argument('--mvtec-root', type=str, required=True, help='MVTec root containing class folders.')
    parser.add_argument('--output-root', type=str, required=True, help='Directory for train.csv and generated empty masks.')
    parser.add_argument('--output-csv', type=str, default='', help='Default: output-root/train.csv')
    parser.add_argument(
        '--prompt-mode',
        type=str,
        choices=['one_per_class', 'random_per_sample', 'all_train'],
        default='one_per_class',
        help='How to choose train/good original images for each test sample.',
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--exclude-good', action='store_true', help='Do not include test/good negative samples.')
    parser.add_argument('--strict', action='store_true', help='Fail if any referenced image or mask is missing.')
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_root / 'train.csv'
    rows = build_rows(args)
    write_csv(rows, output_csv)
    num_good = sum(1 for row in rows if int(row['label']) == 0)
    num_defect = len(rows) - num_good
    print(f'wrote: {output_csv}')
    print(f'rows: {len(rows)} defective: {num_defect} good: {num_good}')
    print(f'empty masks: {output_root / "empty_masks"}')


if __name__ == '__main__':
    main()
