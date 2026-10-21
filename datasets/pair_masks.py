"""Datasets and manifest readers for prompt/query/mask image triples."""
# 0721修改

import csv
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import adjust_brightness, affine, center_crop, gaussian_blur, hflip, pad, pil_to_tensor, resize, vflip

   
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
MASK_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MASK_PIXEL_THRESHOLD = 128.0


def read_manifest_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
    elif path.suffix.lower() in {'.jsonl', '.ndjson'}:
        with path.open('r', encoding='utf-8') as file:
            rows = [json.loads(line) for line in file if line.strip()]
    elif path.suffix.lower() == '.json':
        with path.open('r', encoding='utf-8') as file:
            rows = json.load(file)
        if isinstance(rows, dict):
            rows = rows.get('data', rows.get('samples', []))
    else:
        raise ValueError(f'Unsupported manifest format: {path}')

    if not rows:
        raise ValueError(f'No samples found in: {path}')
    return rows


def row_paths(row):
    original = row.get('original') or row.get('prompt') or row.get('reference') or row.get('normal')
    changed = row.get('changed') or row.get('query') or row.get('image') or row.get('defect')
    mask = row.get('mask') or row.get('maskname') or row.get('gt')
    if not (original and changed and mask):
        raise ValueError(
            'Each manifest row must contain original/changed/mask '
            '(aliases: prompt/query/gt are also accepted).'
        )
    return original, changed, mask


def read_pair_triples(path):
    return [row_paths(row) for row in read_manifest_rows(path)]


def scan_pair_directories(original_dir, changed_dir, mask_dir):
    original_dir = Path(original_dir)
    changed_dir = Path(changed_dir)
    mask_dir = Path(mask_dir)
    triples = []
    masks_by_stem = {
        path.relative_to(mask_dir).with_suffix('').as_posix(): path
        for path in mask_dir.rglob('*')
        if path.suffix.lower() in MASK_EXTENSIONS
    }

    for changed in sorted(changed_dir.rglob('*')):
        if changed.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = changed.relative_to(changed_dir)
        original = original_dir / relative
        if not original.exists():
            original = original_dir / changed.name
        mask = mask_dir / relative
        if not mask.exists():
            mask = masks_by_stem.get(relative.with_suffix('').as_posix())
        if original.exists() and mask is not None and Path(mask).exists():
            triples.append((str(original), str(changed), str(mask)))

    if not triples:
        raise ValueError('No triples found from original_dir/changed_dir/mask_dir.')
    return triples


def resolve_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def resolve_test_image_path(root, path):
    """Resolve a manifest image by its stem, regardless of file extension."""
    path = resolve_path(root, path)
    if path.is_file():
        return path
    matches = sorted(
        candidate for candidate in path.parent.iterdir()
        if candidate.is_file() and candidate.stem == path.stem
    ) if path.parent.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f'Multiple files match image name {path}: {matches}')
    raise FileNotFoundError(f'No image found with name {path.stem!r} in {path.parent}')


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def read_mask(path):
    with Image.open(path) as image:
        return image.convert('L')


def read_binary_mask(path):
    mask = pil_to_tensor(read_mask(path)).float()
    return (mask >= MASK_PIXEL_THRESHOLD).float()


def resize_binary_mask(mask, image_size):
    mask = resize(
        mask,
        image_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return (mask >= 0.5).float()


def affine_padding(image_size, angle, translate, scale, minimum=0):
    """Conservative padding needed to keep the output crop inside valid pixels."""
    width, height = image_size
    radians = math.radians(abs(angle))
    cos_a, sin_a = abs(math.cos(radians)), abs(math.sin(radians))
    safe_scale = max(float(scale), 1e-6)
    needed_x = (
        cos_a * width / 2 + sin_a * height / 2 + abs(translate[0])
    ) / safe_scale - width / 2
    needed_y = (
        sin_a * width / 2 + cos_a * height / 2 + abs(translate[1])
    ) / safe_scale - height / 2
    return max(int(minimum), int(math.ceil(max(0.0, needed_x, needed_y))) + 4)


def affine_image_without_black_border(image, angle, translate, scale, minimum_padding=0):
    """Apply affine augmentation on reflection padding and crop to the input size."""
    width, height = image.size
    padding = affine_padding(image.size, angle, translate, scale, minimum_padding)
    transformed = image
    for _ in range(6):
        expanded = pad(image, padding, padding_mode='reflect')
        transformed = affine(
            expanded,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        transformed = center_crop(transformed, [height, width])

        # Verify that no pixel in the final crop came from affine's constant fill.
        valid = Image.new('L', expanded.size, color=255)
        valid = affine(
            valid,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        valid = center_crop(valid, [height, width])
        if valid.getextrema()[0] == 255:
            return transformed, True, padding
        padding = max(padding + 8, int(math.ceil(padding * 1.5)))
    return transformed, False, padding


def affine_mask_on_padded_canvas(mask, angle, translate, scale, padding):
    """Apply the same affine transform to a mask while preserving its semantics."""
    width, height = mask.size
    expanded = pad(mask, padding, fill=0, padding_mode='constant')
    transformed = affine(
        expanded,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    return center_crop(transformed, [height, width])


class PairMaskTrainDataset(Dataset):
    """Training dataset with synchronized pair/mask augmentation."""

    def __init__(
        self,
        triples,
        root,
        image_size,
        augment=False,
        hflip_p=0.0,
        vflip_p=0.0,
        light_p=0.0,
        light_brightness=(0.8, 1.2),
        gaussian_blur_p=0.0,
        gaussian_blur_kernel_size=5,
        gaussian_blur_sigma=(0.1, 2.0),
    ):
        self.triples = triples
        self.root = root
        self.image_size = [image_size, image_size]
        self.augment = augment
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        if not 0.0 <= light_p <= 1.0:
            raise ValueError('light_p must be in [0, 1].')
        if light_brightness[0] <= 0 or light_brightness[1] < light_brightness[0]:
            raise ValueError('light_brightness must be positive and ordered min <= max.')
        self.light_p = light_p
        self.light_brightness = light_brightness
        if not 0.0 <= gaussian_blur_p <= 1.0:
            raise ValueError('gaussian_blur_p must be in [0, 1].')
        if gaussian_blur_kernel_size <= 0 or gaussian_blur_kernel_size % 2 == 0:
            raise ValueError('gaussian_blur_kernel_size must be a positive odd integer.')
        if gaussian_blur_sigma[0] <= 0 or gaussian_blur_sigma[1] < gaussian_blur_sigma[0]:
            raise ValueError('gaussian_blur_sigma must be positive and ordered min <= max.')
        self.gaussian_blur_p = gaussian_blur_p
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        original_path, changed_path, mask_path = self.triples[index]
        original_path = resolve_path(self.root, original_path)
        changed_path = resolve_path(self.root, changed_path)
        mask_path = resolve_path(self.root, mask_path)

        original = resize(read_rgb(original_path), self.image_size)
        changed = resize(read_rgb(changed_path), self.image_size)
        mask = resize_binary_mask(read_binary_mask(mask_path), self.image_size)

        if self.augment:
            if random.random() < self.hflip_p:
                original, changed, mask = hflip(original), hflip(changed), hflip(mask)
            if random.random() < self.vflip_p:
                original, changed, mask = vflip(original), vflip(changed), vflip(mask)
            # Lighting and Gaussian blur are sampled independently.
            if random.random() < self.light_p:
                brightness = random.uniform(*self.light_brightness)
                original = adjust_brightness(original, brightness)
                changed = adjust_brightness(changed, brightness)
            if random.random() < self.gaussian_blur_p:
                sigma = random.uniform(*self.gaussian_blur_sigma)
                kernel_size = [
                    self.gaussian_blur_kernel_size,
                    self.gaussian_blur_kernel_size,
                ]
                blur_sigma = [sigma, sigma]
                original = gaussian_blur(original, kernel_size, blur_sigma)
                changed = gaussian_blur(changed, kernel_size, blur_sigma)

        prompt_tensor = pil_to_tensor(original).float() / 255.0
        query_tensor = pil_to_tensor(changed).float() / 255.0
        mask_tensor = mask
        if not (prompt_tensor.shape[-2:] == query_tensor.shape[-2:] == mask_tensor.shape[-2:]):
            raise RuntimeError('Prompt, query, and mask resolutions must be identical.')

        return {
            'prompt_image': prompt_tensor,
            'query_image': query_tensor,
            'query_mask': mask_tensor,
            'original_path': str(original_path),
            'changed_path': str(changed_path),
            'mask_path': str(mask_path),
        }


class PairMaskTestDataset(Dataset):
    """Test dataset that derives each image-level label from its mask."""

    def __init__(self, data_root, manifest, image_size):
        self.data_root = Path(data_root)
        self.rows = read_manifest_rows(manifest)
        self.image_size = [image_size, image_size]
        for row in self.rows:
            row_paths(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        original, changed, mask = row_paths(row)
        prompt_path = resolve_test_image_path(self.data_root, original)
        query_path = resolve_test_image_path(self.data_root, changed)
        mask_path = resolve_test_image_path(self.data_root, mask)

        prompt = pil_to_tensor(read_rgb(prompt_path)).float() / 255.0
        query = pil_to_tensor(read_rgb(query_path)).float() / 255.0
        mask_tensor = read_binary_mask(mask_path)
        label = int(torch.any(mask_tensor).item())

        prompt = resize(prompt, self.image_size, antialias=True)
        query = resize(query, self.image_size, antialias=True)
        mask_tensor = resize_binary_mask(mask_tensor, self.image_size)

        return {
            'query_image': query,
            'prompt_image': prompt,
            'cls_name': row.get('clsname') or 'pair_masks',
            'query_filename': str(query_path),
            'prompt_filename': str(prompt_path),
            'query_label': label,
            'query_mask': mask_tensor,
            'prompt_mask': mask_tensor,
        }
