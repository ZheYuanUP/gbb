"""CSV-backed test dataset for Hedian original/query/mask triples."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize


class HedianPairDataset(Dataset):
    def __init__(self, data_root, manifest, image_size):
        self.data_root = Path(data_root)
        self.image_size = image_size
        with Path(manifest).open('r', encoding='utf-8-sig', newline='') as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f'No samples found in: {manifest}')

        required = {'original', 'changed', 'mask'}
        missing = required.difference(self.rows[0])
        if missing:
            raise ValueError(f'Missing CSV columns: {sorted(missing)}')

    def __len__(self):
        return len(self.rows)

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    @staticmethod
    def read_image(path):
        with Image.open(path) as image:
            return pil_to_tensor(image.convert('RGB')).float() / 255.0

    @staticmethod
    def read_mask(path):
        with Image.open(path) as image:
            return (pil_to_tensor(image.convert('L')) > 0).float()

    def __getitem__(self, index):
        row = self.rows[index]
        prompt_path = self.resolve(row['original'])
        query_path = self.resolve(row['changed'])
        mask_path = self.resolve(row['mask'])

        query = self.read_image(query_path)
        prompt = self.read_image(prompt_path)
        mask = self.read_mask(mask_path)
        # Derive the image-level ground truth from the original mask. An all-zero
        # mask is normal; any non-zero pixel marks the query as anomalous.
        label = int(torch.any(mask).item())
        size = [self.image_size, self.image_size]
        query = resize(query, size, antialias=True)
        prompt = resize(prompt, size, antialias=True)
        mask = resize(
            mask,
            size,
            interpolation=InterpolationMode.NEAREST,
        )

        class_name = row.get('clsname') or 'hedian'
        return {
            'query_image': query,
            'prompt_image': prompt,
            'cls_name': class_name,
            'query_filename': str(query_path),
            'prompt_filename': str(prompt_path),
            'query_label': label,
            'query_mask': mask,
            'prompt_mask': mask,
        }
