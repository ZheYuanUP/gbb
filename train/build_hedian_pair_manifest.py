#!/usr/bin/env python3
"""Build a MetaUAS training manifest for dataset/hedian.

Expected layout:
  dataset/hedian/
    prompt/   # original/reference images
    prompt_offset/  # optional offset views of prompt images
    query/    # changed images
    mask/     # binary masks for query images

Files may be placed in matching subdirectories. Matching names can either be
identical or use role suffixes, for example 001_original.jpg, 001_query.jpg,
and 001_mask.png. An offset prompt drops the final numeric sequence suffix from
its prompt name, for example craw_near_1_3.png -> craw_near_1.png. Available
offset images are selected independently for the train and test manifests;
if an offset image is absent, the original prompt image is always retained.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
TRAIN_OFFSET_PROBABILITY = 0.4
TEST_OFFSET_PROBABILITY = 0.7
ROLE_SUFFIXES = (
    '_original',
    '-original',
    '_prompt',
    '-prompt',
    '_reference',
    '-reference',
    '_changed',
    '-changed',
    '_query',
    '-query',
    '_mask',
    '-mask',
    '_gt',
    '-gt',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build original/changed/mask CSV manifest for the Hedian dataset.'
    )
    parser.add_argument('--data-root', default='dataset/hedian', help='Hedian dataset root.')
    parser.add_argument('--prompt-dir', default='prompt', help='Prompt directory under data-root.')
    parser.add_argument(
        '--prompt-offset-dir',
        default='prompt_offset',
        help=(
            'Optional offset-prompt directory under data-root. Matching images are '
            'selected according to the train/test probabilities; missing matches '
            'fall back to prompt.'
        ),
    )
    parser.add_argument('--query-dir', default='query', help='Query directory under data-root.')
    parser.add_argument('--mask-dir', default='mask', help='Mask directory under data-root.')
    parser.add_argument(
        '--train-csv',
        '--output-csv',
        dest='train_csv',
        default='',
        help='Training manifest path. Default: <data-root>/train.csv',
    )
    parser.add_argument(
        '--test-csv',
        default='',
        help='Test manifest path. Default: <data-root>/test.csv',
    )
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--no-size-check',
        action='store_true',
        help='Allow a query and its mask to have different image sizes.',
    )
    return parser.parse_args()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalized_stem(stem):
    value = stem.lower()
    changed = True
    while changed:
        changed = False
        for suffix in ROLE_SUFFIXES:
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                changed = True
                break
    return value


def relative_key(path, root):
    relative = path.relative_to(root)
    parent = relative.parent.as_posix().lower()
    stem = normalized_stem(relative.stem)
    return f'{parent}/{stem}' if parent != '.' else stem


def build_indexes(paths, root):
    by_relative = defaultdict(list)
    by_stem = defaultdict(list)
    for path in paths:
        by_relative[relative_key(path, root)].append(path)
        by_stem[normalized_stem(path.stem)].append(path)
    return by_relative, by_stem


def offset_stem(prompt_stem):
    """Map craw_near_1_3 to craw_near_1; keep non-sequenced names unchanged."""
    match = re.fullmatch(r'(.+)_\d+', normalized_stem(prompt_stem))
    return match.group(1) if match else normalized_stem(prompt_stem)


def find_offset(prompt, prompt_root, by_relative, by_stem):
    """Return a prompt's offset view, or None when that view is unavailable."""
    relative = prompt.relative_to(prompt_root)
    parent = relative.parent.as_posix().lower()
    stem = offset_stem(relative.stem)
    rel_key = f'{parent}/{stem}' if parent != '.' else stem
    candidates = by_relative.get(rel_key, [])
    if not candidates:
        candidates = by_stem.get(stem, [])

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    joined = '\n  '.join(str(path) for path in candidates)
    raise ValueError(f'Ambiguous prompt_offset for prompt {prompt}:\n  {joined}')


def find_match(query, query_root, role, by_relative, by_stem):
    rel_key = relative_key(query, query_root)
    candidates = by_relative.get(rel_key, [])
    if not candidates:
        candidates = by_stem.get(normalized_stem(query.stem), [])

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f'No {role} found for query: {query}')
    joined = '\n  '.join(str(path) for path in candidates)
    raise ValueError(f'Ambiguous {role} for query {query}:\n  {joined}')


def verify_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f'Cannot read image {path}: {exc}') from exc


def image_size(path):
    with Image.open(path) as image:
        return image.size


def mask_has_anomaly(path):
    with Image.open(path) as image:
        extrema = image.convert('L').getextrema()
    return int(extrema[1] > 0)


def csv_path(path, data_root):
    """Use paths relative to data-root so the dataset remains relocatable."""
    return path.relative_to(data_root).as_posix()


def build_rows(
    data_root,
    prompt_root,
    query_root,
    mask_root,
    check_size,
    prompt_offset_root=None,
):
    if prompt_offset_root is None:
        prompt_offset_root = data_root / 'prompt_offset'
    prompts = sorted(path.resolve() for path in prompt_root.rglob('*') if is_image(path))
    prompt_offsets = sorted(
        path.resolve() for path in prompt_offset_root.rglob('*') if is_image(path)
    ) if prompt_offset_root.is_dir() else []
    queries = sorted(path.resolve() for path in query_root.rglob('*') if is_image(path))
    masks = sorted(path.resolve() for path in mask_root.rglob('*') if is_image(path))

    if not prompts:
        raise ValueError(f'No prompt images found in: {prompt_root}')
    if not queries:
        raise ValueError(f'No query images found in: {query_root}')
    if not masks:
        raise ValueError(f'No mask images found in: {mask_root}')

    prompt_by_relative, prompt_by_stem = build_indexes(prompts, prompt_root)
    offset_by_relative, offset_by_stem = build_indexes(
        prompt_offsets, prompt_offset_root
    )
    mask_by_relative, mask_by_stem = build_indexes(masks, mask_root)
    rows = []

    for query in queries:
        prompt = find_match(
            query, query_root, 'prompt', prompt_by_relative, prompt_by_stem
        )
        prompt_offset = find_offset(
            prompt,
            prompt_root,
            offset_by_relative,
            offset_by_stem,
        )
        mask = find_match(query, query_root, 'mask', mask_by_relative, mask_by_stem)

        for path in (prompt, query, mask):
            verify_image(path)
        if prompt_offset is not None:
            verify_image(prompt_offset)
        if check_size and image_size(query) != image_size(mask):
            raise ValueError(
                f'Query and mask sizes differ: {query} {image_size(query)} != '
                f'{mask} {image_size(mask)}'
            )

        relative_query = query.relative_to(query_root)
        class_name = relative_query.parts[0] if len(relative_query.parts) > 1 else 'hedian'
        label = mask_has_anomaly(mask)
        rows.append(
            {
                'original': csv_path(prompt, data_root),
                '_prompt_offset': (
                    csv_path(prompt_offset, data_root) if prompt_offset is not None else ''
                ),
                'changed': csv_path(query, data_root),
                'mask': csv_path(mask, data_root),
                'clsname': class_name,
                'label': label,
                'label_name': 'defective' if label else 'good',
            }
        )

    return rows


def split_rows(rows, test_ratio, seed):
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f'--test-ratio must be between 0 and 1, got {test_ratio}')
    if len(rows) < 2:
        raise ValueError('At least two paired samples are required for a train/test split.')

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_ratio))
    test_count = min(test_count, len(shuffled) - 1)
    return shuffled[test_count:], shuffled[:test_count]


def select_prompt_offsets(rows, probability, seed):
    """Select available offset prompts reproducibly and remove the private CSV field."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f'Offset probability must be between 0 and 1, got {probability}'
        )

    rng = random.Random(seed)
    selected_rows = []
    selected_count = 0
    available_count = 0
    for source_row in rows:
        row = dict(source_row)
        prompt_offset = row.pop('_prompt_offset', '')
        if prompt_offset:
            available_count += 1
            if rng.random() < probability:
                row['original'] = prompt_offset
                selected_count += 1
        selected_rows.append(row)
    return selected_rows, selected_count, available_count


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'original',
                'changed',
                'mask',
                'clsname',
                'label',
                'label_name',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    prompt_root = (data_root / args.prompt_dir).resolve()
    prompt_offset_root = (data_root / args.prompt_offset_dir).resolve()
    query_root = (data_root / args.query_dir).resolve()
    mask_root = (data_root / args.mask_dir).resolve()
    train_csv = Path(args.train_csv).resolve() if args.train_csv else data_root / 'train.csv'
    test_csv = Path(args.test_csv).resolve() if args.test_csv else data_root / 'test.csv'

    for role, directory in (
        ('prompt', prompt_root),
        ('query', query_root),
        ('mask', mask_root),
    ):
        if not directory.is_dir():
            raise NotADirectoryError(f'{role} directory does not exist: {directory}')

    rows = build_rows(
        data_root,
        prompt_root,
        query_root,
        mask_root,
        check_size=not args.no_size_check,
        prompt_offset_root=prompt_offset_root,
    )
    train_rows, test_rows = split_rows(rows, args.test_ratio, args.seed)
    train_rows, train_offset_count, train_offset_available = select_prompt_offsets(
        train_rows,
        TRAIN_OFFSET_PROBABILITY,
        args.seed,
    )
    test_rows, test_offset_count, test_offset_available = select_prompt_offsets(
        test_rows,
        TEST_OFFSET_PROBABILITY,
        args.seed + 1,
    )
    write_csv(train_rows, train_csv)
    write_csv(test_rows, test_csv)

    print(f'train: {train_csv} ({len(train_rows)} rows)')
    print(f'test:  {test_csv} ({len(test_rows)} rows)')
    print(
        f'offset prompts: train={train_offset_count}/{train_offset_available} '
        f'(p={TRAIN_OFFSET_PROBABILITY:.0%}), '
        f'test={test_offset_count}/{test_offset_available} '
        f'(p={TEST_OFFSET_PROBABILITY:.0%})'
    )
    print(f'split: {1.0 - args.test_ratio:.1%}/{args.test_ratio:.1%} seed={args.seed}')
    print(f'data root: {data_root}')


if __name__ == '__main__':
    main()
