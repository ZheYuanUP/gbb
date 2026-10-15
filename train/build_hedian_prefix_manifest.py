#!/usr/bin/env python3
"""Build leakage-free MetaUAS manifests for selected Hedian devices.

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
offset images are selected independently for the train, validation, and test manifests;
if an offset image is absent, the original prompt image is always retained.

Filename convention:
  <device>_<source_id>_<variant_id>.<extension>

For example, purple_craw_1_35 is variant 35 generated from source image 1 on
device purple_craw. All variants from the same source image are assigned to
the same split to prevent data leakage, while the split ratios target the
number of generated images rather than the number of source-image groups.
Images from devices not listed in USED_DEVICES are skipped.
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
VAL_OFFSET_PROBABILITY = 0.7
TEST_OFFSET_PROBABILITY = 0.7
# Only these devices participate in this manifest build.
USED_DEVICES = ('purple_craw',)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
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
            'selected according to the split probabilities; missing matches '
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
        '--val-csv',
        default='',
        help='Validation manifest path. Default: <data-root>/val.csv',
    )
    parser.add_argument(
        '--test-csv',
        default='',
        help='Test manifest path. Default: <data-root>/test.csv',
    )
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


def sample_identity(path):
    """Return (device, source_id, variant_id), or None for an invalid name."""
    match = re.fullmatch(
        r'(?P<device>.+)_(?P<source_id>\d+)_(?P<variant_id>\d+)',
        normalized_stem(path.stem),
    )
    if match is None:
        return None
    return (
        match.group('device'),
        int(match.group('source_id')),
        int(match.group('variant_id')),
    )


def source_group(path):
    """Return the device/source-image group used as the split unit."""
    identity = sample_identity(path)
    if identity is None:
        return None
    device, source_id, _ = identity
    if device not in USED_DEVICES:
        return None
    return device, source_id


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
    all_queries = sorted(path.resolve() for path in query_root.rglob('*') if is_image(path))
    queries = [path for path in all_queries if source_group(path) is not None]
    skipped_names = [path.name for path in all_queries if source_group(path) is None]
    masks = sorted(path.resolve() for path in mask_root.rglob('*') if is_image(path))

    if not prompts:
        raise ValueError(f'No prompt images found in: {prompt_root}')
    if not all_queries:
        raise ValueError(f'No query images found in: {query_root}')
    if not queries:
        raise ValueError(
            'No query images matched the required filename convention for devices: '
            f'{USED_DEVICES}'
        )
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

    return rows, skipped_names


def split_rows_by_source(rows, seed):
    """Split intact source groups while balancing the number of manifest rows."""
    ratios = (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    if any(ratio <= 0.0 for ratio in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError(
            'TRAIN_RATIO, VAL_RATIO and TEST_RATIO must be positive and sum to 1.'
        )

    rows_by_group = defaultdict(list)
    for row in rows:
        group = source_group(Path(row['changed']))
        if group is None:
            raise ValueError(f"Cannot derive source group from: {row['changed']}")
        rows_by_group[group].append(row)

    groups = sorted(rows_by_group)
    if len(groups) < 3:
        raise ValueError(
            'At least three original-image groups are required for non-empty '
            f'train/validation/test splits, found {len(groups)}.'
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Place large source groups first. The shuffle above provides seeded,
    # reproducible tie-breaking for groups with the same number of variants.
    groups.sort(key=lambda group: len(rows_by_group[group]), reverse=True)

    total_rows = sum(len(group_rows) for group_rows in rows_by_group.values())
    target_rows = [total_rows * ratio for ratio in ratios]
    selected_groups = [set(), set(), set()]
    selected_row_counts = [0, 0, 0]

    for position, group in enumerate(groups):
        group_size = len(rows_by_group[group])
        empty_splits = [index for index, selected in enumerate(selected_groups) if not selected]
        groups_left = len(groups) - position
        # Reserve the final groups for any still-empty split.
        candidates = empty_splits if groups_left == len(empty_splits) else range(3)
        split_index = min(
            candidates,
            key=lambda index: (
                (selected_row_counts[index] + group_size) / target_rows[index],
                index,
            ),
        )
        selected_groups[split_index].add(group)
        selected_row_counts[split_index] += group_size

    train_groups, val_groups, test_groups = selected_groups

    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError('Source-image group leakage detected between splits.')

    def collect(selected_groups):
        return [row for group in groups if group in selected_groups for row in rows_by_group[group]]

    return (
        collect(train_groups),
        collect(val_groups),
        collect(test_groups),
        (len(train_groups), len(val_groups), len(test_groups)),
    )


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
    val_csv = Path(args.val_csv).resolve() if args.val_csv else data_root / 'val.csv'
    test_csv = Path(args.test_csv).resolve() if args.test_csv else data_root / 'test.csv'

    for role, directory in (
        ('prompt', prompt_root),
        ('query', query_root),
        ('mask', mask_root),
    ):
        if not directory.is_dir():
            raise NotADirectoryError(f'{role} directory does not exist: {directory}')

    rows, skipped_names = build_rows(
        data_root,
        prompt_root,
        query_root,
        mask_root,
        check_size=not args.no_size_check,
        prompt_offset_root=prompt_offset_root,
    )
    train_rows, val_rows, test_rows, group_counts = split_rows_by_source(rows, args.seed)
    train_rows, train_offset_count, train_offset_available = select_prompt_offsets(
        train_rows,
        TRAIN_OFFSET_PROBABILITY,
        args.seed,
    )
    val_rows, val_offset_count, val_offset_available = select_prompt_offsets(
        val_rows,
        VAL_OFFSET_PROBABILITY,
        args.seed + 1,
    )
    test_rows, test_offset_count, test_offset_available = select_prompt_offsets(
        test_rows,
        TEST_OFFSET_PROBABILITY,
        args.seed + 2,
    )
    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)
    write_csv(test_rows, test_csv)

    print(f'train: {train_csv} ({len(train_rows)} rows)')
    print(f'val:   {val_csv} ({len(val_rows)} rows)')
    print(f'test:  {test_csv} ({len(test_rows)} rows)')
    print(f'skipped unmatched query images: {len(skipped_names)}')
    print(
        f'offset prompts: train={train_offset_count}/{train_offset_available} '
        f'(p={TRAIN_OFFSET_PROBABILITY:.0%}), '
        f'val={val_offset_count}/{val_offset_available} '
        f'(p={VAL_OFFSET_PROBABILITY:.0%}), '
        f'test={test_offset_count}/{test_offset_available} '
        f'(p={TEST_OFFSET_PROBABILITY:.0%})'
    )
    print(
        f'devices={USED_DEVICES}; source groups: '
        f'train={group_counts[0]}, val={group_counts[1]}, test={group_counts[2]}; '
        f'ratios={TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}; seed={args.seed}'
    )
    total_rows = len(train_rows) + len(val_rows) + len(test_rows)
    print(
        'actual image ratios: '
        f'train={len(train_rows) / total_rows:.1%}, '
        f'val={len(val_rows) / total_rows:.1%}, '
        f'test={len(test_rows) / total_rows:.1%}'
    )
    print(f'data root: {data_root}')


if __name__ == '__main__':
    main()
