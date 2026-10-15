#!/usr/bin/env python3
"""Append prompt/query/mask datasets to a Hedian training manifest.

The Hedian manifests must already have been generated.  Every extra dataset is
matched by filename (role suffixes are ignored) and appended to training using
absolute paths.  The Hedian validation and test manifests are deliberately
never modified.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
ROLE_SUFFIXES = (
    '_original', '-original', '_prompt', '-prompt', '_reference', '-reference',
    '_changed', '-changed', '_query', '-query', '_mask', '-mask', '_gt', '-gt',
)
FIELDNAMES = ('original', 'changed', 'mask', 'clsname', 'label', 'label_name')
DIRECTORY_ALIASES = {
    'prompt': ('prompt', 'original', 'reference'),
    'query': ('query', 'changed'),
    'mask': ('mask', 'masks', 'gt'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append extra pair datasets to an existing Hedian train CSV file.'
    )
    parser.add_argument(
        '--extra-root',
        action='append',
        required=True,
        help='Extra dataset root; repeat this option for multiple datasets.',
    )
    parser.add_argument('--train-csv', required=True, help='Existing Hedian train CSV.')
    parser.add_argument(
        '--no-size-check',
        action='store_true',
        help='Allow a query and mask to have different image sizes.',
    )
    return parser.parse_args()


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


def image_paths(root):
    return sorted(
        path.resolve()
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_role_directory(data_root, role):
    matches = [data_root / name for name in DIRECTORY_ALIASES[role] if (data_root / name).is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        aliases = ', '.join(DIRECTORY_ALIASES[role])
        raise NotADirectoryError(
            f'No {role} directory found under {data_root}; expected one of: {aliases}'
        )
    raise ValueError(f'Ambiguous {role} directories under {data_root}: {matches}')


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


def find_match(query, query_root, role, by_relative, by_stem):
    candidates = by_relative.get(relative_key(query, query_root), [])
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
        return int(image.convert('L').getextrema()[1] > 0)


def build_rows(data_root, check_size=True):
    prompt_root = find_role_directory(data_root, 'prompt')
    query_root = find_role_directory(data_root, 'query')
    mask_root = find_role_directory(data_root, 'mask')
    prompts = image_paths(prompt_root)
    queries = image_paths(query_root)
    masks = image_paths(mask_root)
    if not prompts or not queries or not masks:
        raise ValueError(f'Empty prompt/query/mask image directory under: {data_root}')

    prompt_by_relative, prompt_by_stem = build_indexes(prompts, prompt_root)
    mask_by_relative, mask_by_stem = build_indexes(masks, mask_root)
    rows = []
    for query in queries:
        prompt = find_match(query, query_root, 'prompt', prompt_by_relative, prompt_by_stem)
        mask = find_match(query, query_root, 'mask', mask_by_relative, mask_by_stem)
        for path in (prompt, query, mask):
            verify_image(path)
        if check_size and image_size(query) != image_size(mask):
            raise ValueError(
                f'Query and mask sizes differ: {query} {image_size(query)} != '
                f'{mask} {image_size(mask)}'
            )
        relative_query = query.relative_to(query_root)
        subclass = relative_query.parts[0] if len(relative_query.parts) > 1 else 'default'
        label = mask_has_anomaly(mask)
        rows.append({
            'original': str(prompt),
            'changed': str(query),
            'mask': str(mask),
            'clsname': f'{data_root.name}/{subclass}',
            'label': label,
            'label_name': 'defective' if label else 'good',
        })
    return rows


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'Existing Hedian manifest is empty: {path}')
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in FIELDNAMES} for row in rows)


def main():
    args = parse_args()
    train_csv = Path(args.train_csv).resolve()
    train_rows = read_csv(train_csv)
    hedian_train_count = len(train_rows)

    for root_value in args.extra_root:
        data_root = Path(root_value).resolve()
        if not data_root.is_dir():
            raise NotADirectoryError(f'Extra dataset does not exist: {data_root}')
        rows = build_rows(data_root, check_size=not args.no_size_check)
        train_rows.extend(rows)
        print(f'{data_root.name}: total={len(rows)}, train={len(rows)}')

    write_csv(train_rows, train_csv)
    print(f'combined train={len(train_rows)} (Hedian={hedian_train_count})')
    print('Hedian validation and test manifests were not modified.')


if __name__ == '__main__':
    main()
