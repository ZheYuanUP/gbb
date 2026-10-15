#!/usr/bin/env python3
"""Build one fold of device-level cross-validation manifests.

A device is assigned by matching ``<device>_`` at the beginning of each query
filename. For fold N, ALL_DEVICES[N] is used for testing, the following device
is used for validation, and the remaining ten devices are used for training.
Prompt/query/mask files are matched by their remaining full sample name, using
the same matching rules as build_hedian_prefix_manifest.py.

Expected dataset layout::

    /home_ssd/lwj/hedianDataSet/
      prompt/
      query/
      mask/

Example filename::

    collet2_device1_gobal_1_1.png

With the configuration below, that filename belongs to ``collet2_device1``.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Reuse the original manifest matching utilities.
    from train.build_hedian_prefix_manifest import (
        build_indexes,
        csv_path,
        find_match,
        image_size,
        is_image,
        mask_has_anomaly,
        normalized_stem,
        verify_image,
        write_csv,
    )
except ModuleNotFoundError:
    # Direct script execution: ``python train/build_hedian_device_manifest.py``.
    from build_hedian_prefix_manifest import (
        build_indexes,
        csv_path,
        find_match,
        image_size,
        is_image,
        mask_has_anomaly,
        normalized_stem,
        verify_image,
        write_csv,
    )


# ---------------------------------------------------------------------------
# Device cross-validation configuration
# ---------------------------------------------------------------------------
# The tuple order defines the fold number. For example, fold 0 tests
# collet2_device3 and validates on collet2_device4. Every device is used once
# for testing and once for validation across all 12 folds.
ALL_DEVICES = (
    'collet2_device3',
    'collet2_device4',
    'collet2_device6',
    'collet2_device8',
    'collet2_device9',
    'collet2_device10',
    'collet2_device11',
    'craw',
    'purple_craw',
    'valve_seat',
    'collet2_device2',
    'collet2_device1',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build Hedian CSV manifests using explicit device-prefix splits.'
    )
    parser.add_argument(
        '--data-root',
        default='/home_ssd/lwj/hedianDataSet',
        help='Dataset root containing prompt/query/mask directories.',
    )
    parser.add_argument('--prompt-dir', default='prompt')
    parser.add_argument('--query-dir', default='query')
    parser.add_argument('--mask-dir', default='mask')
    parser.add_argument(
        '--fold',
        type=int,
        help=f'Zero-based test fold in the range 0..{len(ALL_DEVICES) - 1}.',
    )
    parser.add_argument(
        '--list-folds',
        action='store_true',
        help='Print all fold-to-device assignments and exit.',
    )
    parser.add_argument(
        '--train-csv',
        default='',
        help='Default: <data-root>/device_splits/fold_NN/train.csv',
    )
    parser.add_argument(
        '--val-csv',
        default='',
        help='Default: <data-root>/device_splits/fold_NN/val.csv',
    )
    parser.add_argument(
        '--test-csv',
        default='',
        help='Default: <data-root>/device_splits/fold_NN/test.csv',
    )
    parser.add_argument(
        '--no-size-check',
        action='store_true',
        help='Allow a query and its mask to have different image sizes.',
    )
    return parser.parse_args()


def normalize_devices(devices, split_name):
    if isinstance(devices, str):
        raise TypeError(
            f'{split_name}_DEVICES must be a tuple. For one device, include a '
            "trailing comma, for example ('device1',)."
        )
    normalized = tuple(str(device).strip().lower() for device in devices)
    if not normalized:
        raise ValueError(f'{split_name}_DEVICES is empty; configure it at the top of this script.')
    if any(not device for device in normalized):
        raise ValueError(f'{split_name}_DEVICES contains an empty device name.')
    if len(set(normalized)) != len(normalized):
        raise ValueError(f'{split_name}_DEVICES contains duplicate device names.')
    return normalized


def fold_device_splits(fold):
    all_devices = normalize_devices(ALL_DEVICES, 'ALL')
    if not 0 <= fold < len(all_devices):
        raise ValueError(
            f'Fold must be between 0 and {len(all_devices) - 1}, got {fold}.'
        )
    test_device = all_devices[fold]
    val_device = all_devices[(fold + 1) % len(all_devices)]
    train_devices = tuple(
        device for device in all_devices if device not in {test_device, val_device}
    )
    return {
        'train': train_devices,
        'val': (val_device,),
        'test': (test_device,),
    }


def validate_device_splits(split_devices):
    split_devices = {
        split_name: normalize_devices(devices, split_name.upper())
        for split_name, devices in split_devices.items()
    }
    owners = {}
    for split_name, devices in split_devices.items():
        for device in devices:
            if device in owners:
                raise ValueError(
                    f'Device {device!r} appears in both {owners[device]} and {split_name}.'
                )
            owners[device] = split_name
    return split_devices, owners


def write_split_config(path, args, split_devices, counts_by_device, csv_paths):
    value = {
        'fold': args.fold,
        'fold_number': args.fold + 1,
        'num_folds': len(ALL_DEVICES),
        'strategy': 'one test device, next device validation, remaining devices training',
        'data_root': str(Path(args.data_root).resolve()),
        'train_devices': list(split_devices['train']),
        'val_devices': list(split_devices['val']),
        'test_devices': list(split_devices['test']),
        'sample_counts_by_device': dict(sorted(counts_by_device.items())),
        'manifests': {name: str(csv_path) for name, csv_path in csv_paths.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')


def matching_device(path, devices):
    """Return the sole configured device prefix matched by this filename."""
    stem = normalized_stem(path.stem)
    matches = [device for device in devices if stem.startswith(f'{device}_')]
    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous device prefixes for {path.name}: {matches}. '
            'Use non-overlapping, complete device prefixes.'
        )
    return matches[0] if matches else None


def collect_rows(data_root, prompt_root, query_root, mask_root, owners, check_size):
    prompts = sorted(path.resolve() for path in prompt_root.rglob('*') if is_image(path))
    queries = sorted(path.resolve() for path in query_root.rglob('*') if is_image(path))
    masks = sorted(path.resolve() for path in mask_root.rglob('*') if is_image(path))

    if not prompts:
        raise ValueError(f'No prompt images found in: {prompt_root}')
    if not queries:
        raise ValueError(f'No query images found in: {query_root}')
    if not masks:
        raise ValueError(f'No mask images found in: {mask_root}')

    prompt_by_relative, prompt_by_stem = build_indexes(prompts, prompt_root)
    mask_by_relative, mask_by_stem = build_indexes(masks, mask_root)
    rows_by_split = {'train': [], 'val': [], 'test': []}
    counts_by_device = Counter()
    skipped = []

    for query in queries:
        device = matching_device(query, owners)
        if device is None:
            skipped.append(query.name)
            continue

        prompt = find_match(query, query_root, 'prompt', prompt_by_relative, prompt_by_stem)
        mask = find_match(query, query_root, 'mask', mask_by_relative, mask_by_stem)
        for path in (prompt, query, mask):
            verify_image(path)
        if check_size and image_size(query) != image_size(mask):
            raise ValueError(
                f'Query and mask sizes differ: {query} {image_size(query)} != '
                f'{mask} {image_size(mask)}'
            )

        label = mask_has_anomaly(mask)
        split_name = owners[device]
        rows_by_split[split_name].append(
            {
                'original': csv_path(prompt, data_root),
                'changed': csv_path(query, data_root),
                'mask': csv_path(mask, data_root),
                'clsname': device,
                'label': label,
                'label_name': 'defective' if label else 'good',
            }
        )
        counts_by_device[device] += 1

    return rows_by_split, counts_by_device, skipped


def require_configured_devices(counts_by_device, split_devices):
    missing = [
        device
        for devices in split_devices.values()
        for device in devices
        if counts_by_device[device] == 0
    ]
    if missing:
        raise ValueError(
            'No query images matched these configured device prefixes: '
            + ', '.join(missing)
        )


def main():
    args = parse_args()
    if args.list_folds:
        for fold in range(len(ALL_DEVICES)):
            split_devices = fold_device_splits(fold)
            print(
                f'fold {fold:02d}: test={split_devices["test"][0]}, '
                f'val={split_devices["val"][0]}, '
                f'train={",".join(split_devices["train"])}'
            )
        return
    if args.fold is None:
        raise ValueError('Provide --fold N, or use --list-folds to inspect all assignments.')

    data_root = Path(args.data_root).resolve()
    prompt_root = (data_root / args.prompt_dir).resolve()
    query_root = (data_root / args.query_dir).resolve()
    mask_root = (data_root / args.mask_dir).resolve()
    default_split_root = data_root / 'device_splits' / f'fold_{args.fold:02d}'
    train_csv = Path(args.train_csv).resolve() if args.train_csv else default_split_root / 'train.csv'
    val_csv = Path(args.val_csv).resolve() if args.val_csv else default_split_root / 'val.csv'
    test_csv = Path(args.test_csv).resolve() if args.test_csv else default_split_root / 'test.csv'

    for role, directory in (
        ('prompt', prompt_root),
        ('query', query_root),
        ('mask', mask_root),
    ):
        if not directory.is_dir():
            raise NotADirectoryError(f'{role} directory does not exist: {directory}')

    split_devices, owners = validate_device_splits(fold_device_splits(args.fold))
    rows_by_split, counts_by_device, skipped = collect_rows(
        data_root,
        prompt_root,
        query_root,
        mask_root,
        owners,
        check_size=not args.no_size_check,
    )
    require_configured_devices(counts_by_device, split_devices)

    csv_paths = {
        'train': train_csv,
        'val': val_csv,
        'test': test_csv,
    }
    print(f'fold: {args.fold} ({args.fold + 1}/{len(ALL_DEVICES)})')
    print(f"test device: {split_devices['test'][0]}")
    print(f"validation device: {split_devices['val'][0]}")
    print('training devices: ' + ', '.join(split_devices['train']))

    for split_name, output_csv in csv_paths.items():
        rows = rows_by_split[split_name]
        if not rows:
            raise ValueError(f'No rows assigned to the {split_name} split.')
        write_csv(rows, output_csv)
        print(f'{split_name}: {output_csv} ({len(rows)} rows)')
        for device in split_devices[split_name]:
            print(f'  {device}: {counts_by_device[device]} rows')

    print(f'skipped images from unconfigured devices: {len(skipped)}')
    if skipped:
        for name in skipped[:20]:
            print(f'  skipped: {name}')
        if len(skipped) > 20:
            print(f'  ... and {len(skipped) - 20} more')
    split_config = train_csv.parent / 'split_config.json'
    write_split_config(split_config, args, split_devices, counts_by_device, csv_paths)
    print(f'split config: {split_config}')
    print(f'data root: {data_root}')


if __name__ == '__main__':
    main()
