#!/usr/bin/env python3
# 0721修改

'''
 # @ Description: Train MetaUAS from original/changed/mask image triples.
 '''

import argparse
import csv
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API.*',
    category=UserWarning,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.pair_masks import PairMaskTrainDataset, read_pair_triples, scan_pair_directories
from metauas import MetaUAS, set_random_seed


def ddp_setup():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, 0, 0, 1, device
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    return True, local_rank, global_rank, world_size, torch.device(f'cuda:{local_rank}')


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        # Rank 0 may still be writing validation previews while the other ranks
        # have finished their final epoch. Wait for every rank before tearing
        # down NCCL so one process cannot destroy the group prematurely.
        dist.barrier()
        dist.destroy_process_group()


def is_rank0(rank):
    return rank == 0


def build_model(dropout_p=0.2):
    return MetaUAS(
        encoder_name='efficientnet-b4',
        decoder_name='unet',
        encoder_depth=5,
        decoder_depth=5,
        num_alignment_layers=3,
        alignment_type='sa',
        fusion_policy='cat',
        dropout_p=dropout_p,
    )


def save_state(path, epoch, model, optimizer, scheduler, best_loss, best_dice, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            'epoch': epoch,
            'model': module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_dice': best_dice,
            'args': vars(args),
        },
        path,
    )


def load_initial_weights(path, model, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=True)


def load_state(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    module = model.module if isinstance(model, DDP) else model
    module.load_state_dict(ckpt['model'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return (
        int(ckpt.get('epoch', -1)) + 1,
        float(ckpt.get('best_loss', float('inf'))),
        float(ckpt.get('best_dice', float('-inf'))),
    )


def reduce_mean(value, count, device):
    t = torch.tensor([float(value), float(count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / max(t[1], 1.0)).item()


def reduce_sums(values, device):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def binary_counts(pred, target, threshold=0.5):
    predicted = pred >= threshold
    expected = target >= 0.5
    return (
        torch.logical_and(predicted, expected).sum().item(),
        torch.logical_and(predicted, ~expected).sum().item(),
        torch.logical_and(~predicted, expected).sum().item(),
    )


def segmentation_metrics(tp, fp, fn):
    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    return dice, iou


def segmentation_loss(pred, target, loss_type='bce', gamma=2.0, alpha=None):
    """Compute BCE or focal loss from the model's sigmoid probabilities."""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    if loss_type == 'bce':
        return bce.mean()
    if loss_type != 'focal':
        raise ValueError(f'Unknown loss type: {loss_type}')
    pt = pred * target + (1.0 - pred) * (1.0 - target)
    loss = (1.0 - pt).pow(gamma) * bce
    if alpha is not None:
        loss = (alpha * target + (1.0 - alpha) * (1.0 - target)) * loss
    return loss.mean()


@torch.no_grad()
def run_eval(model, loader, device, bce_weight, loss_type='bce', focal_gamma=2.0, focal_alpha=None):
    model.eval()
    total_loss, total_n, total_tp, total_fp, total_fn = 0.0, 0, 0, 0, 0
    for data in loader:
        batch = {
            'query_image': data['query_image'].to(device, non_blocking=True),
            'prompt_image': data['prompt_image'].to(device, non_blocking=True),
        }
        mask = data['query_mask'].to(device, non_blocking=True)
        pred = model(batch)
        loss = bce_weight * segmentation_loss(pred, mask, loss_type, focal_gamma, focal_alpha)
        total_loss += loss.item() * mask.size(0)
        total_n += mask.size(0)
        tp, fp, fn = binary_counts(pred, mask)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    loss = reduce_mean(total_loss, total_n, device)
    tp, fp, fn = reduce_sums((total_tp, total_fp, total_fn), device)
    dice, iou = segmentation_metrics(tp, fp, fn)
    return {'loss': loss, 'dice': dice, 'iou': iou}


def source_group_name(changed_path):
    # Extra datasets are written to the combined manifest with absolute paths.
    # Their split is sample-random and their filenames have no grouping
    # convention, so only apply Hedian's numeric source grouping to relative
    # Hedian paths.
    if Path(changed_path).is_absolute():
        return str(Path(changed_path).resolve())
    stem = Path(changed_path).stem.lower()
    for suffix in ('_changed', '_query', '-changed', '-query'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    match = re.fullmatch(r'(.+_\d+)_\d+', stem)
    return match.group(1) if match else stem


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def append_history(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history_csv, curves_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with Path(history_csv).open('r', encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return
    epochs = [int(row['epoch']) for row in rows]
    curves_dir = Path(curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)

    for filename, ylabel, train_key, val_key in (
        ('loss_curve.png', 'loss', 'train_loss', 'val_loss'),
        ('dice_curve.png', 'Dice', 'train_dice', 'val_dice'),
        ('iou_curve.png', 'IoU', 'train_iou', 'val_iou'),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [float(row[train_key]) for row in rows], label=train_key)
        if rows[0].get(val_key):
            plt.plot(epochs, [float(row[val_key]) for row in rows], label=val_key)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(curves_dir / filename, dpi=150)
        plt.close()


def tensor_rgb_image(tensor):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode='RGB')


def tensor_mask_image(tensor):
    array = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode='L')


@torch.no_grad()
def save_validation_previews(model, dataset, indices, device, output_dir, threshold):
    module = model.module if isinstance(model, DDP) else model
    module.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []

    for index in indices:
        data = dataset[index]
        batch = {
            'query_image': data['query_image'].unsqueeze(0).to(device),
            'prompt_image': data['prompt_image'].unsqueeze(0).to(device),
        }
        target = data['query_mask']
        pred = module(batch)[0].detach().cpu()
        binary = (pred >= threshold).float()
        tp, fp, fn = binary_counts(pred, target)
        dice, iou = segmentation_metrics(tp, fp, fn)
        sample_dir = output_dir / f'sample_{index:04d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        tensor_rgb_image(data['query_image']).save(sample_dir / 'query.png')
        tensor_mask_image(target).save(sample_dir / 'ground_truth.png')
        tensor_mask_image(pred).save(sample_dir / 'heatmap.png')
        tensor_mask_image(binary).save(sample_dir / 'prediction.png')
        prediction_rows.append(
            {
                'dataset_index': index,
                'changed_path': data['changed_path'],
                'source_group': source_group_name(data['changed_path']),
                'label': int(torch.any(target > 0).item()),
                'score': float(pred.max().item()),
                'prediction': int(torch.any(binary > 0).item()),
                'pixel_dice': dice,
                'pixel_iou': iou,
            }
        )

    with (output_dir / 'predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)


def parse_args():
    p = argparse.ArgumentParser(description='Train MetaUAS on original/changed/mask triples.')
    p.add_argument('--data-root', type=str, default='.')
    p.add_argument('--manifest', type=str, default='')
    p.add_argument('--val-manifest', type=str, default='')
    p.add_argument('--test-manifest', type=str, default='')
    p.add_argument('--original-dir', type=str, default='')
    p.add_argument('--changed-dir', type=str, default='')
    p.add_argument('--mask-dir', type=str, default='')
    p.add_argument('--target-size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--val-batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.005)
    p.add_argument('--dropout-p', type=float, default=0.2)
    p.add_argument('--scheduler', choices=['constant', 'cosine', 'exponential'], default='constant')
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--eta-min', type=float, default=1e-6)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--bce-weight', type=float, default=1.0)
    p.add_argument('--loss-type', choices=['bce', 'focal'], default='bce')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--focal-alpha', type=float, default=None,
                   help='Optional positive-pixel weight in [0, 1]; negative pixels use 1-alpha.')
    p.add_argument(
        '--dice-weight',
        type=float,
        default=0.0,
        help='Deprecated compatibility option; Dice loss is disabled and this value is ignored.',
    )
    p.add_argument('--best-path', type=str, default='checkpoints/best_model_pair_masks.pth')
    p.add_argument('--best-dice-path', type=str, default='checkpoints/best_dice_pair_masks.pth')
    p.add_argument('--training-state-path', type=str, default='checkpoints/training_state_latest.pth')
    p.add_argument('--init-checkpoint', type=str, default='')
    p.add_argument('--resume-from', type=str, default='')
    p.add_argument('--checkpoint-prefix', type=str, default='checkpoints/metauas_pair_masks_epoch')
    p.add_argument('--no-epoch-checkpoints', action='store_true')
    p.add_argument('--checkpoint-every', type=int, default=5)
    p.add_argument('--experiment-dir', type=str, default='experiments/pair_masks')
    p.add_argument('--preview-count', type=int, default=16)
    p.add_argument('--preview-every', type=int, default=5)
    p.add_argument('--prediction-threshold', type=float, default=0.5)
    p.add_argument('--aug-hflip-p', type=float, default=0.0)
    p.add_argument('--aug-vflip-p', type=float, default=0.0)
    p.add_argument('--aug-light-p', type=float, default=0.2)
    p.add_argument('--aug-light-brightness-min', type=float, default=0.8)
    p.add_argument('--aug-light-brightness-max', type=float, default=1.2)
    p.add_argument('--aug-gaussian-blur-p', type=float, default=0.2)
    p.add_argument('--aug-gaussian-blur-kernel-size', type=int, default=5)
    p.add_argument('--aug-gaussian-blur-sigma-min', type=float, default=0.1)
    p.add_argument('--aug-gaussian-blur-sigma-max', type=float, default=2.0)
    args = p.parse_args()
    if not np.isfinite(args.focal_gamma) or args.focal_gamma < 0:
        p.error('--focal-gamma must be finite and non-negative')
    if args.focal_alpha is not None and not 0 <= args.focal_alpha <= 1:
        p.error('--focal-alpha must be in [0, 1]')
    # Keep accepting the legacy CLI option so existing launch scripts work,
    # but record the effective configuration truthfully.
    args.dice_weight = 0.0
    return args


def main():
    args = parse_args()
    is_ddp, local_rank, global_rank, world_size, device = ddp_setup()
    set_random_seed(args.seed + global_rank)

    if args.manifest:
        triples = read_pair_triples(args.manifest)
    else:
        if not (args.original_dir and args.changed_dir and args.mask_dir):
            raise ValueError('Provide --manifest or all of --original-dir/--changed-dir/--mask-dir.')
        triples = scan_pair_directories(args.original_dir, args.changed_dir, args.mask_dir)

    if args.val_manifest:
        train_triples = list(triples)
        val_triples = read_pair_triples(args.val_manifest)
    else:
        val_count = int(len(triples) * args.val_ratio)
        train_count = len(triples) - val_count
        if train_count <= 0:
            raise ValueError('No training samples after split; reduce --val-ratio.')
        shuffled = list(triples)
        random.Random(args.seed).shuffle(shuffled)
        train_triples = shuffled[:train_count]
        val_triples = shuffled[train_count:]

    train_count = len(train_triples)
    val_count = len(val_triples)
    if args.val_manifest:
        train_groups = {source_group_name(changed) for _, changed, _ in train_triples}
        val_groups = {source_group_name(changed) for _, changed, _ in val_triples}
        overlap = sorted(train_groups & val_groups)
        if overlap:
            raise ValueError(
                'Source-image leakage between train and validation manifests: '
                + ', '.join(overlap[:10])
            )

    train_dataset = PairMaskTrainDataset(
        train_triples,
        root=args.data_root,
        image_size=args.target_size,
        augment=True,
        hflip_p=args.aug_hflip_p,
        vflip_p=args.aug_vflip_p,
        light_p=args.aug_light_p,
        light_brightness=(
            args.aug_light_brightness_min,
            args.aug_light_brightness_max,
        ),
        gaussian_blur_p=args.aug_gaussian_blur_p,
        gaussian_blur_kernel_size=args.aug_gaussian_blur_kernel_size,
        gaussian_blur_sigma=(
            args.aug_gaussian_blur_sigma_min,
            args.aug_gaussian_blur_sigma_max,
        ),
    )
    val_dataset = PairMaskTrainDataset(
        val_triples,
        root=args.data_root,
        image_size=args.target_size,
        augment=False,
    )

    experiment_dir = Path(args.experiment_dir).resolve()
    history_csv = experiment_dir / 'training_history.csv'
    preview_count = min(max(args.preview_count, 0), val_count)
    preview_indices = sorted(
        random.Random(args.seed).sample(range(val_count), preview_count)
    ) if preview_count else []

    if is_rank0(global_rank):
        write_json(experiment_dir / 'config.json', vars(args))
        train_group_names = sorted({source_group_name(item[1]) for item in train_triples})
        val_group_names = sorted({source_group_name(item[1]) for item in val_triples})
        split_summary = {
            'seed': args.seed,
            'train_manifest': str(Path(args.manifest).resolve()) if args.manifest else '',
            'val_manifest': str(Path(args.val_manifest).resolve()) if args.val_manifest else '',
            'test_manifest': str(Path(args.test_manifest).resolve()) if args.test_manifest else '',
            'train_images': train_count,
            'val_images': val_count,
            'train_source_groups': len(train_group_names),
            'val_source_groups': len(val_group_names),
            'train_source_group_names': train_group_names,
            'val_source_group_names': val_group_names,
            'preview_indices': preview_indices,
        }
        if args.test_manifest:
            test_triples = read_pair_triples(args.test_manifest)
            test_group_names = sorted({source_group_name(item[1]) for item in test_triples})
            split_summary['test_images'] = len(test_triples)
            split_summary['test_source_groups'] = len(test_group_names)
            split_summary['test_source_group_names'] = test_group_names
        write_json(experiment_dir / 'split_summary.json', split_summary)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if is_ddp and val_count > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_count > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model = build_model(dropout_p=args.dropout_p).to(device)
    if args.init_checkpoint and not args.resume_from:
        load_initial_weights(args.init_checkpoint, model, device)
        if is_rank0(global_rank):
            print(f'[init] loaded model weights from {args.init_checkpoint}', flush=True)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.scheduler == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    start_epoch, best_loss, best_dice = 0, float('inf'), float('-inf')
    if args.resume_from:
        start_epoch, best_loss, best_dice = load_state(
            args.resume_from, model, optimizer, scheduler, device
        )
    elif is_rank0(global_rank) and history_csv.exists():
        history_csv.unlink()

    if is_rank0(global_rank):
        print(
            f'[train_pair_masks] samples={len(triples)} train={train_count} val={val_count} '
            f'world_size={world_size} batch_per_gpu={args.batch_size} img={args.target_size}',
            flush=True,
        )
        print(
            f'[train_pair_masks] lr={args.lr} wd={args.weight_decay} scheduler={args.scheduler} '
            f'loss={args.loss_type} loss_weight={args.bce_weight} '
            f'focal_gamma={args.focal_gamma} focal_alpha={args.focal_alpha} dice_loss=disabled',
            flush=True,
        )
        print(
            f'[train_pair_masks] light_p={args.aug_light_p} '
            f'brightness=[{args.aug_light_brightness_min}, {args.aug_light_brightness_max}]',
            flush=True,
        )
        print(
            f'[train_pair_masks] gaussian_blur_p={args.aug_gaussian_blur_p} '
            f'kernel={args.aug_gaussian_blur_kernel_size} '
            f'sigma=[{args.aug_gaussian_blur_sigma_min}, {args.aug_gaussian_blur_sigma_max}]',
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        # The ImageNet encoder is frozen; keep its normalization statistics fixed
        # while trainable alignment, decoder, head, and dropout use training mode.
        base_model = model.module if isinstance(model, DDP) else model
        base_model.encoder.eval()
        total_loss, total_n, total_tp, total_fp, total_fn = 0.0, 0, 0, 0, 0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', ncols=100, disable=not is_rank0(global_rank))
        for data in progress:
            batch = {
                'query_image': data['query_image'].to(device, non_blocking=True),
                'prompt_image': data['prompt_image'].to(device, non_blocking=True),
            }
            mask = data['query_mask'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = args.bce_weight * segmentation_loss(
                pred, mask, args.loss_type, args.focal_gamma, args.focal_alpha
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mask.size(0)
            total_n += mask.size(0)
            tp, fp, fn = binary_counts(pred.detach(), mask)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            progress.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        train_loss = reduce_mean(total_loss, total_n, device)
        train_tp, train_fp, train_fn = reduce_sums(
            (total_tp, total_fp, total_fn), device
        )
        train_dice, train_iou = segmentation_metrics(train_tp, train_fp, train_fn)
        val_metrics = (
            run_eval(model, val_loader, device, args.bce_weight,
                     args.loss_type, args.focal_gamma, args.focal_alpha)
            if val_loader is not None else None
        )
        save_metric = val_metrics['loss'] if val_metrics is not None else train_loss

        if is_rank0(global_rank):
            if save_metric < best_loss:
                best_loss = save_metric
                Path(args.best_path).parent.mkdir(parents=True, exist_ok=True)
                module = model.module if isinstance(model, DDP) else model
                torch.save(module.state_dict(), args.best_path)
                print(f'[best] loss={best_loss:.6f} -> {args.best_path}', flush=True)

            if val_metrics is not None and val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                Path(args.best_dice_path).parent.mkdir(parents=True, exist_ok=True)
                module = model.module if isinstance(model, DDP) else model
                torch.save(module.state_dict(), args.best_dice_path)
                print(
                    f'[best] dice={best_dice:.6f} -> {args.best_dice_path}',
                    flush=True,
                )

            if args.training_state_path:
                save_state(
                    args.training_state_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_loss,
                    best_dice,
                    args,
                )
            if (
                not args.no_epoch_checkpoints
                and args.checkpoint_every > 0
                and (epoch + 1) % args.checkpoint_every == 0
            ):
                module = model.module if isinstance(model, DDP) else model
                epoch_path = f'{args.checkpoint_prefix}_{epoch + 1:03d}.pth'
                Path(epoch_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(module.state_dict(), epoch_path)

            lr = scheduler.get_last_lr()[0]
            history_row = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'] if val_metrics else '',
                'train_dice': train_dice,
                'val_dice': val_metrics['dice'] if val_metrics else '',
                'train_iou': train_iou,
                'val_iou': val_metrics['iou'] if val_metrics else '',
                'learning_rate': lr,
                'generalization_gap_loss': (
                    val_metrics['loss'] - train_loss if val_metrics else ''
                ),
                'generalization_gap_dice': (
                    train_dice - val_metrics['dice'] if val_metrics else ''
                ),
            }
            append_history(history_csv, history_row)
            plot_history(history_csv, experiment_dir / 'curves')

            should_preview = preview_indices and (
                epoch == start_epoch
                or epoch + 1 == args.epochs
                or (args.preview_every > 0 and (epoch + 1) % args.preview_every == 0)
            )
            if should_preview:
                save_validation_previews(
                    model,
                    val_dataset,
                    preview_indices,
                    device,
                    experiment_dir / 'predictions' / f'epoch_{epoch + 1:03d}',
                    args.prediction_threshold,
                )

            msg = (
                f'Epoch {epoch + 1}, train_loss={train_loss:.6f}, '
                f'train_dice={train_dice:.6f}, train_iou={train_iou:.6f}, '
                f'lr={lr:.8f}'
            )
            if val_metrics is not None:
                msg += (
                    f", val_loss={val_metrics['loss']:.6f}, "
                    f"val_dice={val_metrics['dice']:.6f}, "
                    f"val_iou={val_metrics['iou']:.6f}"
                )
            print(msg, flush=True)

    # Drop optimizer gradient buffers and cached CUDA blocks before NCCL teardown.
    # Model parameters remain available in the checkpoints already written above.
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    ddp_cleanup()


if __name__ == '__main__':
    main()
