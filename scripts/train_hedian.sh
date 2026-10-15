#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
export PYTHONPATH="$REPO"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning"

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1
MASTER_PORT=29501
NPROC=2

# Dataset settings
DATA_ROOT=/home_ssd/lwj/hedianDataSet
PIAD_ROOT=/home_ssd/lwj/PIAD_dataset
MANIFEST=$DATA_ROOT/train.csv
VAL_MANIFEST=$DATA_ROOT/val.csv
TEST_MANIFEST=$DATA_ROOT/test.csv

# Training settings
EPOCHS=25
BATCH_SIZE=24
VAL_BATCH_SIZE=24
NUM_WORKERS=8
LR=1e-4
SCHEDULER=constant
WEIGHT_DECAY=0.005
TARGET_SIZE=512
VAL_RATIO=0
SEED=1
GAMMA=0.95
ETA_MIN=1e-6

# Loss settings
BCE_WEIGHT=1.0
DICE_WEIGHT=1.0

# Augmentation settings
AUG_HFLIP_P=0.0
AUG_VFLIP_P=0.0
AUG_LIGHT_P=0.2
AUG_LIGHT_BRIGHTNESS_MIN=0.8
AUG_LIGHT_BRIGHTNESS_MAX=1.2
AUG_GAUSSIAN_BLUR_P=0.2
AUG_GAUSSIAN_BLUR_KERNEL_SIZE=5
AUG_GAUSSIAN_BLUR_SIGMA_MIN=0.1
AUG_GAUSSIAN_BLUR_SIGMA_MAX=2.0

# Checkpoint and experiment output settings
INIT_CHECKPOINT=$REPO/weights/metauas-512.ckpt
RUN_TIMESTAMP=$1
EXPERIMENT_ROOT=/home_ssd/lwj/gbb/experiments
EXPERIMENT_DIR=$EXPERIMENT_ROOT/${RUN_TIMESTAMP}_purple_craw_seed${SEED}
LOCAL_CHECKPOINT_DIR=$REPO/checkpoints/${RUN_TIMESTAMP}_purple_craw_seed${SEED}
BEST_PATH=$EXPERIMENT_DIR/checkpoints/best_val_loss.pth
BEST_DICE_PATH=$EXPERIMENT_DIR/checkpoints/best_val_dice.pth
TRAINING_STATE_PATH=$EXPERIMENT_DIR/checkpoints/latest.pth
CHECKPOINT_PREFIX=$EXPERIMENT_DIR/checkpoints/epoch

# Build reproducible leakage-free train, validation, and test manifests.
python train/build_hedian_prefix_manifest.py \
  --data-root "$DATA_ROOT" \
  --train-csv "$MANIFEST" \
  --val-csv "$VAL_MANIFEST" \
  --test-csv "$TEST_MANIFEST" \
  --seed $SEED

# Keep Hedian's original 70/15/15 split, then append all PIAD samples to
# training only. val.csv and test.csv remain Hedian-only.
python train/merge_extra_pair_datasets.py \
  --extra-root "$PIAD_ROOT" \
  --train-csv "$MANIFEST"

torchrun \
  --nproc_per_node $NPROC \
  --master_port $MASTER_PORT \
  train/train_pair_masks.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --target-size $TARGET_SIZE \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --val-batch-size $VAL_BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --scheduler $SCHEDULER \
  --gamma $GAMMA \
  --eta-min $ETA_MIN \
  --weight-decay $WEIGHT_DECAY \
  --val-ratio $VAL_RATIO \
  --seed $SEED \
  --bce-weight $BCE_WEIGHT \
  --dice-weight $DICE_WEIGHT \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --best-path "$BEST_PATH" \
  --best-dice-path "$BEST_DICE_PATH" \
  --training-state-path "$TRAINING_STATE_PATH" \
  --experiment-dir "$EXPERIMENT_DIR" \
  --checkpoint-every 5 \
  --preview-count 16 \
  --preview-every 5 \
  --aug-hflip-p $AUG_HFLIP_P \
  --aug-vflip-p $AUG_VFLIP_P \
  --aug-light-p $AUG_LIGHT_P \
  --aug-light-brightness-min $AUG_LIGHT_BRIGHTNESS_MIN \
  --aug-light-brightness-max $AUG_LIGHT_BRIGHTNESS_MAX \
  --aug-gaussian-blur-p $AUG_GAUSSIAN_BLUR_P \
  --aug-gaussian-blur-kernel-size $AUG_GAUSSIAN_BLUR_KERNEL_SIZE \
  --aug-gaussian-blur-sigma-min $AUG_GAUSSIAN_BLUR_SIGMA_MIN \
  --aug-gaussian-blur-sigma-max $AUG_GAUSSIAN_BLUR_SIGMA_MAX \
  --checkpoint-prefix "$CHECKPOINT_PREFIX"

# Keep a second copy under the current repository's checkpoints directory.
mkdir -p "$LOCAL_CHECKPOINT_DIR"
cp -a "$EXPERIMENT_DIR/checkpoints/." "$LOCAL_CHECKPOINT_DIR/"
echo "Checkpoint copy: $LOCAL_CHECKPOINT_DIR"
