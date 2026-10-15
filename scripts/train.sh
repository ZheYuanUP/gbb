#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"

# Silence dependency startup warnings from pytorch_lightning / TensorFlow.
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning"

# ========= GPU =========
export CUDA_VISIBLE_DEVICES=0,1
MASTER_PORT=29501
NPROC=2

# ========= Dataset =========
MVTEC_ROOT=/home_ssd/lwj/mvtec_ad

# Generated pair manifest and empty masks will be saved here.
DATA_ROOT=./data/MVTec-AD-pairs
MANIFEST=$DATA_ROOT/train.csv

# How to select train/good images as original prompts:
# one_per_class, random_per_sample, or all_train.
PROMPT_MODE=one_per_class

# ========= Training =========
EPOCHS=50
BATCH_SIZE=24
VAL_BATCH_SIZE=24
NUM_WORKERS=8
LR=1e-4
SCHEDULER=constant
WEIGHT_DECAY=0.005
TARGET_SIZE=512
VAL_RATIO=0.1
SEED=1
GAMMA=0.95
ETA_MIN=1e-6

# ========= Loss =========
BCE_WEIGHT=1.0
DICE_WEIGHT=1.0

# ========= Augmentation =========
AUG_HFLIP_P=0.0
AUG_VFLIP_P=0.0
AUG_LIGHT_P=0.2
AUG_LIGHT_BRIGHTNESS_MIN=0.8
AUG_LIGHT_BRIGHTNESS_MAX=1.2
AUG_GAUSSIAN_BLUR_P=0.2
AUG_GAUSSIAN_BLUR_KERNEL_SIZE=5
AUG_GAUSSIAN_BLUR_SIGMA_MIN=0.1
AUG_GAUSSIAN_BLUR_SIGMA_MAX=2.0

# ========= Save =========
BEST_PATH=checkpoints/best_model_pair_masks.pth
TRAINING_STATE_PATH=checkpoints/training_state_latest.pth
# CHECKPOINT_PREFIX=checkpoints/metauas_pair_masks_epoch
CHECKPOINT_PREFIX=/home_ssd/lwj/gbb/checkpoints/epochs/metauas_pair_masks_epoch

python train/build_mvtec_pair_manifest.py \
  --mvtec-root "$MVTEC_ROOT" \
  --output-root "$DATA_ROOT" \
  --output-csv "$MANIFEST" \
  --prompt-mode $PROMPT_MODE \
  --seed $SEED \
  --strict

torchrun \
  --nproc_per_node $NPROC \
  --master_port $MASTER_PORT \
  train/train_pair_masks.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
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
  --best-path "$BEST_PATH" \
  --training-state-path "$TRAINING_STATE_PATH" \
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
