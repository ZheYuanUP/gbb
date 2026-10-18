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

# Fold and dataset settings
RUN_TIMESTAMP=${1:?Usage: bash scripts_k_fold/train_hedian.sh RUN_TIMESTAMP FOLD}
FOLD=${2:?Usage: bash scripts_k_fold/train_hedian.sh RUN_TIMESTAMP FOLD}
if ! [[ "$FOLD" =~ ^([0-9]|1[01])$ ]]; then
  echo "Error: FOLD must be an integer from 0 to 11, got: $FOLD" >&2
  exit 2
fi
FOLD_PAD=$(printf '%02d' "$FOLD")
DATA_ROOT=/home_ssd/lwj/hedianDataSet
PIAD_ROOT=/home_ssd/lwj/PIAD_dataset
SPLIT_DIR=$DATA_ROOT/device_splits/fold_${FOLD_PAD}
MANIFEST=$SPLIT_DIR/train.csv
VAL_MANIFEST=$SPLIT_DIR/val.csv
TEST_MANIFEST=$SPLIT_DIR/test.csv

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
# Environment overrides propagate through run_device_folds.sh and
# train_and_test_hedian.sh to every selected fold.
LOSS_TYPE=${LOSS_TYPE:-bce}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_ALPHA=${FOCAL_ALPHA:-}
BCE_WEIGHT=1.0
DICE_WEIGHT=1.0

if [[ "$LOSS_TYPE" != bce && "$LOSS_TYPE" != focal ]]; then
  echo "Error: LOSS_TYPE must be bce or focal, got: $LOSS_TYPE" >&2
  exit 2
fi
LOSS_ARGS=(--loss-type "$LOSS_TYPE" --focal-gamma "$FOCAL_GAMMA")
if [[ -n "$FOCAL_ALPHA" ]]; then
  LOSS_ARGS+=(--focal-alpha "$FOCAL_ALPHA")
fi
echo "Fold $FOLD loss: $LOSS_TYPE, gamma=$FOCAL_GAMMA, alpha=${FOCAL_ALPHA:-disabled}, weight=$BCE_WEIGHT"

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

# Fold-specific output settings
INIT_CHECKPOINT=$REPO/weights/metauas-512.ckpt
RUN_NAME=${RUN_TIMESTAMP}_fold${FOLD_PAD}_seed${SEED}
EXPERIMENT_DIR=$REPO/experiments_k_fold/$RUN_NAME
LOCAL_CHECKPOINT_DIR=$REPO/checkpoints_k_fold/$RUN_NAME
BEST_PATH=$EXPERIMENT_DIR/checkpoints/best_val_loss.pth
BEST_DICE_PATH=$EXPERIMENT_DIR/checkpoints/best_val_dice.pth
TRAINING_STATE_PATH=$EXPERIMENT_DIR/checkpoints/latest.pth
CHECKPOINT_PREFIX=$EXPERIMENT_DIR/checkpoints/epoch

python train_k_fold/build_hedian_device_manifest.py \
  --data-root "$DATA_ROOT" \
  --fold "$FOLD" \
  --train-csv "$MANIFEST" \
  --val-csv "$VAL_MANIFEST" \
  --test-csv "$TEST_MANIFEST"

# Preserve the existing behavior: PIAD is appended to training only.
python train/merge_extra_pair_datasets.py \
  --extra-root "$PIAD_ROOT" \
  --train-csv "$MANIFEST"

mkdir -p "$EXPERIMENT_DIR/manifests"
cp "$MANIFEST" "$VAL_MANIFEST" "$TEST_MANIFEST" \
  "$SPLIT_DIR/split_config.json" "$EXPERIMENT_DIR/manifests/"

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
  "${LOSS_ARGS[@]}" \
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

mkdir -p "$LOCAL_CHECKPOINT_DIR"
cp -a "$EXPERIMENT_DIR/checkpoints/." "$LOCAL_CHECKPOINT_DIR/"
cp "$SPLIT_DIR/split_config.json" "$LOCAL_CHECKPOINT_DIR/split_config.json"
echo "Checkpoint copy: $LOCAL_CHECKPOINT_DIR"
echo "Fold configuration: $LOCAL_CHECKPOINT_DIR/split_config.json"
