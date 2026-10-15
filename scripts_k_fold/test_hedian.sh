#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
export PYTHONPATH="$REPO"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning"

export CUDA_VISIBLE_DEVICES=0

# ========= Fold =========

CHECKPOINT=${1:?Usage: bash scripts_k_fold/test_hedian.sh CHECKPOINT FOLD [RUN_NAME]}
FOLD=${2:?Usage: bash scripts_k_fold/test_hedian.sh CHECKPOINT FOLD [RUN_NAME]}
if ! [[ "$FOLD" =~ ^([0-9]|1[01])$ ]]; then
  echo "Error: FOLD must be an integer from 0 to 11, got: $FOLD" >&2
  exit 2
fi
FOLD_PAD=$(printf '%02d' "$FOLD")

# ========= Dataset =========

DATA_ROOT=/home_ssd/lwj/hedianDataSet
SPLIT_DIR=$DATA_ROOT/device_splits/fold_${FOLD_PAD}
TEST_CSV=$SPLIT_DIR/test.csv

# ========= Model =========

IMG_SIZE=512
SEED=1

# ========= Output =========

RUN_NAME=${3:-$(date +%Y%m%d_%H%M%S)_fold${FOLD_PAD}_seed${SEED}}
SAVE_PATH=$REPO/result_k_fold/$RUN_NAME

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$TEST_CSV" ]]; then
  echo "Error: fold test manifest does not exist: $TEST_CSV" >&2
  exit 2
fi

mkdir -p "$SAVE_PATH"

EVAL_METRICS=(
  I-AUROC
  I-AP
  I-F1max
  P-AUROC
  P-AP
  P-F1max
  P-AUPRO
  I-Accuracy
  R-Accuracy
  R-FPR
)

python test/test.py \
  --checkpoint "$CHECKPOINT" \
  --seed $SEED \
  --img_size $IMG_SIZE \
  --image_dir "$DATA_ROOT" \
  --test_csv "$TEST_CSV" \
  --save_path "$SAVE_PATH" \
  --pred_threshold 0.2 \
  --min_gt_coverage 0.2 \
  --eval_metrics "${EVAL_METRICS[@]}" \
  2>&1 | tee "$SAVE_PATH/evaluation_results.txt"

# Save the device assignment with the test results when available.
if [[ -f "$SPLIT_DIR/split_config.json" ]]; then
  cp "$SPLIT_DIR/split_config.json" "$SAVE_PATH/split_config.json"
fi
