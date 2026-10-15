#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"
export PYTHONPATH="$REPO"

# Silence dependency startup warnings from pytorch_lightning / TensorFlow.
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning"

# ========= GPU =========
# Inference runs on one GPU. Change this to 1 to use the second GPU.
export CUDA_VISIBLE_DEVICES=0

# ========= Dataset =========
# test.csv directly defines every prompt/query/mask image triple.
DATA_ROOT=/home_ssd/lwj/hedianDataSet
TEST_CSV=$DATA_ROOT/test.csv

# ========= Model =========
CHECKPOINT=$1
IMG_SIZE=512
SEED=1

# ========= Output =========
SAVE_PATH=$REPO/result/hedian_$(date +%m%d_%H%M)

python test/test.py \
  --checkpoint "$CHECKPOINT" \
  --seed $SEED \
  --img_size $IMG_SIZE \
  --image_dir "$DATA_ROOT" \
  --test_csv "$TEST_CSV" \
  --save_path "$SAVE_PATH" \
  --pred_threshold 0.7 \
  --min_gt_coverage 0.2 \
  --eval_metrics \
    I-AUROC \
    I-AP \
    I-F1max \
    P-AUROC \
    P-AP \
    P-F1max \
    P-AUPRO \
    I-Accuracy \
    R-Accuracy \
    R-FPR
