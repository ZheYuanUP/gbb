#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
# Loss settings are inherited by training in every selected fold.
# Example: LOSS_TYPE=focal FOCAL_GAMMA=2 bash scripts_k_fold/train_and_test_hedian.sh 0
FOLD=${1:?Usage: bash scripts_k_fold/train_and_test_hedian.sh FOLD}
if ! [[ "$FOLD" =~ ^([0-9]|1[01])$ ]]; then
  echo "Error: FOLD must be an integer from 0 to 11, got: $FOLD" >&2
  exit 2
fi

FOLD_PAD=$(printf '%02d' "$FOLD")
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_TIMESTAMP}_fold${FOLD_PAD}_seed1
EXPERIMENT_DIR=$REPO/experiments_k_fold/$RUN_NAME
BEST_DICE_PATH=$EXPERIMENT_DIR/checkpoints/best_val_dice.pth

bash "$SCRIPT_DIR/train_hedian.sh" "$RUN_TIMESTAMP" "$FOLD"
bash "$SCRIPT_DIR/test_hedian.sh" "$BEST_DICE_PATH" "$FOLD" "$RUN_NAME"
