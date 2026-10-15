#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

# Use one run id for training and testing so the test always loads the model
# produced by this invocation instead of an older, hard-coded checkpoint.
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=1
EXPERIMENT_DIR=/home_ssd/lwj/gbb/experiments/${RUN_TIMESTAMP}_purple_craw_seed${SEED}
BEST_DICE_PATH=$EXPERIMENT_DIR/checkpoints/best_val_dice.pth

bash "$SCRIPT_DIR/train_hedian.sh" "$RUN_TIMESTAMP"
bash "$SCRIPT_DIR/test_hedian.sh" "$BEST_DICE_PATH"
