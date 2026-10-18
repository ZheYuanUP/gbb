#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loss settings are inherited by training in every selected fold.
# Example: LOSS_TYPE=focal FOCAL_GAMMA=2 bash scripts_k_fold/run_device_folds.sh 0
# With no arguments, run all 12 folds sequentially. Pass fold numbers to run a
# subset, for example: bash scripts_k_fold/run_device_folds.sh 0 3 7
if [[ $# -eq 0 ]]; then
  FOLDS=(0 1 2 3 4 5 6 7 8 9 10 11)
else
  FOLDS=("$@")
fi

for fold in "${FOLDS[@]}"; do
  if ! [[ "$fold" =~ ^([0-9]|1[01])$ ]]; then
    echo "Error: fold must be an integer from 0 to 11, got: $fold" >&2
    exit 2
  fi
done

for fold in "${FOLDS[@]}"; do
  echo "============================================================"
  echo "Starting device fold $fold"
  echo "============================================================"
  bash "$SCRIPT_DIR/train_and_test_hedian.sh" "$fold"
done
