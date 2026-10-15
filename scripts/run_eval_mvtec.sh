#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ========= Config =========
MVTEC_ROOT="/home_ssd/lwj/mvtec_ad"
CKPT="checkpoints/20260812_114359_purple_craw_seed1/best_val_dice.pth"
IMAGE_SIZE=512
GPU_ID=0
RUN_TIME="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="results/mvtec_eval/topk1_${RUN_TIME}"

TEST_JSON="${REPO_ROOT}/data/MVTec-AD/test.json"
TOPK_JSON="${REPO_ROOT}/data/MVTec-AD/test-train-top10pair-eb4.json"

# ========= Path check =========
for required_path in "$MVTEC_ROOT" "$CKPT" "$TEST_JSON" "$TOPK_JSON"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$RESULTS_ROOT"

echo "=== MVTec AD: Top-1 prompt evaluation ==="
echo "Checkpoint : $CKPT"
echo "MVTec root : $MVTEC_ROOT"
echo "Image size : $IMAGE_SIZE"
echo "GPU        : $GPU_ID"
echo "Results    : $RESULTS_ROOT"

# test-train-top10pair-eb4.json 中每张测试图对应的第一张正常图
# 将被 ADDataset 作为 Top-1 Prompt。
CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "${REPO_ROOT}/test/test.py" \
    --checkpoint "$CKPT" \
    --seed 1 \
    --img_size "$IMAGE_SIZE" \
    --image_dir "$MVTEC_ROOT" \
    --test_json "$TEST_JSON" \
    --prompt_json "$TOPK_JSON" \
    --save_path "$RESULTS_ROOT" \
    --eval_metrics \
        I-AUROC I-AP I-F1max \
        P-AUROC P-AP P-F1max P-AUPRO \
    2>&1 | tee "${RESULTS_ROOT}/evaluation_results.txt"

echo ""
echo "MVTec Top-1 evaluation complete."
echo "Results: $RESULTS_ROOT"
