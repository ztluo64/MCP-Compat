#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLIP_PATH="${CLIP_PATH:-openai/clip-vit-base-patch32}"
GPU_BASE="${GPU_BASE:-0}"
GPU_COMPAT="${GPU_COMPAT:-1}"

SEEDS=(42 777 2025)

TRAIN_CSV="tri_view_compat/outputs/splits/train.csv"
VAL_CSV="tri_view_compat/outputs/splits/val.csv"
TEST_CSV="tri_view_compat/outputs/splits/test.csv"
BALANCED_CSV="tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv"

ROOT_OUT="tri_view_compat/outputs/three_seed_stability"
LOG_DIR="logs/three_seed_stability"

mkdir -p "$LOG_DIR"

run_base() {
    for seed in "${SEEDS[@]}"; do
        OUT="${ROOT_OUT}/mcp_base/seed${seed}"

        echo "============================================================"
        echo "MCP-Base seed=${seed}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES="$GPU_BASE" python -m tri_view_compat.train_triview \
          --train_csv "$TRAIN_CSV" \
          --val_csv "$VAL_CSV" \
          --test_csv "$TEST_CSV" \
          --clip_name "$CLIP_PATH" \
          --output_dir "$OUT" \
          --epochs 10 \
          --batch_size 256 \
          --num_workers 8 \
          --lr 1e-3 \
          --weight_decay 1e-4 \
          --dropout 0.2 \
          --no_geo \
          --seed "$seed"

        CUDA_VISIBLE_DEVICES="$GPU_BASE" python -m tri_view_compat.tools.eval_threshold_calibrated \
          --ckpt_dir "$OUT" \
          --model_type baseline \
          --val_csv "$VAL_CSV" \
          --test_csv "$TEST_CSV" \
          --batch_size 256 \
          --num_workers 8 \
          --seed "$seed"

        cp "$OUT/threshold_metrics.json" \
           "$OUT/threshold_metrics_original.json"

        CUDA_VISIBLE_DEVICES="$GPU_BASE" python -m tri_view_compat.tools.eval_threshold_calibrated \
          --ckpt_dir "$OUT" \
          --model_type baseline \
          --val_csv "$VAL_CSV" \
          --test_csv "$BALANCED_CSV" \
          --batch_size 256 \
          --num_workers 8 \
          --seed "$seed"

        cp "$OUT/threshold_metrics.json" \
           "$OUT/threshold_metrics_balanced.json"
    done
}

run_compat() {
    for seed in "${SEEDS[@]}"; do
        OUT="${ROOT_OUT}/mcp_compat/seed${seed}"

        echo "============================================================"
        echo "MCP-Compat seed=${seed}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES="$GPU_COMPAT" python -m tri_view_compat.train_triview_prior_v2 \
          --train_csv "$TRAIN_CSV" \
          --val_csv "$VAL_CSV" \
          --test_csv "$TEST_CSV" \
          --clip_name "$CLIP_PATH" \
          --output_dir "$OUT" \
          --epochs 10 \
          --batch_size 256 \
          --num_workers 8 \
          --lr 1e-3 \
          --weight_decay 1e-4 \
          --prior_loss_weight 0.2 \
          --no_geo \
          --seed "$seed"

        CUDA_VISIBLE_DEVICES="$GPU_COMPAT" python -m tri_view_compat.tools.eval_threshold_calibrated \
          --ckpt_dir "$OUT" \
          --model_type prior_v2 \
          --val_csv "$VAL_CSV" \
          --test_csv "$TEST_CSV" \
          --batch_size 256 \
          --num_workers 8 \
          --seed "$seed"

        cp "$OUT/threshold_metrics.json" \
           "$OUT/threshold_metrics_original.json"

        CUDA_VISIBLE_DEVICES="$GPU_COMPAT" python -m tri_view_compat.tools.eval_threshold_calibrated \
          --ckpt_dir "$OUT" \
          --model_type prior_v2 \
          --val_csv "$VAL_CSV" \
          --test_csv "$BALANCED_CSV" \
          --batch_size 256 \
          --num_workers 8 \
          --seed "$seed"

        cp "$OUT/threshold_metrics.json" \
           "$OUT/threshold_metrics_balanced.json"
    done
}

run_base > "$LOG_DIR/mcp_base.log" 2>&1 &
PID_BASE=$!

run_compat > "$LOG_DIR/mcp_compat.log" 2>&1 &
PID_COMPAT=$!

wait "$PID_BASE"
wait "$PID_COMPAT"

python -m tri_view_compat.tools.summarize_three_seed \
  --root "$ROOT_OUT" \
  --seeds "${SEEDS[@]}"

echo
echo "Three-seed stability experiment completed."
