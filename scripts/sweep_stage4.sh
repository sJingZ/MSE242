#!/usr/bin/env bash
# Stage-4 robustness check (validation-based selection): re-run the two NEW
# top-ranked configs by validation composite score across extra seeds, so the
# winner is picked on seed-averaged VALIDATION metrics (Sharpe is high-variance).
# Appends to experiments.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=H100
COMMON="--model cnn_lstm --gpu $GPU --max-epochs 60 --markets all --eval-stride 10"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

for SEED in 0 7; do
  # Finalist X (val #1): stride5 + lr1e3 + hidden64
  run "s4-X-stride5-lr1e3-s${SEED}"   --patience 6 --train-stride 5  --lr 1e-3 --hidden 64 --seed $SEED
  # Finalist Y (val #2): stride10 + lr2e3 + hidden64
  run "s4-Y-stride10-lr2e3-s${SEED}"  --patience 6 --train-stride 10 --lr 2e-3 --hidden 64 --seed $SEED
done

echo ""
echo "######## STAGE-4 SWEEP DONE ########"
