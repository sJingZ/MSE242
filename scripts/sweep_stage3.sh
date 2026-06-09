#!/usr/bin/env bash
# Stage-3 robustness check: re-run the two finalist configs across extra seeds
# to see which is stable (Sharpe is high-variance). Appends to experiments.jsonl.
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
  # Finalist A: stride10 + lr2e3 + patience8
  run "s3-A-stride10-lr2e3-s${SEED}"      --patience 8 --train-stride 10 --lr 2e-3 --hidden 64  --seed $SEED
  # Finalist B: stride5 + lr2e3 + hidden128
  run "s3-B-stride5-lr2e3-h128-s${SEED}"  --patience 6 --train-stride 5  --lr 2e-3 --hidden 128 --seed $SEED
done

echo ""
echo "######## STAGE-3 SWEEP DONE ########"
