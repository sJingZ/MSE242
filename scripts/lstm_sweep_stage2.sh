#!/usr/bin/env bash
# Stage-2 refinement for the vanilla LSTM: fix the winning learning-rate /
# train-stride from stage-1, then sweep model capacity -- depth (num_layers,
# incl. the paper's "LSTM (3)" row) and width (hidden) -- AND regularization.
# Stage-1 showed the higher-lr runs learn but overfit (in-sample R2 up, OOS R2
# negative), so weight_decay is a first-class lever here, not an afterthought.
# Dropout is only active when num_layers > 1 (PyTorch convention), so we add a
# little for the deeper nets. Appends to results/experiments.jsonl (tags "ls2-").
#   Inspect with:  python scripts/analyze_experiments.py --prefix ls2-
set -uo pipefail
cd "$(dirname "$0")/.."

# >>> EDIT THESE after reading stage-1 (analyze_experiments.py --prefix ls1-) <<<
BEST_LR=2e-3        # winning learning rate from stage-1
BEST_STRIDE=10      # winning train_stride from stage-1

GPU=A10G
COMMON="--model lstm --gpu $GPU --max-epochs 60 --patience 8 --markets all --eval-stride 10 --seed 42 --lr $BEST_LR --train-stride $BEST_STRIDE"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# tag                     extra hyper-params
# --- depth (single-layer baseline + stacked LSTMs) ---
run ls2-nl1-h128          --num-layers 1 --hidden 128
run ls2-nl2-h128          --num-layers 2 --hidden 128 --dropout 0.1
run ls2-nl3-h128          --num-layers 3 --hidden 128 --dropout 0.1   # paper "LSTM (3)" row
# --- width ---
run ls2-nl1-h256          --num-layers 1 --hidden 256
run ls2-nl2-h256          --num-layers 2 --hidden 256 --dropout 0.1
# --- regularization (combat the overfit seen at higher lr in stage-1).
#     These run at $BEST_LR; the goal is to turn the negative OOS R2 positive
#     without losing the in-sample signal. If your BEST_LR ended up tiny (1e-5)
#     and nothing overfit, these are low-value -- skip or re-point at a higher lr.
run ls2-wd1e4-nl1-h128    --num-layers 1 --hidden 128 --weight-decay 1e-4
run ls2-wd1e3-nl1-h128    --num-layers 1 --hidden 128 --weight-decay 1e-3
run ls2-wd1e4-nl1-h256    --num-layers 1 --hidden 256 --weight-decay 1e-4
# --- even more data (train_stride 5, ~285k windows) at the best single-layer net ---
run ls2-stride5-nl1-h128  --num-layers 1 --hidden 128 --train-stride 5

echo ""
echo "######## STAGE-2 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_experiments.py --prefix ls2-"
