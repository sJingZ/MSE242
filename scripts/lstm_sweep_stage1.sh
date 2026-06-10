#!/usr/bin/env bash
# Stage-1 hyper-parameter sweep for the vanilla LSTM on Modal.
# Goal: find the right learning-rate x data-volume regime. The baseline run
# (lr=1e-5, the paper's Nasdaq default) barely trained -- train_loss only fell
# 1.005 -> 0.990 over 50 epochs and never early-stopped. lr is the #1 lever here.
# Each run appends one line to results/experiments.jsonl (tags prefixed "ls1-"
# so they don't collide with the CNN-LSTM "s1-" runs).
#   Inspect with:  python scripts/analyze_experiments.py --prefix ls1-
set -uo pipefail
cd "$(dirname "$0")/.."

# A10G is plenty for a ~80K-param LSTM (cheaper than H100). Bump --patience/
# --max-epochs vs the paper: at a sane lr the model converges and we want early
# stopping to do real work instead of being cut off by the epoch budget.
GPU=A10G
COMMON="--model lstm --gpu $GPU --max-epochs 60 --patience 8 --markets all --eval-stride 10 --hidden 128 --seed 42"

run () {  # $1 = tag, rest = extra args
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# tag                 extra hyper-params
# --- lr sweep at the baseline data volume (train_stride 25, ~57k windows) ---
run ls1-lr1e4-st25    --lr 1e-4 --train-stride 25
run ls1-lr3e4-st25    --lr 3e-4 --train-stride 25
run ls1-lr1e3-st25    --lr 1e-3 --train-stride 25
run ls1-lr2e3-st25    --lr 2e-3 --train-stride 25
# --- best-guess lrs at higher data volume (train_stride 10, ~143k windows) ---
run ls1-lr3e4-st10    --lr 3e-4 --train-stride 10
run ls1-lr1e3-st10    --lr 1e-3 --train-stride 10
run ls1-lr2e3-st10    --lr 2e-3 --train-stride 10
# --- control: the paper's Nasdaq default lr, with the extra data ---
run ls1-lr1e5-st10    --lr 1e-5 --train-stride 10

echo ""
echo "######## STAGE-1 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_experiments.py --prefix ls1-"
