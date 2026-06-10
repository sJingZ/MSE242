#!/usr/bin/env bash
# Stage-1 hyper-parameter sweep for the LSTM-MLP on Modal.
# Goal: fix the under-training seen in the lstm-mlp-baseline run. At lr=1e-5
# (the paper's Nasdaq default) train_loss only fell 1.004 -> 0.989 over 50
# epochs, val_loss was still dropping at epoch 50, and early stopping never
# fired -- i.e. mse_model ~= mse_naive (the model barely beat predicting the
# mean). lr is the #1 lever; this sweep shifts the grid *up* and also tries the
# higher data volume (train_stride 10). MLP head is held at the 1x64 default.
# Per-run JSON/.pt/.npz are stored under results/runs/lstm_mlp_tuning/; each run
# also appends one line to the global results/experiments.jsonl (tags "ml1-").
#   Inspect with:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml1-
set -uo pipefail
cd "$(dirname "$0")/.."

# A10G is plenty for a ~85K-param LSTM-MLP (cheaper than H100). Bump
# --patience/--max-epochs vs the paper so that at a sane lr the model converges
# and early stopping does real work instead of being cut off by the epoch budget.
GPU=A10G
COMMON="--model lstm_mlp --gpu $GPU --max-epochs 60 --patience 8 --markets all \
--eval-stride 10 --hidden 128 --mlp-hidden 64 --mlp-layers 1 --seed 42 \
--runs-subdir lstm_mlp_tuning"

run () {  # $1 = tag, rest = extra args
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# tag                 extra hyper-params
# --- lr sweep at the baseline data volume (train_stride 25, ~57k windows) ---
run ml1-lr5e5-st25    --lr 5e-5 --train-stride 25
run ml1-lr1e4-st25    --lr 1e-4 --train-stride 25
run ml1-lr5e4-st25    --lr 5e-4 --train-stride 25
run ml1-lr1e3-st25    --lr 1e-3 --train-stride 25
run ml1-lr2e3-st25    --lr 2e-3 --train-stride 25
# --- best-guess lrs at higher data volume (train_stride 10, ~143k windows) ---
run ml1-lr5e4-st10    --lr 5e-4 --train-stride 10
run ml1-lr1e3-st10    --lr 1e-3 --train-stride 10
# --- control: the paper's Nasdaq default lr (reproduces the under-trained baseline) ---
run ml1-lr1e5-st25    --lr 1e-5 --train-stride 25

echo ""
echo "######## STAGE-1 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml1-"
