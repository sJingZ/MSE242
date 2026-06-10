#!/usr/bin/env bash
# Stage-2 refinement for the LSTM-MLP: fix the winning learning-rate /
# train-stride from stage-1, then sweep capacity along BOTH axes that the
# LSTM-MLP exposes -- the recurrent trunk (hidden, num_layers) and the MLP head
# (mlp_hidden, mlp_layers, mlp_activation). Dropout between LSTM layers is only
# active when num_layers > 1 (PyTorch convention), so we add a little for the
# deeper trunks. Per-run files -> results/runs/lstm_mlp_tuning/; appends to the
# global results/experiments.jsonl (tags "ml2-").
#   Inspect with:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml2-
set -uo pipefail
cd "$(dirname "$0")/.."

# Stage-1 winners (val composite, ml1-lr1e3-st10): lr=1e-3, train_stride=10.
BEST_LR=1e-3        # winning learning rate from stage-1
BEST_STRIDE=10      # winning train_stride from stage-1

GPU=A10G
COMMON="--model lstm_mlp --gpu $GPU --max-epochs 60 --patience 8 --markets all \
--eval-stride 10 --seed 42 --lr $BEST_LR --train-stride $BEST_STRIDE \
--runs-subdir lstm_mlp_tuning"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# tag                     extra hyper-params
# --- trunk width (MLP head fixed at the 1x64 default) ---
run ml2-h64-mlp1x64       --hidden 64  --mlp-hidden 64 --mlp-layers 1
run ml2-h128-mlp1x64      --hidden 128 --mlp-hidden 64 --mlp-layers 1   # ~ stage-1 winner
run ml2-h256-mlp1x64      --hidden 256 --mlp-hidden 64 --mlp-layers 1
# --- trunk depth (stacked LSTMs, hidden 128) ---
run ml2-nl2-h128          --hidden 128 --num-layers 2 --dropout 0.1 --mlp-hidden 64 --mlp-layers 1
run ml2-nl3-h128          --hidden 128 --num-layers 3 --dropout 0.1 --mlp-hidden 64 --mlp-layers 1
# --- MLP head width / depth (trunk fixed at hidden 128) ---
run ml2-mlp1x128          --hidden 128 --mlp-hidden 128 --mlp-layers 1
run ml2-mlp2x64           --hidden 128 --mlp-hidden 64  --mlp-layers 2
run ml2-mlp2x128          --hidden 128 --mlp-hidden 128 --mlp-layers 2
# --- head activation variant ---
run ml2-mlp1x64-gelu      --hidden 128 --mlp-hidden 64  --mlp-layers 1 --mlp-activation gelu
# --- regularization probe (expect little: val->test gap is distribution shift, not overfit) ---
run ml2-wd1e4             --hidden 128 --mlp-hidden 64  --mlp-layers 1 --weight-decay 1e-4

echo ""
echo "######## STAGE-2 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml2-"
