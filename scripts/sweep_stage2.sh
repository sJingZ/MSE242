#!/usr/bin/env bash
# Stage-2 refinement: focus on the winning levers from stage-1
# (more data via smaller train_stride, + higher lr). Appends to experiments.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=H100
COMMON="--model cnn_lstm --gpu $GPU --max-epochs 60 --markets all --eval-stride 10 --seed 42"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# More data (stride 5 ~ 285k windows) and lr / capacity combos.
run s2-stride5-lr1e3          --patience 6 --train-stride 5  --lr 1e-3 --hidden 64
run s2-stride10-lr2e3         --patience 6 --train-stride 10 --lr 2e-3 --hidden 64
run s2-stride5-lr2e3          --patience 6 --train-stride 5  --lr 2e-3 --hidden 64
run s2-stride5-lr2e3-h128     --patience 6 --train-stride 5  --lr 2e-3 --hidden 128
run s2-stride10-lr2e3-pat8    --patience 8 --train-stride 10 --lr 2e-3 --hidden 64

echo ""
echo "######## STAGE-2 SWEEP DONE ########"
