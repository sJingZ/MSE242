#!/usr/bin/env bash
# Stage-1 hyper-parameter sweep for the CNN-LSTM on H100 (Modal).
# Each run appends one line to results/experiments.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=H100
COMMON="--model cnn_lstm --gpu $GPU --max-epochs 50 --patience 5 --markets all --eval-stride 10 --seed 42"

run () {  # $1 = tag, rest = extra args
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

# tag                              extra hyper-params
run s1-base                        --lr 1e-3 --weight-decay 0    --hidden 64  --train-stride 25
run s1-wd1e4                       --lr 1e-3 --weight-decay 1e-4  --hidden 64  --train-stride 25
run s1-wd1e3                       --lr 1e-3 --weight-decay 1e-3  --hidden 64  --train-stride 25
run s1-lr3e4                       --lr 3e-4 --weight-decay 0     --hidden 64  --train-stride 25
run s1-lr2e3                       --lr 2e-3 --weight-decay 0     --hidden 64  --train-stride 25
run s1-h128                        --lr 1e-3 --weight-decay 0     --hidden 128 --train-stride 25
run s1-h128-inc128-wd1e4           --lr 1e-3 --weight-decay 1e-4  --hidden 128 --inception-filters 128 --train-stride 25
run s1-stride10                    --lr 1e-3 --weight-decay 0     --hidden 64  --train-stride 10

echo ""
echo "######## STAGE-1 SWEEP DONE ########"
