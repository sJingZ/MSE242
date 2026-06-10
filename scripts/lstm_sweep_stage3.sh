#!/usr/bin/env bash
# Stage-3 robustness check for the vanilla LSTM: re-run the two finalist configs
# across extra seeds to see which is stable. Sharpe is the high-variance metric
# (the CNN-LSTM finalist moved +-0.04 across seeds), so a single-seed winner can
# mislead. Seed 42 already exists from stage-2; this adds seeds 0 and 7 so you
# can report a 3-seed mean+-std like results/BEST_EXPERIMENT.md.
# Appends to results/experiments.jsonl (tags prefixed "ls3-").
#   Inspect with:  python scripts/analyze_experiments.py --prefix ls3-
set -uo pipefail
cd "$(dirname "$0")/.."

# >>> EDIT THESE after reading stage-2 (analyze_experiments.py --prefix ls2-) <<<
BEST_LR=2e-3        # winning learning rate
BEST_STRIDE=10      # winning train_stride

GPU=A10G
COMMON="--model lstm --gpu $GPU --max-epochs 60 --patience 8 --markets all --eval-stride 10 --lr $BEST_LR --train-stride $BEST_STRIDE"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

for SEED in 0 7; do
  # Finalist A: single-layer, hidden 128  (<<< edit to your stage-2 winner)
  run "ls3-A-nl1-h128-s${SEED}"  --num-layers 1 --hidden 128 --seed $SEED
  # Finalist B: stacked 3-layer, hidden 128  (<<< edit to your stage-2 runner-up)
  run "ls3-B-nl3-h128-s${SEED}"  --num-layers 3 --hidden 128 --dropout 0.1 --seed $SEED
done

echo ""
echo "######## STAGE-3 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_experiments.py --prefix ls3-"
