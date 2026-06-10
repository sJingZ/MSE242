#!/usr/bin/env bash
# Stage-3 robustness check for the LSTM-MLP: re-run the two finalist configs
# across extra seeds to see which is stable. Sharpe is the high-variance metric,
# so a single-seed winner can mislead -- with a ~85K-param model on noisy
# Polymarket data, small best_val differences are mostly seed noise. Seed 42
# already exists from stage-2; this adds seeds 0 and 7 so you can report a
# 3-seed mean+-std (use --group on the analyzer). Per-run files ->
# results/runs/lstm_mlp_tuning/; appends to results/experiments.jsonl (tags "ml3-").
#   Inspect with:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml3- --group
set -uo pipefail
cd "$(dirname "$0")/.."

# >>> EDIT THESE after reading stage-2 (analyze_lstm_mlp_experiments.py --prefix ml2-) <<<
BEST_LR=1e-3        # winning learning rate
BEST_STRIDE=10      # winning train_stride

GPU=A10G
COMMON="--model lstm_mlp --gpu $GPU --max-epochs 60 --patience 8 --markets all \
--eval-stride 10 --lr $BEST_LR --train-stride $BEST_STRIDE \
--runs-subdir lstm_mlp_tuning"

run () {
  local tag="$1"; shift
  echo ""
  echo "######## RUN: $tag ########"
  modal run src/lstm_modal_app.py $COMMON --tag "$tag" "$@" 2>&1 || echo "[FAILED] $tag"
}

for SEED in 0 7; do
  # Finalist A: trunk hidden 128, head 1x64  (<<< edit to your stage-2 winner)
  run "ml3-A-h128-mlp1x64-s${SEED}"  --hidden 128 --mlp-hidden 64 --mlp-layers 1 --seed $SEED
  # Finalist B: trunk hidden 128, head 2x128  (<<< edit to your stage-2 runner-up)
  run "ml3-B-h128-mlp2x128-s${SEED}" --hidden 128 --mlp-hidden 128 --mlp-layers 2 --seed $SEED
done

echo ""
echo "######## STAGE-3 SWEEP DONE ########"
echo "Rank the runs:  python scripts/analyze_lstm_mlp_experiments.py --prefix ml3- --group"
