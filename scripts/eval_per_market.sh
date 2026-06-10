#!/usr/bin/env bash
# Re-train the BEST CNN-LSTM config (s2-stride5-lr1e3) once on all 12 markets,
# then report TEST metrics for each market separately (--per-market-eval).
#
# This is the exact winning configuration from results/BEST_EXPERIMENT.md, with
# two additions:
#   --per-market-eval : also logs metrics["per_market"][<market>] on the test split
#   --save-model      : pulls the trained checkpoint (.pt) back locally
#
# The single trained model is evaluated on every market using the GLOBAL target
# standardization + global naive baseline, so per-market numbers are comparable.
# The per-market block lands in the run JSON under results/runs/cnn_lstm_*.json
# (and one appended line in results/experiments.jsonl).
#
# Optionally pass extra seeds to average out noise, e.g.:
#   bash scripts/eval_per_market.sh 42 0 7
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-H100}"
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(42); fi

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "######## PER-MARKET EVAL  (seed=$seed, gpu=$GPU) ########"
  modal run src/modal_app.py \
    --model cnn_lstm --gpu "$GPU" \
    --markets all --train-stride 5 --eval-stride 10 \
    --lr 1e-3 --hidden 64 --weight-decay 0 \
    --max-epochs 60 --patience 6 \
    --seed "$seed" \
    --per-market-eval --save-model \
    --tag "permarket-seed$seed" 2>&1 || echo "[FAILED] seed=$seed"
done

echo ""
echo "######## DONE — now run: python scripts/analyze_market_groups.py ########"
