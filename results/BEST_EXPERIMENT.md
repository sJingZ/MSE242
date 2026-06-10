# Best Experiment — CNN-LSTM Order-Flow Forecaster

**Experiment id:** `cnnlstm-valbest-20260610-025536`
**Tag:** `s2-stride5-lr1e3`  ·  **Device:** `cuda (H100, via Modal)`  ·  **Timestamp (UTC):** `2026-06-10T02:55:36`
**Model:** CNN-LSTM (OF, Blocks 2-5 + Inception + LSTM), 125,125 parameters
**Run detail:** `results/runs/cnn_lstm_20260610T025536_9748420000.json`
**Checkpoint:** not persisted for sweep runs — regenerate deterministically with
`modal run src/modal_app.py --model cnn_lstm --gpu H100 --train-stride 5 --lr 1e-3 --hidden 64 --patience 6 --max-epochs 60 --markets all --eval-stride 10 --seed 42 --save-model`

This run is the winner of a 4-stage, 21-run hyper-parameter search (see `scripts/sweep_stage{1,2,3,4}.sh`). It was selected by composite **validation** score and confirmed across 3 random seeds, then reported once on the held-out test split (seed-averaged test: R²=0.0017±0.0007, DirAcc=0.6225±0.0068, Sharpe=0.5666±0.033). The numbers below are for the canonical seed-42 run.

---

## 0. Selection criterion — which metric picked the "best" run?

The best run was **not** chosen by a single metric, and — importantly — it was
**not** chosen on the test set. Model/hyper-parameter selection is done entirely
on the **validation** split; the **test** split is held out and reported exactly
**once**, for the single winning configuration. This avoids letting the test set
leak into model selection (which would make the reported numbers optimistic).

**Composite score** (`scripts/analyze_experiments.py`), all terms use the
per-horizon **mean** on the **validation** set:

```
score = (directional_accuracy_mean − 0.5) × 2   # 0.50→0, 1.00→1
        + sharpe_mean                            # already ~0–1 scale
        + r2_oos_mean × 100                       # rescaled (R² ~0.01–0.02 → ~1–2)
```

| Term | Why this scaling |
|---|---|
| `(DirAcc − 0.5) × 2` | subtract the 0.5 chance baseline, scale so 0.5→1.0 maps to 0→1 |
| `+ Sharpe_mean` | added directly (already on a comparable 0–1 scale) |
| `+ R²_OS_mean × 100` | R² is only ~0.01–0.02 on val, so ×100 makes it comparable to the other two |

**Three-step selection.**
1. **Rank by validation composite score** across the sweep → `s2-stride5-lr1e3`
   (stride5 + lr1e-3 + hidden64) ranked **#1** on the validation split, with
   `s2-stride10-lr2e3` a close **#2** (val score 2.841 vs 2.821 on seed 42).
2. **Multi-seed confirmation** (Sharpe is noisy): re-ran the top-2 configs on
   seeds 0 & 7 (`scripts/sweep_stage4.sh`) and compared their **seed-averaged
   validation** metrics. The chosen config (stride5 + lr1e-3 + hidden64) won
   and was the more stable: **val score 2.710 ± 0.126** vs `s2-stride10-lr2e3`'s
   **2.625 ± 0.147**.
3. **Report test once.** Only after the config was frozen on validation did we
   evaluate it on the held-out test split — the numbers reported in the rest of
   this document (seed-averaged test: R²=0.0017, DirAcc=0.6225, Sharpe=0.5666)
   are that single, final test report, not a quantity that was optimized over.

> Note: the per-run `metrics["validation"]` block is logged by the trainer
> (`src/cnn_lstm.py`) and is what `analyze_experiments.py` ranks on. Runs logged
> before this change (the earlier `best-final`/`full-run`/`s*` records) have no
> validation block and are automatically excluded from the ranking.

---

## 1. Parameter settings


| Group         | Parameter             | Value                | Note                                         |
| ------------- | --------------------- | -------------------- | -------------------------------------------- |
| **Data**      | `markets`             | all (12 NBA markets) | windows never cross market boundaries        |
|               | `window`              | 100                  | look-back ticks fed to the model             |
|               | `of_dim`              | 20                   | order-flow features (10 bid + 10 ask levels) |
|               | `train_stride`        | **5**                | **key tuned lever** — 285,857 train windows  |
|               | `eval_stride`         | 10                   | val=30,513 / test=30,615 windows             |
|               | `standardize_targets` | true                 | targets z-scored with train mean/std         |
| **Model**     | `hidden`              | 64                   | LSTM hidden size                             |
|               | `cnn_filters`         | 32                   | conv block channels                          |
|               | `inception_filters`   | 64                   | inception sub-block channels                 |
|               | `batchnorm`           | true                 | stabilizes training                          |
| **Training**  | `lr`                  | 1e-3                 | Adam (default kept; lr2e-3 ranked #2 on val) |
|               | `weight_decay`        | 0                    | regularization didn't help (see search)      |
|               | `batch_size`          | 256                  |                                              |
|               | `max_epochs`          | 60                   |                                              |
|               | `patience`            | 6                    | early stopping on val loss                   |
|               | `grad_clip`           | 1.0                  | max grad norm                                |
|               | `seed`                | 42                   |                                              |
| **Benchmark** | `linear_benchmark`    | Ridge (α=1)          | fit on flattened windows                     |
|               | `sharpe_ann_factor`   | √252 ≈ 15.87         | annualization factor                         |


Horizons predicted jointly: **h = 1, 2, 3, 5, 10** ticks ahead.

The dominant winning lever was **data volume**: dropping `train_stride` from the
stage-1 default of 25 down to **5** quadruples the training windows (57k → 286k)
and gave the biggest, most consistent validation gains. Raising the learning rate
to 2e-3 helped some configs (it was the runner-up), but the most stable winner on
validation kept the default `lr = 1e-3`.

---

## 2. Train & Validation loss per epoch

Loss = **MSE on standardized targets** (`nn.MSELoss`). Because targets are
z-scored with the train statistics, a model that just predicts the mean scores
a loss of **≈ 1.0** — so `train_loss < 1.0` is the bar for "learning something".


| epoch | train_loss | val_loss    | note                               |
| ----- | ---------- | ----------- | ---------------------------------- |
| 1     | 0.99279    | 3.40797     |                                    |
| 2     | 0.98093    | 3.38878     |                                    |
| 3     | 0.97361    | 3.40009     |                                    |
| 4     | 0.96452    | **3.38861** | ← best val (checkpoint saved here) |
| 5     | 0.95664    | 3.39731     |                                    |
| 6     | 0.94151    | 3.40895     |                                    |
| 7     | 0.92874    | 3.43607     |                                    |
| 8     | 0.91664    | 3.42286     |                                    |
| 9     | 0.89667    | 3.41005     |                                    |
| 10    | 0.87848    | 3.44048     | early stop (no val gain for 6)     |


**Reading the curve**

- `train_loss` falls smoothly from 0.993 → 0.878 (the model is learning real structure).
- `val_loss` bottoms at **3.389 (epoch 4)** — epoch 2 is nearly identical (3.3888) — then drifts up → early stopping restores the epoch-4 weights.
- The ~3.5× train/val gap is **not classic overfitting** (weight decay didn't help in the sweep); it reflects **non-stationarity / distribution shift** between the train and val periods — an inherent property of these market series.

See `results/viz_07_results_s2-stride5-lr1e3.png` for the plotted curve and
per-horizon bars (regenerate any time with
`python scripts/plot_best_experiment.py --tag s2-stride5-lr1e3 --gpu H100`).

---

## 3. The three metrics

All metrics are computed **per horizon** in `compute_metrics` (`src/cnn_lstm.py`),
then averaged across the 5 horizons. Predictions are first un-standardized back
to raw return units; the **naive benchmark** is the constant train-mean predictor.
Selection used the **validation** values; the **test** values below are the
final, report-once held-out numbers.

### 3.1 R²_OS — Out-of-Sample R²

**(a) Definition.** The fraction of return variance the model explains *relative
to the naive constant-mean predictor*.

**(b) How it is computed.**

```
R²_OS = 1 − MSE_model / MSE_naive
MSE_model = mean( (pred − true)² )
MSE_naive = mean( (train_mean − true)² )
```

`> 0` means the model beats "always predict the average return".

**(c) Interpretation (this run).** Validation **R²_OS = +0.0159** (positive on all
5 horizons). On the held-out **test** set the mean is **+0.0007**, positive on
**4 of 5 horizons** (h=1:−0.0009, h=2:+0.0019, h=3:+0.0004, h=5:+0.0002, h=10:+0.0021;
seed-averaged test R²=+0.0017±0.0007). In high-frequency return forecasting, R² is
intrinsically tiny (signal-to-noise is very low), so a small-but-positive R²_OS is
a **genuine edge** — and it stands in sharp contrast to the Ridge benchmark, whose
R²_OS is **negative on every horizon** (it overfits and loses to naive). Do not
expect values like 0.5 here; that would signal a data leak, not skill.

### 3.2 Directional Accuracy

**(a) Definition.** How often the model gets the **sign** (up vs. down) of the future return right — measured **only on ticks that actually moved** (`true ≠ 0`), since most ticks are flat and scoring those just rewards predicting zero.

**(b) How it is computed.**

```
moved = (true != 0)
directional_accuracy = mean( sign(pred[moved]) == sign(true[moved]) )
```

`directional_coverage` (reported alongside) is the fraction of moved ticks, i.e. the sample size this metric is computed over.

**(c) Interpretation (this run).** Validation **DirAcc = 0.6529**; held-out **test
DirAcc = 0.6134**, i.e. **~61%** correct direction vs. a **0.50 random baseline** —
clearly above chance on every horizon (h=1 is strongest at **0.696**, decaying to
0.562 at h=10, as expected: nearer events are more predictable). This is the most
robust of the three metrics across seeds (seed-averaged test 0.6225 ± 0.0068) and
is the clearest evidence the model captures directional information in the order flow.

### 3.3 Sharpe Ratio (annualized)

**(a) Definition.** The risk-adjusted return of a simple strategy that takes a 1-unit long/short position in the model's predicted direction each tick.

**(b) How it is computed.**

```
pnl    = sign(pred) * true_return          # per-tick P&L
Sharpe = mean(pnl) / std(pnl) * ann_factor # ann_factor = √252 ≈ 15.87
```

`mean_pnl` (the un-annualized average P&L) is reported next to it.

**(c) Interpretation (this run).** Validation **Sharpe = +0.948**; held-out **test
Sharpe = +0.5514**, positive on all horizons (range 0.416–0.693). A positive Sharpe
means the directional strategy has positive risk-adjusted expectancy; ~0.55
annualized is a **modest but real** signal (well above the Ridge benchmark's ~0.12).
Sharpe is the **noisiest** metric across seeds (seed-averaged test 0.5666 ± 0.033),
so treat the level as indicative rather than exact, and note this is a frictionless
figure — it ignores transaction costs and slippage.

---

## Summary


| Metric                      | This run (TEST, seed 42) | Seed-averaged TEST | Random / naive | Ridge benchmark   |
| --------------------------- | ------------------------ | ------------------ | -------------- | ----------------- |
| R²_OS (mean)                | **+0.0007**              | +0.0017 ± 0.0007   | 0.0            | −0.012 (negative) |
| Directional accuracy (mean) | **0.6134**               | 0.6225 ± 0.0068    | 0.50           | 0.557             |
| Sharpe annualized (mean)    | **+0.5514**              | 0.5666 ± 0.033     | 0.0            | 0.118             |

For reference, the winning config's **validation** metrics (used for selection)
were R²=+0.0159, DirAcc=0.6529, Sharpe=+0.948 (seed 42).

**Bottom line:** selected purely on validation and reported once on test, the
tuned CNN-LSTM beats both the naive mean predictor (positive mean R²_OS, positive
on 4/5 horizons) and the linear benchmark on all three metrics, with ~61%
directional accuracy and a ~0.57 annualized Sharpe. The signal is small in
absolute terms — expected for tick-level market microstructure — but consistent
and statistically credible across seeds.
