"""Compute IN-SAMPLE Ridge-benchmark metrics (the logs only stored OOS).

Reproduces the exact linear benchmark used by the best CNN-LSTM run
(seed 42, window 100, train_stride 5, all markets, n_fit=40000), validates the
reproduction against the recorded `linear_benchmark_oos`, then evaluates the
SAME fitted Ridge model on the training split to obtain in-sample R2/DirAcc/Sharpe.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cnn_lstm import (  # noqa: E402
    Config, load_markets, build_index, gather_targets, compute_metrics,
)
from sklearn.linear_model import Ridge  # noqa: E402

cfg = Config(
    data_dir=str(ROOT / "data" / "processed" / "of"),
    window=100,
    markets="all",
    train_stride=5,
    eval_stride=10,
    max_train_windows=0,
    standardize_targets=True,
    linear_fit_windows=40000,
    seed=42,
)

markets, _ = load_markets(cfg)
W, OFD = cfg.window, cfg.of_dim
rng = np.random.default_rng(cfg.seed)  # matches main(): build_index(train) does not consume rng when max_windows=0

train_idx = build_index(markets, W, "train", cfg.train_stride, rng=rng,
                        max_windows=cfg.max_train_windows)
test_idx = build_index(markets, W, "test", cfg.eval_stride)
print(f"train windows={len(train_idx):,}  test windows={len(test_idx):,}")

train_targets = gather_targets(markets, train_idx)
t_mu = train_targets.mean(axis=0).astype(np.float32)  # naive baseline (train mean)


def flatten(index):
    X = np.empty((len(index), W * OFD), np.float32)
    for i, (m, t) in enumerate(index):
        X[i] = markets[m]["of_norm"][t - W + 1 : t + 1].reshape(-1)
    return X


# --- replicate the exact fit subsample used by linear_benchmark() ---
fit_idx = train_idx
if cfg.linear_fit_windows and len(train_idx) > cfg.linear_fit_windows:
    sel = rng.choice(len(train_idx), size=cfg.linear_fit_windows, replace=False)
    fit_idx = train_idx[np.sort(sel)]
print(f"Ridge fit windows={len(fit_idx):,}")

Xtr = flatten(fit_idx)
Ytr = gather_targets(markets, fit_idx)
ridge = Ridge(alpha=1.0)
ridge.fit(Xtr, Ytr)


def predict_chunked(index, chunk=20000):
    preds = []
    for s in range(0, len(index), chunk):
        preds.append(ridge.predict(flatten(index[s : s + chunk])).astype(np.float32))
    return np.concatenate(preds, axis=0)


# --- (sanity) reproduce the recorded OUT-OF-SAMPLE benchmark ---
te_pred = predict_chunked(test_idx)
te_true = gather_targets(markets, test_idx)
naive_te = np.broadcast_to(t_mu, te_true.shape)
oos = compute_metrics(te_pred, te_true, naive_te, cfg.horizons, cfg.sharpe_ann_factor)

# --- IN-SAMPLE: same Ridge model evaluated on the full training split ---
tr_pred = predict_chunked(train_idx)
naive_tr = np.broadcast_to(t_mu, train_targets.shape)
ins = compute_metrics(tr_pred, train_targets, naive_tr, cfg.horizons, cfg.sharpe_ann_factor)

hk = [str(h) for h in cfg.horizons]


def show(title, d):
    print(f"\n{title}")
    print("  R2_OS ", [round(d['r2_oos'][h], 4) for h in hk], "mean", round(d['r2_oos_mean'], 4))
    print("  DirAcc", [round(d['directional_accuracy'][h], 3) for h in hk], "mean", round(d['directional_accuracy_mean'], 4))
    print("  Sharpe", [round(d['sharpe'][h], 3) for h in hk], "mean", round(d['sharpe_mean'], 4))


show("RIDGE OUT-OF-SAMPLE (should match recorded linear_benchmark_oos)", oos)
show("RIDGE IN-SAMPLE (new)", ins)
