#!/usr/bin/env python3
"""LSTM for Polymarket Order-Flow (OF) mid-price-return forecasting.

Implements the paper's vanilla LSTM architecture (Kolm, Turiel & Westray 2021,
Table 2 "LSTM" row) in PyTorch. Mirrors the structure of ``cnn_lstm.py`` so
the two scripts share Config field names, CLI flags, and results JSON schema
-- which means a single Modal wrapper can run either model just by changing
the import.

Architecture (OF input, shapes shown as [batch x time x feature]):
    Input  [B, 100, 20]
    LSTM   (num_layers=1, hidden=128, batch_first=True)
        -> last hidden state [B, 128]
    Linear (128 -> H)
        -> alpha term structure [B, H]

The forget-gate bias is initialized to 1.0 (Gers et al. 2000) for both
``bias_ih`` and ``bias_hh`` chunks, matching the paper's setup.

Run from the CLI, e.g.::

    python src/lstm.py --max-epochs 50 --batch-size 256 --lr 1e-5
    python src/lstm.py --quick                      # tiny fast smoke run
    python src/lstm.py --markets "Suns vs Thunder__Suns" --hidden 256
    python src/lstm.py --num-layers 3 --tag stacked  # paper's "LSTM (3)" row

Import and call programmatically (Colab notebook cell, Modal function, ...):

    from lstm import run_experiment, Config
    out = run_experiment(Config(max_epochs=50, tag="baseline"))
    record = out["record"]

Every run appends a full record (config + all metrics) to
``results/experiments.jsonl`` and writes a detailed per-run JSON to
``results/runs/lstm_<timestamp>.json``.

Paper-faithful defaults for the LSTM row (Table 2):
    lr=1e-5, batch_size=256, max_epochs=50, patience=5, hidden=128.
Note the 1e-5 learning rate -- two orders of magnitude smaller than the
CNN-LSTM's 1e-3. Inherit at your own risk.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install it with `pip install torch`.\n"
        f"Import error: {e}"
    )


# Used in record_experiment() filenames + record["model"] field.
MODEL_NAME = "LSTM"
RESULTS_FILE_PREFIX = "lstm"


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """All inputs to ``run_experiment``. Field names mirror ``cnn_lstm.py``
    wherever the meaning is shared, so a Modal wrapper can pass the same
    Config dict to either script."""

    # Paths
    data_dir: str = "data/processed/of"
    results_dir: str = "results"

    # Data / windowing
    window: int = 100
    markets: str = "all"  # "all" or comma-separated market keys
    train_stride: int = 25  # subsample stride for training windows (overlapping)
    eval_stride: int = 10   # subsample stride for val/test windows
    max_train_windows: int = 0  # 0 = no cap; else random-cap train windows

    # Model (paper Table 2, LSTM row)
    hidden: int = 128       # ~80K LSTM params -> matches paper's ~1e5 total
    num_layers: int = 1     # use 3 for the paper's "LSTM (3)" row
    dropout: float = 0.0    # only applied when num_layers > 1 (PyTorch convention)
    forget_bias_init: float = 1.0  # forget-gate bias init (Gers et al. 2000)

    # Training (paper Table 2)
    batch_size: int = 256
    lr: float = 1e-5        # !!! lower than CNN-LSTM's 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    patience: int = 5       # early-stopping patience (val loss)
    grad_clip: float = 1.0  # max grad norm (<=0 disables)
    standardize_targets: bool = True

    # Benchmarks / misc
    linear_benchmark: bool = True
    linear_fit_windows: int = 40000  # subsample size for the linear benchmark
    sharpe_ann_factor: float = math.sqrt(252)
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | mps
    num_workers: int = 0
    tag: str = ""  # free-form experiment label

    # Filled in at load time (not user-set)
    horizons: list = field(default_factory=list)
    of_dim: int = 20


# --------------------------------------------------------------------------- #
# Data loading + windowing                                                    #
# (kept structurally identical to cnn_lstm.py so the same processed data      #
# files work, and so we can later refactor to a shared `data.py` module.)     #
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Slices per-market OF sequences into (window, of_dim) input matrices on demand.

    Windows never cross market boundaries. A window ending at index ``t`` belongs to
    the split of ``t`` and is kept only if its target row has no NaN (enough future
    steps exist). Targets are optionally standardized using train-set statistics.
    """

    def __init__(self, markets, index, window, of_dim, target_mu=None, target_sigma=None):
        self.markets = markets  # list of dicts with 'of_norm', 'returns'
        self.index = index      # (N, 2) int array of (market_idx, end_t)
        self.window = window
        self.of_dim = of_dim
        self.target_mu = target_mu
        self.target_sigma = target_sigma

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m, t = self.index[i]
        d = self.markets[m]
        x = d["of_norm"][t - self.window + 1 : t + 1]  # (window, of_dim)
        y = d["returns"][t]                            # (H,)
        if self.target_mu is not None:
            y = (y - self.target_mu) / self.target_sigma
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )


def load_markets(cfg: Config):
    """Load processed npz files + global config. Returns (markets, run_config).

    Mutates ``cfg`` to fill in ``cfg.horizons`` and ``cfg.of_dim`` from the
    pipeline's ``_config.json``.
    """
    data_dir = Path(cfg.data_dir)
    config_path = data_dir / "_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_path}. Run the OF pipeline notebook first."
        )
    proc_cfg = json.loads(config_path.read_text())
    cfg.horizons = list(proc_cfg["horizons"])
    cfg.of_dim = int(proc_cfg["of_dim"])
    if cfg.window != int(proc_cfg["window"]):
        print(f"[warn] requested window={cfg.window} differs from pipeline "
              f"window={proc_cfg['window']}; using {cfg.window}.")

    # Select markets.
    all_entries = proc_cfg["markets"]
    if cfg.markets.strip().lower() == "all":
        selected = all_entries
    else:
        wanted = {m.strip() for m in cfg.markets.split(",") if m.strip()}
        selected = [e for e in all_entries if e["key"] in wanted]
        missing = wanted - {e["key"] for e in selected}
        if missing:
            raise SystemExit(f"Unknown market key(s): {sorted(missing)}\n"
                             f"Available: {[e['key'] for e in all_entries]}")
    if not selected:
        raise SystemExit("No markets selected.")

    markets = []
    for e in selected:
        npz = np.load(data_dir / e["file"], allow_pickle=True)
        markets.append({
            "key": e["key"],
            "of_norm": npz["of_norm"].astype(np.float32),
            "returns": npz["returns"].astype(np.float32),
            "split": npz["split"].astype(str),
        })
    return markets, proc_cfg


def build_index(markets, window, split_name, stride, rng=None, max_windows=0):
    """Build a (N, 2) int array of (market_idx, end_t) for a given split."""
    rows = []
    for mi, d in enumerate(markets):
        of = d["of_norm"]
        ret = d["returns"]
        split = d["split"]
        n = len(of)
        # valid end indices: split match + target has no NaN
        finite = ~np.isnan(ret).any(axis=1)
        cand = np.arange(window - 1, n, stride, dtype=np.int64)
        cand = cand[(split[cand] == split_name) & finite[cand]]
        if cand.size:
            rows.append(np.stack([np.full(cand.shape, mi, np.int64), cand], axis=1))
    if not rows:
        return np.empty((0, 2), np.int64)
    idx = np.concatenate(rows, axis=0)
    if max_windows and len(idx) > max_windows and rng is not None:
        sel = rng.choice(len(idx), size=max_windows, replace=False)
        idx = idx[np.sort(sel)]
    return idx


def gather_targets(markets, index):
    """Return the (N, H) raw (unstandardized) targets for an index array."""
    return np.stack([markets[m]["returns"][t] for m, t in index], axis=0)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class LSTMModel(nn.Module):
    """Vanilla LSTM for OF input. (N, window, of_dim) -> (N, H).

    Paper Table 2 "LSTM" row: ~1.0e5 params with hidden=128. The forget-gate
    bias is initialized to ``cfg.forget_bias_init`` (paper uses 1.0).
    """

    def __init__(self, of_dim: int, n_horizons: int, hidden: int = 128,
                 num_layers: int = 1, dropout: float = 0.0,
                 forget_bias_init: float = 1.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=of_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, n_horizons)
        self._init_lstm_forget_bias(forget_bias_init)

    def _init_lstm_forget_bias(self, value: float):
        """Init the forget-gate bias to ``value`` (PyTorch gate order: i, f, g, o).

        PyTorch's LSTM has two bias vectors per layer (``bias_ih_lN`` and
        ``bias_hh_lN``); the effective forget-gate bias is the sum of their
        f-chunks. We zero all biases, then set each f-chunk to value/2 so the
        sum equals ``value``.
        """
        for name, p in self.lstm.named_parameters():
            if name.startswith("bias"):
                nn.init.zeros_(p)
                h = self.lstm.hidden_size
                p.data[h:2 * h] = value / 2.0

    def forward(self, x):
        # x: (N, window, of_dim)
        out, _ = self.lstm(x)            # (N, window, hidden)
        return self.fc(out[:, -1, :])    # last time step -> (N, H)


def build_model(cfg: Config, n_horizons: int, device: torch.device) -> nn.Module:
    """Construct the model and move it to ``device``. Reseeds torch immediately
    before construction so the init is reproducible regardless of how much
    randomness has been consumed earlier in the run."""
    torch.manual_seed(cfg.seed)
    return LSTMModel(
        of_dim=cfg.of_dim,
        n_horizons=n_horizons,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        forget_bias_init=cfg.forget_bias_init,
    ).to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# (verbatim from cnn_lstm.py to keep the results JSON schema identical)       #
# --------------------------------------------------------------------------- #
def compute_metrics(pred, true, benchmark_pred, horizons, ann_factor):
    """Per-horizon R2_OS, directional accuracy, Sharpe, and MSEs.

    pred, true, benchmark_pred : (N, H) arrays (raw return units).
    benchmark_pred is the naive constant prediction (e.g. train mean per horizon).
    """
    out = {"r2_oos": {}, "directional_accuracy": {}, "directional_coverage": {},
           "sharpe": {}, "mean_pnl": {}, "mse_model": {}, "mse_naive": {}}
    for j, h in enumerate(horizons):
        hk = str(h)
        p, y = pred[:, j], true[:, j]
        b = benchmark_pred[:, j]
        mse_m = float(np.mean((p - y) ** 2))
        mse_b = float(np.mean((b - y) ** 2))
        out["mse_model"][hk] = mse_m
        out["mse_naive"][hk] = mse_b
        out["r2_oos"][hk] = float(1.0 - mse_m / mse_b) if mse_b > 0 else float("nan")

        # Directional accuracy only over non-flat ground-truth ticks (the
        # trading-relevant subset): most ticks have exactly-zero return, and
        # scoring those would just measure how often the model outputs ~0.
        moved = y != 0
        out["directional_coverage"][hk] = float(np.mean(moved))
        if moved.any():
            out["directional_accuracy"][hk] = float(
                np.mean(np.sign(p[moved]) == np.sign(y[moved])))
        else:
            out["directional_accuracy"][hk] = float("nan")

        pnl = np.sign(p) * y  # long/short 1 unit in predicted direction
        mu, sd = float(pnl.mean()), float(pnl.std())
        out["mean_pnl"][hk] = mu
        out["sharpe"][hk] = float(mu / sd * ann_factor) if sd > 0 else float("nan")

    def _mean(d):
        vals = [v for v in d.values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out["r2_oos_mean"] = _mean(out["r2_oos"])
    out["directional_accuracy_mean"] = _mean(out["directional_accuracy"])
    out["sharpe_mean"] = _mean(out["sharpe"])
    return out


# --------------------------------------------------------------------------- #
# Train / eval loops                                                          #
# --------------------------------------------------------------------------- #
def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, criterion, optimizer=None, grad_clip=0.0):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = criterion(pred, yb)
        lval = float(loss.detach())
        if train:
            # Abort immediately on a non-finite batch (e.g. an MPS backend bug)
            # so the caller can fall back to CPU without wasting a whole epoch.
            if not math.isfinite(lval):
                return float("nan")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = xb.size(0)
        total += lval * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


def train_loop(model, train_loader, val_loader, device, cfg, has_val):
    """Train with early stopping. Returns (history, best_state, best_val, diverged)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    diverged = False
    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, optimizer, cfg.grad_clip)
        va = run_epoch(model, val_loader, device, criterion) if has_val else float("nan")
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        improved = va < best_val - 1e-7
        if improved:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"epoch {epoch:>3}/{cfg.max_epochs}  train={tr:.5f}  val={va:.5f}"
              f"  best={best_val:.5f}{' *' if improved else ''}")
        if not math.isfinite(tr):
            print("[error] non-finite training loss; aborting this attempt.")
            diverged = True
            break
        if cfg.patience and bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {cfg.patience}).")
            break
    return history, best_state, best_val, diverged


def linear_benchmark(markets, train_idx, test_idx, window, horizons,
                     n_fit, rng):
    """Ridge linear regression on flattened windows -> per-horizon predictions.

    Returns (test_pred, n_fit_used) or (None, 0) if sklearn is unavailable.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print("[warn] scikit-learn not available; skipping linear benchmark.")
        return None, 0

    fit_idx = train_idx
    if n_fit and len(train_idx) > n_fit:
        sel = rng.choice(len(train_idx), size=n_fit, replace=False)
        fit_idx = train_idx[np.sort(sel)]

    def flatten(index):
        X = np.empty((len(index), window * markets[0]["of_norm"].shape[1]), np.float32)
        for i, (m, t) in enumerate(index):
            X[i] = markets[m]["of_norm"][t - window + 1 : t + 1].reshape(-1)
        return X

    Xtr = flatten(fit_idx)
    Ytr = gather_targets(markets, fit_idx)
    Xte = flatten(test_idx)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, Ytr)
    return model.predict(Xte).astype(np.float32), len(fit_idx)


# --------------------------------------------------------------------------- #
# Experiment recording                                                        #
# --------------------------------------------------------------------------- #
def record_experiment(cfg: Config, record: dict):
    results_dir = Path(cfg.results_dir)
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = record["timestamp"].replace(":", "").replace("-", "").replace(".", "_")
    safe_ts = re.sub(r"[^0-9A-Za-z_]+", "", ts)
    detail_path = runs_dir / f"{RESULTS_FILE_PREFIX}_{safe_ts}.json"
    detail_path.write_text(json.dumps(record, indent=2, default=_json_default))

    log_path = results_dir / "experiments.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return detail_path, log_path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def print_metric_table(horizons, in_sample, out_sample, bench_lin=None):
    hk = [str(h) for h in horizons]
    def row(label, d, key, fmt="{:>10.4f}"):
        cells = "".join(fmt.format(d[key][h]) for h in hk)
        print(f"  {label:<26}{cells}")
    header = "  " + " " * 26 + "".join(f"{('h='+h):>10}" for h in hk)
    print("\n" + "=" * len(header))
    print("OUT-OF-SAMPLE (test)")
    print(header)
    row("R2_OS", out_sample, "r2_oos")
    row("Directional accuracy", out_sample, "directional_accuracy")
    row("Sharpe (annualized)", out_sample, "sharpe")
    row("MSE model", out_sample, "mse_model", "{:>10.2e}")
    row("MSE naive (benchmark)", out_sample, "mse_naive", "{:>10.2e}")
    print(f"  -> mean R2_OS={out_sample['r2_oos_mean']:.4f}  "
          f"mean DirAcc={out_sample['directional_accuracy_mean']:.4f}  "
          f"mean Sharpe={out_sample['sharpe_mean']:.4f}")
    print("\nIN-SAMPLE (train)")
    print(header)
    row("R2_OS", in_sample, "r2_oos")
    row("Directional accuracy", in_sample, "directional_accuracy")
    if bench_lin is not None:
        print("\nLINEAR BENCHMARK (Ridge, out-of-sample)")
        print(header)
        row("R2_OS", bench_lin, "r2_oos")
        row("Directional accuracy", bench_lin, "directional_accuracy")
    print("=" * len(header))


# --------------------------------------------------------------------------- #
# Programmatic entrypoint                                                     #
# Matches cnn_lstm.run_experiment's signature exactly so modal_app.py can     #
# dispatch to either module with no model-specific branching.                 #
# --------------------------------------------------------------------------- #
def run_experiment(cfg: Config, *, write: bool = True, return_model: bool = False) -> dict:
    """Train + evaluate one LSTM run and return its full result payload.

    Parameters
    ----------
    cfg : Config
        Fully-populated run configuration.
    write : bool
        If True, persist the record to ``cfg.results_dir`` (experiments.jsonl +
        runs/<ts>.json). Remote workers set this False and let the local caller
        write, so results always land on the local machine.
    return_model : bool
        If True, also return the trained model's ``state_dict`` bytes so the
        caller can save the checkpoint (used to pull weights back from Modal).

    Returns
    -------
    dict
        ``{"record": <record dict>, "model_state": <bytes|None>}``.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # --- data ---
    markets, proc_cfg = load_markets(cfg)
    print(f"Loaded {len(markets)} market(s); horizons={cfg.horizons}")

    train_idx = build_index(markets, cfg.window, "train", cfg.train_stride,
                            rng=rng, max_windows=cfg.max_train_windows)
    val_idx = build_index(markets, cfg.window, "val", cfg.eval_stride)
    test_idx = build_index(markets, cfg.window, "test", cfg.eval_stride)
    print(f"Windows: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("Not enough windows; reduce stride or window.")

    # Target standardization stats from train targets (raw returns).
    train_targets = gather_targets(markets, train_idx)
    if cfg.standardize_targets:
        t_mu = train_targets.mean(axis=0).astype(np.float32)
        t_sigma = train_targets.std(axis=0).astype(np.float32)
        t_sigma[t_sigma < 1e-12] = 1.0
    else:
        t_mu = np.zeros(train_targets.shape[1], np.float32)
        t_sigma = np.ones(train_targets.shape[1], np.float32)

    def mk_loader(idx, shuffle):
        return DataLoader(
            WindowDataset(markets, idx, cfg.window, cfg.of_dim, t_mu, t_sigma),
            batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, drop_last=False)
    train_loader = mk_loader(train_idx, True)
    val_loader   = mk_loader(val_idx,   False)
    test_loader  = mk_loader(test_idx,  False)

    # --- model ---
    n_horizons = len(cfg.horizons)
    model = build_model(cfg, n_horizons, device)
    n_params = count_params(model)
    print(f"Model parameters: {n_params:,}")

    # --- train (with one automatic CPU fallback if a GPU backend diverges) ---
    has_val = len(val_idx) > 0
    t0 = time.time()
    history, best_state, best_val, diverged = train_loop(
        model, train_loader, val_loader, device, cfg, has_val)
    if diverged and device.type != "cpu":
        print(f"[warn] training diverged on {device.type}; retrying on cpu. "
              f"(This is usually a GPU-backend numerical issue, not a model bug.)")
        device = torch.device("cpu")
        model = build_model(cfg, n_horizons, device)
        history, best_state, best_val, diverged = train_loop(
            model, train_loader, val_loader, device, cfg, has_val)
    train_secs = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate ---
    def eval_split(loader, idx):
        pred_std = predict(model, loader, device)
        pred = pred_std * t_sigma + t_mu          # un-standardize
        true = gather_targets(markets, idx)
        naive = np.broadcast_to(t_mu, true.shape) # constant train-mean predictor
        return compute_metrics(pred, true, naive, cfg.horizons, cfg.sharpe_ann_factor)

    out_sample = eval_split(test_loader, test_idx)
    # NOTE: train_loader is shuffled (for SGD), so its prediction order would
    # not line up with gather_targets(train_idx). Use a fresh un-shuffled
    # loader so in-sample preds/targets stay aligned. (Same fix as cnn_lstm.)
    in_sample  = eval_split(mk_loader(train_idx, False), train_idx)

    bench_lin = None
    if cfg.linear_benchmark:
        lin_pred, n_fit = linear_benchmark(
            markets, train_idx, test_idx, cfg.window, cfg.horizons,
            cfg.linear_fit_windows, rng)
        if lin_pred is not None:
            true = gather_targets(markets, test_idx)
            naive = np.broadcast_to(t_mu, true.shape)
            bench_lin = compute_metrics(lin_pred, true, naive, cfg.horizons,
                                        cfg.sharpe_ann_factor)
            bench_lin["linear_fit_windows_used"] = n_fit

    print_metric_table(cfg.horizons, in_sample, out_sample, bench_lin)

    # --- record ---
    record = {
        "model": f"{MODEL_NAME} (OF, num_layers={cfg.num_layers}, hidden={cfg.hidden})",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": cfg.tag,
        "device": str(device),
        "n_params": n_params,
        "horizons": cfg.horizons,
        "config": asdict(cfg),
        "data": {
            "markets": [m["key"] for m in markets],
            "n_windows": {"train": int(len(train_idx)), "val": int(len(val_idx)),
                          "test": int(len(test_idx))},
            "target_mu": t_mu.tolist(),
            "target_sigma": t_sigma.tolist(),
        },
        "training": {
            "best_val_loss": best_val if math.isfinite(best_val) else None,
            "epochs_run": len(history["train_loss"]),
            "train_seconds": round(train_secs, 1),
            "history": history,
        },
        "metrics": {
            "out_of_sample": out_sample,
            "in_sample": in_sample,
            "linear_benchmark_oos": bench_lin,
        },
    }
    detail_path, log_path = (None, None)
    if write:
        detail_path, log_path = record_experiment(cfg, record)
        print(f"\nSaved run detail        -> {detail_path}")
        print(f"Appended to experiment log -> {log_path}")

    model_state = None
    if return_model:
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        model_state = buf.getvalue()
    return {"record": record, "model_state": model_state}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_config_from_args(argv: Optional[list] = None) -> Config:
    """Parse argv into a Config. Pass ``argv=None`` to read from sys.argv
    (default behaviour); pass a list to override (handy for tests)."""
    p = argparse.ArgumentParser(
        description="Train + evaluate an LSTM on Polymarket OF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = Config()
    # paths
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--results-dir", default=d.results_dir)
    # data
    p.add_argument("--window", type=int, default=d.window)
    p.add_argument("--markets", default=d.markets,
                   help='"all" or comma-separated market keys')
    p.add_argument("--train-stride", type=int, default=d.train_stride)
    p.add_argument("--eval-stride", type=int, default=d.eval_stride)
    p.add_argument("--max-train-windows", type=int, default=d.max_train_windows)
    # model
    p.add_argument("--hidden", type=int, default=d.hidden,
                   help="LSTM hidden size (paper LSTM row uses ~128 for ~1e5 params)")
    p.add_argument("--num-layers", type=int, default=d.num_layers,
                   help='number of stacked LSTM layers (use 3 for paper "LSTM (3)" row)')
    p.add_argument("--dropout", type=float, default=d.dropout,
                   help="dropout between LSTM layers (only used when num_layers > 1)")
    p.add_argument("--forget-bias-init", type=float, default=d.forget_bias_init)
    # training
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr,
                   help="learning rate. Paper LSTM row uses 1e-5 (NOT 1e-3 like CNN-LSTM)")
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--no-standardize-targets", dest="standardize_targets",
                   action="store_false")
    # benchmarks / misc
    p.add_argument("--no-linear-benchmark", dest="linear_benchmark",
                   action="store_false")
    p.add_argument("--linear-fit-windows", type=int, default=d.linear_fit_windows)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--quick", action="store_true",
                   help="tiny fast run for smoke-testing (few epochs/windows)")
    args = p.parse_args(argv)

    cfg = Config(
        data_dir=args.data_dir, results_dir=args.results_dir, window=args.window,
        markets=args.markets, train_stride=args.train_stride,
        eval_stride=args.eval_stride, max_train_windows=args.max_train_windows,
        hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout,
        forget_bias_init=args.forget_bias_init,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience, grad_clip=args.grad_clip,
        standardize_targets=args.standardize_targets,
        linear_benchmark=args.linear_benchmark,
        linear_fit_windows=args.linear_fit_windows, seed=args.seed,
        device=args.device, num_workers=args.num_workers, tag=args.tag)

    if args.quick:
        cfg.max_epochs = min(cfg.max_epochs, 2)
        cfg.train_stride = max(cfg.train_stride, 200)
        cfg.eval_stride = max(cfg.eval_stride, 200)
        cfg.max_train_windows = cfg.max_train_windows or 4000
        cfg.linear_fit_windows = min(cfg.linear_fit_windows, 4000)
        if cfg.tag == "":
            cfg.tag = "quick"
    return cfg


def main():
    cfg = build_config_from_args()
    run_experiment(cfg, write=True)


if __name__ == "__main__":
    main()