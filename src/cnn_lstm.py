#!/usr/bin/env python3
"""CNN-LSTM for Polymarket Order-Flow (OF) mid-price-return forecasting.

Implements the paper's CNN-LSTM architecture (OF input, Blocks 2-5 + Inception + LSTM)
in PyTorch, trains it on the processed per-market OF arrays produced by the
`colab_of_pipeline` notebook, and evaluates it with several metrics against benchmarks.

Run from the CLI, e.g.::

    python src/cnn_lstm.py --max-epochs 50 --batch-size 256 --lr 1e-3
    python src/cnn_lstm.py --quick                 # tiny fast smoke run
    python src/cnn_lstm.py --markets "Suns vs Thunder__Suns" --hidden 128

Every run appends a full record (config + all metrics) to
``results/experiments.jsonl`` and writes a detailed per-run JSON to
``results/runs/cnn_lstm_<timestamp>.json``.

Architecture (OF input, shapes shown as [time x feature x channels]):
    Input            [100 x 20 x 1]
    Block 2a  Conv (1x2) s(1x2)  ->  [100 x 10 x 32]   merge bid/ask
    Block 2b  Conv (4x1) same x2 ->  [100 x 10 x 32]   short-term time patterns
    Block 3   Conv (1x10)        ->  [100 x  1 x 32]   aggregate the 10 levels
    Inception 3 parallel subblocks (3-tick / 5-tick / maxpool), concat
                                 ->  [100 x 192]
    LSTM (1 layer, hidden=64)    ->  last hidden state
    Dense                        ->  [H] (one prediction per horizon)
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


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Paths
    data_dir: str = "data/processed/of"
    results_dir: str = "results"
    # Data / windowing
    window: int = 100
    markets: str = "all"            # "all" or comma-separated market keys
    train_stride: int = 25          # subsample stride for training windows (overlapping)
    eval_stride: int = 10           # subsample stride for val/test windows
    max_train_windows: int = 0      # 0 = no cap; else random-cap train windows
    # Model
    hidden: int = 64
    cnn_filters: int = 32
    inception_filters: int = 64
    batchnorm: bool = True          # BatchNorm in conv blocks (stabilizes MPS + speeds convergence)
    # Training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    patience: int = 5               # early-stopping patience (val loss)
    grad_clip: float = 1.0          # max grad norm (<=0 disables)
    standardize_targets: bool = True
    # Benchmarks / misc
    per_market_eval: bool = False   # also report test metrics per individual market
    linear_benchmark: bool = True
    linear_fit_windows: int = 40000  # subsample size for the linear benchmark
    sharpe_ann_factor: float = math.sqrt(252)
    seed: int = 42
    device: str = "auto"            # auto | cpu | cuda | mps
    num_workers: int = 0
    tag: str = ""                   # free-form experiment label

    # Filled in at load time (not user-set)
    horizons: list = field(default_factory=list)
    of_dim: int = 20


# --------------------------------------------------------------------------- #
# Data loading + windowing                                                    #
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Slices per-market OF sequences into (window, of_dim) input matrices on demand.

    Windows never cross market boundaries. A window ending at index ``t`` belongs to
    the split of ``t`` and is kept only if its target row has no NaN (enough future
    steps exist). Targets are optionally standardized using train-set statistics.
    """

    def __init__(self, markets, index, window, of_dim, target_mu=None, target_sigma=None):
        self.markets = markets            # list of dicts with 'of_norm', 'returns'
        self.index = index                # (N, 2) int array of (market_idx, end_t)
        self.window = window
        self.of_dim = of_dim
        self.target_mu = target_mu
        self.target_sigma = target_sigma

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        m, t = self.index[i]
        d = self.markets[m]
        x = d["of_norm"][t - self.window + 1 : t + 1]          # (window, of_dim)
        y = d["returns"][t]                                    # (H,)
        if self.target_mu is not None:
            y = (y - self.target_mu) / self.target_sigma
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )


def load_markets(cfg: Config):
    """Load processed npz files + global config. Returns (markets, run_config)."""
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
def _conv_block(in_c, out_c, kernel, stride=(1, 1), same_time=False,
                use_bn=True, leaky=0.01):
    """Conv2d -> (BatchNorm2d) -> LeakyReLU.

    When ``same_time`` is set, the time-axis kernel ``(kt, 1)`` is given explicit
    'same' zero-padding via ZeroPad2d instead of ``padding='same'``. PyTorch's MPS
    (Apple Silicon) backend mis-handles 'same' padding with even kernels and can
    emit NaN/inf, so we pad manually (output length stays == input length on every
    backend). BatchNorm bounds activations, which both speeds convergence and
    prevents the activation/gradient blow-up that overflows the MPS backend.
    """
    layers = []
    if same_time:
        kt = kernel[0]
        top = (kt - 1) // 2
        bottom = (kt - 1) - top
        layers.append(nn.ZeroPad2d((0, 0, top, bottom)))   # (left, right, top, bottom)
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=(kt, 1), stride=stride))
    else:
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(leaky))
    return nn.Sequential(*layers)


class CNNLSTM(nn.Module):
    """CNN-LSTM for OF input. Input: (N, window, of_dim) -> output: (N, H)."""

    def __init__(self, window, of_dim, n_horizons, cnn_filters=32,
                 inception_filters=64, hidden=64, leaky=0.01, use_bn=True):
        super().__init__()
        f = cnn_filters
        g = inception_filters
        cb = lambda i, o, k, **kw: _conv_block(i, o, k, use_bn=use_bn, leaky=leaky, **kw)

        # Block 2a: merge bid/ask (feature 20 -> 10), channels 1 -> f
        self.block2a = cb(1, f, (1, 2), stride=(1, 2))
        # Block 2b: two (4x1) same-padding conv over the time axis
        self.block2b = nn.Sequential(
            cb(f, f, (4, 1), same_time=True),
            cb(f, f, (4, 1), same_time=True))
        # Block 3: aggregate the 10 levels (feature 10 -> 1)
        self.block3 = cb(f, f, (1, of_dim // 2))

        # Inception: three parallel subblocks, each ends with g channels
        self.inc1 = nn.Sequential(cb(f, g, (1, 1)), cb(g, g, (3, 1), same_time=True))
        self.inc2 = nn.Sequential(cb(f, g, (1, 1)), cb(g, g, (5, 1), same_time=True))
        self.inc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            cb(f, g, (1, 1)))

        self.lstm = nn.LSTM(input_size=3 * g, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_horizons)

        self._init_lstm_forget_bias()

    def _init_lstm_forget_bias(self):
        """Init the forget-gate bias to ~1 (PyTorch gate order: i, f, g, o)."""
        h = self.lstm.hidden_size
        for name, p in self.lstm.named_parameters():
            if name.startswith("bias"):
                nn.init.zeros_(p)
                p.data[h:2 * h] = 0.5   # bias_ih + bias_hh forget chunks sum to 1

    def forward(self, x):
        # x: (N, window, of_dim) -> (N, 1, window, of_dim)
        x = x.unsqueeze(1)
        x = self.block2a(x)
        x = self.block2b(x)
        x = self.block3(x)              # (N, f, window, 1)

        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)  # (N,3g,window,1)
        x = x.squeeze(-1).permute(0, 2, 1)   # (N, window, 3g)

        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])        # last time step -> (N, H)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
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

        pnl = np.sign(p) * y               # long/short 1 unit in predicted direction
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
              f"  best={best_val:.5f}{'  *' if improved else ''}")
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
    detail_path = runs_dir / f"cnn_lstm_{safe_ts}.json"
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
# Main                                                                        #
# --------------------------------------------------------------------------- #
def build_config_from_args() -> Config:
    p = argparse.ArgumentParser(
        description="Train + evaluate a CNN-LSTM on Polymarket OF data.",
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
    p.add_argument("--hidden", type=int, default=d.hidden)
    p.add_argument("--cnn-filters", type=int, default=d.cnn_filters)
    p.add_argument("--inception-filters", type=int, default=d.inception_filters)
    p.add_argument("--no-batchnorm", dest="batchnorm", action="store_false",
                   help="disable BatchNorm (recover the exact-spec architecture)")
    # training
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--no-standardize-targets", dest="standardize_targets",
                   action="store_false")
    # benchmarks / misc
    p.add_argument("--per-market-eval", dest="per_market_eval",
                   action="store_true",
                   help="also report test metrics for each market separately")
    p.add_argument("--no-linear-benchmark", dest="linear_benchmark",
                   action="store_false")
    p.add_argument("--linear-fit-windows", type=int, default=d.linear_fit_windows)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--quick", action="store_true",
                   help="tiny fast run for smoke-testing (few epochs/windows)")
    args = p.parse_args()

    cfg = Config(
        data_dir=args.data_dir, results_dir=args.results_dir, window=args.window,
        markets=args.markets, train_stride=args.train_stride,
        eval_stride=args.eval_stride, max_train_windows=args.max_train_windows,
        hidden=args.hidden, cnn_filters=args.cnn_filters,
        inception_filters=args.inception_filters, batchnorm=args.batchnorm,
        batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
        patience=args.patience, grad_clip=args.grad_clip,
        standardize_targets=args.standardize_targets,
        per_market_eval=args.per_market_eval,
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


def run_experiment(cfg: Config, *, write: bool = True, return_model: bool = False):
    """Train + evaluate one CNN-LSTM run and return its full result record.

    This holds the entire experiment body so it can be driven either from the
    local CLI (:func:`main`) or from a remote worker (e.g. the Modal app in
    ``src/modal_app.py``).

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

    mk = lambda idx, shuffle: DataLoader(
        WindowDataset(markets, idx, cfg.window, cfg.of_dim, t_mu, t_sigma),
        batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
        drop_last=False)
    train_loader = mk(train_idx, True)
    val_loader = mk(val_idx, False)
    test_loader = mk(test_idx, False)

    # --- model ---
    def build_model(dev):
        torch.manual_seed(cfg.seed)
        return CNNLSTM(cfg.window, cfg.of_dim, len(cfg.horizons),
                       cnn_filters=cfg.cnn_filters,
                       inception_filters=cfg.inception_filters,
                       hidden=cfg.hidden, use_bn=cfg.batchnorm).to(dev)

    model = build_model(device)
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
        model = build_model(device)
        history, best_state, best_val, diverged = train_loop(
            model, train_loader, val_loader, device, cfg, has_val)
    train_secs = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- evaluate ---
    def eval_split(loader, idx):
        pred_std = predict(model, loader, device)
        pred = pred_std * t_sigma + t_mu             # un-standardize
        true = gather_targets(markets, idx)
        naive = np.broadcast_to(t_mu, true.shape)    # constant train-mean predictor
        return compute_metrics(pred, true, naive, cfg.horizons, cfg.sharpe_ann_factor)

    out_sample = eval_split(test_loader, test_idx)
    # Validation metrics drive model/hyper-parameter selection in the sweep;
    # the test split is reported only once, for the finally-chosen config.
    validation = eval_split(val_loader, val_idx) if has_val else None
    # NOTE: train_loader is shuffled (for SGD), so its prediction order would not
    # line up with gather_targets(train_idx). Use a fresh un-shuffled loader so
    # in-sample preds/targets stay aligned.
    in_sample = eval_split(mk(train_idx, False), train_idx)

    # --- per-market test metrics (optional) ---
    # Evaluate the SAME trained model on each market's test windows separately,
    # reusing the GLOBAL target standardization (t_mu/t_sigma) and the GLOBAL
    # naive baseline (train mean) so the per-market numbers stay comparable to
    # each other and to the pooled `out_of_sample` block above.
    per_market = None
    if cfg.per_market_eval and len(test_idx):
        per_market = {}
        for mi in np.unique(test_idx[:, 0]):
            sub = test_idx[test_idx[:, 0] == mi]
            sub_metrics = eval_split(mk(sub, False), sub)
            true_h1 = gather_targets(markets, sub)[:, 0]
            per_market[markets[int(mi)]["key"]] = {
                "n_windows": int(len(sub)),
                # realized vol of the per-tick mid change on this market's test
                # windows (same quantity scripts/group_markets.py groups on).
                "realized_vol_test": float(np.std(true_h1.astype(np.float64))),
                "r2_oos_mean": sub_metrics["r2_oos_mean"],
                "directional_accuracy_mean": sub_metrics["directional_accuracy_mean"],
                "sharpe_mean": sub_metrics["sharpe_mean"],
                "metrics": sub_metrics,
            }
        print("\nPER-MARKET (test) — mean over horizons")
        print(f"  {'market':<34}{'n':>9}{'R2_OS':>10}{'DirAcc':>10}{'Sharpe':>10}")
        for k, v in per_market.items():
            print(f"  {k:<34}{v['n_windows']:>9,}{v['r2_oos_mean']:>10.4f}"
                  f"{v['directional_accuracy_mean']:>10.4f}{v['sharpe_mean']:>10.4f}")

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
        "model": "CNN-LSTM (OF, Blocks 2-5 + Inception + LSTM)",
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
            "validation": validation,
            "out_of_sample": out_sample,
            "in_sample": in_sample,
            "per_market": per_market,
            "linear_benchmark_oos": bench_lin,
        },
    }
    if write:
        detail_path, log_path = record_experiment(cfg, record)
        print(f"\nSaved run detail -> {detail_path}")
        print(f"Appended to experiment log -> {log_path}")

    model_state = None
    if return_model:
        import io
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        model_state = buf.getvalue()
    return {"record": record, "model_state": model_state}


def main():
    cfg = build_config_from_args()
    run_experiment(cfg, write=True)


if __name__ == "__main__":
    main()
