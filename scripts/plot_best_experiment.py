#!/usr/bin/env python3
"""Regenerate the viz_07 results dashboard for a single experiment record.

Standalone reimplementation of the `show_experiment` cell in
`notebooks/data_quality_viz.ipynb`, so the figure can be regenerated for any run
in `results/experiments.jsonl` without opening the notebook.

Usage:
    python scripts/plot_best_experiment.py                       # newest record
    python scripts/plot_best_experiment.py --tag s2-stride5-lr1e3
    python scripts/plot_best_experiment.py --tag s2-stride5-lr1e3 --gpu H100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOG = RESULTS_DIR / "experiments.jsonl"


def load_experiments(path=None):
    path = Path(path) if path else LOG
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"No experiments found in {path}")
    return recs


def experiment_id(rec):
    tag = rec.get("tag", "run")
    ts = rec["timestamp"].replace("-", "").replace(":", "")[:15]  # YYYYMMDDTHHMMSS
    return f"cnnlstm-{tag}-{ts}"


def pick(recs, tag=None, run_index=None):
    if run_index is not None:
        return recs[run_index]
    if tag is not None:
        matches = [r for r in recs if r.get("tag") == tag]
        if not matches:
            raise SystemExit(f"No run with tag={tag!r}. Tags: "
                             f"{sorted({r.get('tag') for r in recs})}")
        return matches[-1]
    return recs[-1]


def show_experiment(rec, gpu_label=None, save=True):
    horizons = rec["horizons"]
    hk = [str(h) for h in horizons]
    cfg = rec["config"]
    hist = rec["training"]["history"]
    tr, va = hist["train_loss"], hist["val_loss"]
    epochs = np.arange(1, len(tr) + 1)
    best_ep = int(np.nanargmin(va)) + 1 if va else 0

    m = rec["metrics"]
    oos = m["out_of_sample"]
    val = m.get("validation")
    lin = m.get("linear_benchmark_oos")

    run_id = experiment_id(rec)
    device = rec.get("device", "?")
    dev_str = f"{device}" + (f" ({gpu_label})" if gpu_label else "")
    nw = rec["data"]["n_windows"]

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)
    ax_info = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1:])
    ax_r2 = fig.add_subplot(gs[1, 0])
    ax_da = fig.add_subplot(gs[1, 1])
    ax_sh = fig.add_subplot(gs[1, 2])

    # ── Run summary panel ───────────────────────────────────────────────────
    ax_info.axis("off")
    info_lines = [
        ("Tag", cfg.get("tag", "")),
        ("Device", dev_str),
        ("Params", f"{rec.get('n_params', 0):,}"),
        ("Markets", f"{len(rec['data']['markets'])}  (train {nw['train']:,})"),
        ("train_stride", str(cfg.get("train_stride"))),
        ("lr / wd", f"{cfg.get('lr'):g} / {cfg.get('weight_decay'):g}"),
        ("hidden", str(cfg.get("hidden"))),
        ("Epochs run", f"{rec['training']['epochs_run']} / {cfg['max_epochs']}"
                       f"  (best @ {best_ep})"),
        ("best val loss", f"{rec['training']['best_val_loss']:.4f}"),
    ]
    if val:
        info_lines += [
            ("VAL R²_OS", f"{val['r2_oos_mean']:+.4f}"),
            ("VAL DirAcc", f"{val['directional_accuracy_mean']:.4f}"),
            ("VAL Sharpe", f"{val['sharpe_mean']:+.4f}"),
        ]
    info_lines += [
        ("TEST R²_OS", f"{oos['r2_oos_mean']:+.4f}"),
        ("TEST DirAcc", f"{oos['directional_accuracy_mean']:.4f}"),
        ("TEST Sharpe", f"{oos['sharpe_mean']:+.4f}"),
    ]
    ax_info.set_title("Run summary", fontsize=12, fontweight="bold", loc="left")
    y = 1.0
    for k, v in info_lines:
        ax_info.text(0.0, y, f"{k}", fontsize=9.5, va="top",
                     color="#555", fontfamily="monospace")
        ax_info.text(0.50, y, f"{v}", fontsize=9.5, va="top",
                     fontweight="bold", fontfamily="monospace")
        y -= 1.0 / (len(info_lines) + 1)

    # ── Train / Val loss ────────────────────────────────────────────────────
    ax_loss.plot(epochs, tr, "-o", ms=4, label="train loss", color="#1f77b4")
    ax_loss.plot(epochs, va, "-s", ms=4, label="val loss", color="#d62728")
    if best_ep:
        ax_loss.axvline(best_ep, ls="--", color="gray", lw=1)
        ax_loss.scatter([best_ep], [va[best_ep - 1]], s=120, facecolors="none",
                        edgecolors="green", lw=1.8,
                        label=f"best val (epoch {best_ep})")
    ax_loss.axhline(1.0, ls=":", color="#888", lw=1)
    ax_loss.text(epochs[-1], 1.0, " predict-mean baseline (=1.0)", fontsize=8,
                 va="bottom", ha="right", color="#888")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE loss (standardized targets)")
    ax_loss.set_title("Train / Val loss vs. epoch", fontweight="bold")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # ── Per-horizon bars: model vs Ridge ────────────────────────────────────
    x = np.arange(len(hk))
    w = 0.38

    ax_r2.bar(x - w / 2, [oos["r2_oos"][h] for h in hk], w, label="CNN-LSTM",
              color="#1f77b4")
    if lin:
        ax_r2.bar(x + w / 2, [lin["r2_oos"][h] for h in hk], w, label="Ridge",
                  color="#ff7f0e")
    ax_r2.axhline(0, color="k", lw=0.8)
    ax_r2.set_xticks(x); ax_r2.set_xticklabels([f"h={h}" for h in hk])
    ax_r2.set_xlabel("horizon"); ax_r2.set_title("R²_OS (vs naive)")
    ax_r2.legend(fontsize=8); ax_r2.grid(alpha=0.3, axis="y")

    ax_da.bar(x - w / 2, [oos["directional_accuracy"][h] for h in hk], w,
              label="CNN-LSTM", color="#1f77b4")
    if lin:
        ax_da.bar(x + w / 2, [lin["directional_accuracy"][h] for h in hk], w,
                  label="Ridge", color="#ff7f0e")
    ax_da.axhline(0.5, color="k", lw=0.8, ls="--")
    ax_da.set_ylim(0.45, max(0.75, max(oos["directional_accuracy"].values()) + 0.05))
    ax_da.set_xticks(x); ax_da.set_xticklabels([f"h={h}" for h in hk])
    ax_da.set_xlabel("horizon"); ax_da.set_title("Directional accuracy")
    ax_da.legend(fontsize=8); ax_da.grid(alpha=0.3, axis="y")

    ax_sh.bar(x - w / 2, [oos["sharpe"][h] for h in hk], w, label="CNN-LSTM",
              color="#1f77b4")
    if lin:
        ax_sh.bar(x + w / 2, [lin["sharpe"][h] for h in hk], w, label="Ridge",
                  color="#ff7f0e")
    ax_sh.axhline(0, color="k", lw=0.8)
    ax_sh.set_xticks(x); ax_sh.set_xticklabels([f"h={h}" for h in hk])
    ax_sh.set_xlabel("horizon"); ax_sh.set_title("Sharpe (annualized)")
    ax_sh.legend(fontsize=8); ax_sh.grid(alpha=0.3, axis="y")

    fig.suptitle(f"CNN-LSTM training results — {run_id}", fontsize=14,
                 fontweight="bold")

    if save:
        out = RESULTS_DIR / f"viz_07_results_{cfg.get('tag', 'run')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="record tag to plot (default: newest)")
    ap.add_argument("--run-index", type=int, default=None)
    ap.add_argument("--gpu", default=None, help="GPU label for annotation, e.g. H100")
    args = ap.parse_args()

    recs = load_experiments()
    rec = pick(recs, tag=args.tag, run_index=args.run_index)
    show_experiment(rec, gpu_label=args.gpu)


if __name__ == "__main__":
    main()
