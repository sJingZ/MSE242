#!/usr/bin/env python3
"""Rank CNN-LSTM runs in results/experiments.jsonl by out-of-sample metrics.

Usage:
    python scripts/analyze_experiments.py                 # all runs
    python scripts/analyze_experiments.py --prefix s1-    # only tags starting s1-
    python scripts/analyze_experiments.py --top 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "experiments.jsonl"


def load(prefix=None):
    recs = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if prefix and not str(r.get("tag", "")).startswith(prefix):
            continue
        recs.append(r)
    return recs


def row(r):
    c = r["config"]
    oos = r["metrics"]["out_of_sample"]
    ins = r["metrics"]["in_sample"]
    tr = r["training"]["history"]["train_loss"]
    return {
        "tag": r.get("tag", ""),
        "lr": c["lr"],
        "wd": c["weight_decay"],
        "hidden": c["hidden"],
        "inc": c["inception_filters"],
        "tr_stride": c["train_stride"],
        "epochs": r["training"]["epochs_run"],
        "best_val": r["training"]["best_val_loss"],
        "final_train": tr[-1] if tr else float("nan"),
        "oos_r2": oos["r2_oos_mean"],
        "oos_dir": oos["directional_accuracy_mean"],
        "oos_sharpe": oos["sharpe_mean"],
        "ins_r2": ins["r2_oos_mean"],
        # composite score: standardized-ish blend (dir-0.5 emphasised + sharpe + r2 scaled)
        "score": (oos["directional_accuracy_mean"] - 0.5) * 2
                 + oos["sharpe_mean"]
                 + oos["r2_oos_mean"] * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    recs = load(args.prefix)
    if not recs:
        raise SystemExit("No matching runs.")
    rows = sorted((row(r) for r in recs), key=lambda d: d["score"], reverse=True)
    if args.top:
        rows = rows[: args.top]

    hdr = (f"{'rank':>4} {'tag':<22} {'lr':>6} {'wd':>6} {'hid':>4} {'inc':>4} "
           f"{'strd':>5} {'ep':>3} {'best_val':>9} {'oos_r2':>9} {'oos_dir':>8} "
           f"{'oos_shrp':>9} {'ins_r2':>9} {'score':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, d in enumerate(rows, 1):
        print(f"{i:>4} {d['tag']:<22} {d['lr']:>6.0e} {d['wd']:>6.0e} "
              f"{d['hidden']:>4} {d['inc']:>4} {d['tr_stride']:>5} {d['epochs']:>3} "
              f"{d['best_val']:>9.4f} {d['oos_r2']:>+9.4f} {d['oos_dir']:>8.4f} "
              f"{d['oos_sharpe']:>+9.4f} {d['ins_r2']:>+9.4f} {d['score']:>8.4f}")

    best = rows[0]
    print("\nBEST by composite score:", best["tag"])
    print(f"  lr={best['lr']:g} wd={best['wd']:g} hidden={best['hidden']} "
          f"inc={best['inc']} train_stride={best['tr_stride']}")
    print(f"  OOS: R2={best['oos_r2']:+.4f}  DirAcc={best['oos_dir']:.4f}  "
          f"Sharpe={best['oos_sharpe']:+.4f}")


if __name__ == "__main__":
    main()
