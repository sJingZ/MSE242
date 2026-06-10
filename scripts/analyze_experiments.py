#!/usr/bin/env python3
"""Rank CNN-LSTM runs in results/experiments.jsonl by VALIDATION metrics.

Model/hyper-parameter selection is done on the *validation* split only. The
*test* (out-of-sample) split is treated as a held-out set and reported exactly
once, for the single winning configuration — never used to rank the sweep.

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


def composite_score(m):
    """Composite selection score from a metrics block (val or test).

    score = (dir_acc - 0.5) * 2   # 0.50 -> 0, 1.00 -> 1 (chance baseline removed)
            + sharpe              # already ~0-1 scale, added directly
            + r2_oos * 100        # R^2 ~0.003, rescaled to be comparable
    """
    return ((m["directional_accuracy_mean"] - 0.5) * 2
            + m["sharpe_mean"]
            + m["r2_oos_mean"] * 100)


def row(r):
    c = r["config"]
    metrics = r["metrics"]
    val = metrics.get("validation")        # may be missing on pre-change runs
    test = metrics["out_of_sample"]
    tr = r["training"]["history"]["train_loss"]
    return {
        "tag": r.get("tag", ""),
        "lr": c["lr"],
        "wd": c["weight_decay"],
        "hidden": c["hidden"],
        "inc": c.get("inception_filters", "-"),
        "tr_stride": c["train_stride"],
        "epochs": r["training"]["epochs_run"],
        "best_val": r["training"]["best_val_loss"],
        "final_train": tr[-1] if tr else float("nan"),
        "has_val": val is not None,
        # ranking metrics: VALIDATION split
        "val_r2": val["r2_oos_mean"] if val else float("nan"),
        "val_dir": val["directional_accuracy_mean"] if val else float("nan"),
        "val_sharpe": val["sharpe_mean"] if val else float("nan"),
        "val_score": composite_score(val) if val else float("nan"),
        # reported once, for the winner only: TEST split
        "test_r2": test["r2_oos_mean"],
        "test_dir": test["directional_accuracy_mean"],
        "test_sharpe": test["sharpe_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    recs = load(args.prefix)
    if not recs:
        raise SystemExit("No matching runs.")

    rows = [row(r) for r in recs]
    rankable = [d for d in rows if d["has_val"]]
    skipped = [d for d in rows if not d["has_val"]]

    if not rankable:
        raise SystemExit(
            "No runs have validation metrics logged. Re-run the sweep with the "
            "updated trainer (src/cnn_lstm.py now logs metrics['validation']) "
            "so the ranking can be based on the validation split.")

    rankable.sort(key=lambda d: d["val_score"], reverse=True)
    shown = rankable[: args.top] if args.top else rankable

    hdr = (f"{'rank':>4} {'tag':<22} {'lr':>6} {'wd':>6} {'hid':>4} {'inc':>4} "
           f"{'strd':>5} {'ep':>3} {'best_val':>9} {'val_r2':>9} {'val_dir':>8} "
           f"{'val_shrp':>9} {'val_score':>9}")
    print("Ranking by VALIDATION composite score (test split NOT used here):\n")
    print(hdr)
    print("-" * len(hdr))
    for i, d in enumerate(shown, 1):
        inc_str = f"{d['inc']:>4}"
        print(f"{i:>4} {d['tag']:<22} {d['lr']:>6.0e} {d['wd']:>6.0e} "
              f"{d['hidden']:>4} {inc_str} {d['tr_stride']:>5} {d['epochs']:>3} "
              f"{d['best_val']:>9.4f} {d['val_r2']:>+9.4f} {d['val_dir']:>8.4f} "
              f"{d['val_sharpe']:>+9.4f} {d['val_score']:>9.4f}")

    if skipped:
        print(f"\n[note] {len(skipped)} run(s) lack validation metrics and were "
              f"excluded from ranking (logged before the val-based change): "
              f"{', '.join(sorted(d['tag'] for d in skipped))}")

    best = rankable[0]
    print("\n" + "=" * 60)
    print("SELECTED by validation composite score:", best["tag"])
    print(f"  lr={best['lr']:g} wd={best['wd']:g} hidden={best['hidden']} "
          f"inc={best['inc']} train_stride={best['tr_stride']}")
    print(f"  VAL : R2={best['val_r2']:+.4f}  DirAcc={best['val_dir']:.4f}  "
          f"Sharpe={best['val_sharpe']:+.4f}")
    print("\nFINAL held-out TEST report (reported once, for the winner only):")
    print(f"  TEST: R2={best['test_r2']:+.4f}  DirAcc={best['test_dir']:.4f}  "
          f"Sharpe={best['test_sharpe']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
