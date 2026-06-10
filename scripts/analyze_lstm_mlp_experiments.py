#!/usr/bin/env python3
"""Rank LSTM-MLP runs in results/experiments.jsonl for MODEL SELECTION.

Model selection is done on the VALIDATION split; the test (out_of_sample) split
is shown for reporting only. Ranking on test would leak it -- the headline test
number stops being an unbiased estimate of generalization the moment you pick a
config because it scored well there. So the composite score and the sort order
use validation metrics; the oos_* columns are there to read off the final number
for whichever config validation already chose.

Selects only LSTM-MLP records and surfaces both the trunk levers (lr, weight
decay, hidden, num_layers, dropout, train_stride) and the MLP-head levers
(mlp_hidden, mlp_layers, mlp_activation), plus ``ftrain`` (final train_loss) to
spot under-training (value near 1.0 == still predicting the mean).

Requires runs produced after the val-metrics patch to lstm_mlp.py (records with
``metrics.validation``). Older runs lack it and are listed separately at the
bottom -- re-run them to rank on validation.

Usage:
    python scripts/analyze_lstm_mlp_experiments.py                 # all LSTM-MLP runs
    python scripts/analyze_lstm_mlp_experiments.py --prefix ml1-   # one sweep stage
    python scripts/analyze_lstm_mlp_experiments.py --top 5
    python scripts/analyze_lstm_mlp_experiments.py --prefix ml3- --group
        # aggregate across seeds -> per-config mean+-std (for the stage-3 check)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "experiments.jsonl"


def is_lstm_mlp(rec) -> bool:
    """LSTM-MLP records only (model string starts 'LSTM-MLP (')."""
    return str(rec.get("model", "")).startswith("LSTM-MLP (")


def load(prefix=None):
    recs = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not is_lstm_mlp(r):
            continue
        if prefix and not str(r.get("tag", "")).startswith(prefix):
            continue
        recs.append(r)
    return recs


def composite(m) -> float:
    """Composite score from a metrics block (validation OR out_of_sample).
    Same blend as the sibling analyzers: directional edge over 0.5 (x2), plus
    Sharpe, plus R2 scaled up (it lives near ~1e-3)."""
    return ((m["directional_accuracy_mean"] - 0.5) * 2
            + m["sharpe_mean"]
            + m["r2_oos_mean"] * 100)


def row(r):
    c = r["config"]
    val = r["metrics"].get("validation")   # None for pre-patch runs
    oos = r["metrics"]["out_of_sample"]
    tr = r["training"]["history"]["train_loss"]
    return {
        "tag": r.get("tag", ""),
        "lr": c["lr"],
        "wd": c["weight_decay"],
        "hidden": c["hidden"],
        "nl": c.get("num_layers", 1),
        "drop": c.get("dropout", 0.0),
        "mlp_h": c.get("mlp_hidden", 0),
        "mlp_l": c.get("mlp_layers", 0),
        "mlp_act": c.get("mlp_activation", "relu"),
        "tr_stride": c["train_stride"],
        "seed": c.get("seed", 0),
        "final_train": tr[-1] if tr else float("nan"),
        "has_val": val is not None,
        # selection metrics (validation) -- NaN if this run predates the patch
        "val_r2": val["r2_oos_mean"] if val else float("nan"),
        "val_dir": val["directional_accuracy_mean"] if val else float("nan"),
        "val_sharpe": val["sharpe_mean"] if val else float("nan"),
        # report-only metrics (test)
        "oos_r2": oos["r2_oos_mean"],
        "oos_dir": oos["directional_accuracy_mean"],
        "oos_sharpe": oos["sharpe_mean"],
        # rank key: validation composite (None-val -> -inf, sorts last)
        "score": composite(val) if val else float("-inf"),
    }


# Hyper-params that define a "config" (everything except the seed).
_CONFIG_KEYS = ("lr", "wd", "hidden", "nl", "drop", "mlp_h", "mlp_l",
                "mlp_act", "tr_stride")

_HDR = (f"{'rank':>4} {'tag':<24} {'lr':>6} {'wd':>6} {'hid':>4} {'nl':>3} "
        f"{'drop':>5} {'head':>7} {'act':>5} {'strd':>5} {'sd':>3} {'ftrain':>8} "
        f"| {'val_r2':>9} {'val_dir':>8} {'val_shrp':>9} "
        f"| {'oos_r2':>9} {'oos_dir':>8} {'oos_shrp':>9} {'score':>8}")


def _head(d):
    return f"{d['mlp_l']}x{d['mlp_h']}"


def print_runs(rows):
    ranked = [d for d in rows if d["has_val"]]
    unranked = [d for d in rows if not d["has_val"]]

    print(_HDR)
    print("-" * len(_HDR))
    for i, d in enumerate(ranked, 1):
        print(f"{i:>4} {d['tag']:<24} {d['lr']:>6.0e} {d['wd']:>6.0e} "
              f"{d['hidden']:>4} {d['nl']:>3} {d['drop']:>5.2f} {_head(d):>7} "
              f"{d['mlp_act']:>5} {d['tr_stride']:>5} {d['seed']:>3} "
              f"{d['final_train']:>8.4f} "
              f"| {d['val_r2']:>+9.4f} {d['val_dir']:>8.4f} {d['val_sharpe']:>+9.4f} "
              f"| {d['oos_r2']:>+9.4f} {d['oos_dir']:>8.4f} {d['oos_sharpe']:>+9.4f} "
              f"{d['score']:>8.4f}")

    if unranked:
        print(f"\n[!] {len(unranked)} run(s) have no validation metrics "
              f"(pre-patch) and are NOT ranked -- re-run to select on validation:")
        for d in unranked:
            print(f"      {d['tag']:<24} (lr={d['lr']:g} strd={d['tr_stride']} "
                  f"oos_r2={d['oos_r2']:+.4f}, report-only)")
    return ranked


def _mean_std(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    if len(vals) == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


def print_grouped(rows):
    """Aggregate runs sharing hyper-params (ignoring seed) -> mean+-std. Selection
    is on validation; test means are shown for the final report. Useful for the
    stage-3 robustness check across seeds."""
    rows = [d for d in rows if d["has_val"]]
    if not rows:
        raise SystemExit("No runs with validation metrics to group. Re-run the "
                         "sweeps after the val-metrics patch.")
    groups = {}
    for d in rows:
        groups.setdefault(tuple(d[k] for k in _CONFIG_KEYS), []).append(d)

    agg = []
    for key, members in groups.items():
        vr2 = _mean_std([m["val_r2"] for m in members])
        vdir = _mean_std([m["val_dir"] for m in members])
        vsh = _mean_std([m["val_sharpe"] for m in members])
        or2 = _mean_std([m["oos_r2"] for m in members])
        odir = _mean_std([m["oos_dir"] for m in members])
        osh = _mean_std([m["oos_sharpe"] for m in members])
        agg.append({
            "key": dict(zip(_CONFIG_KEYS, key)),
            "n": len(members),
            "vr2": vr2, "vdir": vdir, "vsh": vsh,
            "or2": or2, "odir": odir, "osh": osh,
            "score": sum(m["score"] for m in members) / len(members),
        })
    agg.sort(key=lambda a: a["score"], reverse=True)

    hdr = (f"{'rank':>4} {'lr':>6} {'wd':>6} {'hid':>4} {'nl':>3} {'drop':>5} "
           f"{'head':>7} {'act':>5} {'strd':>5} {'n':>2} "
           f"| {'val_r2 (mu+-sd)':>20} {'val_dir (mu+-sd)':>20} "
           f"| {'oos_r2 (mu+-sd)':>20} {'oos_dir (mu+-sd)':>20} {'score':>8}")
    print("VALIDATION-selected (test columns are report-only)\n")
    print(hdr)
    print("-" * len(hdr))
    for i, a in enumerate(agg, 1):
        k = a["key"]
        head = f"{k['mlp_l']}x{k['mlp_h']}"
        print(f"{i:>4} {k['lr']:>6.0e} {k['wd']:>6.0e} {k['hidden']:>4} "
              f"{k['nl']:>3} {k['drop']:>5.2f} {head:>7} {k['mlp_act']:>5} "
              f"{k['tr_stride']:>5} {a['n']:>2} "
              f"| {a['vr2'][0]:>+9.4f}+-{a['vr2'][1]:<8.4f} "
              f"{a['vdir'][0]:>9.4f}+-{a['vdir'][1]:<8.4f} "
              f"| {a['or2'][0]:>+9.4f}+-{a['or2'][1]:<8.4f} "
              f"{a['odir'][0]:>9.4f}+-{a['odir'][1]:<8.4f} {a['score']:>8.4f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="only tags starting with this")
    ap.add_argument("--top", type=int, default=0, help="show only the top N")
    ap.add_argument("--group", action="store_true",
                    help="aggregate across seeds -> per-config mean+-std")
    args = ap.parse_args()

    recs = load(args.prefix)
    if not recs:
        raise SystemExit("No matching LSTM-MLP runs in results/experiments.jsonl.")

    rows = sorted((row(r) for r in recs), key=lambda d: d["score"], reverse=True)

    if args.group:
        agg = print_grouped(rows)
        best = agg[0]["key"]
        print(f"\nBEST config by VALIDATION (seed-averaged): lr={best['lr']:g} "
              f"wd={best['wd']:g} hidden={best['hidden']} num_layers={best['nl']} "
              f"dropout={best['drop']:g} mlp={best['mlp_l']}x{best['mlp_h']} "
              f"act={best['mlp_act']} train_stride={best['tr_stride']}")
        print(f"  -> reported test: R2={agg[0]['or2'][0]:+.4f}  "
              f"DirAcc={agg[0]['odir'][0]:.4f}  Sharpe={agg[0]['osh'][0]:+.4f}")
        return

    if args.top:
        rows = rows[: args.top]
    ranked = print_runs(rows)

    if ranked:
        best = ranked[0]
        print("\nBEST by VALIDATION composite:", best["tag"])
        print(f"  lr={best['lr']:g} wd={best['wd']:g} hidden={best['hidden']} "
              f"num_layers={best['nl']} dropout={best['drop']:g} "
              f"mlp={best['mlp_l']}x{best['mlp_h']} act={best['mlp_act']} "
              f"train_stride={best['tr_stride']} (seed {best['seed']})")
        print(f"  VAL : R2={best['val_r2']:+.4f}  DirAcc={best['val_dir']:.4f}  "
              f"Sharpe={best['val_sharpe']:+.4f}  | final_train={best['final_train']:.4f}")
        print(f"  TEST (report-only): R2={best['oos_r2']:+.4f}  "
              f"DirAcc={best['oos_dir']:.4f}  Sharpe={best['oos_sharpe']:+.4f}")


if __name__ == "__main__":
    main()
