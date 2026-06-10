#!/usr/bin/env python3
"""Compare CNN-LSTM test performance across the 3 market-variability groups.

Pipeline (run these in order):
    1. python scripts/group_markets.py              # -> results/market_groups.json
    2. bash   scripts/eval_per_market.sh [seeds]    # -> per-market metrics (Modal)
    3. python scripts/analyze_market_groups.py      # this script

What this does
--------------
Joins the per-market TEST metrics produced by the per-market eval run(s) with the
Low/Mid/High variability grouping, then answers the question:

    H0: model test performance is independent of price-curve variability.
    H1: model test performance depends on variability (better/worse on
        high- vs low-volatility markets).

It reports the comparison two ways:
  - **Group aggregates** (Low/Mid/High mean of each metric) — for presentation.
  - **Spearman rank correlation** between each market's realized vol and each
    metric across all 12 markets — the actual test (uses all 12 points, more
    powerful than a 3-group ANOVA at n=4/group). Reported vs both the full-series
    grouping vol and the test-segment vol.

Outputs:
    results/viz_09_group_comparison.png
    results/MARKET_GROUP_ANALYSIS.md

Usage:
    python scripts/analyze_market_groups.py
    python scripts/analyze_market_groups.py --tag-prefix permarket
    python scripts/analyze_market_groups.py --run results/runs/cnn_lstm_XXXX.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOG = RESULTS_DIR / "experiments.jsonl"
GROUPS_JSON = RESULTS_DIR / "market_groups.json"

GROUP_NAMES = ["Low", "Mid", "High"]
GROUP_COLORS = {"Low": "#4C9F70", "Mid": "#E1A33A", "High": "#C5413B"}
METRICS = [
    ("r2_oos_mean", "R\u00b2_OS"),
    ("directional_accuracy_mean", "Directional accuracy"),
    ("sharpe_mean", "Sharpe (annualized)"),
]


def short_key(key: str) -> str:
    return key.split("__")[0]


def load_groups() -> dict:
    if not GROUPS_JSON.exists():
        raise SystemExit(f"Missing {GROUPS_JSON}. Run scripts/group_markets.py first.")
    payload = json.loads(GROUPS_JSON.read_text())
    return {m["key"]: m for m in payload["markets"]}


def load_per_market_records(run_path: str | None, tag_prefix: str | None) -> list[dict]:
    """Return the list of run records that carry a metrics['per_market'] block."""
    if run_path:
        rec = json.loads(Path(run_path).read_text())
        if not rec.get("metrics", {}).get("per_market"):
            raise SystemExit(f"{run_path} has no metrics.per_market block.")
        return [rec]
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}. Run scripts/eval_per_market.sh first.")
    recs = []
    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("metrics", {}).get("per_market"):
            continue
        if tag_prefix and not str(rec.get("tag", "")).startswith(tag_prefix):
            continue
        recs.append(rec)
    if not recs:
        raise SystemExit(
            "No runs with a per_market block found in experiments.jsonl.\n"
            "Run `bash scripts/eval_per_market.sh` (it passes --per-market-eval)."
        )
    return recs


def aggregate_over_seeds(recs: list[dict]) -> dict:
    """Average each per-market metric across the supplied run records (seeds)."""
    collected = defaultdict(lambda: defaultdict(list))
    n_win = {}
    vol_test = {}
    seeds = []
    for rec in recs:
        seeds.append(rec.get("config", {}).get("seed"))
        for key, pm in rec["metrics"]["per_market"].items():
            for mk, _ in METRICS:
                collected[key][mk].append(pm[mk])
            n_win[key] = pm["n_windows"]
            vol_test[key] = pm.get("realized_vol_test")
    out = {}
    for key, md in collected.items():
        out[key] = {
            "n_windows": n_win[key],
            "realized_vol_test": vol_test[key],
            "n_seeds": len(recs),
        }
        for mk, _ in METRICS:
            vals = np.array(md[mk], float)
            out[key][mk] = float(np.mean(vals))
            out[key][mk + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return out, seeds


def build_table(groups: dict, agg: dict) -> list[dict]:
    """One row per market: group + grouping vol + averaged metrics."""
    rows = []
    for key, g in groups.items():
        if key not in agg:
            continue
        r = {
            "key": key,
            "group": g["group"],
            "realized_vol": g["realized_vol"],          # full-series (grouping)
            "realized_vol_test": agg[key]["realized_vol_test"],
            "n_windows": agg[key]["n_windows"],
            "n_seeds": agg[key]["n_seeds"],
        }
        for mk, _ in METRICS:
            r[mk] = agg[key][mk]
            r[mk + "_std"] = agg[key][mk + "_std"]
        rows.append(r)
    rows.sort(key=lambda r: r["realized_vol"])
    return rows


def group_means(rows: list[dict]) -> dict:
    """Per-group simple + window-weighted mean of each metric."""
    out = {}
    for g in GROUP_NAMES:
        members = [r for r in rows if r["group"] == g]
        if not members:
            continue
        w = np.array([r["n_windows"] for r in members], float)
        gm = {"n_markets": len(members), "n_windows": int(w.sum())}
        for mk, _ in METRICS:
            v = np.array([r[mk] for r in members], float)
            gm[mk] = float(v.mean())
            gm[mk + "_wmean"] = float(np.average(v, weights=w))
            gm[mk + "_std"] = float(v.std())
        out[g] = gm
    return out


def spearman_tests(rows: list[dict]) -> dict:
    """Spearman rho + p of each metric vs vol (full-series and test-segment)."""
    res = {}
    for vol_field in ("realized_vol", "realized_vol_test"):
        x = np.array([r[vol_field] for r in rows], float)
        res[vol_field] = {}
        for mk, _ in METRICS:
            y = np.array([r[mk] for r in rows], float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                rho, p = stats.spearmanr(x[ok], y[ok])
            else:
                rho, p = float("nan"), float("nan")
            res[vol_field][mk] = {"rho": float(rho), "p": float(p), "n": int(ok.sum())}
    return res


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #
def make_plot(rows, gmeans, spear, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: group-mean bars per metric.
    for ax, (mk, label) in zip(axes[0], METRICS):
        present = [g for g in GROUP_NAMES if g in gmeans]
        means = [gmeans[g][mk] for g in present]
        errs = [gmeans[g][mk + "_std"] for g in present]
        colors = [GROUP_COLORS[g] for g in present]
        ax.bar(present, means, yerr=errs, capsize=5, color=colors, alpha=0.9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{label} — group mean")
        ax.set_xlabel("variability group")
        ax.grid(axis="y", alpha=0.3)

    # Row 2: scatter metric vs full-series realized vol, colored by group.
    for ax, (mk, label) in zip(axes[1], METRICS):
        for g in GROUP_NAMES:
            members = [r for r in rows if r["group"] == g]
            if not members:
                continue
            ax.scatter([r["realized_vol"] for r in members],
                       [r[mk] for r in members],
                       color=GROUP_COLORS[g], label=g, s=55, edgecolor="k", lw=0.4)
        st = spear["realized_vol"][mk]
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(f"{label} vs realized vol\nSpearman \u03c1={st['rho']:.2f}, p={st['p']:.3f}")
        ax.set_xlabel("realized vol (full series)")
        ax.grid(alpha=0.3)
    axes[1][0].legend(title="Group", fontsize=8)

    fig.suptitle("CNN-LSTM test performance vs market price-curve variability",
                 fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def verdict(rho: float, p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < alpha:
        direction = "higher" if rho > 0 else "lower"
        return f"**reject H0** (p={p:.3f}): performance is {direction} on more-volatile markets"
    return f"cannot reject H0 (p={p:.3f}): no significant dependence"


def write_report(rows, gmeans, spear, seeds, out_md: Path) -> None:
    lines = []
    lines.append("# Market-group analysis — does CNN-LSTM performance depend on price-curve variability?\n")
    seeds_str = ", ".join(str(s) for s in seeds)
    lines.append(f"**Seeds averaged:** {len(seeds)} ({seeds_str})  ·  "
                 f"**Markets:** {len(rows)}  ·  "
                 f"**Grouping metric:** realized vol = std(diff(mid)), full series\n")
    lines.append("Groups are equal-size tertiles (Low/Mid/High) of full-series "
                 "realized volatility (see `scripts/group_markets.py`). The single "
                 "all-markets CNN-LSTM checkpoint is evaluated on each market's test "
                 "split with global standardization + global naive baseline.\n")

    # Per-market table.
    lines.append("## Per-market test metrics\n")
    lines.append("| Group | Market | realized_vol | vol_test | n_win | R\u00b2_OS | DirAcc | Sharpe |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['group']} | {short_key(r['key'])} | {r['realized_vol']:.6f} | "
            f"{r['realized_vol_test']:.6f} | {r['n_windows']:,} | "
            f"{r['r2_oos_mean']:.4f} | {r['directional_accuracy_mean']:.4f} | "
            f"{r['sharpe_mean']:.4f} |")
    lines.append("")

    # Group aggregates.
    lines.append("## Group aggregates (mean across markets)\n")
    lines.append("| Group | n_mkts | R\u00b2_OS | DirAcc | Sharpe | R\u00b2_OS (w) | DirAcc (w) | Sharpe (w) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for g in GROUP_NAMES:
        if g not in gmeans:
            continue
        m = gmeans[g]
        lines.append(
            f"| {g} | {m['n_markets']} | {m['r2_oos_mean']:.4f} | "
            f"{m['directional_accuracy_mean']:.4f} | {m['sharpe_mean']:.4f} | "
            f"{m['r2_oos_mean_wmean']:.4f} | {m['directional_accuracy_mean_wmean']:.4f} | "
            f"{m['sharpe_mean_wmean']:.4f} |")
    lines.append("\n*(w) = window-weighted mean.*\n")

    # Spearman test.
    lines.append("## Hypothesis test — Spearman rank correlation (vol vs metric, n=%d)\n" % len(rows))
    for vol_field, title in (("realized_vol", "full-series realized vol"),
                             ("realized_vol_test", "test-segment realized vol")):
        lines.append(f"### vs {title}\n")
        lines.append("| Metric | Spearman \u03c1 | p-value | verdict |")
        lines.append("|---|---:|---:|---|")
        for mk, label in METRICS:
            st = spear[vol_field][mk]
            lines.append(f"| {label} | {st['rho']:.3f} | {st['p']:.3f} | "
                         f"{verdict(st['rho'], st['p'])} |")
        lines.append("")

    # Auto conclusion (checks BOTH the grouping vol and the test-segment vol).
    lines.append("## Bottom line\n")
    any_sig = False
    for vol_field, title in (("realized_vol", "full-series vol"),
                             ("realized_vol_test", "test-segment vol")):
        for mk, label in METRICS:
            st = spear[vol_field][mk]
            if np.isfinite(st["p"]) and st["p"] < 0.05:
                any_sig = True
                direction = "higher" if st["rho"] > 0 else "lower"
                lines.append(f"- **{label}** (vs {title}): significantly {direction} on "
                             f"more-volatile markets (\u03c1={st['rho']:.2f}, p={st['p']:.3f}).")
    if not any_sig:
        lines.append("- No metric shows a statistically significant (p<0.05) monotone "
                     "dependence on price-curve variability across the 12 markets. "
                     "At this sample size (n=12) we **cannot reject H0**: the model's "
                     "test performance looks broadly comparable across Low/Mid/High "
                     "volatility groups. Treat group-mean differences as exploratory.")
    else:
        lines.append("- Metrics not listed above show no significant (p<0.05) dependence "
                     "on variability; with n=12 treat all group-mean differences as "
                     "exploratory.")
    lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="explicit run JSON (else scan experiments.jsonl)")
    ap.add_argument("--tag-prefix", default=None,
                    help="only use experiments.jsonl runs whose tag starts with this")
    args = ap.parse_args()

    groups = load_groups()
    recs = load_per_market_records(args.run, args.tag_prefix)
    agg, seeds = aggregate_over_seeds(recs)
    rows = build_table(groups, agg)
    if not rows:
        raise SystemExit("No overlap between grouped markets and per-market metrics.")
    gmeans = group_means(rows)
    spear = spearman_tests(rows)

    out_png = RESULTS_DIR / "viz_09_group_comparison.png"
    out_md = RESULTS_DIR / "MARKET_GROUP_ANALYSIS.md"
    make_plot(rows, gmeans, spear, out_png)
    write_report(rows, gmeans, spear, seeds, out_md)

    # Console summary.
    print(f"Seeds used: {seeds}")
    print(f"\n{'group':<6}{'R2_OS':>10}{'DirAcc':>10}{'Sharpe':>10}  (group mean)")
    for g in GROUP_NAMES:
        if g not in gmeans:
            continue
        m = gmeans[g]
        print(f"{g:<6}{m['r2_oos_mean']:>10.4f}{m['directional_accuracy_mean']:>10.4f}"
              f"{m['sharpe_mean']:>10.4f}")
    print("\nSpearman (full-series vol vs metric):")
    for mk, label in METRICS:
        st = spear["realized_vol"][mk]
        print(f"  {label:<22} rho={st['rho']:+.3f}  p={st['p']:.3f}")
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
