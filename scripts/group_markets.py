#!/usr/bin/env python3
"""Group the 12 NBA markets into 3 variability tiers by realized volatility.

Hypothesis context
------------------
The mid-price curves in `results/viz_02_midprice_series.png` look very different
across markets (some sit flat near an extreme, some trend, some swing wildly).
We want to *test* whether the CNN-LSTM's test-set performance depends on how
much a market's price curve actually moves. To do that we first need an
objective, reproducible grouping of the markets by "how much the curve varies".

Variability metric
-------------------
We use **realized volatility = std of the per-tick mid-price change**
(`std(diff(mid))`). The processed targets are absolute mid changes
(`mid[t+k]-mid[t]`), so this is exactly the volatility of what the model
forecasts, in the model's own units. Because `mid` is a 0-1 probability, the
metric is directly comparable across markets without rescaling.

We rank the 12 markets by full-series realized vol and split into 3 equal
tertiles (4 markets each): Low / Mid / High variability.

Output
------
- `results/market_groups.json`  : per-market vol + group assignment
- `results/viz_08_market_groups.png` : ranked-vol bar chart + grouped curves

Run:
    python scripts/group_markets.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "of"
RESULTS_DIR = ROOT / "results"

GROUP_NAMES = ["Low", "Mid", "High"]
GROUP_COLORS = {"Low": "#4C9F70", "Mid": "#E1A33A", "High": "#C5413B"}


def realized_vol(mid: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Std of per-tick mid-price changes (realized volatility) over `mask`."""
    diffs = np.diff(mid.astype(np.float64))
    if mask is not None:
        # diff[i] sits between tick i and i+1; attribute it to the ending tick.
        diffs = diffs[mask[1:]]
    diffs = diffs[np.isfinite(diffs)]
    return float(np.std(diffs)) if diffs.size else float("nan")


def load_config() -> dict:
    cfg_path = DATA_DIR / "_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}. Run the OF pipeline notebook first.")
    return json.loads(cfg_path.read_text())


def compute_market_stats(entries: list[dict]) -> list[dict]:
    stats = []
    for e in entries:
        npz = np.load(DATA_DIR / e["file"], allow_pickle=True)
        mid = npz["mid"].astype(np.float64)
        split = npz["split"].astype(str)
        stats.append({
            "key": e["key"],
            "file": e["file"],
            "n": int(len(mid)),
            "realized_vol": realized_vol(mid),                       # full series (grouping metric)
            "realized_vol_train": realized_vol(mid, split == "train"),
            "realized_vol_test": realized_vol(mid, split == "test"),
            "mid_min": float(np.nanmin(mid)),
            "mid_max": float(np.nanmax(mid)),
            "mid_range": float(np.nanmax(mid) - np.nanmin(mid)),
            "_mid": mid,            # kept only for plotting; stripped before JSON
            "_split": split,
        })
    return stats


def assign_groups(stats: list[dict], n_groups: int = 3) -> list[dict]:
    """Rank by realized_vol and split into `n_groups` equal-size tertiles."""
    order = sorted(stats, key=lambda s: s["realized_vol"])
    n = len(order)
    # Equal-size buckets; remainder spills into the higher tiers.
    base = n // n_groups
    sizes = [base] * n_groups
    for i in range(n - base * n_groups):
        sizes[-(i + 1)] += 1
    labels = []
    for gi, sz in enumerate(sizes):
        labels += [GROUP_NAMES[gi]] * sz
    for rank, (s, lab) in enumerate(zip(order, labels)):
        s["vol_rank"] = rank
        s["group"] = lab
    return order


def short_key(key: str) -> str:
    return key.split("__")[0]


def plot_groups(stats_sorted: list[dict], out_path: Path) -> None:
    fig, (ax_bar, ax_curves) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.15]})

    # --- left: ranked realized-vol bars colored by group ---
    labels = [short_key(s["key"]) for s in stats_sorted]
    vols = [s["realized_vol"] for s in stats_sorted]
    colors = [GROUP_COLORS[s["group"]] for s in stats_sorted]
    ypos = np.arange(len(stats_sorted))
    ax_bar.barh(ypos, vols, color=colors)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Realized volatility  =  std(diff(mid))  [full series]")
    ax_bar.set_title("Markets ranked by price-curve variability")
    ax_bar.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    bar_handles = [Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_NAMES]
    ax_bar.legend(handles=bar_handles, title="Group", loc="lower right")

    # --- right: mid-price curves, colored by group (normalized x to [0,1]) ---
    for s in stats_sorted:
        mid = s["_mid"]
        x = np.linspace(0, 1, len(mid))
        ax_curves.plot(x, mid, color=GROUP_COLORS[s["group"]], alpha=0.55, lw=0.8)
    ax_curves.set_xlabel("normalized time (per market)")
    ax_curves.set_ylabel("mid-price")
    ax_curves.set_title("Mid-price curves colored by variability group")
    handles = [plt.Line2D([], [], color=GROUP_COLORS[g], label=g) for g in GROUP_NAMES]
    ax_curves.legend(handles=handles, title="Group", loc="upper left")

    fig.suptitle("Market grouping by realized volatility (3 tiers, 4 markets each)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-groups", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    stats = compute_market_stats(cfg["markets"])
    stats_sorted = assign_groups(stats, n_groups=args.n_groups)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "market_groups.json"
    serializable = [{k: v for k, v in s.items() if not k.startswith("_")}
                    for s in stats_sorted]
    payload = {
        "metric": "realized_vol = std(diff(mid)) over full series",
        "n_groups": args.n_groups,
        "group_names": GROUP_NAMES[:args.n_groups],
        "markets": serializable,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    out_png = RESULTS_DIR / "viz_08_market_groups.png"
    plot_groups(stats_sorted, out_png)

    # --- console summary ---
    print(f"{'group':<6} {'realized_vol':>13} {'vol_test':>10} {'n':>9}  market")
    print("-" * 72)
    for s in stats_sorted:
        print(f"{s['group']:<6} {s['realized_vol']:>13.6f} "
              f"{s['realized_vol_test']:>10.6f} {s['n']:>9,}  {short_key(s['key'])}")
    print("-" * 72)
    for g in GROUP_NAMES[:args.n_groups]:
        members = [s for s in stats_sorted if s["group"] == g]
        mvol = np.mean([s["realized_vol"] for s in members])
        print(f"  {g:<5} group: {len(members)} markets, mean realized_vol={mvol:.6f}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
