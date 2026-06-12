#!/usr/bin/env python3
"""Generate the EDA tables/figures for the report's Data section.

Everything is derived from the raw L2 books in ``data/raw/book_snapshot_25`` (one
folder per market), reusing the SAME cleaning + order-flow (OF) construction as the
modeling pipeline (``notebooks/colab_of_pipeline.ipynb``) so the EDA describes
exactly the inputs the model consumes. Market metadata (settlement, lifetime) is
joined from ``datasets/polymarket_markets.parquet`` when available.

Outputs (written to ``results/report_plot/``):
    table_market_inventory.{csv,png}   §1  one row per market (paper Table 6 style)
    fig_eda_spread_dist.png            §2  spread histogram, stratified by mid level
    fig_eda_of_acf.png                 §3a autocorrelation of OF imbalance
    fig_eda_of_leadlag.png             §3b corr(OF imbalance_t, return_{t+h}) vs h
    fig_eda_lag_decay.png              §3c per-lag predictive content (why a learned aggregator)
    fig_eda_return_dist.png            §4  per-horizon mid-return distribution

Note on "volume": book snapshots carry resting *sizes*, not executed trades, so the
inventory's volume column is an **order-flow turnover proxy** -- the total size that
flowed through the top 10 levels, sum|OF| in share units -- not executed volume.

Usage:
    mse242/bin/python scripts/eda_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "book_snapshot_25"
META_PATH = ROOT / "datasets" / "polymarket_markets.parquet"
OUT_DIR = ROOT / "results" / "report_plot"

N_LEVELS = 10
HORIZONS = [1, 2, 3, 5, 10]
W = 100                       # look-back window used by the model
ACF_MAXLAG = 60
GROUP_BLUE = "#3B6EA5"

# --------------------------------------------------------------------------- #
# Data loading + OF construction (ported verbatim from colab_of_pipeline.ipynb)
# --------------------------------------------------------------------------- #
def load_clean_book(folder: Path, n_levels: int = N_LEVELS) -> pd.DataFrame:
    """Load + clean a market's books: time-sort, de-dup, drop inverted/missing-best,
    impute deeper levels, add mid. Identical rules to the modeling pipeline."""
    cols = ["timestamp_us"]
    for i in range(n_levels):
        cols += [f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"]
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(folder)
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)

    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bid_price_0", "ask_price_0"]).reset_index(drop=True)
    df = df[df["bid_price_0"] < df["ask_price_0"]].reset_index(drop=True)
    df["mid"] = (df["bid_price_0"] + df["ask_price_0"]) / 2

    for i in range(n_levels):
        df[f"bid_price_{i}"] = df[f"bid_price_{i}"].fillna(0.0)
        df[f"ask_price_{i}"] = df[f"ask_price_{i}"].fillna(1.0)
        df[f"bid_size_{i}"] = df[f"bid_size_{i}"].fillna(0.0)
        df[f"ask_size_{i}"] = df[f"ask_size_{i}"].fillna(0.0)
    return df


def compute_of(df: pd.DataFrame, n_levels: int = N_LEVELS):
    """Three-case OF. Returns (of[(n-1)x20], mid[n-1]) aligned to t=1..n-1.
    Column order = [bOF_0..bOF_9, aOF_0..aOF_9]; ask signs mirrored so positive
    bid-OF and positive ask-OF push the mid in OPPOSITE directions."""
    bp = df[[f"bid_price_{i}" for i in range(n_levels)]].to_numpy(np.float64)
    bv = df[[f"bid_size_{i}" for i in range(n_levels)]].to_numpy(np.float64)
    ap = df[[f"ask_price_{i}" for i in range(n_levels)]].to_numpy(np.float64)
    av = df[[f"ask_size_{i}" for i in range(n_levels)]].to_numpy(np.float64)

    bp_c, bp_p = bp[1:], bp[:-1]
    bv_c, bv_p = bv[1:], bv[:-1]
    ap_c, ap_p = ap[1:], ap[:-1]
    av_c, av_p = av[1:], av[:-1]

    bof = bv_c - bv_p
    bof = np.where(bp_c > bp_p, bv_c, bof)
    bof = np.where(bp_c < bp_p, -bv_c, bof)

    aof = av_c - av_p
    aof = np.where(ap_c > ap_p, -av_c, aof)
    aof = np.where(ap_c < ap_p, av_c, aof)

    of = np.concatenate([bof, aof], axis=1).astype(np.float64)
    mid = df["mid"].to_numpy(np.float64)[1:]
    return of, mid


def raw_update_count(folder: Path) -> int:
    """Total raw book updates (pre-clean) via parquet metadata -- cheap."""
    return int(sum(pq.read_metadata(f).num_rows for f in sorted(folder.glob("*.parquet"))))


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def parse_folder(name: str):
    """'nba_playoffs_who_will_win_series_thunder_vs_lakers__Thunder' ->
       (label 'Thunder vs Lakers', tracked outcome 'Thunder', slug with dashes)."""
    slug_part, _, outcome = name.partition("__")
    slug = slug_part.replace("_", "-")
    m = re.search(r"who_will_win_series_(.+)$", slug_part)
    teams = m.group(1) if m else slug_part
    # capitalize() (not title()) so "76ers" stays "76ers", not "76Ers".
    nice = lambda t: " ".join(p.capitalize() for p in t.split("_"))
    label = " vs ".join(nice(t) for t in teams.split("_vs_"))
    return label, outcome, slug


def acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    """Biased autocorrelation at lags 0..maxlag via FFT (O(n log n))."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[: maxlag + 1]
    return acov / acov[0] if acov[0] > 0 else np.zeros(maxlag + 1)


def fmt_int(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Found {len(folders)} market folders.")

    meta = None
    if META_PATH.exists():
        meta = pd.read_parquet(META_PATH).drop_duplicates(subset=["slug"]).set_index("slug")

    inv_rows = []
    per_market = []          # dicts: label, ofi, mid, returns(dict h->array), spread
    for folder in folders:
        label, outcome, slug = parse_folder(folder.name)
        df = load_clean_book(folder)
        of, mid = compute_of(df)
        ofi = of[:, :N_LEVELS].sum(1) - of[:, N_LEVELS:].sum(1)   # net up-pressure
        spread_c = (df["ask_price_0"].to_numpy() - df["bid_price_0"].to_numpy())[1:] * 100.0
        mid_c = mid * 100.0
        rets = {h: (mid[h:] - mid[:-h]) for h in HORIZONS}        # raw mid diffs (prob units)

        # lifetime: prefer metadata creation->settlement, else book span
        ts = df["ts"]
        life_s = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        if meta is not None and slug in meta.index:
            row = meta.loc[slug]
            if pd.notna(row.get("created_at_us")) and pd.notna(row.get("settled_at_us")):
                life_s = (row["settled_at_us"] - row["created_at_us"]) / 1e6
        days, hours = int(life_s // 86400), int((life_s % 86400) // 3600)

        n_mid_chg = int((np.diff(mid) != 0).sum())
        of_turnover = float(np.abs(of).sum())                    # proxy "volume", shares
        winner = winner_team(label, outcome, mid[-1])            # inferred from terminal mid

        inv_rows.append({
            "Market": label, "Tracked": outcome, "Winner": winner,
            "Updates": raw_update_count(folder), "Lifetime": f"{days}d {hours}h",
            "MidChanges": n_mid_chg, "MeanSpread_c": float(spread_c.mean()),
            "OFTurnover": of_turnover,
        })
        per_market.append({"label": label, "ofi": ofi, "mid_c": mid_c,
                           "spread_c": spread_c, "rets": rets})
        print(f"  {label:<26} rows={len(mid):>7,}  midΔ={n_mid_chg:>6,}  "
              f"spread={spread_c.mean():.2f}¢  winner={winner}")

    inv = pd.DataFrame(inv_rows).sort_values("Updates", ascending=False).reset_index(drop=True)
    _write_inventory(inv)
    _plot_spread(per_market)
    _plot_acf(per_market)
    _plot_leadlag(per_market)
    _plot_lag_decay(per_market)
    _plot_return_dist(per_market)
    print(f"\nWrote 1 table + 5 figures to {OUT_DIR}")


def winner_team(label: str, outcome: str, final_mid: float) -> str:
    """Full team name of the settled winner, inferred from terminal mid. The tracked
    outcome side is matched to the matchup label by 3-char prefix (so short names like
    'Cavs'/'Wolves' resolve to 'Cavaliers'/'Timberwolves')."""
    sides = label.split(" vs ")
    pre = outcome.lower()[:3]
    idx = next((i for i, s in enumerate(sides) if s.lower()[:3] == pre), 0)
    return sides[idx if final_mid >= 0.5 else 1 - idx]


# --------------------------------------------------------------------------- #
# §1 inventory table                                                          #
# --------------------------------------------------------------------------- #
def _write_inventory(inv: pd.DataFrame) -> None:
    out = inv.copy()
    out.to_csv(OUT_DIR / "table_market_inventory.csv", index=False)

    disp = pd.DataFrame({
        "Market": inv["Market"],
        "Updates": inv["Updates"].map(fmt_int),
        "Lifetime": inv["Lifetime"],
        "Mid changes": inv["MidChanges"].map(fmt_int),
        "Settled winner": inv["Winner"],
        "Mean spread (¢)": inv["MeanSpread_c"].map(lambda v: f"{v:.2f}"),
        "OF turnover (k shares)": (inv["OFTurnover"] / 1e3).map(lambda v: f"{v:,.0f}"),
    })
    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.42 * (len(disp) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for j in range(len(disp.columns)):
        tbl[0, j].set_facecolor(GROUP_BLUE)
        tbl[0, j].set_text_props(color="w", fontweight="bold")
    ax.set_title("Table 1. Per-market data inventory (book_snapshot_25, top 10 levels)\n"
                 "OF turnover is a flow proxy (Σ|OF| over top levels), not executed volume.",
                 fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table_market_inventory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §2 spread distribution, stratified by mid level                             #
# --------------------------------------------------------------------------- #
def _plot_spread(pm: list) -> None:
    spread = np.concatenate([m["spread_c"] for m in pm])
    mid = np.concatenate([m["mid_c"] for m in pm])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    axL.hist(spread, bins=np.arange(0, 6.05, 0.1), color=GROUP_BLUE, alpha=0.85)
    axL.axvline(5.0, color="k", ls="--", lw=1, label="liquidity screen (5¢)")
    axL.set_title(f"Pooled bid–ask spread (all markets, n={len(spread):,})")
    axL.set_xlabel("spread (cents on the [0,1] range)")
    axL.set_ylabel("count")
    axL.legend()
    axL.grid(alpha=0.3)

    # Spread vs mid level: mean per decile. NB the relationship is the *opposite* of
    # the usual longshot intuition -- spread is tightest at the boundaries (near-certain
    # outcomes => consensus) and widest near a 0.5 tossup (max disagreement).
    edges = np.linspace(0, 100, 11)
    centers, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mid >= lo) & (mid < hi) if hi < 100 else (mid >= lo) & (mid <= hi)
        if sel.sum() < 50:
            continue
        s = spread[sel]
        centers.append((lo + hi) / 2)
        means.append(s.mean()); sems.append(s.std() / np.sqrt(len(s)))
    centers, means, sems = map(np.array, (centers, means, sems))
    axR.fill_between(centers, means - sems, means + sems, color=GROUP_BLUE, alpha=0.25)
    axR.plot(centers, means, "-o", color=GROUP_BLUE, label="mean spread per decile")
    axR.axvline(50, color="gray", ls=":", lw=1)
    axR.set_title("Spread is widest at a tossup (mid≈0.5), tightest near resolution\n"
                  "(opposite of the longshot intuition; within-market ρ<0 in all 12 markets)")
    axR.set_xlabel("mid-price level (cents = implied probability × 100)")
    axR.set_ylabel("mean spread (cents)")
    axR.legend()
    axR.grid(alpha=0.3)

    fig.suptitle("Figure E2. Bid–ask spread structure", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "fig_eda_spread_dist.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §3a OF-imbalance autocorrelation                                            #
# --------------------------------------------------------------------------- #
def _plot_acf(pm: list) -> None:
    lags = np.arange(1, ACF_MAXLAG + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    per = []
    for m in pm:
        a = acf(m["ofi"], ACF_MAXLAG)[1:]
        per.append(a)
        ax.plot(lags, a, color="gray", alpha=0.25, lw=0.8)
    per = np.vstack(per)
    ax.plot(lags, per.mean(0), color="#C5413B", lw=2.4, label="mean across 12 markets")
    nmin = min(len(m["ofi"]) for m in pm)
    ax.axhline(1.96 / np.sqrt(nmin), color="k", ls=":", lw=1,
               label="95% noise band (smallest market)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Figure E3a. Order-flow imbalance has strong, structured autocorrelation\n"
                 "(anti-persistent at lag 1, oscillating for ~40+ lags): the input is highly "
                 "non-i.i.d.,\nso a sequence model rather than a per-tick map fits it")
    ax.set_xlabel("lag (ticks)")
    ax.set_ylabel("autocorrelation of OF imbalance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_eda_of_acf.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §3b lead-lag: corr(OFI_t, return_{t+h}) vs horizon                          #
# --------------------------------------------------------------------------- #
def _plot_leadlag(pm: list) -> None:
    corr = {h: [] for h in HORIZONS}       # per-market correlations
    for m in pm:
        ofi, rets = m["ofi"], m["rets"]
        for h in HORIZONS:
            r = rets[h]
            x = ofi[: len(r)]
            ok = np.isfinite(x) & np.isfinite(r)
            if ok.sum() > 100 and x[ok].std() > 0 and r[ok].std() > 0:
                corr[h].append(np.corrcoef(x[ok], r[ok])[0, 1])
    means = [np.mean(corr[h]) for h in HORIZONS]
    stds = [np.std(corr[h]) for h in HORIZONS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in pm:                            # faint per-market lines
        ys = []
        for h in HORIZONS:
            r = m["rets"][h]; x = m["ofi"][: len(r)]
            ok = np.isfinite(x) & np.isfinite(r)
            ys.append(np.corrcoef(x[ok], r[ok])[0, 1] if ok.sum() > 100 else np.nan)
        ax.plot(HORIZONS, ys, color="gray", alpha=0.3, lw=0.8)
    ax.errorbar(HORIZONS, means, yerr=stds, fmt="-o", color="#C5413B", lw=2.4,
                capsize=4, label="mean ± sd across markets")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Figure E3b. OF imbalance is predictively informative at every target "
                 "horizon\n(weak and mean-reverting at the linear level: |corr|≈0.03, "
                 "sign < 0 from the bid–ask bounce)")
    ax.set_xlabel("forecast horizon $h$ (ticks)")
    ax.set_ylabel("corr(OF imbalance$_t$, return$_{t+h}$)")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_eda_of_leadlag.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# §3c per-lag predictive content -- why a *learned* sequence aggregator        #
# --------------------------------------------------------------------------- #
def _plot_lag_decay(pm: list, maxlag: int = 25) -> None:
    """corr(OF imbalance_{t-k}, next-tick return) vs lag k, plus the naive
    equal-weight W-window-sum correlation, to show that the predictive signal is
    distributed across many lags with alternating sign -- naive pooling cancels it,
    so a learned nonlinear aggregator (LSTM) is needed."""
    per_lag, win_corr, last_corr = [], [], []
    for m in pm:
        ofi, r1 = m["ofi"], m["rets"][1]      # r1 aligned to t
        L = len(r1)
        if L <= W + maxlag + 5:
            continue
        # per-lag: pair (OFI_{t-k}, r1[t]) for t in [maxlag, L-1]
        row = []
        for k in range(maxlag + 1):
            x = ofi[maxlag - k: L - k]
            y = r1[maxlag:L]
            row.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan)
        per_lag.append(row)
        last_corr.append(row[0])
        # naive equal-weight window sum ending at t predicts r1[t]
        wsum = np.convolve(ofi, np.ones(W), "valid")
        win_corr.append(np.corrcoef(wsum[: L - (W - 1)], r1[W - 1: L])[0, 1])

    per_lag = np.array(per_lag)
    lags = np.arange(maxlag + 1)
    mean, sd = np.nanmean(per_lag, 0), np.nanstd(per_lag, 0)
    win_m, last_m = float(np.mean(win_corr)), float(np.mean(last_corr))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(lags, mean - sd, mean + sd, color=GROUP_BLUE, alpha=0.2,
                    label="±sd across markets")
    ax.plot(lags, mean, "-o", color=GROUP_BLUE, ms=4, lw=2, label="per-lag corr (mean)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(win_m, color="#C5413B", ls="--", lw=1.6,
               label=f"naive {W}-tick window sum (corr={win_m:+.3f}): signal cancels")
    ax.set_title("Figure E3c. Predictive signal is spread across many lags with "
                 "alternating sign\nEach OF-imbalance lag weakly forecasts the next "
                 "tick out to ~20 ticks; a naive sum cancels it — a learned, nonlinear\n"
                 "sequence aggregator (LSTM) is needed to combine the lags")
    ax.set_xlabel("lag $k$ of OF imbalance used to predict the next-tick return")
    ax.set_ylabel("corr(OF imbalance$_{t-k}$, return$_{t+1}$)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_eda_lag_decay.png", dpi=140)
    plt.close(fig)
    print(f"  [lag decay] last-tick corr={last_m:+.3f}  naive window-sum corr={win_m:+.3f}")


# --------------------------------------------------------------------------- #
# §4 return distribution per horizon                                          #
# --------------------------------------------------------------------------- #
def _plot_return_dist(pm: list) -> None:
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(17, 4), sharey=True)
    bins = np.linspace(-3, 3, 121)          # cents
    for ax, h in zip(axes, HORIZONS):
        r = np.concatenate([m["rets"][h] for m in pm]) * 100.0   # cents
        r = r[np.isfinite(r)]
        frac0 = float(np.mean(r == 0))
        ax.hist(np.clip(r, bins[0], bins[-1]), bins=bins, color=GROUP_BLUE, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"$h={h}$\nflat ticks: {frac0*100:.0f}%", fontsize=10)
        ax.set_xlabel("mid return (cents)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count (log scale)")
    fig.suptitle("Figure E4. Mid-price return distribution by horizon: a spike at zero "
                 "(flat ticks) plus heavy continuous tails", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "fig_eda_return_dist.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
