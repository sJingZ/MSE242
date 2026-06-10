# Market-group analysis — does CNN-LSTM performance depend on price-curve variability?

**Seeds averaged:** 1 (42)  ·  **Markets:** 12  ·  **Grouping metric:** realized vol = std(diff(mid)), full series

Groups are equal-size tertiles (Low/Mid/High) of full-series realized volatility (see `scripts/group_markets.py`). The single all-markets CNN-LSTM checkpoint is evaluated on each market's test split with global standardization + global naive baseline.

## Per-market test metrics

| Group | Market | realized_vol | vol_test | n_win | R²_OS | DirAcc | Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | Suns vs Thunder | 0.001909 | 0.000498 | 583 | -0.0848 | 0.5814 | 0.3000 |
| Low | Lakers vs Rockets | 0.002125 | 0.000894 | 5,656 | -0.1754 | 0.5892 | 0.7438 |
| Low | Thunder vs Lakers | 0.002185 | 0.000332 | 680 | -0.9249 | 0.6580 | 3.0933 |
| Low | Spurs vs Timberwolves | 0.002287 | 0.001288 | 1,697 | 0.0185 | 0.6227 | 0.9805 |
| Mid | Timberwolves vs Nuggets | 0.002330 | 0.003103 | 4,947 | 0.0425 | 0.6947 | 1.6064 |
| Mid | 76ers vs Celtics | 0.002401 | 0.003909 | 3,953 | 0.0104 | 0.5716 | 1.0600 |
| Mid | Pistons vs Magic | 0.002520 | 0.001501 | 3,568 | -0.1434 | 0.6886 | 1.4637 |
| Mid | Knicks vs 76ers | 0.003976 | 0.000295 | 598 | -0.4988 | 0.6278 | 1.1921 |
| High | Knicks vs Hawks | 0.004564 | 0.000951 | 1,744 | -0.0021 | 0.5664 | 0.3644 |
| High | Spurs vs Trail Blazers | 0.004674 | 0.000467 | 1,723 | -0.3885 | 0.5920 | 0.5059 |
| High | Cavaliers vs Pistons | 0.007065 | 0.010027 | 1,532 | 0.0077 | 0.6048 | 1.1554 |
| High | Raptors vs Cavaliers | 0.008168 | 0.024527 | 3,934 | 0.0023 | 0.6321 | 0.9179 |

## Group aggregates (mean across markets)

| Group | n_mkts | R²_OS | DirAcc | Sharpe | R²_OS (w) | DirAcc (w) | Sharpe (w) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 4 | -0.2916 | 0.6128 | 1.2794 | -0.1902 | 0.6007 | 0.9458 |
| Mid | 4 | -0.1473 | 0.6457 | 1.3306 | -0.0427 | 0.6527 | 1.3832 |
| High | 4 | -0.0951 | 0.5988 | 0.7359 | -0.0730 | 0.6069 | 0.7711 |

*(w) = window-weighted mean.*

## Hypothesis test — Spearman rank correlation (vol vs metric, n=12)

### vs full-series realized vol

| Metric | Spearman ρ | p-value | verdict |
|---|---:|---:|---|
| R²_OS | 0.091 | 0.779 | cannot reject H0 (p=0.779): no significant dependence |
| Directional accuracy | 0.056 | 0.863 | cannot reject H0 (p=0.863): no significant dependence |
| Sharpe (annualized) | -0.021 | 0.948 | cannot reject H0 (p=0.948): no significant dependence |

### vs test-segment realized vol

| Metric | Spearman ρ | p-value | verdict |
|---|---:|---:|---|
| R²_OS | 0.783 | 0.003 | **reject H0** (p=0.003): performance is higher on more-volatile markets |
| Directional accuracy | 0.091 | 0.779 | cannot reject H0 (p=0.779): no significant dependence |
| Sharpe (annualized) | 0.070 | 0.829 | cannot reject H0 (p=0.829): no significant dependence |

## Bottom line

- **R²_OS** (vs test-segment vol): significantly higher on more-volatile markets (ρ=0.78, p=0.003).
- Metrics not listed above show no significant (p<0.05) dependence on variability; with n=12 treat all group-mean differences as exploratory.
