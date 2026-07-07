"""Performance attribution: annual breakdown, sector exposure, worst periods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_attribution(
    port_returns: pd.DataFrame,
    periods_per_year: int,
) -> pd.DataFrame:
    """Per-year Sharpe, return, vol, max drawdown, hit rate, % invested."""
    rows = []
    for year in sorted(port_returns.index.year.unique()):
        yr = port_returns[port_returns.index.year == year]
        if len(yr) < 2:
            continue
        r       = yr["net_return"]
        ann_ret = r.mean() * periods_per_year
        ann_vol = r.std() * np.sqrt(periods_per_year)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = (1 + r).cumprod()
        max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
        hit     = (yr["gross_return"] > 0).mean()
        invested = yr["in_regime"].mean() if "in_regime" in yr.columns else 1.0
        rows.append({
            "year":        year,
            "ann_return":  round(ann_ret, 4),
            "ann_vol":     round(ann_vol, 4),
            "sharpe":      round(sharpe, 3),
            "max_dd":      round(max_dd, 4),
            "hit_rate":    round(hit, 3),
            "pct_invested": round(invested, 3),
            "n_periods":   len(yr),
        })
    return pd.DataFrame(rows).set_index("year")


def sector_exposure(
    predictions: pd.DataFrame,
    sectors: pd.Series,
    top_q: float = 0.2,
) -> pd.Series:
    """Average fraction of the long portfolio in each GICS sector."""
    counts: dict[str, int] = {}
    dates = predictions.index.get_level_values("date").unique()
    for date in dates:
        scores  = predictions.xs(date, level="date")["score"]
        n_long  = max(1, int(round(len(scores) * top_q)))
        top     = scores.nlargest(n_long).index
        for ticker in top:
            sec = sectors.get(ticker, "Unknown")
            counts[sec] = counts.get(sec, 0) + 1
    total = sum(counts.values())
    s = pd.Series({k: v / total for k, v in counts.items()}).sort_values(ascending=False)
    s.name = "avg_weight"
    return s


def worst_periods(port_returns: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n worst individual rebalance periods by net return."""
    return (
        port_returns[["gross_return", "cost", "net_return"]]
        .nsmallest(n, "net_return")
        .round(4)
    )


def best_periods(port_returns: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n best individual rebalance periods by net return."""
    return (
        port_returns[["gross_return", "cost", "net_return"]]
        .nlargest(n, "net_return")
        .round(4)
    )


def print_attribution(
    port_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    sectors: pd.Series,
    periods_per_year: int,
    top_q: float = 0.2,
) -> None:
    """Print a full attribution tearsheet to stdout."""
    sep = "-" * 55

    print(sep)
    print("ANNUAL ATTRIBUTION")
    print(sep)
    ann = annual_attribution(port_returns, periods_per_year)
    print(ann.to_string())

    print()
    print(sep)
    print("AVERAGE SECTOR EXPOSURE (long leg)")
    print(sep)
    exp = sector_exposure(predictions, sectors, top_q=top_q)
    for sec, wt in exp.items():
        print(f"  {sec:<35s} {wt:.1%}")

    print()
    print(sep)
    print("WORST 5 PERIODS")
    print(sep)
    print(worst_periods(port_returns).to_string())

    print()
    print(sep)
    print("BEST 5 PERIODS")
    print(sep)
    print(best_periods(port_returns).to_string())
