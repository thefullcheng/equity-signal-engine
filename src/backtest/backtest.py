"""Long-short portfolio backtest with transaction costs and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _construct_weights(scores: pd.Series, top_q: float) -> pd.Series:
    """Equal-weight long top quintile, short bottom quintile; zero otherwise."""
    n = len(scores)
    n_leg = max(1, int(round(n * top_q)))
    ranked = scores.rank(ascending=True)
    weights = pd.Series(0.0, index=scores.index)
    weights[ranked > (n - n_leg)] =  1.0 / n_leg   # long leg
    weights[ranked <= n_leg]       = -1.0 / n_leg   # short leg
    return weights


def run_backtest(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    costs_bps: float = 10,
    top_q: float = 0.2,
) -> pd.DataFrame:
    """Compute per-period gross and net returns.

    Parameters
    ----------
    predictions : (date, ticker) MultiIndex, column 'score'
    labels      : (date, ticker) MultiIndex, column 'fwd_return'
    costs_bps   : one-way transaction cost in basis points
    top_q       : fraction of universe in each leg

    Returns
    -------
    DataFrame indexed by date with: gross_return, cost, net_return,
    long_return, short_return.
    """
    pred_dates = predictions.index.get_level_values("date").unique().sort_values()
    label_dates = set(labels.index.get_level_values("date"))
    prev_weights: pd.Series | None = None
    results = []

    for date in pred_dates:
        scores = predictions.xs(date, level="date")["score"]
        weights = _construct_weights(scores, top_q)

        if date in label_dates:
            fwd = labels.xs(date, level="date")["fwd_return"]
            common = weights.index.intersection(fwd.index)
            w = weights[common]
            f = fwd[common]
            gross      = (w * f).sum()
            long_ret   = (w[w > 0] * f[w > 0]).sum()
            short_ret  = (w[w < 0] * f[w < 0]).sum()
        else:
            gross = long_ret = short_ret = np.nan

        if prev_weights is not None:
            prev = prev_weights.reindex(weights.index, fill_value=0.0)
            turnover = (weights - prev).abs().sum()
        else:
            turnover = weights.abs().sum()
        cost = turnover * costs_bps / 10_000

        results.append({
            "date":         date,
            "gross_return": gross,
            "cost":         cost,
            "net_return":   gross - cost if not np.isnan(gross) else np.nan,
            "long_return":  long_ret,
            "short_return": short_ret,
        })
        prev_weights = weights

    return pd.DataFrame(results).set_index("date").dropna()


def compute_metrics(port_returns: pd.DataFrame, periods_per_year: int = 52) -> dict:
    """Annualized performance metrics from a column of period net returns."""
    r = port_returns["net_return"]
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr    = cum.iloc[-1] ** (periods_per_year / len(r)) - 1
    return {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe,  3),
        "max_drawdown": round(max_dd,  4),
        "cagr":         round(cagr,    4),
        "hit_rate":     round((port_returns["gross_return"] > 0).mean(), 4),
        "n_periods":    len(r),
    }


def information_coefficient(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.Series:
    """Spearman rank correlation between predicted score and realized return."""
    label_dates = set(labels.index.get_level_values("date"))
    ics: dict[pd.Timestamp, float] = {}

    for date in predictions.index.get_level_values("date").unique():
        if date not in label_dates:
            continue
        scores = predictions.xs(date, level="date")["score"]
        fwd    = labels.xs(date, level="date")["fwd_return"].dropna()
        common = scores.index.intersection(fwd.index)
        if len(common) < 10:
            continue
        ics[date] = scores[common].corr(fwd[common], method="spearman")

    return pd.Series(ics, name="ic").sort_index()
