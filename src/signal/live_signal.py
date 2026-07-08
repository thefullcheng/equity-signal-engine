"""Generate the current portfolio signal from the latest model predictions."""

from __future__ import annotations

import pandas as pd


def latest_signal(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    sectors: pd.Series,
    top_q: float = 0.20,
) -> pd.DataFrame:
    """Return a ranked DataFrame for the most recent rebalance date.

    Columns: rank, score, sector, plus all feature values available on that date.
    """
    latest_date = predictions.index.get_level_values("date").max()
    scores = predictions.xs(latest_date, level="date")["score"].sort_values(ascending=False)

    n_long = max(1, int(round(len(scores) * top_q)))

    df = scores.rename("score").to_frame()
    df["rank"]     = range(1, len(df) + 1)
    df["long_leg"] = df["rank"] <= n_long
    df["sector"]   = sectors.reindex(df.index).fillna("Unknown")

    # Attach feature values for the latest date
    if latest_date in features.index.get_level_values("date"):
        feat_snap = features.xs(latest_date, level="date")
        df = df.join(feat_snap, how="left")

    df.index.name = "ticker"
    return df[["rank", "long_leg", "score", "sector"] +
              [c for c in df.columns if c not in {"rank", "long_leg", "score", "sector"}]]


def latest_momentum_signal(
    features: pd.DataFrame,
    sectors: pd.Series,
    top_q: float = 0.20,
) -> pd.DataFrame:
    """Rank by mom_12_1 alone on the most recent feature date -- no model,
    no fundamentals. A naive 12-1 momentum sort, for live comparison against
    the full model (see README, Statistical significance: the full pipeline
    barely beats this in backtest).
    """
    latest_date = features.index.get_level_values("date").max()
    scores = (
        features.xs(latest_date, level="date")["mom_12_1"]
        .dropna()
        .sort_values(ascending=False)
    )

    n_long = max(1, int(round(len(scores) * top_q)))

    df = scores.rename("score").to_frame()
    df["rank"]     = range(1, len(df) + 1)
    df["long_leg"] = df["rank"] <= n_long
    df["sector"]   = sectors.reindex(df.index).fillna("Unknown")

    df.index.name = "ticker"
    return df[["rank", "long_leg", "score", "sector"]]
