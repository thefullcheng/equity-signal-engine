"""Compute and cache the feature panel. RUN ON YOUR MACHINE.

    python -m src.features.build_features --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.prices import load_panel
from src.features.features import build_feature_panel, rank_normalize, rebalance_dates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _join_ranked_panel(features: pd.DataFrame, panel_path: Path, label: str) -> pd.DataFrame:
    """Rank-normalize each column of a (date, ticker) panel cross-sectionally
    on each date, then left-join into `features`. Shared by fundamentals and
    insider-trading joins below.
    """
    if not panel_path.exists():
        logger.info("No %s found — skipping %s features", panel_path, label)
        return features

    logger.info("Joining %s features ...", label)
    panel = pd.read_parquet(panel_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "ticker"])

    cols = list(panel.columns)
    wide = panel.unstack(level="ticker")   # (date, feature-ticker wide)
    ranked_frames = []
    for col in cols:
        if col in wide.columns.get_level_values(0):
            w = wide[col]                  # date × ticker
            ranked = rank_normalize(w)      # cross-sectional rank
            ranked.index.name = "date"
            s = ranked.stack(future_stack=True)
            s.name = col
            ranked_frames.append(s)

    if not ranked_frames:
        return features

    ranked = pd.concat(ranked_frames, axis=1)
    ranked.index.names = ["date", "ticker"]
    features = features.join(ranked, how="left")
    logger.info("Added %s columns: %s", label, cols)
    return features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    price_cache   = Path(cfg["data"]["raw_dir"]) / "prices"

    prices  = pd.read_parquet(processed_dir / "prices_clean.parquet")
    returns = pd.read_parquet(processed_dir / "returns.parquet")
    prices.index  = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)

    logger.info("Loading volume panel ...")
    tickers = list(prices.columns)
    volumes = load_panel(tickers, cache_dir=price_cache, price_col="Volume")
    volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    freq  = cfg["labels"]["horizon_days"]
    dates = rebalance_dates(prices.index, freq=freq)
    logger.info("Rebalance dates: %d  (%s .. %s)",
                len(dates), dates[0].date(), dates[-1].date())

    logger.info("Computing price features ...")
    features = build_feature_panel(prices, returns, volumes, dates)

    features = _join_ranked_panel(features, processed_dir / "fundamentals_panel.parquet", "fundamental")
    features = _join_ranked_panel(features, processed_dir / "insider_panel.parquet", "insider")

    logger.info(
        "Feature panel: %d observations  %d tickers  %d features",
        len(features),
        features.index.get_level_values("ticker").nunique(),
        features.shape[1],
    )

    out = processed_dir / "features.parquet"
    features.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
