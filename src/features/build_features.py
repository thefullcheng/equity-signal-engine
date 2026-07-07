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
from src.features.features import build_feature_panel, rebalance_dates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    logger.info("Computing features ...")
    features = build_feature_panel(prices, returns, volumes, dates)
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
