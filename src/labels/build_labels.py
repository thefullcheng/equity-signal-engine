"""Compute forward return labels. RUN ON YOUR MACHINE.

    python -m src.labels.build_labels --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.features.features import rebalance_dates
from src.labels.labels import build_label_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg     = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    horizon = cfg["labels"]["horizon_days"]

    prices = pd.read_parquet(processed_dir / "prices_clean.parquet")
    prices.index = pd.to_datetime(prices.index)

    dates  = rebalance_dates(prices.index, freq=horizon)
    labels = build_label_panel(prices, dates, horizon=horizon)

    logger.info(
        "Labels: %d observations  dates: %s .. %s",
        len(labels),
        labels.index.get_level_values("date").min().date(),
        labels.index.get_level_values("date").max().date(),
    )

    out = processed_dir / "labels.parquet"
    labels.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
