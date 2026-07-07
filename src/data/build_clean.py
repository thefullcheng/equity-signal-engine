"""Clean the price panel and compute daily log returns. RUN ON YOUR MACHINE.

    python -m src.data.build_clean --config config/config.yaml

Reads prices_panel.parquet and coverage.parquet, applies the point-in-time
membership mask, forward-fills short data gaps, and writes:
  data/processed/prices_clean.parquet  -- cleaned (date x ticker) Close panel
  data/processed/returns.parquet       -- daily log returns

Requires Phase 2b outputs (prices_panel.parquet, coverage.parquet).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.clean import clean_panel, filter_by_coverage, log_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--max-fill", type=int, default=5,
                    help="Max consecutive NaN days to forward-fill (default: 5)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])

    coverage = pd.read_parquet(processed_dir / "coverage.parquet")
    panel = pd.read_parquet(processed_dir / "prices_panel.parquet")
    membership = pd.read_parquet(processed_dir / "membership.parquet")
    panel.index = pd.to_datetime(panel.index)

    logger.info("Loaded panel: %s  dates: %s .. %s",
                panel.shape, panel.index.min().date(), panel.index.max().date())

    filtered = filter_by_coverage(panel, coverage)

    clean = clean_panel(filtered, membership, max_fill=args.max_fill)

    n_nan = int(clean.isna().sum().sum())
    n_total = clean.size
    logger.info(
        "Clean panel: %d tickers x %d dates  NaN cells: %d (%.1f%%)",
        clean.shape[1], clean.shape[0], n_nan, 100 * n_nan / n_total,
    )

    returns = log_returns(clean)

    clean_path = processed_dir / "prices_clean.parquet"
    returns_path = processed_dir / "returns.parquet"
    clean.to_parquet(clean_path)
    returns.to_parquet(returns_path)
    logger.info("Wrote %s", clean_path)
    logger.info("Wrote %s", returns_path)

    active_today = int(clean.iloc[-1].notna().sum())
    logger.info(
        "Active tickers on %s: %d / %d",
        clean.index[-1].date(), active_today, clean.shape[1],
    )


if __name__ == "__main__":
    main()
