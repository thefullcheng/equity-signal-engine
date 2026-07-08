"""Fetch SEC insider-trading (Form 4) bulk data and build the point-in-time
feature panel. RUN ON YOUR MACHINE.

    python -m src.data.build_insider --config config/config.yaml

Uses SEC's bulk quarterly Form 3/4/5 structured datasets -- no API key
required, no per-company rate limiting (one ZIP per quarter, not one request
per ticker). Estimated runtime: a few minutes for ~65 quarters (2010-2026).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.insider import build_insider_panel, fetch_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--force-refresh", action="store_true",
                    help="Re-download even if a quarter's cache exists")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir       = Path(cfg["data"]["raw_dir"])
    insider_cache = raw_dir / "insider"

    prices = pd.read_parquet(processed_dir / "prices_clean.parquet")
    prices.index = pd.to_datetime(prices.index)
    tickers = list(prices.columns)
    logger.info("Universe: %d tickers", len(tickers))

    start = cfg["dates"]["start"]
    end   = str(prices.index.max().date())

    logger.info("Fetching SEC bulk Form 3/4/5 datasets (%s .. %s) ...", start, end)
    ok, fail = fetch_all(start, end, insider_cache, force_refresh=args.force_refresh)
    logger.info("Fetch complete: %d ok  %d failed/skipped", ok, fail)

    from src.features.features import rebalance_dates
    horizon = cfg["labels"]["horizon_days"]
    dates   = rebalance_dates(prices.index, freq=horizon)
    logger.info("Building point-in-time panel (%d rebalance dates) ...", len(dates))

    panel = build_insider_panel(tickers, insider_cache, dates)

    if panel.empty:
        logger.error("Insider panel is empty — check cache at %s", insider_cache)
        return

    n_tickers = panel.index.get_level_values("ticker").nunique()
    n_dates   = panel.index.get_level_values("date").nunique()
    logger.info(
        "Panel: %d obs  %d tickers  %d dates  columns=%s",
        len(panel), n_tickers, n_dates, list(panel.columns),
    )

    years = pd.Series(panel.index.get_level_values("date")).dt.year
    logger.info("Coverage by year:\n%s", years.value_counts().sort_index().to_string())

    out = processed_dir / "insider_panel.parquet"
    panel.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
