"""Fetch prices for all S&P 500 tickers and run coverage audit. RUN ON YOUR MACHINE.

    python -m src.data.build_prices --config config/config.yaml

Downloads (and caches) daily adjusted OHLCV for all 814 historical S&P 500
members, assembles a (date × ticker) Close panel, and writes a per-ticker
coverage audit. Already-cached tickers are skipped so re-runs are incremental.

Network required. Expect ~2-5 minutes for a cold run over all 814 tickers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.prices import audit_coverage, fetch_all, load_panel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="Tickers per yfinance batch request (default: 50)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"] or pd.Timestamp.today().strftime("%Y-%m-%d")
    processed_dir = Path(cfg["data"]["processed_dir"])
    price_cache = Path(cfg["data"]["raw_dir"]) / "prices"
    processed_dir.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(processed_dir / "membership.parquet")
    tickers = sorted(membership["ticker"].unique().tolist())
    logger.info("Universe: %d unique tickers  window: %s .. %s", len(tickers), start, end)

    fetch_all(tickers, start=start, end=end, cache_dir=price_cache,
              chunk_size=args.chunk_size)

    logger.info("Assembling Close panel ...")
    panel = load_panel(tickers, cache_dir=price_cache)
    panel_path = processed_dir / "prices_panel.parquet"
    panel.to_parquet(panel_path)
    logger.info(
        "Wrote %s  shape=%s  dates=%s .. %s",
        panel_path, panel.shape,
        panel.index.min().date() if not panel.empty else "n/a",
        panel.index.max().date() if not panel.empty else "n/a",
    )

    logger.info("Running coverage audit ...")
    coverage = audit_coverage(membership, cache_dir=price_cache, start=start, end=end)
    cov_path = processed_dir / "coverage.parquet"
    coverage.to_parquet(cov_path, index=False)

    counts = coverage["status"].value_counts()
    logger.info("Coverage summary:")
    for status in ("full", "partial", "empty"):
        n = int(counts.get(status, 0))
        logger.info("  %-8s %d tickers (%.1f%%)", status, n, 100 * n / len(coverage))

    empty = coverage.loc[coverage["status"] == "empty", "ticker"].tolist()
    if empty:
        preview = ", ".join(empty[:20])
        tail = f" ... and {len(empty) - 20} more" if len(empty) > 20 else ""
        logger.info("Empty tickers (%d): %s%s", len(empty), preview, tail)

    logger.info("Wrote %s", cov_path)


if __name__ == "__main__":
    main()
