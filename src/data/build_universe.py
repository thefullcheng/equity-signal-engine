"""Build the point-in-time membership table. RUN THIS ON YOUR MACHINE.

    python -m src.data.build_universe --config config/config.yaml

Fetches (and caches) the Wikipedia S&P 500 page, reconstructs membership
intervals, and writes ``data/processed/membership.parquet`` plus a small
human-readable summary. Network required on first run only; afterward the
cached HTML snapshot is used unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.universe import (
    build_membership,
    fetch_wikipedia_tables,
    members_on,
    normalize_ticker,
    parse_change_events,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-download Wikipedia instead of using cache")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    floor = cfg["universe"]["floor_date"]
    cache = cfg["data"]["wikipedia_cache"]
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    current_raw, changes_raw = fetch_wikipedia_tables(cache, force_refresh=args.refresh)

    sym_col = next(c for c in current_raw.columns if "Symbol" in str(c))
    current = [normalize_ticker(t) for t in current_raw[sym_col].dropna()]
    logger.info("Current constituents parsed: %d", len(current))

    events = parse_change_events(changes_raw)
    logger.info("Change events parsed: %d (%s .. %s)",
                len(events), events['date'].min().date(), events['date'].max().date())

    membership = build_membership(current, events, floor_date=floor)

    out_file = out_dir / "membership.parquet"
    membership.to_parquet(out_file, index=False)
    logger.info("Wrote %s (%d intervals, %d unique tickers)",
                out_file, len(membership), membership['ticker'].nunique())

    # Sanity summary: membership count on a few probe dates should hover ~500.
    for probe in ["2012-06-01", "2016-06-01", "2020-06-01", "2024-06-01"]:
        n = len(members_on(membership, probe))
        flag = "" if 480 <= n <= 520 else "   <-- INVESTIGATE (expected ~500)"
        logger.info("Members on %s: %d%s", probe, n, flag)


if __name__ == "__main__":
    main()
