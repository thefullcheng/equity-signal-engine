"""Price download, caching, panel assembly, and coverage audit.

Fetches daily OHLCV (adjusted) from yfinance for all historical S&P 500
tickers, caches each ticker to an individual parquet under data/raw/prices/,
then assembles a (date × ticker) Close panel. The coverage audit compares
available price days to expected trading days within each ticker's membership
window and classifies each ticker as full / partial / empty so downstream
steps can handle the delisted-ticker gap explicitly rather than silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FULL_THRESHOLD = 0.95  # coverage_pct >= this → 'full'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _fetch_chunk(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Download one chunk of tickers; return {ticker: ohlcv_df}.

    Tickers with no available data (delisted, renamed) are omitted from the
    result dict so callers never write empty parquet files to the cache.
    yfinance returns a flat DataFrame for a single ticker and a (field, ticker)
    MultiIndex DataFrame for multiple tickers; both cases are handled.
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                if not df.empty:
                    result[ticker] = df
            except KeyError:
                pass
    else:
        # Flat columns: yfinance returned a single-ticker frame
        ticker = tickers[0]
        df = raw.dropna(how="all")
        if not df.empty:
            result[ticker] = df

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    chunk_size: int = 50,
) -> None:
    """Download all tickers, writing {ticker}.parquet to cache_dir.

    Already-cached tickers are skipped so re-runs are incremental. A chunk
    that raises an exception is logged and skipped rather than aborting the
    full run.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [t for t in tickers if not (cache_dir / f"{t}.parquet").exists()]
    logger.info(
        "%d tickers to fetch (%d already cached), chunk_size=%d",
        len(missing), len(tickers) - len(missing), chunk_size,
    )

    chunks = list(_chunk(missing, chunk_size))
    for i, chunk in enumerate(chunks, 1):
        logger.info("Chunk %d/%d (%d tickers) ...", i, len(chunks), len(chunk))
        try:
            data = _fetch_chunk(chunk, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chunk %d failed (%s) — skipping", i, exc)
            continue

        for ticker, df in data.items():
            df.to_parquet(cache_dir / f"{ticker}.parquet")

        n_empty = len(chunk) - len(data)
        if n_empty:
            logger.info("  %d ticker(s) returned no data (delisted/unknown)", n_empty)

    logger.info("Fetch complete — cache: %s", cache_dir)


def load_panel(
    tickers: list[str],
    cache_dir: str | Path,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Assemble a (date × ticker) panel from the per-ticker cache.

    Tickers with no cache file appear as all-NaN columns; callers decide
    whether to drop or impute. The date index is the union of all dates
    present across all cache files.
    """
    cache_dir = Path(cache_dir)
    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if price_col in df.columns and not df.empty:
                series[ticker] = df[price_col].rename(ticker)

    if not series:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=sorted(tickers))

    panel = pd.concat(series.values(), axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.reindex(columns=sorted(tickers))
    return panel


def audit_coverage(
    membership: pd.DataFrame,
    cache_dir: str | Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Per-ticker coverage stats: price days vs expected membership window.

    Returns a DataFrame sorted by coverage_pct ascending (worst-covered
    tickers first) with columns: ticker, membership_start, membership_end,
    price_start, price_end, price_days, expected_days, coverage_pct, status.

    expected_days uses pd.bdate_range as a business-day approximation of
    trading days; it over-counts slightly around holidays but is consistent
    and avoids a market-calendar dependency at this stage.
    """
    cache_dir = Path(cache_dir)
    sample_start = pd.Timestamp(start)
    sample_end = pd.Timestamp(end)

    rows = []
    for ticker, grp in membership.groupby("ticker"):
        mem_start = max(grp["start"].min(), sample_start)
        raw_end = grp["end"].max()
        mem_end = sample_end if pd.isna(raw_end) else min(raw_end, sample_end)

        expected_days = len(pd.bdate_range(mem_start, mem_end))

        path = cache_dir / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            df = df[(df.index >= sample_start) & (df.index <= sample_end)]
            price_days = int(df["Close"].notna().sum()) if "Close" in df.columns else 0
            price_start = df.index.min() if not df.empty else pd.NaT
            price_end = df.index.max() if not df.empty else pd.NaT
        else:
            price_days, price_start, price_end = 0, pd.NaT, pd.NaT

        coverage_pct = round(price_days / expected_days, 4) if expected_days > 0 else 0.0
        if price_days == 0:
            status = "empty"
        elif coverage_pct >= FULL_THRESHOLD:
            status = "full"
        else:
            status = "partial"

        rows.append({
            "ticker": ticker,
            "membership_start": mem_start,
            "membership_end": mem_end,
            "price_start": price_start,
            "price_end": price_end,
            "price_days": price_days,
            "expected_days": expected_days,
            "coverage_pct": coverage_pct,
            "status": status,
        })

    return pd.DataFrame(rows).sort_values("coverage_pct").reset_index(drop=True)
