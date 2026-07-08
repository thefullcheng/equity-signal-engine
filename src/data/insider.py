"""SEC EDGAR insider trading (Form 4) fetch and point-in-time panel construction.

Uses SEC's bulk quarterly Form 3/4/5 structured datasets (no key required):
    https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip

Each quarter's ZIP contains normalized TSVs. We only need two:
    SUBMISSION.tsv     -- one row per filing: accession number, filing date,
                           issuer CIK/ticker. Filing date is the point-in-time
                           availability date (Form 4 has a strict 2-business-
                           day filing deadline after the transaction, so this
                           lag is small and low-risk relative to the
                           fundamentals filing-date issue fixed earlier).
    NONDERIV_TRANS.tsv -- one row per non-derivative (common stock)
                           transaction within a filing: transaction code,
                           shares, price.

Critical filter: raw Form 4 data is dominated by routine compensation
mechanics (option exercises 'M', tax withholding 'F', grants 'A', gifts 'G')
that carry no signal about an insider's actual view. Only TRANS_CODE in
{'P', 'S'} -- genuine open-market purchases and sales -- are used.
"""

from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "equity-signal-engine research@example.com"}
_BASE_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{quarter}_form345.zip"
)

# Genuine open-market transactions only -- excludes grants, exercises, tax
# withholding, gifts, and other compensation-mechanic codes.
_OPEN_MARKET_CODES = {"P", "S"}


def quarters_between(start: str, end: str) -> list[tuple[int, int]]:
    """Return [(year, quarter), ...] spanning start..end inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = []
    y, q = start_ts.year, (start_ts.month - 1) // 3 + 1
    while (y, q) <= (end_ts.year, (end_ts.month - 1) // 3 + 1):
        out.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def _fetch_one_quarter(year: int, quarter: int, cache_dir: Path, force_refresh: bool) -> bool:
    """Download one quarter's ZIP, extract only the two TSVs we need."""
    sub_out = cache_dir / f"{year}q{quarter}_SUBMISSION.parquet"
    trans_out = cache_dir / f"{year}q{quarter}_NONDERIV_TRANS.parquet"
    if sub_out.exists() and trans_out.exists() and not force_refresh:
        return True

    url = _BASE_URL.format(year=year, quarter=quarter)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60)
        if resp.status_code == 404:
            logger.debug("No data for %dq%d (not yet published)", year, quarter)
            return False
        resp.raise_for_status()

        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            with zf.open("SUBMISSION.tsv") as f:
                sub = pd.read_csv(f, sep="\t", low_memory=False)
            with zf.open("NONDERIV_TRANS.tsv") as f:
                trans = pd.read_csv(f, sep="\t", low_memory=False)

        sub = sub[sub["DOCUMENT_TYPE"] == "4"][
            ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]
        ]
        trans = trans[trans["TRANS_CODE"].isin(_OPEN_MARKET_CODES)][
            ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]
        ]

        cache_dir.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(sub_out)
        trans.to_parquet(trans_out)
        return True

    except Exception as exc:
        logger.debug("Failed %dq%d: %s", year, quarter, exc)
        return False


def fetch_all(
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> tuple[int, int]:
    """Fetch all quarterly datasets spanning start..end. Returns (ok, fail)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    quarters = quarters_between(start, end)
    ok = fail = 0
    for i, (year, quarter) in enumerate(quarters, 1):
        success = _fetch_one_quarter(year, quarter, cache_dir, force_refresh)
        if success:
            ok += 1
        else:
            fail += 1
        if i % 10 == 0 or i == len(quarters):
            logger.info("  %d / %d quarters  (ok=%d  fail=%d)", i, len(quarters), ok, fail)
    return ok, fail


def _load_all_transactions(cache_dir: Path, tickers: set[str]) -> pd.DataFrame:
    """Concatenate all cached quarters into one long DataFrame of signed,
    dollar-valued, open-market insider transactions, restricted to `tickers`.
    """
    sub_files = sorted(cache_dir.glob("*_SUBMISSION.parquet"))
    frames = []
    for sub_path in sub_files:
        trans_path = Path(str(sub_path).replace("_SUBMISSION.parquet", "_NONDERIV_TRANS.parquet"))
        if not trans_path.exists():
            continue
        sub = pd.read_parquet(sub_path)
        sub = sub[sub["ISSUERTRADINGSYMBOL"].isin(tickers)]
        if sub.empty:
            continue
        trans = pd.read_parquet(trans_path)
        merged = trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["ticker", "filing_date", "signed_flag"])

    out = pd.concat(frames, ignore_index=True)
    out["filing_date"] = pd.to_datetime(out["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    out = out.dropna(subset=["filing_date"])
    out["signed_flag"] = out["TRANS_CODE"].map({"P": 1, "S": -1})
    out = out.rename(columns={"ISSUERTRADINGSYMBOL": "ticker"})
    return out[["ticker", "filing_date", "signed_flag"]]


def build_insider_panel(
    tickers: list[str],
    cache_dir: Path,
    rebalance_dates: pd.DatetimeIndex,
    window_days: int = 90,
) -> pd.DataFrame:
    """Point-in-time insider-trading feature indexed by (date, ticker).

    Feature: net open-market buy/sell balance over the trailing `window_days`
    calendar days, using only filings with filing_date strictly before the
    rebalance date -- (n_buys - n_sells) / (n_buys + n_sells), bounded
    [-1, 1] and scale-free (a simple count ratio, not dollar-weighted, so it
    isn't dominated by company size / insider wealth).
    """
    tx = _load_all_transactions(cache_dir, set(tickers))
    if tx.empty:
        logger.warning("No insider transactions found — check cache at %s", cache_dir)
        return pd.DataFrame(columns=["insider_score"],
                             index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]))

    records: list[dict] = []
    window = pd.Timedelta(days=window_days)

    for ticker, grp in tx.groupby("ticker"):
        grp = grp.sort_values("filing_date")
        dates_arr = grp["filing_date"].values
        flags_arr = grp["signed_flag"].values

        for date in rebalance_dates:
            lo = date - window
            mask = (dates_arr >= lo.to_datetime64()) & (dates_arr < date.to_datetime64())
            if not mask.any():
                continue
            flags = flags_arr[mask]
            n_buy  = (flags == 1).sum()
            n_sell = (flags == -1).sum()
            total = n_buy + n_sell
            if total == 0:
                continue
            records.append({
                "date": date,
                "ticker": ticker,
                "insider_score": float(n_buy - n_sell) / total,
            })

    if not records:
        return pd.DataFrame(columns=["insider_score"],
                             index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]))

    out = pd.DataFrame(records).set_index(["date", "ticker"]).sort_index()
    return out
