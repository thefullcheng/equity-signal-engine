"""Fama-French 5-factor + momentum daily data from Ken French's data library
(free, no key required): https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/

Daily (not monthly) factors are used and compounded over each backtest
period's exact date range, rather than approximating a 20-trading-day
rebalance period as a calendar month -- the periods don't line up with
month boundaries and drift over time.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "equity-signal-engine research@example.com"}
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _fetch_csv_from_zip(url: str, cache_path: Path, force_refresh: bool = False) -> str:
    if cache_path.exists() and not force_refresh:
        return cache_path.read_text()
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def _parse_daily_factor_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV: a few header/description lines, a header
    row, then `YYYYMMDD,val1,val2,...` rows, then a trailing copyright line.
    Values are in percentage points; converted to decimals here.
    """
    lines = text.splitlines()
    data_lines = [l for l in lines if l[:8].strip().isdigit() and len(l[:8].strip()) == 8]
    n_data_cols = len(data_lines[0].split(",")) - 1  # excluding the date column

    header_idx = next(i for i, l in enumerate(lines) if l.startswith(","))
    header_labels = [c.strip() for c in lines[header_idx].split(",")][1 : 1 + n_data_cols]

    # Some files (e.g. the momentum factor's) have a blank header row -- fall
    # back to a placeholder name, and de-duplicate either way so df[col]
    # can't return multiple columns for the same label.
    seen: dict[str, int] = {}
    cols = ["date"]
    for i, label in enumerate(header_labels):
        name = label if label else f"factor_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        cols.append(name)

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        rows.append(parts[: len(cols)])

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date").dropna(how="all")


def fetch_daily_factors(cache_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    """Return daily factor returns (decimals) indexed by date, columns:
    Mkt-RF, SMB, HML, RMW, CMA, RF, MOM.
    """
    ff5_text = _fetch_csv_from_zip(_FF5_URL, cache_dir / "ff5_daily.csv", force_refresh)
    mom_text = _fetch_csv_from_zip(_MOM_URL, cache_dir / "mom_daily.csv", force_refresh)

    ff5 = _parse_daily_factor_csv(ff5_text)
    mom = _parse_daily_factor_csv(mom_text)
    mom = mom.rename(columns={mom.columns[0]: "MOM"})[["MOM"]]

    return ff5.join(mom, how="inner")


def compound_factor_returns(
    daily_factors: pd.DataFrame,
    period_starts: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compound daily factor returns into period returns matching a strategy's
    exact rebalance-date boundaries: period i covers (period_starts[i],
    period_starts[i+1]], i.e. trading days strictly after the rebalance date
    through and including the next one -- the same window a forward-return
    label covers. The final period_start has no successor and is dropped.
    """
    starts = sorted(period_starts)
    records = []
    for start, end in zip(starts[:-1], starts[1:]):
        window = daily_factors[(daily_factors.index > start) & (daily_factors.index <= end)]
        if window.empty:
            continue
        compounded = (1 + window).prod() - 1
        compounded.name = start
        records.append(compounded)

    return pd.DataFrame(records)
