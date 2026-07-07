"""Point-in-time S&P 500 universe construction.

Why this module exists
----------------------
Backtesting on *today's* index members is survivorship bias: current members
are, by construction, firms that survived and grew. We instead reconstruct
which tickers were in the S&P 500 on each historical date, using Wikipedia's
constituent change log, and only ever expose the universe *as it was known at
the time*.

Method: backward reconstruction. We know today's members. Walking the change
log from newest to oldest, each "added" event tells us a ticker was NOT a
member before that date, and each "removed" event tells us it WAS. This yields
an interval table (ticker, start, end).

Known, documented limitations (see README):
  * Wikipedia's change log is incomplete in the distant past; we only trust it
    back to ``floor_date`` (config: universe.floor_date, default 2010-01-01).
    Tickers with no observed addition are assigned start = floor_date.
  * Ticker renames (e.g. FB -> META) sometimes appear as remove+add pairs and
    sometimes as silent edits; we treat the symbols as distinct and rely on
    the price layer's coverage audit to quantify the damage.
  * Delisted tickers may have no retrievable price history on yfinance. The
    coverage audit in ``src/data/coverage.py`` (Phase 2, step 2) measures this.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# ---------------------------------------------------------------------------
# Ticker normalization
# ---------------------------------------------------------------------------

def normalize_ticker(raw: str) -> str:
    """Map a Wikipedia-style ticker to yfinance's convention.

    Wikipedia writes share classes with a dot (BRK.B, BF.B); yfinance uses a
    dash (BRK-B, BF-B). We also strip whitespace and footnote junk. Keeping a
    single canonical symbol everywhere prevents silent join failures between
    the membership table and the price panel.
    """
    t = str(raw).strip().upper()
    t = re.sub(r"\[.*?\]", "", t)  # strip wiki footnote markers like "[1]"
    t = t.replace(".", "-")
    return t


# ---------------------------------------------------------------------------
# Wikipedia fetch (network side -- runs on YOUR machine, not in CI/tests)
# ---------------------------------------------------------------------------

def fetch_wikipedia_tables(
    cache_dir: str | Path,
    url: str = WIKI_URL,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or load cached) current-constituents and changes tables.

    We cache the raw HTML, not the parsed frames, so that parser fixes can be
    re-applied to the same snapshot -- reproducibility beats convenience here.

    Returns
    -------
    (current, changes) : raw pandas frames straight from ``pd.read_html``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sp500_wikipedia.html"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached Wikipedia snapshot: %s", cache_file)
        html = cache_file.read_text(encoding="utf-8")
    else:
        logger.info("Fetching %s", url)
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        cache_file.write_text(html, encoding="utf-8")
        logger.info("Cached snapshot to %s", cache_file)

    from io import StringIO

    tables = pd.read_html(StringIO(html))
    # Table 0 = current constituents; table 1 = selected changes. Wikipedia
    # layout occasionally shifts, so verify by column signature, not position.
    current = _find_table(tables, {"Symbol", "Security"})
    changes = _find_table(tables, {"Date"}, fuzzy_any={"Added", "Removed"})
    return current, changes


def _find_table(
    tables: list[pd.DataFrame],
    required: set[str],
    fuzzy_any: set[str] | None = None,
) -> pd.DataFrame:
    """Locate a table by column names so a Wikipedia reshuffle fails loudly."""
    for t in tables:
        cols = {str(c[0]) if isinstance(c, tuple) else str(c) for c in t.columns}
        flat = " ".join(
            " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            for c in t.columns
        )
        if required.issubset(cols) or all(r in flat for r in required):
            if fuzzy_any is None or any(f in flat for f in fuzzy_any):
                return t
    raise ValueError(
        f"No Wikipedia table matched columns {required}. "
        "The page layout may have changed; inspect the cached HTML."
    )


# ---------------------------------------------------------------------------
# Parsing the changes table into tidy events
# ---------------------------------------------------------------------------

def parse_change_events(changes_raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten the (multi-indexed) Wikipedia changes table into tidy events.

    Output columns: date (Timestamp), ticker (str), action ('add'|'remove').
    One physical row of the wiki table can yield up to two events.
    """
    df = changes_raw.copy()
    # Wikipedia uses a two-level header: ('Date',''), ('Added','Ticker'), ...
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x) != "nan").strip("_")
            for col in df.columns
        ]
    df.columns = [str(c) for c in df.columns]

    date_col = _first_matching(df.columns, "Date")
    add_col = _first_matching(df.columns, "Added_Ticker", "Added Ticker", "Added")
    rem_col = _first_matching(df.columns, "Removed_Ticker", "Removed Ticker", "Removed")

    events = []
    for _, row in df.iterrows():
        date = pd.to_datetime(str(row[date_col]), errors="coerce")
        if pd.isna(date):
            continue  # malformed row; change log has a few
        for col, action in ((add_col, "add"), (rem_col, "remove")):
            val = row.get(col)
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                events.append(
                    {"date": date, "ticker": normalize_ticker(val), "action": action}
                )
    out = pd.DataFrame(events)
    if out.empty:
        raise ValueError("Parsed zero change events -- inspect the changes table.")
    return out.sort_values("date").reset_index(drop=True)


def _first_matching(columns, *candidates: str) -> str:
    for cand in candidates:
        for c in columns:
            if cand.lower() in str(c).lower():
                return c
    raise KeyError(f"None of {candidates} found in columns {list(columns)}")


# ---------------------------------------------------------------------------
# Backward reconstruction of membership intervals
# ---------------------------------------------------------------------------

@dataclass
class _OpenInterval:
    end: pd.Timestamp | None  # None = still a member today


def build_membership(
    current_tickers: list[str],
    events: pd.DataFrame,
    floor_date: str | pd.Timestamp = "2010-01-01",
) -> pd.DataFrame:
    """Reconstruct (ticker, start, end) membership intervals.

    Semantics: a ticker is a member on date d iff start <= d < end
    (end is exclusive: the removal date is the first day OUT of the index;
    an addition date is the first day IN). ``end`` is NaT for current members.

    Walking the change log backward from today:
      * an 'add' event at date d closes the ticker's open interval with
        start = d (before d it was not a member),
      * a 'remove' event at date d opens an interval ending at d (before d it
        WAS a member; we don't yet know since when).
    Tickers still open when we run out of events get start = floor_date --
    an explicit, documented approximation, not silent truncation.
    """
    floor_date = pd.Timestamp(floor_date)
    active: dict[str, _OpenInterval] = {
        normalize_ticker(t): _OpenInterval(end=None) for t in current_tickers
    }
    intervals: list[dict] = []

    ev = events.sort_values("date", ascending=False)
    for _, e in ev.iterrows():
        t, d, action = e["ticker"], e["date"], e["action"]
        if action == "add":
            if t in active:
                iv = active.pop(t)
                intervals.append({"ticker": t, "start": d, "end": iv.end})
            else:
                # Added but not active while walking backward => it was later
                # removed by an event we should already have seen. If we
                # didn't, the log is inconsistent for this ticker (often a
                # rename). Log and skip rather than corrupt the table.
                logger.warning(
                    "Inconsistent log: add of %s on %s with no open interval "
                    "(possible rename); skipping event.", t, d.date(),
                )
        elif action == "remove":
            if t in active:
                logger.warning(
                    "Inconsistent log: remove of %s on %s while already open "
                    "(overlap); keeping earlier removal.", t, d.date(),
                )
            else:
                active[t] = _OpenInterval(end=d)
        else:  # pragma: no cover
            raise ValueError(f"Unknown action {action!r}")

    # Anything still open predates the change log's coverage.
    for t, iv in active.items():
        intervals.append({"ticker": t, "start": floor_date, "end": iv.end})

    out = pd.DataFrame(intervals)
    out["end"] = pd.to_datetime(out["end"])  # NaT-safe
    out = out.sort_values(["ticker", "start"]).reset_index(drop=True)

    # Drop intervals that end before they start or end before floor_date --
    # they carry no information for our sample window.
    bad = out["end"].notna() & (out["end"] <= out["start"])
    if bad.any():
        logger.warning("Dropping %d degenerate intervals.", int(bad.sum()))
        out = out[~bad].reset_index(drop=True)
    out = out[(out["end"].isna()) | (out["end"] > floor_date)].reset_index(drop=True)
    return out


def members_on(membership: pd.DataFrame, date: str | pd.Timestamp) -> list[str]:
    """Tickers that were index members on ``date`` (point-in-time query)."""
    d = pd.Timestamp(date)
    m = membership
    mask = (m["start"] <= d) & (m["end"].isna() | (m["end"] > d))
    return sorted(m.loc[mask, "ticker"].unique())


def membership_over(
    membership: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Boolean panel (index=dates, columns=tickers): was ticker a member?

    Used downstream to mask features/labels so a stock can never enter the
    training set on dates it wasn't in the index -- that would be lookahead
    through the universe definition itself.
    """
    tickers = sorted(membership["ticker"].unique())
    panel = pd.DataFrame(False, index=dates, columns=tickers)
    for _, row in membership.iterrows():
        end = row["end"] if pd.notna(row["end"]) else dates[-1] + pd.Timedelta(days=1)
        mask = (dates >= row["start"]) & (dates < end)
        panel.loc[mask, row["ticker"]] = True
    return panel
