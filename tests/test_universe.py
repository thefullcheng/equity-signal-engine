"""Tests for point-in-time universe construction.

These run entirely on synthetic fixtures -- no network, no Wikipedia. The
scenarios encode the guarantees the rest of the pipeline relies on, above all:
a ticker must NEVER appear as a member before its addition date (that would be
survivorship/lookahead leaking in through the universe itself).
"""

import pandas as pd
import pytest

from src.data.universe import (
    build_membership,
    members_on,
    membership_over,
    normalize_ticker,
    parse_change_events,
)

FLOOR = "2010-01-01"


# ---------------------------------------------------------------------------
# Fixtures: a tiny synthetic index history
# ---------------------------------------------------------------------------
# Today's members: A, B, C, R2 (R2 was removed once and later re-added).
# Change log (chronological):
#   2015-03-02  add B      remove D    (D was a member from before the log)
#   2018-07-16  add C      remove E
#   2019-05-01  add R2     remove F
#   2020-09-21  remove R2  add G
#   2021-11-08  add R2     remove G
CURRENT = ["A", "B", "C", "R2"]


def make_events() -> pd.DataFrame:
    rows = [
        ("2015-03-02", "B", "add"), ("2015-03-02", "D", "remove"),
        ("2018-07-16", "C", "add"), ("2018-07-16", "E", "remove"),
        ("2019-05-01", "R2", "add"), ("2019-05-01", "F", "remove"),
        ("2020-09-21", "G", "add"), ("2020-09-21", "R2", "remove"),
        ("2021-11-08", "R2", "add"), ("2021-11-08", "G", "remove"),
    ]
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "ticker": t, "action": a} for d, t, a in rows]
    )


@pytest.fixture
def membership() -> pd.DataFrame:
    return build_membership(CURRENT, make_events(), floor_date=FLOOR)


# ---------------------------------------------------------------------------
# The core guarantee
# ---------------------------------------------------------------------------

def test_never_member_before_addition(membership):
    """B was added 2015-03-02: it must be absent on any earlier date."""
    assert "B" not in members_on(membership, "2015-03-01")
    assert "B" not in members_on(membership, "2010-06-30")
    assert "B" in members_on(membership, "2015-03-02")  # first day IN


def test_removed_ticker_present_before_not_after(membership):
    """D was removed 2015-03-02: member strictly before, gone from that day."""
    assert "D" in members_on(membership, "2015-03-01")
    assert "D" in members_on(membership, FLOOR)  # open-start -> floor date
    assert "D" not in members_on(membership, "2015-03-02")  # removal day = OUT
    assert "D" not in members_on(membership, "2023-01-01")


def test_relisting_produces_two_intervals(membership):
    """R2: in 2019-05-01..2020-09-21, out, back in from 2021-11-08."""
    ivs = membership[membership["ticker"] == "R2"].sort_values("start")
    assert len(ivs) == 2
    assert "R2" in members_on(membership, "2020-01-15")
    assert "R2" not in members_on(membership, "2021-01-15")  # the gap
    assert "R2" in members_on(membership, "2022-01-15")
    assert "R2" not in members_on(membership, "2019-04-30")


def test_pre_log_members_get_floor_start(membership):
    """A has no add event: its start must be exactly the floor date, i.e. we
    make the approximation explicit instead of inventing an earlier date."""
    a = membership[membership["ticker"] == "A"]
    assert len(a) == 1
    assert a.iloc[0]["start"] == pd.Timestamp(FLOOR)
    assert pd.isna(a.iloc[0]["end"])


def test_membership_matrix_matches_pointwise_queries(membership):
    dates = pd.date_range("2010-01-01", "2023-01-01", freq="90D")
    panel = membership_over(membership, dates)
    for d in dates:
        assert set(panel.columns[panel.loc[d]]) == set(members_on(membership, d))


# ---------------------------------------------------------------------------
# Parsing + normalization
# ---------------------------------------------------------------------------

def test_normalize_ticker_yfinance_convention():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker(" bf.b ") == "BF-B"
    assert normalize_ticker("MMM[1]") == "MMM"


def test_parse_change_events_multiindex_and_partial_rows():
    """Wikipedia serves a 2-level header and rows where only one side is
    populated (an add with no matching remove, or vice versa)."""
    cols = pd.MultiIndex.from_tuples(
        [("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
         ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason")]
    )
    raw = pd.DataFrame(
        [
            ["June 20, 2023", "PANW", "Palo Alto", "DISH", "Dish", "cap change"],
            ["March 15, 2023", "FICO", "Fair Isaac", None, None, "add only"],
            ["not a date", None, None, None, None, "junk row"],
        ],
        columns=cols,
    )
    ev = parse_change_events(raw)
    assert len(ev) == 3  # PANW add, DISH remove, FICO add; junk row dropped
    assert set(ev["action"]) == {"add", "remove"}
    assert ev["date"].is_monotonic_increasing


def test_inconsistent_log_does_not_corrupt_table(caplog):
    """An 'add' with no later removal and not in current members (a rename
    artifact) must be skipped with a warning, not fabricate an interval."""
    events = pd.DataFrame(
        [{"date": pd.Timestamp("2016-01-04"), "ticker": "GHOST", "action": "add"}]
    )
    m = build_membership(["A"], events, floor_date=FLOOR)
    assert "GHOST" not in set(m["ticker"])
    assert "A" in members_on(m, "2012-01-01")
