"""Panel cleaning and alignment for Phase 2c.

Three operations in sequence:
  1. Coverage filter  -- drop tickers yfinance couldn't recover (status='empty').
  2. Membership mask  -- set prices to NaN outside each ticker's point-in-time
                         membership window so no out-of-index data leaks forward.
  3. Gap fill         -- forward-fill up to ``max_fill`` consecutive NaN cells
                         within a membership window (handles holidays / stale
                         feeds), then re-apply the mask so fill can't spill past
                         a membership boundary.

log_returns computes daily log returns from the cleaned panel; the first row
of each contiguous membership window is always NaN (no prior price to compare).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.universe import membership_over

logger = logging.getLogger(__name__)


def filter_by_coverage(
    panel: pd.DataFrame,
    coverage: pd.DataFrame,
    keep_statuses: tuple[str, ...] = ("full", "partial"),
) -> pd.DataFrame:
    """Drop tickers whose coverage status is not in keep_statuses."""
    keep = set(coverage.loc[coverage["status"].isin(keep_statuses), "ticker"])
    cols = [c for c in panel.columns if c in keep]
    dropped = len(panel.columns) - len(cols)
    logger.info(
        "Coverage filter: kept %d tickers, dropped %d (statuses kept: %s)",
        len(cols), dropped, keep_statuses,
    )
    return panel[cols]


def apply_membership_mask(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Set prices to NaN on dates the ticker was not an index member.

    Only builds the mask for tickers present in prices.columns, so it stays
    efficient after the coverage filter has reduced the universe.
    """
    relevant = membership[membership["ticker"].isin(prices.columns)]
    mask = membership_over(relevant, prices.index)
    mask = mask.reindex(columns=prices.columns, fill_value=False)
    return prices.where(mask)


def fill_gaps(prices: pd.DataFrame, max_fill: int = 5) -> pd.DataFrame:
    """Forward-fill at most max_fill consecutive NaN cells within each column.

    Handles short market-data gaps (holidays, feed outages) without bridging
    across genuine membership boundaries. Callers that need boundary safety
    should re-apply apply_membership_mask after this call, or use clean_panel.
    """
    return prices.ffill(limit=max_fill)


def clean_panel(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    max_fill: int = 5,
) -> pd.DataFrame:
    """Mask → fill → re-mask: the full cleaning pipeline in one call.

    The second mask ensures forward-fill never spills a price past the date
    a ticker left the index (which would introduce lookahead).
    """
    masked = apply_membership_mask(prices, membership)
    filled = masked.ffill(limit=max_fill)
    return apply_membership_mask(filled, membership)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_{t-1}).

    The first row of each contiguous membership window is NaN because there
    is no prior price within that window to compute a return against.
    """
    return np.log(prices / prices.shift(1))
