"""Block bootstrap for confidence intervals on the strategy-minus-benchmark
comparison. A single point estimate (e.g. "+1.46%/yr excess") with only 169
non-overlapping periods carries real sampling uncertainty; this makes it
explicit rather than implicit.

Uses a circular moving-block bootstrap (not iid resampling) to preserve any
serial dependence in the return series -- periods aren't necessarily
independent (e.g. performance can cluster in trending vs. choppy regimes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def circular_block_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One bootstrap resample's indices into a length-n series, built from
    contiguous blocks of `block_length`, wrapping around circularly so every
    period has an equal chance of appearing regardless of its position.
    """
    n_blocks = -(-n // block_length)  # ceil
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) % n for s in starts])
    return idx[:n]


def bootstrap_excess_return_and_sharpe_gap(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    block_length: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Block-bootstrap CIs for (a) annualized excess return and (b) the
    Sharpe gap (strategy - benchmark), resampling both series with the same
    block indices each draw so their paired, same-date relationship is
    preserved.
    """
    common = strategy_returns.index.intersection(benchmark_returns.index)
    s = strategy_returns.loc[common].values
    b = benchmark_returns.loc[common].values
    n = len(common)

    rng = np.random.default_rng(seed)
    excess_ann = np.full(n_boot, np.nan)
    sharpe_gap = np.full(n_boot, np.nan)

    for i in range(n_boot):
        idx = circular_block_bootstrap_indices(n, block_length, rng)
        s_i, b_i = s[idx], b[idx]

        excess_ann[i] = (s_i.mean() - b_i.mean()) * periods_per_year

        s_vol = s_i.std()
        b_vol = b_i.std()
        s_sharpe = (s_i.mean() * periods_per_year) / (s_vol * np.sqrt(periods_per_year)) if s_vol > 0 else np.nan
        b_sharpe = (b_i.mean() * periods_per_year) / (b_vol * np.sqrt(periods_per_year)) if b_vol > 0 else np.nan
        sharpe_gap[i] = s_sharpe - b_sharpe

    def ci(arr):
        arr = arr[~np.isnan(arr)]
        return {
            "mean": arr.mean(),
            "ci_lo": np.percentile(arr, 2.5),
            "ci_hi": np.percentile(arr, 97.5),
            "pct_above_zero": (arr > 0).mean(),
        }

    return {
        "excess_return_annualized": ci(excess_ann),
        "sharpe_gap": ci(sharpe_gap),
        "n_periods": n,
        "block_length": block_length,
        "n_boot": n_boot,
    }
