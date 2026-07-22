"""Full-strategy Sharpe permutation null: does the complete, cost-inclusive
strategy (top_q selection, costs, actual turnover) beat naive random
stock-picking? RUN ON YOUR MACHINE.

Different question from an IC permutation null (which only checks raw
ranking accuracy): this shuffles predicted SCORES within each rebalance
date (not just IC pairings), runs the exact same portfolio construction on
each shuffle, and builds a null distribution of Sharpes. Reconstructed as a
real script -- the original predates this repo's current script layout, and
the README documents a real bug in an early version (`.stack(future_stack=
True)` without dropping the resulting NaN padding silently broke the
top-quintile selection for most permutations, producing near-zero turnover
and an invalid p-value). Guarded against that specific failure mode below
with an explicit turnover sanity check on the null distribution itself, not
just trusting the p-value.

    python -m src.backtest.sharpe_permutation --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, run_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_PERM = 1000
MIN_PLAUSIBLE_TURNOVER = 1.0   # fully-reshuffled rankings should churn hard;
                                # anything near 0 means the shuffle silently
                                # isn't taking effect -- see module docstring


def _shuffle_scores_within_date(predictions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute the 'score' column within each date's group independently,
    destroying score<->ticker (and hence score<->outcome) alignment while
    preserving the per-date score distribution and every date's ticker set."""
    shuffled = predictions.copy()
    for date, idx in predictions.groupby(level="date").indices.items():
        shuffled.iloc[idx, shuffled.columns.get_loc("score")] = rng.permutation(
            predictions.iloc[idx]["score"].values
        )
    return shuffled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    top_q         = cfg["portfolio"]["top_q"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")

    observed_bt = run_backtest(predictions, labels, costs_bps=costs_bps, top_q=top_q, long_only=True)
    observed_m  = compute_metrics(observed_bt, periods_per_year=ppy)
    observed_sharpe = observed_m["sharpe"]
    logger.info("Observed strategy: Sharpe=%.3f  avg_turnover=%.4f",
                observed_sharpe, observed_m["avg_turnover"])

    rng = np.random.default_rng(args.seed)
    null_sharpes = np.full(args.n_perm, np.nan)
    null_turnovers = np.full(args.n_perm, np.nan)

    logger.info("Running %d permutations ...", args.n_perm)
    for i in range(args.n_perm):
        shuffled = _shuffle_scores_within_date(predictions, rng)
        bt = run_backtest(shuffled, labels, costs_bps=costs_bps, top_q=top_q, long_only=True)
        m = compute_metrics(bt, periods_per_year=ppy)
        null_sharpes[i] = m["sharpe"]
        null_turnovers[i] = m["avg_turnover"]
        if (i + 1) % 200 == 0:
            logger.info("  %d / %d done", i + 1, args.n_perm)

    # Sanity check against the exact bug the README documents: a broken
    # shuffle collapses turnover toward zero because the "shuffled" ranking
    # ends up identical (or near-identical) to the original each time.
    median_null_turnover = float(np.median(null_turnovers))
    logger.info("Median null turnover: %.3f (observed: %.4f)",
                median_null_turnover, observed_m["avg_turnover"])
    if median_null_turnover < MIN_PLAUSIBLE_TURNOVER:
        raise RuntimeError(
            f"Median null-distribution turnover ({median_null_turnover:.3f}) is "
            f"implausibly low for genuinely reshuffled rankings (expected roughly "
            f"1.5+ for full reshuffling) -- this is the exact failure mode the "
            f"README warns about (shuffle silently not taking effect). Not "
            f"reporting a p-value against a broken null."
        )

    p_value = float((null_sharpes >= observed_sharpe).mean())
    percentile = float((null_sharpes < observed_sharpe).mean() * 100)

    print()
    print("-" * 70)
    print(f"FULL-STRATEGY SHARPE PERMUTATION NULL  (n={args.n_perm}, top_q={top_q})")
    print("-" * 70)
    print(f"Observed Sharpe:          {observed_sharpe:.3f}")
    print(f"Null distribution mean:   {null_sharpes.mean():.3f}")
    print(f"Null distribution std:    {null_sharpes.std():.3f}")
    print(f"Empirical p-value:        {p_value:.4f}" if p_value > 0 else "Empirical p-value:        < 0.001")
    print(f"Observed percentile:      {percentile:.0f}th")
    print(f"Median null turnover:     {median_null_turnover:.3f} (observed: {observed_m['avg_turnover']:.4f})")

    out = processed_dir / "sharpe_permutation.parquet"
    pd.DataFrame({"null_sharpe": null_sharpes, "null_turnover": null_turnovers}).to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
