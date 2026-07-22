"""Block-bootstrap CI on the top_q=0.15 (honestly-selected) vs top_q=0.20
(in-sample default) gap from top_q_holdout.py, same methodology as
build_bootstrap_ci.py. RUN ON YOUR MACHINE.

Primary question is the HOLDOUT-period gap (2020-2026, n~81): that's the
genuinely out-of-sample test, since top_q=0.15 was selected using only
pre-2020 data. Bootstrapping the full 169-period sample would partly
reflect that top_q=0.15 was chosen to fit the pre-2020 portion, so it's
shown only for reference against this repo's existing bootstrap convention,
not as the answer to "does this hold up out of sample."

    python -m src.backtest.top_q_bootstrap_ci --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import run_backtest
from src.backtest.bootstrap import bootstrap_excess_return_and_sharpe_gap

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_START = "2020-01-01"


def _print_ci(title: str, result: dict) -> None:
    er = result["excess_return_annualized"]
    sg = result["sharpe_gap"]
    print()
    print("=" * 70)
    print(f"{title}  (n={result['n_periods']} periods, block={result['block_length']}, "
          f"{result['n_boot']} draws)")
    print("=" * 70)
    print(f"Annualized excess return: mean={er['mean']:+.4f}  "
          f"95% CI=[{er['ci_lo']:+.4f}, {er['ci_hi']:+.4f}]  P(>0)={er['pct_above_zero']*100:.1f}%")
    print(f"Sharpe gap (top_q=0.15 - top_q=0.20): mean={sg['mean']:+.3f}  "
          f"95% CI=[{sg['ci_lo']:+.3f}, {sg['ci_hi']:+.3f}]  P(>0)={sg['pct_above_zero']*100:.1f}%")
    if sg["ci_lo"] <= 0 <= sg["ci_hi"]:
        print("CI spans zero: not statistically distinguishable from top_q=0.20 here.")
    else:
        print("CI does NOT span zero: top_q=0.15's edge holds up under resampling.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--block-length", type=int, default=13,
                    help="Bootstrap block length in periods (default 13 ~= 1 year)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    horizon       = cfg["labels"]["horizon_days"]
    periods_per_year = round(252 / horizon)

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")

    bt_15 = run_backtest(predictions, labels, costs_bps=costs_bps, top_q=0.15, long_only=True)
    bt_20 = run_backtest(predictions, labels, costs_bps=costs_bps, top_q=0.20, long_only=True)

    logger.info("Running block bootstrap (block_length=%d periods, n_boot=%d) ...",
                args.block_length, args.n_boot)

    # Primary: holdout only -- the genuinely out-of-sample window.
    hold_15 = bt_15["net_return"][bt_15.index >= HOLDOUT_START]
    hold_20 = bt_20["net_return"][bt_20.index >= HOLDOUT_START]
    result_holdout = bootstrap_excess_return_and_sharpe_gap(
        hold_15, hold_20, periods_per_year,
        block_length=args.block_length, n_boot=args.n_boot,
    )
    _print_ci(f"HOLDOUT ONLY ({HOLDOUT_START} to end) -- the real test", result_holdout)

    # Reference: full sample, for consistency with the repo's existing
    # bootstrap convention -- NOT the answer to the out-of-sample question,
    # since top_q=0.15 was selected to fit the pre-2020 part of this window.
    result_full = bootstrap_excess_return_and_sharpe_gap(
        bt_15["net_return"], bt_20["net_return"], periods_per_year,
        block_length=args.block_length, n_boot=args.n_boot,
    )
    _print_ci("FULL SAMPLE (reference only, includes the selection window)", result_full)


if __name__ == "__main__":
    main()
