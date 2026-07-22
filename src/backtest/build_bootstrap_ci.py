"""Block-bootstrap confidence intervals on the strategy-vs-benchmark
comparison. RUN ON YOUR MACHINE.

    python -m src.backtest.build_bootstrap_ci --config config/config.yaml

The headline "+1.46%/yr excess" and "Sharpe 0.665 vs 0.570" numbers are
point estimates from 169 periods; this puts confidence intervals around them
so the results table doesn't imply more precision than the sample supports.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--block-length", type=int, default=13,
                    help="Bootstrap block length in periods (default 13 ~= 1 year)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    horizon = cfg["labels"]["horizon_days"]
    periods_per_year = round(252 / horizon)

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    port_returns = run_backtest(predictions, labels, costs_bps=cfg["costs"]["per_side_bps"],
                                top_q=cfg["portfolio"]["top_q"], long_only=True)

    bmark = labels.groupby(level="date")["fwd_return"].mean()

    logger.info("Running block bootstrap (block_length=%d periods, n_boot=%d) ...",
                args.block_length, args.n_boot)
    result = bootstrap_excess_return_and_sharpe_gap(
        port_returns["net_return"], bmark, periods_per_year,
        block_length=args.block_length, n_boot=args.n_boot,
    )

    er = result["excess_return_annualized"]
    sg = result["sharpe_gap"]
    logger.info("")
    logger.info("=" * 70)
    logger.info("BLOCK BOOTSTRAP CI  (n=%d periods, block=%d, %d draws)",
                result["n_periods"], result["block_length"], result["n_boot"])
    logger.info("=" * 70)
    logger.info("Annualized excess return: mean=%.4f  95%% CI=[%.4f, %.4f]  P(>0)=%.1f%%",
                er["mean"], er["ci_lo"], er["ci_hi"], er["pct_above_zero"] * 100)
    logger.info("Sharpe gap (strategy - benchmark): mean=%.3f  95%% CI=[%.3f, %.3f]  P(>0)=%.1f%%",
                sg["mean"], sg["ci_lo"], sg["ci_hi"], sg["pct_above_zero"] * 100)


if __name__ == "__main__":
    main()
