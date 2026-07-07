"""Run backtest and generate performance report. RUN ON YOUR MACHINE.

    python -m src.backtest.build_backtest --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, information_coefficient, run_backtest
from src.report.report import plot_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")

    logger.info("Running backtest (costs: %d bps per side) ...", costs_bps)
    port_returns = run_backtest(predictions, labels, costs_bps=costs_bps)

    metrics = compute_metrics(port_returns)
    logger.info("Performance summary:")
    for k, v in metrics.items():
        logger.info("  %-16s %s", k, v)

    ic = information_coefficient(predictions, labels)
    logger.info("IC  mean=%.4f  std=%.4f  pct>0=%.1f%%",
                ic.mean(), ic.std(), 100 * (ic > 0).mean())

    bt_path = processed_dir / "backtest.parquet"
    port_returns.to_parquet(bt_path)
    logger.info("Wrote %s", bt_path)

    report_path = processed_dir / "report.png"
    plot_report(port_returns, ic, metrics, report_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
