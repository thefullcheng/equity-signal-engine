"""Regress strategy returns on the Fama-French 5 factors + momentum. RUN ON
YOUR MACHINE.

    python -m src.backtest.build_factor_regression --config config/config.yaml

Answers "is this explained by known factor exposures?" -- the standard quant
framing, and a direct extension of the Decomposition section: if MOM/RMW
loadings are significant and alpha isn't, the model's edge (such as it is)
is a repackaging of known factors, not novel stock-selection skill.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import yaml

from src.backtest.backtest import compute_metrics, run_backtest
from src.backtest.factor_data import compound_factor_returns, fetch_daily_factors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    horizon = cfg["labels"]["horizon_days"]
    periods_per_year = round(252 / horizon)

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")
    port_returns = run_backtest(predictions, labels, costs_bps=cfg["costs"]["per_side_bps"],
                                top_q=cfg["portfolio"]["top_q"], long_only=True)

    logger.info("Fetching Fama-French 5 factors + momentum (Ken French data library) ...")
    daily_factors = fetch_daily_factors(raw_dir / "factors", force_refresh=args.force_refresh)
    logger.info("Daily factor data: %s .. %s", daily_factors.index.min().date(),
                daily_factors.index.max().date())

    period_factors = compound_factor_returns(daily_factors, port_returns.index)
    logger.info("Compounded to %d strategy-aligned periods", len(period_factors))

    common = port_returns.index.intersection(period_factors.index)
    logger.info("Regression sample: %d periods (of %d backtest periods)", len(common), len(port_returns))

    y = port_returns.loc[common, "net_return"] - period_factors.loc[common, "RF"]
    X = sm.add_constant(period_factors.loc[common, FACTOR_COLS])

    model = sm.OLS(y, X).fit()

    alpha_period = model.params["const"]
    alpha_annualized = (1 + alpha_period) ** periods_per_year - 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("FACTOR REGRESSION: strategy excess return ~ FF5 + Momentum")
    logger.info("=" * 70)
    logger.info("%-10s %10s %10s %10s", "Factor", "Coef", "t-stat", "p-value")
    logger.info("%-10s %10.4f %10.2f %10.4f", "alpha", alpha_period, model.tvalues["const"],
                model.pvalues["const"])
    for col in FACTOR_COLS:
        logger.info("%-10s %10.4f %10.2f %10.4f", col, model.params[col], model.tvalues[col],
                    model.pvalues[col])
    logger.info("-" * 70)
    logger.info("Annualized alpha: %.4f (%.2f%%/yr)", alpha_annualized, alpha_annualized * 100)
    logger.info("R-squared: %.4f  |  Adj R-squared: %.4f", model.rsquared, model.rsquared_adj)
    logger.info("N periods: %d", int(model.nobs))

    out_path = processed_dir / "factor_regression.txt"
    out_path.write_text(str(model.summary()))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
