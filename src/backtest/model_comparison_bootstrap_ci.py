"""Block-bootstrap CI on the LightGBM-vs-Ridge Sharpe/return gap from
model_comparison.py, same methodology as build_bootstrap_ci.py. RUN ON YOUR
MACHINE.

model_comparison.py found LightGBM beats Ridge by +0.041 Sharpe on an
identical, row-matched sample -- but that's a point estimate from 169
periods, the same size gap this project's own bootstrap already showed can
span zero elsewhere. This settles whether it actually does here.

    python -m src.backtest.model_comparison_bootstrap_ci --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import run_backtest
from src.backtest.bootstrap import bootstrap_excess_return_and_sharpe_gap
from src.models.model import walk_forward_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    features    = pd.read_parquet(processed_dir / "features.parquet")
    labels_rank = pd.read_parquet(processed_dir / "labels_ranked.parquet")
    labels_raw  = pd.read_parquet(processed_dir / "labels.parquet")

    mode      = cfg["universe"]["mode"]
    dev_top_n = cfg["universe"]["dev_top_n"] if mode == "dev" else None
    start             = pd.to_datetime(cfg["dates"]["start"])
    initial_train_end = str((start + pd.DateOffset(years=3)).date())

    returns = {}
    for label, model_type in [("lightgbm_matched", "lightgbm"), ("ridge", "ridge")]:
        logger.info("Walk-forward fitting: %s (row-matched) ...", label)
        preds = walk_forward_predict(
            features, labels_rank,
            initial_train_end=initial_train_end,
            dev_top_n=dev_top_n,
            model_type=model_type,
            drop_na_features=True,   # matched rows for both, see model_comparison.py
        )
        bt = run_backtest(preds, labels_raw, costs_bps=costs_bps, top_q=0.20, long_only=True)
        returns[label] = bt["net_return"]

    logger.info("Running block bootstrap (block_length=%d periods, n_boot=%d) ...",
                args.block_length, args.n_boot)
    result = bootstrap_excess_return_and_sharpe_gap(
        returns["lightgbm_matched"], returns["ridge"], periods_per_year,
        block_length=args.block_length, n_boot=args.n_boot,
    )

    er = result["excess_return_annualized"]
    sg = result["sharpe_gap"]
    print()
    print("=" * 70)
    print(f"BLOCK BOOTSTRAP CI: lightgbm_matched - ridge  "
          f"(n={result['n_periods']} periods, block={result['block_length']}, "
          f"{result['n_boot']} draws)")
    print("=" * 70)
    print(f"Annualized excess return: mean={er['mean']:+.4f}  "
          f"95% CI=[{er['ci_lo']:+.4f}, {er['ci_hi']:+.4f}]  P(>0)={er['pct_above_zero']*100:.1f}%")
    print(f"Sharpe gap (lightgbm - ridge): mean={sg['mean']:+.3f}  "
          f"95% CI=[{sg['ci_lo']:+.3f}, {sg['ci_hi']:+.3f}]  P(>0)={sg['pct_above_zero']*100:.1f}%")
    print()
    if sg["ci_lo"] <= 0 <= sg["ci_hi"]:
        print("CI spans zero: the +0.041 Sharpe point estimate for LightGBM is not "
              "statistically distinguishable from Ridge at 169 periods.")
    else:
        print("CI does NOT span zero: LightGBM's Sharpe edge over Ridge holds up under "
              "resampling, not just a point-estimate artifact.")


if __name__ == "__main__":
    main()
