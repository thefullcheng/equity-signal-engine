"""LightGBM vs. a plain linear (Ridge) model, identical features/labels/
walk-forward split for both. Answers the standing question this repo never
directly tested: is the nonlinear model actually earning its complexity, or
would a one-line linear combination of the same 5 features do just as well?
RUN ON YOUR MACHINE.

    python -m src.backtest.model_comparison --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, information_coefficient, run_backtest
from src.models.model import walk_forward_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    features     = pd.read_parquet(processed_dir / "features.parquet")
    labels_rank  = pd.read_parquet(processed_dir / "labels_ranked.parquet")
    labels_raw   = pd.read_parquet(processed_dir / "labels.parquet")

    mode      = cfg["universe"]["mode"]
    dev_top_n = cfg["universe"]["dev_top_n"] if mode == "dev" else None

    start             = pd.to_datetime(cfg["dates"]["start"])
    initial_train_end = str((start + pd.DateOffset(years=3)).date())

    # Three runs, not two: lightgbm_full is the repo's current default
    # (LightGBM's native NaN handling, all rows). ridge can't use those NaN
    # rows at all. lightgbm_matched re-fits LightGBM on ridge's exact
    # (smaller, complete-case) row set, isolating model architecture from
    # sample composition -- without it, a lightgbm_full-vs-ridge gap can't
    # tell you which one actually caused it.
    runs = [
        ("lightgbm_full",    "lightgbm", False),
        ("lightgbm_matched", "lightgbm", True),
        ("ridge",            "ridge",    True),
    ]

    results = {}
    for label, model_type, drop_na in runs:
        logger.info("Walk-forward fitting: %s ...", label)
        preds = walk_forward_predict(
            features, labels_rank,
            initial_train_end=initial_train_end,
            dev_top_n=dev_top_n,
            model_type=model_type,
            drop_na_features=drop_na,
        )
        bt = run_backtest(preds, labels_raw, costs_bps=costs_bps, top_q=0.20, long_only=True)
        m  = compute_metrics(bt, periods_per_year=ppy)
        ic = information_coefficient(preds, labels_raw)
        t_stat = ic.mean() / (ic.std() / len(ic) ** 0.5) if len(ic) > 1 else float("nan")
        results[label] = {
            "sharpe": m["sharpe"], "cagr": m["cagr"], "ann_vol": m["ann_vol"],
            "max_drawdown": m["max_drawdown"], "avg_turnover": m.get("avg_turnover"),
            "ic_mean": round(ic.mean(), 4), "ic_t_stat": round(t_stat, 2),
            "n_periods": m["n_periods"],
        }

    table = pd.DataFrame(results).T
    print()
    print("-" * 90)
    print("MODEL COMPARISON: identical features, labels, walk-forward split and embargo")
    print("-" * 90)
    print(table.to_string())

    # The fair architecture comparison is lightgbm_matched vs. ridge -- same
    # rows for both. lightgbm_full vs. ridge is shown too but is confounded
    # by sample composition (ridge only ever sees complete-EDGAR-coverage
    # rows), not just model class -- see the drop-rate logged above.
    sharpe_gap = results["lightgbm_matched"]["sharpe"] - results["ridge"]["sharpe"]
    ic_gap = results["lightgbm_matched"]["ic_mean"] - results["ridge"]["ic_mean"]
    print()
    print("Fair comparison (lightgbm_matched vs. ridge -- identical rows for both):")
    if abs(sharpe_gap) <= 0.02 and abs(ic_gap) <= 0.001:
        print(f"  Negligible gap (Sharpe {sharpe_gap:+.3f}, IC {ic_gap:+.4f}). The nonlinear "
              "model isn't earning its complexity -- a linear combination of the same 5 "
              "features does essentially as well, on the same rows.")
    else:
        print(f"  Sharpe {sharpe_gap:+.3f}, IC {ic_gap:+.4f} (lightgbm_matched - ridge). "
              f"{'LightGBM' if sharpe_gap > 0 else 'Ridge'} wins on identical rows -- "
              "a real, not sample-composition-driven, gap.")
    print("(lightgbm_full is shown for reference against the repo's actual default config, "
          "which does use all rows via LightGBM's native NaN handling -- but is not a clean "
          "architecture comparison against ridge for the reason above.)")

    out = processed_dir / "model_comparison.parquet"
    table.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
