"""Held-out validation of feature selection, reconstructed as a real script
-- the original predates this repo's current script layout, and the README
describes its result in prose without a persisted way to reproduce it.

The 5-feature default (mom_12_1, dollar_vol_60, gross_prof, roe, ep_ratio)
was chosen using IC measured over the full 2013-2026 span the backtest also
reports performance over -- in-sample selection. This retrains two separate
walk-forward models -- the 5-feature default, and a 2-feature set
(mom_12_1, roe) using only the features that clear a |t|>=0.5 bar on
pre-2020 IC alone -- then compares both on the full period and the true
2020-2026 holdout neither selection procedure touched. RUN ON YOUR MACHINE.

    python -m src.backtest.feature_selection_holdout --config config/config.yaml
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

HOLDOUT_START = "2020-01-01"
FEATURE_SETS = {
    "5-feature (in-sample selected, current default)":
        ["mom_12_1", "dollar_vol_60", "gross_prof", "roe", "ep_ratio"],
    "2-feature (honestly selected, pre-2020 only)":
        ["mom_12_1", "roe"],
}


def _metrics_on_slice(preds, labels, ppy, costs_bps, top_q, start=None, end=None):
    p = preds
    if start is not None:
        p = p[p.index.get_level_values("date") >= start]
    if end is not None:
        p = p[p.index.get_level_values("date") < end]
    bt = run_backtest(p, labels, costs_bps=costs_bps, top_q=top_q, long_only=True)
    m = compute_metrics(bt, periods_per_year=ppy)
    ic = information_coefficient(p, labels)
    t_stat = ic.mean() / (ic.std() / len(ic) ** 0.5) if len(ic) > 1 else float("nan")
    return {"sharpe": m["sharpe"], "ic_t_stat": round(t_stat, 2), "n_periods": m["n_periods"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    top_q         = cfg["portfolio"]["top_q"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    features    = pd.read_parquet(processed_dir / "features.parquet")
    labels_rank = pd.read_parquet(processed_dir / "labels_ranked.parquet")
    labels_raw  = pd.read_parquet(processed_dir / "labels.parquet")

    mode      = cfg["universe"]["mode"]
    dev_top_n = cfg["universe"]["dev_top_n"] if mode == "dev" else None
    start             = pd.to_datetime(cfg["dates"]["start"])
    initial_train_end = str((start + pd.DateOffset(years=3)).date())

    rows = []
    for label, cols in FEATURE_SETS.items():
        logger.info("Walk-forward fitting: %s ...", label)
        subset_features = features[[c for c in cols if c in features.columns]]
        preds = walk_forward_predict(
            subset_features, labels_rank,
            initial_train_end=initial_train_end,
            dev_top_n=dev_top_n,
            model_type="lightgbm",
        )
        full = _metrics_on_slice(preds, labels_raw, ppy, costs_bps, top_q)
        hold = _metrics_on_slice(preds, labels_raw, ppy, costs_bps, top_q, start=HOLDOUT_START)
        rows.append({
            "feature_set": label,
            "full_sharpe": full["sharpe"], "full_ic_t": full["ic_t_stat"],
            "holdout_sharpe": hold["sharpe"], "holdout_ic_t": hold["ic_t_stat"],
        })

    table = pd.DataFrame(rows).set_index("feature_set")
    print()
    print("-" * 100)
    print(f"FEATURE SELECTION HOLDOUT (top_q={top_q}, holdout={HOLDOUT_START} to end)")
    print("-" * 100)
    print(table.to_string())

    out = processed_dir / "feature_selection_holdout.parquet"
    table.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
