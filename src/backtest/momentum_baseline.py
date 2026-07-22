"""Naive 12-1 momentum-only baseline vs. the full 5-feature LightGBM model,
identical dates/universe/costs/top_q for both. RUN ON YOUR MACHINE.

The comparison the README's Statistical significance section reports (the
whole EDGAR/feature-engineering/LightGBM pipeline earns barely anything over
sorting by trailing 12-month return alone) -- reconstructed as a real,
persisted script since the original never was one.

mom_12_1 in features.parquet is already cross-sectionally rank-normalized to
[0,1] (see build_features.py), so it's usable directly as a "score" with no
extra transformation -- higher rank = stronger momentum.

    python -m src.backtest.momentum_baseline --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, information_coefficient, run_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    model_predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    features          = pd.read_parquet(processed_dir / "features.parquet")
    labels            = pd.read_parquet(processed_dir / "labels.parquet")

    # Same exact (date, ticker) universe the model was scored on -- not
    # every date features.parquet has (that would include the initial
    # training window and dates without a resolved label), so the two
    # backtests cover identical periods, not just identical top_q/costs.
    momentum_predictions = features.loc[model_predictions.index, ["mom_12_1"]].rename(
        columns={"mom_12_1": "score"}
    )

    results = {}
    for label, preds in [("momentum_only", momentum_predictions), ("full_model", model_predictions)]:
        bt = run_backtest(preds, labels, costs_bps=costs_bps, top_q=top_q, long_only=True)
        m  = compute_metrics(bt, periods_per_year=ppy)
        ic = information_coefficient(preds, labels)
        t_stat = ic.mean() / (ic.std() / len(ic) ** 0.5) if len(ic) > 1 else float("nan")
        results[label] = {
            "ic_t_stat": round(t_stat, 2), "sharpe": m["sharpe"], "cagr": m["cagr"],
            "ann_vol": m["ann_vol"], "max_drawdown": m["max_drawdown"],
            "avg_turnover": m.get("avg_turnover"), "n_periods": m["n_periods"],
        }

    table = pd.DataFrame(results).T
    print()
    print("-" * 80)
    print(f"MOMENTUM-ONLY vs. FULL MODEL  (top_q={top_q}, identical dates/costs)")
    print("-" * 80)
    print(table.to_string())

    gap = results["full_model"]["sharpe"] - results["momentum_only"]["sharpe"]
    print()
    print(f"Full model Sharpe - momentum-only Sharpe: {gap:+.3f}")
    print(f"Full model IC t-stat: {results['full_model']['ic_t_stat']}  |  "
          f"Momentum-only IC t-stat: {results['momentum_only']['ic_t_stat']}")

    out = processed_dir / "momentum_baseline.parquet"
    table.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
