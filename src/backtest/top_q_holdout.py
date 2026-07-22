"""Held-out validation of top_q, same methodology as the README's "Feature
selection was in-sample" check and drawdown_cap_holdout.py.

sensitivity.py picked top_q=0.20 by sweeping the full 2013-2026 sample and
reading off the sweep's own baseline -- exactly the in-sample selection
that check warned about, just never applied to this specific parameter.
Here: select top_q using ONLY pre-2020 periods, then evaluate that single,
honestly-selected value on the true 2020-2026 holdout it never saw, against
the in-sample default (0.20) over the same holdout window. RUN ON YOUR
MACHINE.

    python -m src.backtest.top_q_holdout --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.backtest import compute_metrics, run_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_START = "2020-01-01"
GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
INSAMPLE_DEFAULT = 0.20   # what build_backtest.py / sensitivity.py currently use


def _metrics_on_slice(port_returns: pd.DataFrame, ppy: int, start=None, end=None) -> dict:
    sl = port_returns
    if start is not None:
        sl = sl[sl.index >= start]
    if end is not None:
        sl = sl[sl.index < end]
    m = compute_metrics(sl, periods_per_year=ppy)
    m["n_periods"] = len(sl)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg           = yaml.safe_load(Path(args.config).read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    costs_bps     = cfg["costs"]["per_side_bps"]
    horizon       = cfg["labels"]["horizon_days"]
    ppy           = int(round(252 / horizon))

    predictions = pd.read_parquet(processed_dir / "predictions.parquet")
    labels      = pd.read_parquet(processed_dir / "labels.parquet")

    logger.info("Selecting top_q using ONLY periods before %s ...", HOLDOUT_START)
    bts = {}
    rows = []
    for q in GRID:
        bt = run_backtest(predictions, labels, costs_bps=costs_bps, top_q=q, long_only=True)
        bts[q] = bt
        sel_m = _metrics_on_slice(bt, ppy, end=HOLDOUT_START)
        rows.append({"top_q": q, "sel_sharpe": sel_m["sharpe"], "sel_cagr": sel_m["cagr"],
                     "sel_max_dd": sel_m["max_drawdown"], "sel_n": sel_m["n_periods"]})

    sel_results = pd.DataFrame(rows)
    print()
    print("-" * 90)
    print(f"SELECTION WINDOW (periods before {HOLDOUT_START}, n={rows[0]['sel_n']})")
    print("-" * 90)
    print(sel_results.sort_values("sel_sharpe", ascending=False).to_string(index=False))

    best_q = float(sel_results.sort_values("sel_sharpe", ascending=False).iloc[0]["top_q"])
    logger.info("Honestly-selected top_q (best Sharpe, pre-2020 only): %.2f", best_q)

    print()
    print("-" * 90)
    n_hold = _metrics_on_slice(bts[INSAMPLE_DEFAULT], ppy, start=HOLDOUT_START)["n_periods"]
    print(f"HOLDOUT ({HOLDOUT_START} to end, n={n_hold})")
    print("-" * 90)

    holdout_rows = []
    for label, q in [
        (f"honestly-selected (pre-2020 only): top_q={best_q}", best_q),
        (f"in-sample default from full sweep: top_q={INSAMPLE_DEFAULT}", INSAMPLE_DEFAULT),
    ]:
        m = _metrics_on_slice(bts[q], ppy, start=HOLDOUT_START)
        holdout_rows.append({"config": label, "top_q": q,
                             "sharpe": m["sharpe"], "cagr": m["cagr"],
                             "max_drawdown": m["max_drawdown"], "n_periods": m["n_periods"]})

    holdout_results = pd.DataFrame(holdout_rows).set_index("config")
    print(holdout_results.to_string())

    if best_q == INSAMPLE_DEFAULT:
        print("\nHonest selection agrees with the in-sample default -- top_q=0.20 was not "
              "a leaked/overfit choice, at least by this check.")
    else:
        print(f"\nHonest selection (top_q={best_q}) differs from the in-sample default "
              f"(top_q={INSAMPLE_DEFAULT}) -- compare the holdout rows above to see whether "
              "the original choice was actually costing anything out of sample.")

    out = processed_dir / "top_q_holdout.parquet"
    sel_results.to_parquet(out)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
