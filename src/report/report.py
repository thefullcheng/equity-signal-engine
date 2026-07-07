"""Generate the two-panel performance report (equity curve + IC over time)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_report(
    port_returns: pd.DataFrame,
    ic_series: pd.Series,
    metrics: dict,
    output_path: str | Path,
) -> None:
    """Save equity curve and IC chart to output_path (PNG)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"Sharpe {metrics['sharpe']:.2f}  |  CAGR {metrics['cagr']:.1%}  |  "
        f"MaxDD {metrics['max_drawdown']:.1%}  |  Hit {metrics['hit_rate']:.1%}",
        fontsize=11,
    )

    cum = (1 + port_returns["net_return"]).cumprod()
    cum.plot(ax=axes[0], color="steelblue", linewidth=1.2)
    axes[0].set_title("Equity Curve (net of transaction costs)")
    axes[0].set_ylabel("Cumulative return")
    axes[0].axhline(1.0, color="black", linewidth=0.5, linestyle="--")

    ic_series.plot(ax=axes[1], alpha=0.3, color="grey", linewidth=0.8, label="IC")
    ic_series.rolling(12).mean().plot(
        ax=axes[1], color="darkorange", linewidth=1.5, label="12-period rolling mean"
    )
    axes[1].set_title("Information Coefficient (score vs realized return)")
    axes[1].set_ylabel("Spearman IC")
    axes[1].axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
