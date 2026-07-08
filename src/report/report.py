"""Generate the performance report: equity curve vs. benchmark, drawdown,
and IC over time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_report(
    port_returns: pd.DataFrame,
    ic_series: pd.Series,
    metrics: dict,
    output_path: str | Path,
    benchmark_returns: pd.Series | None = None,
) -> None:
    """Save equity curve (vs. benchmark), drawdown, and IC chart to output_path (PNG)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), height_ratios=[2, 1, 1.2])
    fig.suptitle(
        f"Sharpe {metrics['sharpe']:.2f}  |  CAGR {metrics['cagr']:.1%}  |  "
        f"MaxDD {metrics['max_drawdown']:.1%}  |  Hit {metrics['hit_rate']:.1%}",
        fontsize=11,
    )

    cum = (1 + port_returns["net_return"]).cumprod()
    cum.plot(ax=axes[0], color="steelblue", linewidth=1.4, label="Strategy")
    if benchmark_returns is not None:
        bmark_aligned = benchmark_returns.reindex(port_returns.index)
        (1 + bmark_aligned).cumprod().plot(
            ax=axes[0], color="grey", linewidth=1.1, linestyle="--",
            label="Equal-weight benchmark",
        )
    axes[0].set_title("Equity Curve (net of transaction costs, log scale)")
    axes[0].set_ylabel("Cumulative return")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    axes[0].legend()

    drawdown = (cum - cum.cummax()) / cum.cummax()
    drawdown.plot(ax=axes[1], color="firebrick", linewidth=1.0)
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.2)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")

    ic_series.plot(ax=axes[2], alpha=0.3, color="grey", linewidth=0.8, label="IC")
    ic_series.rolling(12).mean().plot(
        ax=axes[2], color="darkorange", linewidth=1.5, label="12-period rolling mean"
    )
    axes[2].set_title("Information Coefficient (score vs realized return)")
    axes[2].set_ylabel("Spearman IC")
    axes[2].axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
