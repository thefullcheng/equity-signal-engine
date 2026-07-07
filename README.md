# Equity Signal Engine

Cross-sectional monthly long-only equity strategy on a point-in-time S&P 500
universe with an overfitting-resistant walk-forward backtest.

## Results (as of latest run)

| Metric | Value |
|---|---|
| Sharpe ratio | **0.842** |
| CAGR | **11.9%** |
| Ann. volatility | 14.7% |
| Max drawdown | -23.9% |
| IC (model vs realized) | +0.012 |
| Excess return vs equal-weight BM | +2.87% / yr |
| Backtest period | 2013 – 2026 (169 periods) |

Strategy: long top-20% by model score, 20-day rebalance, 200-day SMA regime
filter (cash when market below 200d SMA), 10 bps per-side transaction costs.

## Phases completed

- [x] **2a** Point-in-time S&P 500 universe (Wikipedia change log, backward reconstruction)
- [x] **2b** Price ingestion + coverage audit (`src/data/prices.py`)
- [x] **2c** Cleaning & membership masking (`src/data/clean.py`)
- [x] **3a** Cross-sectional features — momentum, volatility, RSI, dollar-vol (`src/features/`)
- [x] **3b** Forward return labels with cross-sectional rank normalisation (`src/labels/`)
- [x] **3c** Walk-forward LightGBM model — annual re-fit, no hyperparameter tuning on test data (`src/models/`)
- [x] **3d** Long-only backtest + equity curve report (`src/backtest/`, `src/report/`)
- [x] **4a** SEC EDGAR fundamental data — 98K point-in-time observations, 2007–2026 (`src/data/edgar.py`)
- [x] **4b** Feature pruning (dropped 3 negative-IC price features); sector-neutral construction (available, not default)
- [x] **5a** Annual attribution tearsheet + sector exposure + worst/best periods (`src/report/attribution.py`)
- [x] **5b** Sensitivity analysis — top_q, cost assumptions, regime window (`src/backtest/sensitivity.py`)
- [x] **5c** Live signal generation — current ranked portfolio (`src/signal/`)
- [x] **5d** Documentation (this file)

## Feature set (8 features, all rank-normalised cross-sectionally)

| Feature | Type | IC | t-stat |
|---|---|---|---|
| `mom_12_1` | Price — 12-1 month momentum | +0.012 | 0.85 |
| `mom_6_1` | Price — 6-1 month momentum | +0.003 | 0.19 |
| `high_52w` | Price — 52-week high proximity | +0.004 | 0.23 |
| `vol_21` | Price — 21-day realised vol | +0.003 | 0.17 |
| `dollar_vol_60` | Price — 60-day avg dollar volume | +0.006 | 0.85 |
| `gross_prof` | Fundamental — gross profit / assets (Novy-Marx) | +0.013 | 1.19 |
| `roe` | Fundamental — net income / equity | **+0.021** | **2.19** |
| `ep_ratio` | Fundamental — earnings / market cap | +0.005 | 0.52 |

## Sensitivity summary

| Dimension | Sharpe range | Baseline |
|---|---|---|
| top_q (0.10 – 0.30) | 0.840 – 0.893 | 0.842 |
| costs_bps (5 – 20) | 0.762 – 0.881 | 0.842 |
| regime window (100 – 250d) | 0.562 – 0.842 | 0.842 |

Strategy is robust to portfolio size and costs; most sensitive to regime filter
window (shorter windows cause whipsawing).

## Quickstart

```bash
pip install -r requirements.txt
pytest                                        # 73 tests

# Run the full pipeline (requires network access):
py -m src.data.build_universe
py -m src.data.build_prices
py -m src.data.build_clean
py -m src.data.build_fundamentals            # ~10 min, SEC EDGAR
py -m src.features.build_features
py -m src.labels.build_labels
py -m src.models.build_model
py -m src.backtest.build_backtest            # prints attribution tearsheet
py -m src.backtest.sensitivity               # parameter sensitivity table
py -m src.signal.build_live_signal           # today's portfolio holdings
```

## Key design decisions

**Point-in-time universe**: membership reconstructed backward from Wikipedia's
S&P 500 change log. Prices outside a ticker's membership window are masked to
NaN, preventing survivorship bias.

**Walk-forward evaluation**: model re-fits annually on an expanding training
window. No hyperparameters are tuned on test data — the biggest silent risk in
rolling-window backtests.

**Ranked training labels**: forward returns are converted to cross-sectional
percentile ranks [0, 1] for model training, giving a stable target distribution
across volatile and quiet markets. Raw log returns are used for backtest P&L.

**Regime filter**: strategy holds cash when the equal-weight universe index is
below its 200-day SMA. Reduces annualised vol from ~18% to ~15%.

**EDGAR fundamentals**: quarterly GAAP data fetched from the public SEC EDGAR
XBRL API (no key required). Filing date used as the availability date — no
extra lag needed.

**Dev mode**: restricts universe to top-100 tickers by trailing dollar volume
for fast iteration. Signal quality drops sharply on smaller stocks (IC ≈ 0 in
full 607-ticker mode vs IC = 0.012 in dev mode).

## Survivorship bias handling

1. Change-log completeness trusted only back to `universe.floor_date` (2010-01-01).
2. Ticker renames may appear as spurious remove/add pairs — inspect coverage audit.
3. Some delisted tickers have no retrievable price history (177 empty of 814 tickers).
