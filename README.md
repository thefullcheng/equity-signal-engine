# Equity Signal Engine (work in progress)

Cross-sectional weekly long-short equity strategy on a point-in-time S&P 500
universe, with an overfitting-resistant walk-forward backtest.

## Status
- [x] Phase 2a: point-in-time universe construction (`src/data/universe.py`)
- [x] Phase 2b: price ingestion + coverage audit (`src/data/prices.py`)
- [x] Phase 2c: cleaning & panel alignment (`src/data/clean.py`)
- [x] Phase 3a: cross-sectional features — mom, vol, RSI, dollar-vol (`src/features/`)
- [x] Phase 3b: 5-day forward return labels (`src/labels/`)
- [x] Phase 3c: walk-forward LightGBM model (`src/models/`)
- [x] Phase 3d: long-short backtest + report (`src/backtest/`, `src/report/`)

## Quickstart
```bash
pip install -r requirements.txt
pytest                                       # run the test suite
python -m src.data.build_universe            # build membership table (network)
```

## Survivorship bias handling (read this)
We reconstruct historical index membership from Wikipedia's constituent
change log (backward reconstruction from current members). Residual
limitations, quantified by the coverage audit in Phase 2b:
1. Change-log completeness is only trusted back to `universe.floor_date`.
2. Ticker renames may appear as spurious remove/add pairs.
3. Some delisted tickers have no retrievable yfinance price history.
