# Equity Signal Engine (work in progress)

Cross-sectional weekly long-short equity strategy on a point-in-time S&P 500
universe, with an overfitting-resistant walk-forward backtest.

## Status
- [x] Phase 2a: point-in-time universe construction (`src/data/universe.py`)
- [ ] Phase 2b: price ingestion + coverage audit
- [ ] Phase 2c: cleaning & panel alignment

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
