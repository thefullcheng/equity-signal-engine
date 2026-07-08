"""Alpaca paper-trading client.

Hardcoded to paper=True -- there is no parameter to point this at live
trading. Switching to real money would need a deliberate, separate change.

Requires environment variables (set these yourself, never paste keys into
chat or commit them). Account 1 (the default, main model strategy):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
Account 2 (e.g. a baseline strategy run in parallel):
    ALPACA_API_KEY_2
    ALPACA_SECRET_KEY_2
"""

from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

load_dotenv()


def get_trading_client(account: int = 1) -> TradingClient:
    suffix = "" if account == 1 else f"_{account}"
    api_key_var    = f"ALPACA_API_KEY{suffix}"
    secret_key_var = f"ALPACA_SECRET_KEY{suffix}"
    api_key    = os.environ.get(api_key_var)
    secret_key = os.environ.get(secret_key_var)
    if not api_key or not secret_key:
        raise RuntimeError(
            f"Set {api_key_var} and {secret_key_var} environment variables "
            "(paper-trading keys from your Alpaca dashboard)."
        )
    return TradingClient(api_key, secret_key, paper=True)
