"""CCXT crypto exchange data loader."""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    import ccxt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ccxt = None


def load_ccxt(
    symbol: str,
    start: str,
    end: str,
    interval: str = "day",
    exchange: str = "binance",
) -> pd.DataFrame:
    """Load OHLCV data from a CCXT exchange.

    Parameters
    ----------
    symbol : str
        Market symbol like ``"BTC/USDT"``.
    start : str
        Start date ``YYYY-MM-DD``.
    end : str
        End date ``YYYY-MM-DD``.
    interval : str, optional
        ``day``, ``hour``, ``minute`` or raw CCXT timeframe.
    exchange : str, optional
        Exchange identifier (e.g. ``"binance"``).
    """
    if ccxt is None:
        raise ImportError(
            "ccxt package is required for load_ccxt. Install with `pip install ccxt`."
        )

    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})

    timeframe_map = {"day": "1d", "hour": "1h", "minute": "1m"}
    tf = timeframe_map.get(interval, interval)

    since = ex.parse8601(f"{start}T00:00:00Z")
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, since=since)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]

    return df.reset_index(drop=True)
