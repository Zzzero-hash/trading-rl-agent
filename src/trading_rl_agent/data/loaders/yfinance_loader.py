"""YFinance data loader."""

from __future__ import annotations

import contextlib

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yf = None


def load_yfinance(
    symbol: str,
    start: str,
    end: str,
    interval: str = "day",
) -> pd.DataFrame:
    """Load OHLCV data using ``yfinance``.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str
        Start date ``YYYY-MM-DD``.
    end : str
        End date ``YYYY-MM-DD``.
    interval : str, optional
        ``day``, ``hour``, ``minute`` or a raw yfinance interval string.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``, ``close`` and ``volume``.
    """
    if yf is None:
        raise ImportError(
            "yfinance package is required for load_yfinance. Install with `pip install yfinance`.",
        )

    interval_map = {"day": "1d", "hour": "1h", "minute": "1m"}
    yf_interval = interval_map.get(interval, interval)

    df = yf.download(symbol, start=start, end=end, interval=yf_interval)
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    with contextlib.suppress(Exception):
        df.index = df.index.tz_localize(None)
    df["timestamp"] = df.index
    return df.reset_index(drop=True)
