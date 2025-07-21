"""Utilities for fetching live market data."""

from __future__ import annotations

import contextlib

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # Allow import when yfinance not installed
    yf = None


def fetch_live_data(
    symbol: str,
    start: str,
    end: str,
    timestep: str = "day",
) -> pd.DataFrame:
    """Fetch live OHLCV data via ``yfinance``.

    Parameters
    ----------
    symbol : str
        The ticker symbol to query.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.
    timestep : str, optional
        Time interval ('day', 'hour', 'minute'), by default "day".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``timestamp``, ``open``, ``high``, ``low``, ``close``
        and ``volume`` columns, mirroring the schema of :func:`fetch_historical_data`.
    """

    if yf is None:
        raise ImportError(
            "yfinance package is required for fetch_live_data. Install with `pip install yfinance`.",
        )

    interval_map = {
        "day": "1d",
        "hour": "1h",
        "minute": "1m",
    }
    interval = interval_map.get(timestep, timestep)

    df = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    with contextlib.suppress(AttributeError):
        df.index = df.index.tz_localize(None)
    df["timestamp"] = df.index
    df.reset_index(drop=True, inplace=True)
    return df
