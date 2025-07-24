"""Alpha Vantage data loader."""


import os

import pandas as pd

try:
    from alpha_vantage.timeseries import TimeSeries
except ModuleNotFoundError:  # pragma: no cover - optional
    TimeSeries = None


def load_alphavantage(
    symbol: str,
    start: str,
    end: str,
    interval: str = "day",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from Alpha Vantage.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str
        Start date ``YYYY-MM-DD``.
    end : str
        End date ``YYYY-MM-DD``.
    interval : str, optional
        ``day``, ``hour``, ``minute`` or raw Alpha Vantage interval.
    api_key : str, optional
        Alpha Vantage API key. Defaults to ``ALPHAVANTAGE_API_KEY`` env var or ``"demo"``.
    """
    if TimeSeries is None:
        raise ImportError(
            "alpha_vantage package is required for load_alphavantage. Install with `pip install alpha_vantage`.",
        )

    if api_key is None:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")

    ts = TimeSeries(key=api_key, output_format="pandas")

    try:
        # Fetch raw data (tuple assumed: DataFrame, metadata)
        if interval == "day":
            raw = ts.get_daily(symbol, outputsize="full")
        else:
            intr_map = {"hour": "60min", "minute": "1min"}
            iv = intr_map.get(interval, interval)
            raw = ts.get_intraday(symbol, interval=iv, outputsize="full")
        # Assume first element is DataFrame
        data = raw[0]
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch data from Alpha Vantage for symbol {symbol}: {e}",
        ) from e

    # Columns like '1. open', etc -> take last part after space
    data.rename(columns=lambda c: c.split(" ")[-1], inplace=True)
    data.index.name = "timestamp"
    data.reset_index(inplace=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    mask = (data["timestamp"] >= pd.to_datetime(start)) & (data["timestamp"] <= pd.to_datetime(end))
    data = data.loc[mask]

    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    data.sort_values("timestamp", inplace=True)
    return data.reset_index(drop=True)
