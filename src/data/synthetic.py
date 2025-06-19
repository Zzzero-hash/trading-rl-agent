"""
Synthetic OHLCV data generator for testing purposes.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def fetch_synthetic_data(
    symbol: str, start: str, end: str, timestep: str = "day"
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data matching pipeline schema.
    :param symbol: ignored in this stub
    :param start: start date (YYYY-MM-DD)
    :param end: end date (YYYY-MM-DD)
    :param timestep: 'day', 'hour', or 'minute'
    :return: DataFrame with ['timestamp','open','high','low','close','volume']
    """
    # Map timestep to pandas frequency
    freq_map = {"day": "D", "hour": "H", "minute": "T"}
    freq = freq_map.get(timestep, timestep)
    # Create date range
    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)
    # Random data
    opens = np.random.uniform(100, 200, size=n)
    highs = opens + np.random.uniform(0, 10, size=n)
    lows = opens - np.random.uniform(0, 10, size=n)
    closes = np.random.uniform(lows, highs)
    volumes = np.random.randint(1000, 10000, size=n)
    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes.astype(int),
        }
    )
    return df.reset_index(drop=True)


def generate_gbm_prices(
    n_days: int, mu: float = 0.0002, sigma: float = 0.01, s0: float = 100.0
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using Geometric Brownian Motion (GBM).

    Args:
        n_days: Number of days to generate
        mu: Drift parameter (daily)
        sigma: Volatility parameter (daily)
        s0: Initial price

    Returns:
        DataFrame with ['timestamp','open','high','low','close','volume']
    """
    # Generate timestamps
    start_date = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate daily returns with drift and volatility
    daily_returns = np.exp(mu + sigma * np.random.normal(0, 1, n_days))

    # Generate close prices
    closes = np.zeros(n_days)
    closes[0] = s0
    for i in range(1, n_days):
        closes[i] = closes[i - 1] * daily_returns[i]

    # Generate open, high, low based on close prices
    opens = closes * np.exp(sigma * np.random.normal(0, 0.5, n_days))

    # High is the max of open and close plus a random amount
    highs = np.maximum(opens, closes) * np.exp(
        sigma * np.abs(np.random.normal(0, 0.5, n_days))
    )

    # Low is the min of open and close minus a random amount
    lows = np.minimum(opens, closes) * np.exp(
        -sigma * np.abs(np.random.normal(0, 0.5, n_days))
    )

    # Generate volume (correlated with price movement)
    price_moves = np.abs(closes - opens) / closes
    volumes = np.random.normal(1000000, 200000, n_days) * (1 + 5 * price_moves)
    volumes = volumes.astype(int)

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    return df
