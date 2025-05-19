"""
Synthetic OHLCV data generator for testing purposes.
"""
import pandas as pd
import numpy as np

def fetch_synthetic_data(symbol: str, start: str, end: str, timestep: str = 'day') -> pd.DataFrame:
    """
    Generate synthetic OHLCV data matching pipeline schema.
    :param symbol: ignored in this stub
    :param start: start date (YYYY-MM-DD)
    :param end: end date (YYYY-MM-DD)
    :param timestep: 'day', 'hour', or 'minute'
    :return: DataFrame with ['timestamp','open','high','low','close','volume']
    """
    # Map timestep to pandas frequency
    freq_map = {'day': 'D', 'hour': 'H', 'minute': 'T'}
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
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int)
    })
    return df.reset_index(drop=True)
