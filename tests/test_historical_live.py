import pytest
import pandas as pd
from src.data.historical import fetch_historical_data

@ pytest.mark.skipif(
    not hasattr(fetch_historical_data, '__call__'),
    reason='Integration test skipped'
)
def test_fetch_historical_live_data():
    """
    Integration test that actually downloads data from Yahoo Finance.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-10'
    # fetch daily data for a known period
    df = fetch_historical_data(symbol, start=start_date, end=end_date, timestep='day')
    # basic validations
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Expected non-empty DataFrame from live data"
    # expected columns
    expected_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    assert set(df.columns) == expected_cols
    # index type and range
    assert df.index.dtype == 'int64'
    assert df['timestamp'].min() >= pd.to_datetime(start_date)
    assert df['timestamp'].max() <= pd.to_datetime(end_date)
    # values should be positive numbers
    assert (df['open'] > 0).all()
    assert (df['close'] > 0).all()