import pandas as pd
import numpy as np
import pytest

from src.data.synthetic import fetch_synthetic_data


def test_synthetic_columns_and_length():
    # Generate daily data for 5 days
    df = fetch_synthetic_data('SYM', '2021-01-01', '2021-01-05', 'day')
    # Check columns
    expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert list(df.columns) == expected_cols
    # Check length
    assert len(df) == 5
    # Timestamp dtype
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    # Numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_high_low_relationship():
    # Ensure high >= max(open, close) and low <= min(open, close)
    df = fetch_synthetic_data('SYM', '2021-01-01', '2021-01-05', 'day')
    high = df['high']
    low = df['low']
    oc_max = df[['open', 'close']].max(axis=1)
    oc_min = df[['open', 'close']].min(axis=1)
    assert (high >= oc_max).all()
    assert (low <= oc_min).all()


def test_volume_integer_positive():
    # Volume should be integer and non-negative
    df = fetch_synthetic_data('SYM', '2021-01-01', '2021-01-02', 'day')
    assert df['volume'].dtype.kind in ('i', 'u')
    assert (df['volume'] >= 0).all()


def test_hourly_timestamp_bounds():
    # Generate hourly data over a single day
    df = fetch_synthetic_data('SYM', '2021-01-01', '2021-01-01', 'hour')
    expected = pd.date_range(start='2021-01-01', end='2021-01-01', freq='H')
    # Ensure first and last timestamps match expected
    assert df['timestamp'].iloc[0] == expected[0]
    assert df['timestamp'].iloc[-1] == expected[-1]
    # Length matches expected
    assert len(df) == len(expected)


def test_minute_frequency_length():
    # Generate minute data for 1 hour
    df = fetch_synthetic_data('SYM', '2021-01-01', '2021-01-01 01:00', 'minute')
    expected = pd.date_range(start='2021-01-01', end='2021-01-01 01:00', freq='T')
    assert len(df) == len(expected)
    assert df['timestamp'].iloc[0] == expected[0]
    assert df['timestamp'].iloc[-1] == expected[-1]
