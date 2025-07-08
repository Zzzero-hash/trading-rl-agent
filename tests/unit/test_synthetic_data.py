import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.synthetic import fetch_synthetic_data


def test_synthetic_columns_and_length():
    # Generate 5 daily samples
    df = fetch_synthetic_data(n_samples=5, timeframe="day")
    # Check columns
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols
    # Check length
    assert len(df) == 5
    # Timestamp dtype
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # Numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_high_low_relationship():
    # Ensure high >= max(open, close) and low <= min(open, close)
    df = fetch_synthetic_data(n_samples=5, timeframe="day")
    high = df["high"]
    low = df["low"]
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    assert (high >= oc_max).all()
    assert (low <= oc_min).all()


def test_volume_integer_positive():
    # Volume should be integer and non-negative
    df = fetch_synthetic_data(n_samples=2, timeframe="day")
    assert df["volume"].dtype.kind in ("i", "u")
    assert (df["volume"] >= 0).all()


def test_hourly_timestamp_bounds():
    # Generate a single hourly sample
    df = fetch_synthetic_data(n_samples=1, timeframe="hour")
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_minute_frequency_length():
    # Generate minute data for 1 hour
    df = fetch_synthetic_data(n_samples=61, timeframe="minute")
    # Verify frequency and length
    diffs = df["timestamp"].diff().dropna().unique()
    assert len(df) == 61
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=1)
