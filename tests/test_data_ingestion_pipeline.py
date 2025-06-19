import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import PipelineConfig, generate_features, split_by_date


def test_sma_computation():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=5, freq="D"),
            "open": np.arange(5),
            "high": np.arange(5),
            "low": np.arange(5),
            "close": [1, 2, 3, 4, 5],
            "volume": 1,
        }
    )
    cfg = PipelineConfig(
        sma_windows=[3], momentum_windows=[], rsi_window=2, vol_window=2
    )
    result = generate_features(df, cfg)
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0]
    assert np.allclose(result["sma_3"].values, expected, equal_nan=True)


def test_split_by_date_no_overlap():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=7, freq="D"),
            "close": np.arange(7),
        }
    )
    train, val, test = split_by_date(df, "2021-01-03", "2021-01-06")
    assert len(train) == 2
    assert len(val) == 3
    assert len(test) == 2
    # ensure no overlap
    all_dates = pd.concat([train, val, test])["timestamp"]
    assert all_dates.is_unique
    assert all_dates.is_monotonic_increasing


def test_generate_features_with_missing_values():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, np.nan, 3.0],
            "volume": 1,
        }
    )
    cfg = PipelineConfig(
        sma_windows=[2], momentum_windows=[1], rsi_window=2, vol_window=2
    )
    result = generate_features(df, cfg)
    # Should keep same number of rows
    assert len(result) == 3
    # SMA should have NaN where insufficient data
    assert result["sma_2"].isnull().sum() >= 1
