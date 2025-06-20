import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_sentiment,
    compute_log_returns,
    compute_rolling_volatility,
    compute_rsi,
    compute_simple_moving_average,
    generate_features,
)


def test_compute_log_returns():
    # price series: [1, e, e^2]
    prices = [1.0, np.e, np.e**2]
    df = pd.DataFrame({"close": prices})
    df_lr = compute_log_returns(df.copy())
    # first should be NaN
    assert np.isnan(df_lr.loc[0, "log_return"])
    # second log_return = log(e/1) = 1
    assert pytest.approx(df_lr.loc[1, "log_return"], rel=1e-6) == 1.0
    # third log_return = log(e^2 / e) = 1
    assert pytest.approx(df_lr.loc[2, "log_return"], rel=1e-6) == 1.0


def test_compute_simple_moving_average():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    df_sma = compute_simple_moving_average(df.copy(), window=3)
    # sma_3 at index 2 = mean([1,2,3]) = 2
    assert pytest.approx(df_sma.loc[2, "sma_3"], rel=1e-6) == 2.0
    # sma_3 at index 4 = mean([3,4,5]) = 4
    assert pytest.approx(df_sma.loc[4, "sma_3"], rel=1e-6) == 4.0


def test_compute_rsi_up_down():
    # Simulate a price series with alternating up/down
    df = pd.DataFrame({"close": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]})
    df_rsi = compute_rsi(df.copy(), window=3)
    # Check that non-NaN RSI values are between 0 and 100
    rsi_vals = df_rsi["rsi_3"]
    valid = rsi_vals.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_rolling_volatility():
    # Log returns alternating between 1 and -1
    lr = [np.nan] + [1.0, -1.0, 1.0, -1.0] * 2
    df = pd.DataFrame({"log_return": lr})
    # volatility window 2: std * sqrt(2)
    df_vol = compute_rolling_volatility(df.copy(), window=2)
    # At index 2: window values [1, -1], std = 1, volatility = 1*sqrt(2)
    assert pytest.approx(df_vol.loc[2, "vol_2"], rel=1e-6) == np.sqrt(2)


def test_add_sentiment():
    df = pd.DataFrame({"close": [1, 2, 3]})
    df_sent = add_sentiment(df.copy(), sentiment_col="sent")
    assert "sent" in df_sent.columns
    # all zeros
    assert (df_sent["sent"] == 0.0).all()


@pytest.mark.parametrize("n_rows", [25, 50])
def test_generate_features_dimensions(n_rows):
    # Create increasing close price
    dates = pd.date_range(start="2021-01-01", periods=n_rows, freq="D")
    prices = np.linspace(1, n_rows, n_rows)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(n_rows),
        }
    )
    df_feat = generate_features(df)
    # After warm-up for all indicators, expect rows = max(0, n_rows - max_indicator_window)
    # MACD slow period is 26, which is the largest warm-up
    max_indicator_window = 26
    if n_rows < max_indicator_window:
        # For insufficient data, the function adjusts window sizes and returns some data
        # The exact number depends on the adjusted windows, so just check it's > 0
        assert len(df_feat) > 0
    else:
        expected_len = n_rows - max_indicator_window
        assert len(df_feat) == expected_len


def test_generate_features_no_nan():
    # Ensure no NaN in core feature columns remains after warm-up
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    prices = np.linspace(10, 40, 30)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(30),
        }
    )
    df_feat = generate_features(df)
    core_cols = (
        ["timestamp", "open", "high", "low", "close", "volume", "log_return"]
        + [f"sma_{w}" for w in [5, 10, 20]]
        + ["rsi_14", "vol_20", "sentiment"]
    )
    assert df_feat[core_cols].notnull().values.all()
