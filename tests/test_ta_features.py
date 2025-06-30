import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_stochastic,
    compute_williams_r,
)


def test_compute_ema_constant():
    # Constant close price should yield EMA equal to price after warmup
    n = 10
    timeperiod = 3
    prices = np.ones(n) * 5.0
    df = pd.DataFrame({"close": prices})
    df_ema = compute_ema(df.copy(), price_col="close", timeperiod=timeperiod)
    # First timeperiod-1 entries are NaN
    assert df_ema[f"ema_{timeperiod}"][: timeperiod - 1].isnull().all()
    # From index timeperiod-1 onward, EMA equals the constant price
    assert (df_ema[f"ema_{timeperiod}"][timeperiod - 1 :] == 5.0).all()


def test_compute_macd_constant():
    # Constant price => MACD line and hist should be zero after NaNs
    n = 100
    prices = np.ones(n) * 10.0
    df = pd.DataFrame({"close": prices})
    df_macd = compute_macd(df.copy(), price_col="close")
    # MACD line and hist should be NaN for initial periods
    assert df_macd["macd_line"][:26].isnull().all()
    # After slow period, MACD line = 0
    valid = df_macd["macd_line"][26:]
    assert np.allclose(valid.fillna(0), 0.0)
    # Signal and hist also zero or NaN before converge
    assert np.allclose(df_macd["macd_signal"][26:].fillna(0), 0.0)
    assert np.allclose(df_macd["macd_hist"][26:].fillna(0), 0.0)


def test_compute_atr_constant():
    # ATR on constant high/low/close yields zeros
    n = 30
    prices = np.linspace(1, n, n)
    df = pd.DataFrame(
        {
            "high": prices,
            "low": prices,
            "close": prices,
        }
    )
    df_atr = compute_atr(df.copy(), timeperiod=5)
    # First timeperiod entries are NaN
    assert df_atr["atr_5"][:5].isnull().all()
    # After that, ATR = 0
    assert (df_atr["atr_5"][5:] == 0.0).all()


def test_compute_bollinger_bands_constant():
    # Bollinger Bands on constant close yields equal bands
    n = 30
    prices = np.ones(n) * 7.0
    df = pd.DataFrame({"close": prices})
    df_bb = compute_bollinger_bands(df.copy(), price_col="close", timeperiod=5)
    # First timeperiod-1 entries are NaN
    assert df_bb["bb_mavg_5"][:4].isnull().all()
    # After that, upper, mavg, lower equal constant price
    assert (df_bb["bb_mavg_5"][4:] == 7.0).all()
    assert (df_bb["bb_upper_5"][4:] == 7.0).all()
    assert (df_bb["bb_lower_5"][4:] == 7.0).all()


def test_compute_stochastic_constant():
    # Constant high/low/close => stochastic undefined (NaN)
    n = 10
    prices = np.ones(n) * 5.0
    df = pd.DataFrame({"high": prices, "low": prices, "close": prices})
    df_stoch = compute_stochastic(
        df.copy(), fastk_period=3, slowk_period=3, slowd_period=3
    )
    assert "stoch_k" in df_stoch.columns and "stoch_d" in df_stoch.columns
    # All values NaN due to zero range
    assert df_stoch["stoch_k"].isnull().all()
    assert df_stoch["stoch_d"].isnull().all()


def test_compute_stochastic_full():
    # Linear ramp high, low=0, close=high => %K and %D == 100 after warm-up
    prices = np.linspace(1, 10, 10)
    df = pd.DataFrame({"high": prices, "low": np.zeros(10), "close": prices})
    # Use minimal periods so warm-up is minimal
    df_stoch = compute_stochastic(
        df.copy(), fastk_period=1, slowk_period=1, slowd_period=1
    )
    # After first period, values are 100
    assert (df_stoch["stoch_k"][1:] == 100.0).all()
    assert (df_stoch["stoch_d"][1:] == 100.0).all()


def test_compute_adx_constant():
    # Constant series => ADX NaN for initial, then zeros
    n = 30
    prices = np.ones(n) * 5.0
    df = pd.DataFrame({"high": prices, "low": prices, "close": prices})
    df_adx = compute_adx(df.copy(), timeperiod=5)
    assert "adx_5" in df_adx.columns
    # First timeperiod entries NaN
    assert df_adx["adx_5"][:5].isnull().all()
    # Afterwards zero
    assert (df_adx["adx_5"][5:].fillna(0) == 0.0).all()


def test_compute_williams_r_constant():
    # Constant series => William %R NaN
    n = 15
    prices = np.ones(n) * 7.0
    df = pd.DataFrame({"high": prices, "low": prices, "close": prices})
    df_wr = compute_williams_r(df.copy(), timeperiod=5)
    assert "wr_5" in df_wr.columns
    # All NaN due to zero range
    assert df_wr["wr_5"].isnull().all()


def test_compute_obv_constant_and_trend():
    # Constant close => OBV zero
    n = 10
    volumes = np.arange(1, n + 1)
    df = pd.DataFrame({"close": np.ones(n) * 5.0, "volume": volumes})
    df_obv = compute_obv(df.copy())
    assert "obv" in df_obv.columns
    assert (df_obv["obv"] == 0).all()
    # Increasing close => OBV increases cumulatively by volume
    df2 = pd.DataFrame({"close": np.arange(n), "volume": volumes})
    df_obv2 = compute_obv(df2.copy())
    expected = np.cumsum(volumes[1:])
    assert (df_obv2["obv"].iloc[1:].values == expected).all()
