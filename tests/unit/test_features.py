import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as pta
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: logging.getLogger(name),
    )
    sys.modules["structlog"] = stub

base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [str(base)]
    sys.modules["trading_rl_agent"] = pkg
if "trading_rl_agent.data" not in sys.modules:
    mod = types.ModuleType("trading_rl_agent.data")
    mod.__path__ = [str(base / "data")]
    sys.modules["trading_rl_agent.data"] = mod

feature_path = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent" / "data" / "features.py"
spec = importlib.util.spec_from_file_location("features", feature_path)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

add_sentiment = features.add_sentiment
generate_features = features.generate_features


def test_compute_log_returns():
    # price series: [1, e, e^2]
    prices = [1.0, np.e, np.e**2]
    df = pd.DataFrame({"close": prices})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    # first should be NaN
    assert np.isnan(df.loc[0, "log_return"])
    # second log_return = log(e/1) = 1
    assert pytest.approx(df.loc[1, "log_return"], rel=1e-6) == 1.0
    # third log_return = log(e^2 / e) = 1
    assert pytest.approx(df.loc[2, "log_return"], rel=1e-6) == 1.0


def test_compute_simple_moving_average():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    df["sma_3"] = df["close"].rolling(3).mean()
    # sma_3 at index 2 = mean([1,2,3]) = 2
    assert pytest.approx(df.loc[2, "sma_3"], rel=1e-6) == 2.0
    # sma_3 at index 4 = mean([3,4,5]) = 4
    assert pytest.approx(df.loc[4, "sma_3"], rel=1e-6) == 4.0


def test_compute_rsi_up_down():
    # Simulate a price series with alternating up/down
    df = pd.DataFrame({"close": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]})
    rsi_vals = pta.rsi(df["close"], length=3)
    # Check that non-NaN RSI values are between 0 and 100
    valid = rsi_vals.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_compute_rolling_volatility():
    # Log returns alternating between 1 and -1
    lr = [np.nan] + [1.0, -1.0, 1.0, -1.0] * 2
    df = pd.DataFrame({"log_return": lr})
    # volatility window 2: std * sqrt(2)
    df["vol_2"] = df["log_return"].rolling(2).std(ddof=0) * np.sqrt(2)
    # At index 2: window values [1, -1], std = 1, volatility = 1*sqrt(2)
    assert pytest.approx(df.loc[2, "vol_2"], rel=1e-6) == np.sqrt(2)


def test_add_sentiment():
    df = pd.DataFrame({"close": [1, 2, 3]})
    df_sent = add_sentiment(df.copy(), sentiment_col="sent")
    assert "sent" in df_sent.columns
    # all zeros
    assert (df_sent["sent"] == 0.0).all()


@pytest.mark.parametrize("n_rows", [40, 50])
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
        },
    )
    df_feat = generate_features(df)
    # Output rows may be fewer due to warm-up trimming
    assert 0 < len(df_feat) <= n_rows


def test_generate_features_no_nan():
    # Ensure no NaN in core feature columns remains after warm-up
    dates = pd.date_range(start="2021-01-01", periods=40, freq="D")
    prices = np.linspace(10, 49, 40)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(40),
        },
    )
    df_feat = generate_features(df)
    core_cols = (
        ["timestamp", "open", "high", "low", "close", "volume", "log_return"]
        + [f"sma_{w}" for w in [5, 10, 20]]
        + ["rsi_14", "vol_20", "sentiment"]
    )
    warmup = 26
    assert df_feat[core_cols].iloc[warmup:].notnull().values.all()
