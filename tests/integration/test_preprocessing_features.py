import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
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
spec_feat = importlib.util.spec_from_file_location("features", feature_path)
features = importlib.util.module_from_spec(spec_feat)
spec_feat.loader.exec_module(features)

preproc_path = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent" / "data" / "preprocessing.py"
spec_pre = importlib.util.spec_from_file_location("preprocessing", preproc_path)
preprocessing = importlib.util.module_from_spec(spec_pre)
spec_pre.loader.exec_module(preprocessing)

compute_bollinger_bands = features.compute_bollinger_bands
compute_macd = features.compute_macd
create_sequences = preprocessing.create_sequences
preprocess_trading_data = preprocessing.preprocess_trading_data


def test_normalize_invalid_method():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        preprocess_trading_data(df, normalize_method="unknown")


def test_preprocess_trading_data_pipeline():
    dates = pd.date_range("2021-01-01", periods=10)
    df = pd.DataFrame(
        {
            "time": dates,
            "open": np.arange(10, 20, dtype=float),
            "high": np.arange(11, 21, dtype=float),
            "low": np.arange(9, 19, dtype=float),
            "close": np.arange(10, 20, dtype=float),
            "volume": np.arange(1, 11, dtype=float),
        },
    )
    seq, tgt, scaler = preprocess_trading_data(
        df,
        sequence_length=3,
        target_column="close",
    )
    # target column is excluded from the sequence features
    assert seq.shape == (7, 3, df.shape[1] - 1)
    assert tgt.shape == (7, 1)
    assert hasattr(scaler, "transform")


def test_feature_generators_basic():
    # Use longer series so pandas_ta indicators have enough data
    data = pd.DataFrame({"close": np.arange(1, 41, dtype=float)})
    df = data.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    assert "log_return" in df.columns

    bb_df = compute_bollinger_bands(data.copy(), price_col="close", timeperiod=3)
    assert bb_df["bb_upper_3"].isna().iloc[:2].all()
    macd_df = compute_macd(data.copy(), price_col="close")
    assert {"macd_line", "macd_signal", "macd_hist"}.issubset(macd_df.columns)


def test_create_sequences_stride_dataframe():
    df = pd.DataFrame(
        {
            "f1": np.arange(6, dtype=float),
            "f2": np.arange(10, 16, dtype=float),
            "target": np.arange(20, 26, dtype=float),
        },
    )
    seq, tgt = create_sequences(df, sequence_length=2, target_column="target", stride=2)

    expected_seq = np.array(
        [
            [[0.0, 10.0], [1.0, 11.0]],
            [[2.0, 12.0], [3.0, 13.0]],
        ],
    )
    expected_tgt = np.array([[22.0], [24.0]])

    assert np.array_equal(seq, expected_seq)
    assert np.array_equal(tgt, expected_tgt)
