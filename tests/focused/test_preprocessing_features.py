import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import normalize_data, preprocess_trading_data
from src.data.features import (
    compute_log_returns,
    compute_bollinger_bands,
    compute_macd,
)


def test_normalize_invalid_method():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        normalize_data(df, method="unknown")


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
        }
    )
    seq, tgt, scaler = preprocess_trading_data(
        df, sequence_length=3, target_column="close"
    )
    # target column is excluded from the sequence features
    assert seq.shape == (7, 3, df.shape[1] - 1)
    assert tgt.shape == (7, 1)
    assert hasattr(scaler, "transform")


def test_feature_generators_basic():
    data = pd.DataFrame({"close": np.arange(1, 6, dtype=float)})
    df = compute_log_returns(data.copy())
    assert "log_return" in df.columns

    series_res = compute_log_returns(data["close"])
    assert len(series_res) == len(data)

    upper, mid, lower = compute_bollinger_bands(data["close"], timeperiod=3)
    assert upper.isna().iloc[:2].all()
    macd_df = compute_macd(data.copy(), price_col="close")
    assert {"macd_line", "macd_signal", "macd_hist"}.issubset(macd_df.columns)
