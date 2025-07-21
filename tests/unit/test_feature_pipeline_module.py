import numpy as np
import pandas as pd
import pytest

from trade_agent.features.technical_indicators import TechnicalIndicators


@pytest.fixture
def ohlcv_sample_data():
    """Provide a simple OHLCV dataframe for testing."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    return pd.DataFrame(
        {
            "open": np.random.uniform(98, 102, size=100),
            "high": np.random.uniform(100, 105, size=100),
            "low": np.random.uniform(95, 100, size=100),
            "close": np.random.uniform(98, 102, size=100),
            "volume": np.random.uniform(1e6, 5e6, size=100),
        },
        index=dates,
    )


def test_technical_indicators_basic(ohlcv_sample_data):
    """Test that technical indicators are added to the dataframe."""
    indicators = TechnicalIndicators()
    result = indicators.calculate_all_indicators(ohlcv_sample_data)

    # Check for a subset of the generated indicators
    expected_cols = ["sma_5", "rsi", "macd", "atr", "obv", "doji"]
    for col in expected_cols:
        assert col in result.columns, f"Column '{col}' not found in result"

    assert result.shape[0] == ohlcv_sample_data.shape[0]
    # Check that no NaNs are introduced in the result for a specific indicator
    # (some indicators will have NaNs at the beginning)
    assert not result["sma_50"].isnull().all()
    assert result["sma_50"].notnull().any()
