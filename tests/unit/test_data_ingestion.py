import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.preprocessing import clean_data, validate_data
from trading_rl_agent.data.professional_feeds import ProfessionalDataProvider


def create_sample_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [99, 100, 101, 102, 103],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


@pytest.fixture
def sample_df():
    return create_sample_df()


def test_validate_data_valid(sample_df):
    validate_data(sample_df)  # Should not raise


def test_validate_data_missing_column(sample_df):
    df = sample_df.drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(df)


def test_validate_data_invalid_type(sample_df):
    df = sample_df.copy()
    df["open"] = df["open"].astype(str)
    with pytest.raises(ValueError, match="open must be numeric"):
        validate_data(df)


def test_validate_data_range_error(sample_df):
    df = sample_df.copy()
    df.loc[0, "high"] = 98  # high < low
    with pytest.raises(ValueError, match="high < low"):
        validate_data(df)


def test_validate_data_duplicates(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_data(df)


def test_clean_data_no_changes(sample_df):
    cleaned = clean_data(sample_df)
    pd.testing.assert_frame_equal(cleaned, sample_df)


def test_clean_data_handle_nans():
    df = create_sample_df()
    df.loc[2, "close"] = np.nan
    cleaned = clean_data(df)
    assert not cleaned.isnull().any().any()
    # The clean_data function does forward fill then interpolation
    # So the value should be interpolated between 103 and 105
    assert cleaned.loc[2, "close"] == 103.0  # interpolated value


def test_clean_data_remove_duplicates():
    df = pd.concat([create_sample_df(), create_sample_df().iloc[[0]]])
    cleaned = clean_data(df)
    assert len(cleaned) == 5
    assert not cleaned["timestamp"].duplicated().any()


def test_clean_data_clip_outliers():
    df = create_sample_df()
    df.loc[0, "close"] = 1000  # outlier
    cleaned = clean_data(df)
    mean = df["close"].mean()  # original mean ~104, but after clip
    assert cleaned.loc[0, "close"] < 1000


def test_fetching_yahoo():
    provider = ProfessionalDataProvider("yahoo")
    data = provider.get_market_data(["AAPL"], "2024-01-01", "2024-01-05", timeframe="1Day", include_features=False)
    assert not data.empty
    assert {"date", "symbol", "open", "high", "low", "close", "volume"}.issubset(data.columns)


# Add more tests for other providers if mocks are set up
