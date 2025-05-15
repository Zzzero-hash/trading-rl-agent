import pandas as pd
import pytest
from src.data.historical import fetch_historical_data, client
from unittest.mock import MagicMock
import os

class DummyBar:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


@pytest.fixture(autouse=True)
def patch_polygon_client(monkeypatch):
    # create dummy bars list
    dummy_response = MagicMock(results=[
        DummyBar(1633036800000, 100, 110, 90, 105, 1000),
        DummyBar(1633123200000, 105, 115, 95, 110, 1500),
        DummyBar(1633209600000, 110, 120, 100, 115, 2000),
    ])
    # stub Finnhub stock_candles to return the dummy data
    monkeypatch.setattr(
        client,
        "stock_candles",
        lambda symbol, timestep, start_ts, end_ts: {
            's': 'ok',
            't': [bar.timestamp // 1000 for bar in dummy_response.results],
            'o': [bar.open for bar in dummy_response.results],
            'h': [bar.high for bar in dummy_response.results],
            'l': [bar.low for bar in dummy_response.results],
            'c': [bar.close for bar in dummy_response.results],
            'v': [bar.volume for bar in dummy_response.results],
        }
    )
    # keep original get_aggs stub for backward compatibility if needed
    monkeypatch.setattr(client, "get_aggs", lambda *args, **kwargs: dummy_response)
    yield


def test_fetch_historical_data_format():
    df = fetch_historical_data("AAPL", start="2020-01-01", end="2020-01-10", timestep="day")
    # DataFrame format
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.dtype == "datetime64[ns]"
    # Correct number of rows and index values
    assert len(df) == 3
    expected = pd.to_datetime([1633036800000, 1633123200000, 1633209600000], unit="ms")
    pd.testing.assert_index_equal(df.index, expected)
    # Spot-check values
    assert df.iloc[1]["high"] == 115
    assert df.iloc[2]["low"] == 100