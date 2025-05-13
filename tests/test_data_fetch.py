import pandas as pd
import pytest
from src.data.historical import fetch_historical_data
from unittest.mock import MagicMock

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
    dummy_response = MagicMock(results=[
        DummyBar(1633036800000, 100, 110, 90, 105, 1000),
        DummyBar(1633123200000, 105, 115, 95, 110, 1500),
        DummyBar(1633209600000, 110, 120, 100, 115, 2000),
    ])
    monkeypatch.setattr(
      hist.client,
      "get_aggs",
      lambda *args, **kwargs: dummy_response
    )
    yield

def test_fetch_historical_data():
    df = fetch_historical_data("FAKE", start="2020-01-01", end="2020-01-10", timestep="day")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.dtype == "datetime64[ns]"
    assert df.iloc[0]["open"] == 10
    assert df.iloc[1]["volume"] == 2000
