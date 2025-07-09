import pytest

from trading_rl_agent.data.historical import fetch_historical_data
from trading_rl_agent.data.live import fetch_live_data


def test_fetch_historical_data_missing_yfinance(monkeypatch):
    """Ensure ImportError is raised when yfinance is unavailable."""
    monkeypatch.setattr("src.data.historical.yf", None)
    monkeypatch.setattr("src.data.historical.client", None)
    with pytest.raises(ImportError, match="yfinance package is required"):
        fetch_historical_data("AAPL", "2023-01-01", "2023-01-02")


def test_fetch_live_data_missing_yfinance(monkeypatch):
    """Ensure ImportError is raised when yfinance is unavailable."""
    monkeypatch.setattr("src.data.live.yf", None)
    with pytest.raises(ImportError, match="yfinance package is required"):
        fetch_live_data("AAPL", "2023-01-01", "2023-01-02")
