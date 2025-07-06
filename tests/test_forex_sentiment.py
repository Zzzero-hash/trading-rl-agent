import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data.forex_sentiment import (
    ForexSentimentData,
    analyze_text_sentiment,
    get_all_forex_sentiment,
    get_forex_sentiment,
    get_yahoo_finance_url_for_forex,
    scrape_yahoo_finance_forex_headlines,
)

SAMPLE_HTML = """
<html><body>
<h3>EUR/USD rises on bullish economic data</h3>
<h3>Bearish outlook for USDJPY as yen strengthens</h3>
<h3>EURUSD trading sideways</h3>
</body></html>
"""


def test_get_yahoo_finance_url_for_forex():
    assert (
        get_yahoo_finance_url_for_forex("EURUSD")
        == "https://finance.yahoo.com/quote/EURUSD=X/news?p=EURUSD=X"
    )


@patch("src.data.forex_sentiment.requests.get")
def test_scrape_yahoo_finance_forex_headlines(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_HTML
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    headlines = scrape_yahoo_finance_forex_headlines("EURUSD")
    assert isinstance(headlines, list)
    assert all(isinstance(h, str) for h in headlines)
    assert len(headlines) > 0


def test_analyze_text_sentiment():
    pos = analyze_text_sentiment("Strong bullish growth expected for EURUSD")
    neg = analyze_text_sentiment("Bearish decline and weak performance")
    neu = analyze_text_sentiment("EURUSD trading sideways")
    assert pos > 0
    assert neg < 0
    assert abs(neu) < NEUTRAL_SENTIMENT_THRESHOLD


@patch("src.data.forex_sentiment.requests.get")
def test_get_forex_sentiment(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_HTML
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    data = get_forex_sentiment("EURUSD")
    assert isinstance(data, list)
    assert all(isinstance(d, ForexSentimentData) for d in data)
    assert all(d.pair == "EURUSD" for d in data)
    assert all(-1.0 <= d.score <= 1.0 for d in data)
    assert all(0.0 <= d.magnitude <= 1.0 for d in data)
    assert all(d.source == "yahoo_finance_scrape" for d in data)


@patch("src.data.forex_sentiment.requests.get")
def test_get_all_forex_sentiment(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_HTML
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    pairs = ["EURUSD", "USDJPY"]
    all_data = get_all_forex_sentiment(pairs)
    assert isinstance(all_data, dict)
    for pair in pairs:
        assert pair in all_data
        assert all(isinstance(d, ForexSentimentData) for d in all_data[pair])


def test_get_forex_sentiment_handles_scrape_failure(monkeypatch):
    """Test that get_forex_sentiment returns neutral sentiment if scraping fails."""
    import requests

    def fail_get(*args, **kwargs):
        raise requests.exceptions.RequestException("Scrape failed")

    monkeypatch.setattr("requests.get", fail_get)
    import importlib
    import sys

    sys.path.insert(0, "./src")
    forex_sentiment = importlib.import_module("data.forex_sentiment")
    data = forex_sentiment.get_forex_sentiment("EURUSD")
    assert len(data) == 1
    d = data[0]
    assert d.pair == "EURUSD"
    assert d.score == 0.0
    assert d.magnitude == 0.0
    assert d.source == "no_sentiment"
