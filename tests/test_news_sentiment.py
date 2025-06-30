import datetime
from pathlib import Path
import sys

import pandas as pd
import pytest

from build_datasets import NEWS_FEEDS, add_news_sentiment


class DummyAnalyzer:
    def polarity_scores(self, text):
        # Return constant compound score based on text length for variation
        return {"compound": 0.5 if len(text) % 2 == 0 else -0.5}


@pytest.fixture(autouse=True)
def patch_sentiment(monkeypatch):
    # Import the build_datasets module first
    import build_datasets

    # Patch HAS_VADER to True and provide mock analyzer
    monkeypatch.setattr("build_datasets.HAS_VADER", True)

    # Set the analyzer class directly on the module
    build_datasets.SentimentIntensityAnalyzer = lambda: DummyAnalyzer()
    yield


def test_add_news_sentiment_empty(monkeypatch, patch_sentiment):
    # Monkeypatch NEWS_FEEDS to include a dummy feed URL
    monkeypatch.setitem(NEWS_FEEDS, "AAPL", ["http://example.com/rss"])

    # Simulate no entries returned
    class DummyFeed:
        entries = []

    monkeypatch.setattr("build_datasets.feedparser.parse", lambda url: DummyFeed())

    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    out = add_news_sentiment(df)
    assert "news_sentiment" in out.columns
    assert all(v == 0.0 for v in out["news_sentiment"])


def test_add_news_sentiment_scores(monkeypatch, patch_sentiment):
    # Monkeypatch news feed with two entries on same date
    monkeypatch.setitem(NEWS_FEEDS, "AAPL", ["http://example.com/rss"])

    class Entry:
        def __init__(self, title, year, month, day):
            self.title = title
            self.summary = ""
            self.published_parsed = (year, month, day, 0, 0, 0, 0, 0, 0)

        def get(self, key, default=""):
            return getattr(self, key, default)

    class DummyFeed:
        def __init__(self):
            self.entries = [Entry("good", 2021, 1, 1), Entry("bad", 2021, 1, 1)]

    monkeypatch.setattr("build_datasets.feedparser.parse", lambda url: DummyFeed())

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-01-01")], "symbol": ["AAPL"]})
    out = add_news_sentiment(df)
    # VADER 0.5 for even len('good ')=5, -0.5 for odd len('bad ')=4 -> avg = 0.0
    assert out["news_sentiment"].iloc[0] == pytest.approx(0.0)
