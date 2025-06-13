import pandas as pd
import datetime
import pytest
from build_datasets import add_news_sentiment, NEWS_FEEDS

def test_add_news_sentiment_empty(monkeypatch):
    # Monkeypatch NEWS_FEEDS to include a dummy feed URL
    monkeypatch.setitem(NEWS_FEEDS, 'AAPL', ['http://example.com/rss'])
    # Simulate no entries returned
    class DummyFeed: pass
    monkeypatch.setattr('build_datasets.feedparser.parse', lambda url: DummyFeed())
    setattr(DummyFeed, 'entries', [])

    df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02')],
        'symbol': ['AAPL', 'AAPL']
    })
    out = add_news_sentiment(df)
    assert 'news_sentiment' in out.columns
    assert all(v == 0.0 for v in out['news_sentiment'])

def test_add_news_sentiment_scores(monkeypatch):
    # Monkeypatch news feed with two entries on same date
    monkeypatch.setitem(NEWS_FEEDS, 'AAPL', ['http://example.com/rss'])
    class Entry:
        def __init__(self, title, year, month, day):
            self.title = title
            self.published_parsed = (year, month, day, 0, 0, 0, 0, 0, 0)
    fake_feed = type('F', (), {})()
    setattr(fake_feed, 'entries', [Entry('good', 2021,1,1), Entry('bad', 2021,1,1)])
    monkeypatch.setattr('build_datasets.feedparser.parse', lambda url: fake_feed)
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2021-01-01')],
        'symbol': ['AAPL']
    })
    out = add_news_sentiment(df)
    # VADER 0.5 for even len('good')=4, -0.5 for odd len('bad')=3 -> avg = 0.0
    assert out['news_sentiment'].iloc[0] == pytest.approx(0.0)
