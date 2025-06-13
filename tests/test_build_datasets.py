import pandas as pd
import datetime
import json
import pytest

from build_datasets import add_hf_sentiment, add_twitter_sentiment
from build_datasets import add_news_sentiment, NEWS_FEEDS

class DummyDataset(list):
    """A dummy dataset that behaves like HF dataset for pd.DataFrame()"""
    pass

class DummyAnalyzer:
    def polarity_scores(self, text):
        # Return constant compound score based on text length for variation
        return {'compound': 0.5 if len(text) % 2 == 0 else -0.5}

@ pytest.fixture(autouse=True)
def patch_hf(monkeypatch):
    # Patch load_dataset to return our dummy dataset
    def fake_load_dataset(name, split):
        # Provide two entries, one matches df, one extra
        return DummyDataset([
            {'symbol': 'AAPL', 'date': '2021-01-01', 'sentiment_score': 0.8},
            {'symbol': 'GOOG', 'date': '2021-01-02', 'sentiment_score': -0.2}
        ])
    monkeypatch.setattr('build_datasets.load_dataset', fake_load_dataset)
    yield

@ pytest.fixture(autouse=True)
def patch_twitter(monkeypatch):
    # Patch VADER analyzer to use DummyAnalyzer
    monkeypatch.setattr('build_datasets.SentimentIntensityAnalyzer', lambda: DummyAnalyzer())
    # Patch subprocess.check_output to return fake tweet JSON lines
    def fake_check_output(cmd, shell, text):
        tweets = [
            json.dumps({'content': 'good news'}),
            json.dumps({'content': 'bad loss'})
        ]
        return "\n".join(tweets)
    monkeypatch.setattr('build_datasets.subprocess.check_output', fake_check_output)
    yield


def test_add_hf_sentiment_merges_scores():
    # Prepare DataFrame with two timestamps, one matching HF and one missing
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-03')],
        'symbol': ['AAPL', 'AAPL']
    })
    out = add_hf_sentiment(df)
    # hf_sentiment should be 0.8 for first, 0.0 default for missing
    assert 'hf_sentiment' in out.columns
    assert out.loc[out['timestamp'] == pd.Timestamp('2021-01-01'), 'hf_sentiment'].iloc[0] == pytest.approx(0.8)
    assert out.loc[out['timestamp'] == pd.Timestamp('2021-01-03'), 'hf_sentiment'].iloc[0] == pytest.approx(0.0)


def test_add_twitter_sentiment_scores(monkeypatch):
    # Prepare DataFrame with one day and one symbol
    ts = pd.Timestamp('2021-02-01 12:00:00')
    df = pd.DataFrame({'timestamp': [ts], 'symbol': ['AAPL']})
    out = add_twitter_sentiment(df)
    # twitter_sentiment should be average of two DummyAnalyzer outputs: good->0.5, bad->-0.5 = 0.0
    assert 'twitter_sentiment' in out.columns
    assert out['twitter_sentiment'].iloc[0] == pytest.approx(0.0)


def test_integration_hf_and_twitter():
    # Combine both operations sequentially
    ts1 = pd.Timestamp('2021-01-01')
    ts2 = pd.Timestamp('2021-02-01')
    df = pd.DataFrame({
        'timestamp': [ts1, ts2],
        'symbol': ['AAPL', 'AAPL']
    })
    intermediate = add_hf_sentiment(df)
    combined = add_twitter_sentiment(intermediate)
    # Ensure both sentiment columns exist and are float
    assert set(['hf_sentiment', 'twitter_sentiment']).issubset(combined.columns)
    assert all(isinstance(v, float) for v in combined['hf_sentiment'])
    assert all(isinstance(v, float) for v in combined['twitter_sentiment'])

def test_integration_full_pipeline(monkeypatch):
    # Patch NEWS_FEEDS with dummy URL and no entries
    monkeypatch.setitem(NEWS_FEEDS, 'AAPL', ['http://example.com/rss'])
    dummy_feed = type('F', (), {})()
    setattr(dummy_feed, 'entries', [])
    monkeypatch.setattr('build_datasets.feedparser.parse', lambda url: dummy_feed)
    # Prepare sample df
    ts1 = pd.Timestamp('2021-01-01')
    ts2 = pd.Timestamp('2021-02-01')
    df = pd.DataFrame({'timestamp': [ts1, ts2], 'symbol': ['AAPL', 'AAPL']})
    out = add_hf_sentiment(df)
    out = add_twitter_sentiment(out)
    out = add_news_sentiment(out)
    # Validate all sentiment columns are present and numeric
    assert set(['hf_sentiment', 'twitter_sentiment', 'news_sentiment']).issubset(out.columns)
    assert all(isinstance(v, float) for v in out['news_sentiment'])
