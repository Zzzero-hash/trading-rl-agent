import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.features import generate_features
from src.data.sentiment import SentimentData
from src.data.synthetic import generate_gbm_prices
from src.nlp import (
    get_sentiment_scores,
    score_news_sentiment,
    score_social_sentiment,
)


@pytest.fixture
def sample_sentiment():
    now = datetime.datetime.now()
    return [
        SentimentData("AAPL", 0.6, 0.8, now, "news"),
        SentimentData("AAPL", 0.2, 0.5, now - datetime.timedelta(hours=1), "news"),
    ]


def test_score_news_sentiment(monkeypatch, sample_sentiment):
    with patch(
        "src.data.sentiment.NewsSentimentProvider.fetch_sentiment",
        return_value=sample_sentiment,
    ):
        score = score_news_sentiment("AAPL")
        assert isinstance(score, float)
        assert score > 0


def test_score_social_sentiment(monkeypatch, sample_sentiment):
    with patch(
        "src.data.sentiment.SocialSentimentProvider.fetch_sentiment",
        return_value=sample_sentiment,
    ):
        score = score_social_sentiment("AAPL")
        assert isinstance(score, float)
        assert score > 0


def test_get_sentiment_scores_pipeline_usage(monkeypatch, sample_sentiment):
    with (
        patch(
            "src.data.sentiment.NewsSentimentProvider.fetch_sentiment",
            return_value=sample_sentiment,
        ),
        patch(
            "src.data.sentiment.SocialSentimentProvider.fetch_sentiment",
            return_value=sample_sentiment,
        ),
    ):
        scores = get_sentiment_scores("AAPL")
        assert set(scores) == {
            "news_sentiment",
            "social_sentiment",
            "aggregate_sentiment",
        }
        df = generate_gbm_prices(n_days=5)
        df["symbol"] = "AAPL"
        feats = generate_features(df)
        feats["sentiment"] = scores["aggregate_sentiment"]
        assert "sentiment" in feats.columns
