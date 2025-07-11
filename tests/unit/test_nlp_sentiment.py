import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from trading_rl_agent.data.features import generate_features
from trading_rl_agent.data.sentiment import SentimentData
from trading_rl_agent.data.synthetic import generate_gbm_prices
from trading_rl_agent.nlp import (
    get_sentiment_scores,
    score_news_sentiment,
    score_social_sentiment,
)


@pytest.fixture
def sample_sentiment_data():
    now = datetime.datetime.now()
    return [
        SentimentData("AAPL", 0.6, 0.8, now, "news"),
        SentimentData("AAPL", 0.2, 0.5, now - datetime.timedelta(hours=1), "news"),
    ]


def test_score_news_sentiment(sample_sentiment_data):
    with patch("trading_rl_agent.nlp.NewsSentimentProvider") as mock_provider:
        mock_instance = mock_provider.return_value
        mock_instance.fetch_sentiment.return_value = sample_sentiment_data

        result = score_news_sentiment("AAPL")
        # The function returns the average of scores
        expected = (0.6 + 0.2) / 2
        assert result == pytest.approx(expected)


def test_score_social_sentiment():
    with patch("trading_rl_agent.nlp.SocialSentimentProvider") as mock_provider:
        mock_instance = mock_provider.return_value
        # Let's use different values for social sentiment
        social_sentiment = [
            SentimentData("AAPL", -0.5, 0.8, datetime.datetime.now(), "social"),
            SentimentData("AAPL", -0.1, 0.5, datetime.datetime.now(), "social"),
        ]
        mock_instance.fetch_sentiment.return_value = social_sentiment

        result = score_social_sentiment("AAPL")
        expected = (-0.5 - 0.1) / 2
        assert result == pytest.approx(expected)


def test_get_sentiment_scores_pipeline_usage(sample_sentiment_data):
    with (
        patch("trading_rl_agent.nlp.NewsSentimentProvider") as mock_news_provider,
        patch(
            "trading_rl_agent.nlp.SocialSentimentProvider",
        ) as mock_social_provider,
    ):
        mock_news_provider.return_value.fetch_sentiment.return_value = sample_sentiment_data

        social_sentiment = [
            SentimentData("AAPL", -0.5, 0.8, datetime.datetime.now(), "social"),
            SentimentData("AAPL", -0.1, 0.5, datetime.datetime.now(), "social"),
        ]
        mock_social_provider.return_value.fetch_sentiment.return_value = social_sentiment

        scores = get_sentiment_scores("AAPL")

        expected_news = (0.6 + 0.2) / 2
        expected_social = (-0.5 - 0.1) / 2
        expected_agg = (expected_news + expected_social) / 2

        assert scores["news_sentiment"] == pytest.approx(expected_news)
        assert scores["social_sentiment"] == pytest.approx(expected_social)
        assert scores["aggregate_sentiment"] == pytest.approx(expected_agg)

        # Test integration with feature generation
        df = generate_gbm_prices(n_days=100)
        df["tic"] = "AAPL"  # generate_features might need this
        df["date"] = pd.to_datetime(
            pd.date_range(start="2024-01-01", periods=100),
        )
        feats = generate_features(df)
        feats["sentiment"] = scores["aggregate_sentiment"]
        assert "sentiment" in feats.columns
        assert not feats["sentiment"].isnull().any()
