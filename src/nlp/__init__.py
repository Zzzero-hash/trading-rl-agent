"""Natural language processing utilities.

This module exposes simple wrappers around the sentiment providers defined in
``src.data.sentiment`` so that other parts of the code base (e.g. the data
pipeline) can obtain sentiment scores without dealing with the provider classes
directly.

Two helper functions are provided for scoring news and social media sentiment
separately.  :func:`get_sentiment_scores` aggregates these and returns a
convenient dictionary that can be easily attached to a dataframe of market
features.
"""

from __future__ import annotations

import datetime
from typing import Dict

from src.data.sentiment import (
    NewsSentimentProvider,
    SentimentData,
    SocialSentimentProvider,
)


def _aggregate_sentiment(data: list[SentimentData], days_back: int) -> float:
    """Aggregate sentiment scores with magnitude and recency weighting."""
    if not data:
        return 0.0

    now = datetime.datetime.now()
    total_score = 0.0
    total_weight = 0.0

    for d in data:
        hours_old = (now - d.timestamp).total_seconds() / 3600
        recency_weight = max(0.1, 1.0 - (hours_old / (days_back * 24)))
        weight = d.magnitude * recency_weight
        total_score += d.score * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def score_news_sentiment(symbol: str, days_back: int = 1) -> float:
    """Return the aggregated news sentiment score for ``symbol``."""

    provider = NewsSentimentProvider()
    data = provider.fetch_sentiment(symbol, days_back)
    return _aggregate_sentiment(data, days_back)


def score_social_sentiment(symbol: str, days_back: int = 1) -> float:
    """Return the aggregated social media sentiment score for ``symbol``."""

    provider = SocialSentimentProvider()
    data = provider.fetch_sentiment(symbol, days_back)
    return _aggregate_sentiment(data, days_back)


def get_sentiment_scores(symbol: str, days_back: int = 1) -> dict[str, float]:
    """Return news, social and overall sentiment scores for ``symbol``.

    The overall score is computed from all available sentiment data using the
    same weighting scheme as the individual scores.
    """

    news_data = NewsSentimentProvider().fetch_sentiment(symbol, days_back)
    social_data = SocialSentimentProvider().fetch_sentiment(symbol, days_back)

    news_score = _aggregate_sentiment(news_data, days_back)
    social_score = _aggregate_sentiment(social_data, days_back)

    combined = news_data + social_data
    overall = _aggregate_sentiment(combined, days_back)

    return {
        "news_sentiment": news_score,
        "social_sentiment": social_score,
        "aggregate_sentiment": overall,
    }


__all__ = [
    "score_news_sentiment",
    "score_social_sentiment",
    "get_sentiment_scores",
]
