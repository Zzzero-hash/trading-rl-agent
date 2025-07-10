"""
NLP utilities for sentiment scoring in Trading RL Agent.
"""

from trading_rl_agent.data.sentiment import (
    NewsSentimentProvider,
    SocialSentimentProvider,
)


def score_news_sentiment(symbol: str) -> float:
    """Fetch and score news sentiment for a symbol."""
    provider = NewsSentimentProvider()
    data = provider.fetch_sentiment(symbol)
    if not data:
        return 0.0
    # Simple average of scores
    return sum(item.score for item in data) / len(data)


def score_social_sentiment(symbol: str) -> float:
    """Fetch and score social sentiment for a symbol."""
    provider = SocialSentimentProvider()
    data = provider.fetch_sentiment(symbol)
    if not data:
        return 0.0
    return sum(item.score for item in data) / len(data)


def get_sentiment_scores(symbol: str) -> dict[str, float]:
    """Get both news and social sentiment scores and their aggregate."""
    news = score_news_sentiment(symbol)
    social = score_social_sentiment(symbol)
    aggregate = (news + social) / 2 if news or social else 0.0
    return {
        "news_sentiment": news,
        "social_sentiment": social,
        "aggregate_sentiment": aggregate,
    }
