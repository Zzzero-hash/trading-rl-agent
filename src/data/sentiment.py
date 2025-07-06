"""Sentiment Analysis Module for Trading RL Agent.

This module provides sentiment analysis capabilities for financial markets,
integrating news sentiment and social media sentiment to enhance trading decisions.
It supports real-time sentiment fetching and historical sentiment analysis.

Example usage:
>>> analyzer = SentimentAnalyzer()
>>> sentiment_score = analyzer.get_symbol_sentiment('AAPL')
>>> news_sentiment = analyzer.analyze_news_sentiment('AAPL', days_back=7)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import datetime
import logging
import time
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup
import requests

logger = logging.getLogger(__name__)

# Global sentiment cache for backward compatibility
sentiment = {}


@dataclass
class SentimentData:
    """Structured sentiment data for a financial symbol."""

    symbol: str
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    magnitude: float  # 0.0 to 1.0 (confidence level)
    timestamp: datetime.datetime
    source: str  # 'news', 'social', 'analyst', etc.
    raw_data: Optional[dict[str, Any]] = None


class SentimentProvider(ABC):
    """Abstract base class for sentiment data providers."""

    @abstractmethod
    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch sentiment data for a symbol."""
        pass


class NewsSentimentProvider(SentimentProvider):
    """News-based sentiment provider using Yahoo Finance scraping only."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # Unused, for test compatibility

    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch news sentiment for a symbol by scraping Yahoo Finance."""
        try:
            return self._scrape_headlines_sentiment(symbol, days_back)
        except Exception as e:
            logger.warning(f"Failed to scrape headlines for {symbol}: {e}")
            # Return mock news sentiment if scraping fails
            return self._get_mock_news_sentiment_static(symbol, days_back)

    @staticmethod
    def _analyze_text_sentiment(text: str) -> float:
        """Analyze text sentiment using the VADER compound score."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            return float(scores.get("compound", 0.0))
        except Exception as exc:  # pragma: no cover - failure path
            logger.warning(f"VADER sentiment failed for text '{text}': {exc}")
            return 0.0

    @staticmethod
    def _get_mock_news_sentiment_static(
        symbol: str, days_back: int
    ) -> list[SentimentData]:
        """Generate mock news sentiment data for testing."""
        import random

        random.seed(hash(symbol))
        sentiment_data = []
        for i in range(min(5, days_back)):
            score = random.uniform(-0.5, 0.8)  # nosec B311
            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=score,
                    magnitude=random.uniform(0.5, 0.9),  # nosec B311
                    timestamp=datetime.datetime.now() - datetime.timedelta(days=i),
                    source="news_mock",
                    raw_data={"mock": True},
                )
            )
        return sentiment_data

    def _scrape_headlines_sentiment(
        self, symbol: str, days_back: int
    ) -> list[SentimentData]:
        """Scrape news headlines from Yahoo Finance and analyze sentiment (robust)."""
        url = f"https://finance.yahoo.com/quote/{symbol}/news?p={symbol}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            resp = requests.get(url, timeout=10, headers=headers)

            # Handle rate limiting gracefully
            if resp.status_code == 429:
                logger.warning(
                    f"Rate limited by Yahoo Finance for {symbol}, using fallback data"
                )
                time.sleep(1)  # Brief pause before fallback
                raise requests.exceptions.HTTPError("Rate limited")

            resp.raise_for_status()

        except (
            requests.exceptions.RequestException,
            requests.exceptions.HTTPError,
        ) as e:
            logger.warning(f"Failed to fetch from Yahoo Finance for {symbol}: {e}")
            # Return fallback mock data instead of raising
            return self._get_mock_news_sentiment_static(symbol, days_back)

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            headlines = set()
            # Try multiple selectors for robustness
            for tag in ["h3", "h2", "a"]:
                for item in soup.find_all(tag):
                    text = item.get_text(strip=True)
                    if text and len(text) > 10:
                        headlines.add(text)

            if not headlines:
                logger.warning(
                    f"No headlines found for {symbol} on Yahoo Finance, using fallback"
                )
                return self._get_mock_news_sentiment_static(symbol, days_back)

            sentiment_data = []
            now = datetime.datetime.now()
            for i, headline in enumerate(list(headlines)[:15]):
                score = NewsSentimentProvider._analyze_text_sentiment(headline)
                sentiment_data.append(
                    SentimentData(
                        symbol=symbol,
                        score=score,
                        magnitude=0.7,
                        timestamp=now - datetime.timedelta(minutes=i * 10),
                        source="news_scrape",
                        raw_data={"headline": headline},
                    )
                )
            return sentiment_data

        except Exception as e:
            logger.warning(f"Failed to parse headlines for {symbol}: {e}")
            return self._get_mock_news_sentiment_static(symbol, days_back)

    def _symbol_to_company(self, symbol: str) -> str:
        """Map a stock symbol to a company name. Uses a static mapping for test/demo purposes."""
        mapping = {
            "AAPL": "Apple Inc",
            "GOOG": "Alphabet Inc",
            "GOOGL": "Google Alphabet",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com, Inc.",
            "TSLA": "Tesla, Inc.",
            "META": "Meta Platforms, Inc.",
            "NFLX": "Netflix, Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "V": "Visa Inc.",
        }
        return mapping.get(symbol.upper(), symbol)


class SocialSentimentProvider(SentimentProvider):
    """Social media sentiment provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch social media sentiment for a symbol."""
        # For now, return mock data - can be extended with Twitter/Reddit APIs
        return self._get_mock_social_sentiment(symbol, days_back)

    def _get_mock_social_sentiment(
        self, symbol: str, days_back: int
    ) -> list[SentimentData]:
        """Generate mock social sentiment data."""
        import random

        random.seed(hash(symbol + "social"))

        sentiment_data = []
        for i in range(min(3, days_back)):
            score = random.uniform(-0.8, 0.6)  # More volatile than news  # nosec B311
            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=score,
                    magnitude=random.uniform(0.3, 0.8),  # nosec B311
                    timestamp=datetime.datetime.now() - datetime.timedelta(days=i),
                    source="social_mock",
                    raw_data={"mock": True},
                )
            )
        return sentiment_data


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    enable_news: bool = True
    enable_social: bool = True
    social_api_key: Optional[str] = None
    news_api_key: Optional[str] = None  # Added for test compatibility
    cache_duration_hours: int = 1
    sentiment_weight: float = 0.2  # Weight in final feature vector


class SentimentAnalyzer:
    """Main sentiment analysis coordinator."""

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self.providers: list[SentimentProvider] = []
        self.sentiment_cache: dict[str, list[SentimentData]] = {}
        if self.config.enable_news:
            self.providers.append(NewsSentimentProvider())  # No API key
        if self.config.enable_social:
            self.providers.append(SocialSentimentProvider(self.config.social_api_key))

    def get_symbol_sentiment(self, symbol: str, days_back: int = 1) -> float:
        """Get aggregated sentiment score for a symbol."""
        sentiment_data = self.fetch_all_sentiment(symbol, days_back)

        if not sentiment_data:
            return 0.0

        # Weight by magnitude and recency
        total_weighted_score = 0.0
        total_weight = 0.0

        now = datetime.datetime.now()
        for data in sentiment_data:
            # Recency weight (more recent = higher weight)
            hours_old = (now - data.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - (hours_old / (days_back * 24)))

            weight = data.magnitude * recency_weight
            total_weighted_score += data.score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def fetch_all_sentiment(
        self, symbol: str, days_back: int = 1
    ) -> list[SentimentData]:
        """Fetch sentiment from all providers."""
        cache_key = f"{symbol}_{days_back}"

        # Check cache
        if cache_key in self.sentiment_cache:
            cached_data = self.sentiment_cache[cache_key]
            if cached_data and self._is_cache_valid(cached_data[0].timestamp):
                return cached_data

        # Fetch fresh data
        all_sentiment = []
        for provider in self.providers:
            try:
                sentiment_data = provider.fetch_sentiment(symbol, days_back)
                all_sentiment.extend(sentiment_data)
            except Exception as e:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed for {symbol}: {e}"
                )

        # Cache results
        self.sentiment_cache[cache_key] = all_sentiment

        # Update global sentiment dictionary for backward compatibility
        if all_sentiment:
            avg_score = sum(d.score for d in all_sentiment) / len(all_sentiment)
            avg_magnitude = sum(d.magnitude for d in all_sentiment) / len(all_sentiment)
            sentiment[symbol] = {
                "score": avg_score,
                "magnitude": avg_magnitude,
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "aggregated",
            }

        return all_sentiment

    def _is_cache_valid(self, timestamp: datetime.datetime) -> bool:
        """Check if cached data is still valid."""
        age_hours = (datetime.datetime.now() - timestamp).total_seconds() / 3600
        return age_hours < self.config.cache_duration_hours

    def get_sentiment_features(
        self, symbols: list[str], days_back: int = 1
    ) -> dict[str, float]:
        """Get sentiment features for multiple symbols suitable for ML models."""
        features = {}
        for symbol in symbols:
            sentiment_score = self.get_symbol_sentiment(symbol, days_back)
            features[f"sentiment_{symbol}"] = sentiment_score
            features[f"sentiment_{symbol}_abs"] = abs(
                sentiment_score
            )  # Magnitude feature
        return features

    def update_sentiment_cache(self, symbol: str, days_back: int = 1) -> None:
        """Manually update sentiment cache for a symbol."""
        self.fetch_all_sentiment(symbol, days_back)


# Default global analyzer instance
_default_analyzer = SentimentAnalyzer()


def get_sentiment_score(symbol: str, days_back: int = 1) -> float:
    """Convenience function to get sentiment score."""
    return _default_analyzer.get_symbol_sentiment(symbol, days_back)


def update_sentiment(symbol: str, days_back: int = 1) -> None:
    """Convenience function to update sentiment."""
    _default_analyzer.update_sentiment_cache(symbol, days_back)
