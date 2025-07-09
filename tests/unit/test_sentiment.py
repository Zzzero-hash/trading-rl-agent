"""
Enhanced tests for the sentiment analysis module.
Tests the new comprehensive sentiment analysis capabilities.
"""

import datetime
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from trading_rl_agent.data.sentiment import (
    NewsSentimentProvider,
    SentimentAnalyzer,
    SentimentConfig,
    SentimentData,
    SocialSentimentProvider,
    get_sentiment_score,
    update_sentiment,
)


class TestSentimentData:
    """Test SentimentData dataclass."""

    def test_sentiment_data_creation(self):
        """Test creating SentimentData objects."""
        timestamp = datetime.datetime.now()
        data = SentimentData(
            symbol="AAPL", score=0.5, magnitude=0.8, timestamp=timestamp, source="news"
        )

        assert data.symbol == "AAPL"
        assert data.score == 0.5
        assert data.magnitude == 0.8
        assert data.timestamp == timestamp
        assert data.source == "news"
        assert data.raw_data is None

    def test_sentiment_data_with_raw_data(self):
        """Test SentimentData with raw data."""
        raw_data = {"title": "Test news", "url": "http://example.com"}
        data = SentimentData(
            symbol="GOOGL",
            score=-0.3,
            magnitude=0.9,
            timestamp=datetime.datetime.now(),
            source="news",
            raw_data=raw_data,
        )

        assert data.raw_data == raw_data


class TestSentimentProviders:
    """Test sentiment provider classes."""

    def test_news_provider_mock_sentiment(self):
        """Test news provider returns mock sentiment data."""
        provider = NewsSentimentProvider()  # No API key = mock mode
        sentiment_data = provider.fetch_sentiment("AAPL", days_back=3)

        assert len(sentiment_data) > 0
        assert len(sentiment_data) <= 5  # Mock returns max 5

        for data in sentiment_data:
            assert isinstance(data, SentimentData)
            assert data.symbol == "AAPL"
            assert -1.0 <= data.score <= 1.0
            assert 0.0 <= data.magnitude <= 1.0
            assert data.source == "news_mock"

    def test_news_provider_symbol_to_company(self):
        """Test symbol to company name conversion."""
        provider = NewsSentimentProvider()

        assert provider._symbol_to_company("AAPL") == "Apple Inc"
        assert provider._symbol_to_company("GOOGL") == "Google Alphabet"
        assert provider._symbol_to_company("UNKNOWN") == "UNKNOWN"

    def test_news_provider_text_sentiment(self):
        """Test text sentiment analysis."""
        provider = NewsSentimentProvider()

        positive_text = "Company reports strong growth and bullish outlook"
        negative_text = "Stock falls amid bearish predictions and weak performance"
        neutral_text = "Company announces new product launch"

        pos_score = provider._analyze_text_sentiment(positive_text)
        neg_score = provider._analyze_text_sentiment(negative_text)
        neu_score = provider._analyze_text_sentiment(neutral_text)

        assert pos_score > 0
        assert neg_score < 0
        assert abs(neu_score) < NEUTRAL_THRESHOLD

    def test_social_provider_mock_sentiment(self):
        """Test social provider returns mock sentiment data."""
        provider = SocialSentimentProvider()
        sentiment_data = provider.fetch_sentiment("TSLA", days_back=2)

        assert len(sentiment_data) > 0
        assert len(sentiment_data) <= 3  # Mock returns max 3

        for data in sentiment_data:
            assert isinstance(data, SentimentData)
            assert data.symbol == "TSLA"
            assert -1.0 <= data.score <= 1.0
            assert 0.0 <= data.magnitude <= 1.0
            assert data.source == "social_mock"


class TestSentimentConfig:
    """Test SentimentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SentimentConfig()

        assert config.enable_news is True
        assert config.enable_social is True
        assert config.news_api_key is None
        assert config.social_api_key is None
        assert config.cache_duration_hours == 1
        assert config.sentiment_weight == 0.2

    def test_custom_config(self):
        """Test custom configuration."""
        config = SentimentConfig(
            enable_news=False,
            enable_social=True,
            news_api_key="test_key",
            cache_duration_hours=6,
            sentiment_weight=0.5,
        )

        assert config.enable_news is False
        assert config.enable_social is True
        assert config.news_api_key == "test_key"
        assert config.cache_duration_hours == 6
        assert config.sentiment_weight == 0.5


class TestSentimentAnalyzer:
    """Test main SentimentAnalyzer class."""

    def test_analyzer_initialization_default(self):
        """Test analyzer initialization with default config."""
        analyzer = SentimentAnalyzer()

        assert len(analyzer.providers) == 2  # News + Social
        assert isinstance(analyzer.providers[0], NewsSentimentProvider)
        assert isinstance(analyzer.providers[1], SocialSentimentProvider)
        assert analyzer.sentiment_cache == {}

    def test_analyzer_initialization_custom_config(self):
        """Test analyzer with custom configuration."""
        config = SentimentConfig(enable_news=True, enable_social=False)
        analyzer = SentimentAnalyzer(config)

        assert len(analyzer.providers) == 1  # News only
        assert isinstance(analyzer.providers[0], NewsSentimentProvider)

    def test_get_symbol_sentiment(self):
        """Test getting aggregated sentiment for a symbol."""
        analyzer = SentimentAnalyzer()
        sentiment_score = analyzer.get_symbol_sentiment("AAPL", days_back=1)

        assert isinstance(sentiment_score, float)
        assert -1.0 <= sentiment_score <= 1.0

    def test_fetch_all_sentiment(self):
        """Test fetching sentiment from all providers."""
        analyzer = SentimentAnalyzer()
        sentiment_data = analyzer.fetch_all_sentiment("MSFT", days_back=2)

        assert len(sentiment_data) > 0
        for data in sentiment_data:
            assert isinstance(data, SentimentData)
            assert data.symbol == "MSFT"

    def test_sentiment_caching(self):
        """Test sentiment data caching."""
        analyzer = SentimentAnalyzer()

        # First call should fetch from providers
        sentiment_data_1 = analyzer.fetch_all_sentiment("NVDA", days_back=1)
        assert len(sentiment_data_1) > 0

        # Second call should use cache
        sentiment_data_2 = analyzer.fetch_all_sentiment("NVDA", days_back=1)
        assert sentiment_data_1 == sentiment_data_2

    def test_get_sentiment_features(self):
        """Test getting sentiment features for multiple symbols."""
        analyzer = SentimentAnalyzer()
        symbols = ["AAPL", "GOOGL"]
        features = analyzer.get_sentiment_features(symbols, days_back=1)

        expected_keys = [
            "sentiment_AAPL",
            "sentiment_AAPL_abs",
            "sentiment_GOOGL",
            "sentiment_GOOGL_abs",
        ]

        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], float)

    def test_cache_validity(self):
        """Test cache validity checking."""
        analyzer = SentimentAnalyzer()

        # Recent timestamp should be valid
        recent_time = datetime.datetime.now() - datetime.timedelta(minutes=30)
        assert analyzer._is_cache_valid(recent_time) is True

        # Old timestamp should be invalid
        old_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        assert analyzer._is_cache_valid(old_time) is False


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_sentiment_score_function(self):
        """Test global get_sentiment_score function."""
        score = get_sentiment_score("AAPL", days_back=1)

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_update_sentiment_function(self):
        """Test global update_sentiment function."""
        # Should not raise an exception
        update_sentiment("TSLA", days_back=1)


class TestSentimentIntegration:
    """Test sentiment module integration with other components."""

    def test_sentiment_with_global_dict_backward_compatibility(self):
        """Test that global sentiment dict is updated for backward compatibility."""
        from trading_rl_agent.data import sentiment

        analyzer = SentimentAnalyzer()
        analyzer.fetch_all_sentiment("META", days_back=1)

        # Check that global sentiment dict was updated
        assert "META" in sentiment.sentiment
        assert "score" in sentiment.sentiment["META"]
        assert "magnitude" in sentiment.sentiment["META"]
        assert "timestamp" in sentiment.sentiment["META"]
        assert "source" in sentiment.sentiment["META"]

    @patch("requests.get")
    def test_news_provider_with_api_key(self, mock_get):
        """Test news provider with actual API key (mocked)."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Apple shows strong growth",
                    "description": "Company beats expectations",
                    "publishedAt": "2024-01-01T10:00:00Z",
                }
            ]
        }
        mock_get.return_value = mock_response

        provider = NewsSentimentProvider(api_key="test_key")
        sentiment_data = provider.fetch_sentiment("AAPL", days_back=1)

        assert len(sentiment_data) > 0
        mock_get.assert_called_once()

    def test_sentiment_error_handling(self):
        """Test sentiment provider error handling."""
        # Test with provider that raises exception
        from trading_rl_agent.data.sentiment import SentimentProvider

        class FailingProvider(SentimentProvider):
            def fetch_sentiment(self, symbol, days_back=1):
                raise Exception("Test error")

        analyzer = SentimentAnalyzer()
        analyzer.providers = [FailingProvider()]

        # Should not raise exception, should return empty list
        sentiment_data = analyzer.fetch_all_sentiment("TEST", days_back=1)
        assert sentiment_data == []

    def test_sentiment_aggregation_weighting(self):
        """Test sentiment score aggregation with magnitude and recency weighting."""
        analyzer = SentimentAnalyzer()

        # Create test sentiment data with different magnitudes and timestamps
        now = datetime.datetime.now()
        test_data = [
            SentimentData("TEST", 0.8, 0.9, now, "news"),  # High confidence, recent
            SentimentData(
                "TEST", -0.5, 0.3, now - datetime.timedelta(hours=12), "social"
            ),  # Low confidence, older
            SentimentData(
                "TEST", 0.2, 0.7, now - datetime.timedelta(hours=1), "news"
            ),  # Medium confidence, recent
        ]

        # Mock the fetch_all_sentiment to return our test data
        analyzer.sentiment_cache["TEST_1"] = test_data

        # Get aggregated score
        score = analyzer.get_symbol_sentiment("TEST", days_back=1)

        # Should be positive (weighted toward high-confidence positive sentiment)
        assert score > 0
        assert isinstance(score, float)


class TestSentimentErrorCases:
    """Test error cases and edge conditions."""

    def test_empty_sentiment_data(self):
        """Test handling of empty sentiment data."""
        analyzer = SentimentAnalyzer()
        analyzer.providers = []  # No providers

        score = analyzer.get_symbol_sentiment("EMPTY", days_back=1)
        assert score == 0.0

    def test_invalid_sentiment_scores(self):
        """Test handling of invalid sentiment scores."""
        # This would be handled by the provider implementation
        # Our mock providers should always return valid scores
        provider = NewsSentimentProvider()
        sentiment_data = provider.fetch_sentiment("TEST", days_back=1)

        for data in sentiment_data:
            assert -1.0 <= data.score <= 1.0
            assert 0.0 <= data.magnitude <= 1.0

    def test_very_old_cache_data(self):
        """Test handling of very old cached data."""
        analyzer = SentimentAnalyzer()

        # Create very old timestamp
        very_old = datetime.datetime.now() - datetime.timedelta(days=10)
        assert analyzer._is_cache_valid(very_old) is False


class TestRealAPISentiment:
    """Test sentiment analysis using the real API and scraping fallback."""

    def test_news_provider_real_api_or_scrape(self):
        """Test news provider with real API key or scraping fallback."""
        import os

        api_key = os.environ.get("NEWSAPI_KEY")
        provider = NewsSentimentProvider(api_key=api_key)
        data = provider.fetch_sentiment("AAPL", days_back=1)
        assert len(data) > 0
        for d in data:
            assert isinstance(d, SentimentData)
            assert d.symbol == "AAPL"
            assert -1.0 <= d.score <= 1.0
            assert 0.0 <= d.magnitude <= 1.0
            assert d.source in ["news", "news_scrape", "news_mock"]

    @patch("src.data.sentiment.requests.get", side_effect=Exception("API fail"))
    def test_news_provider_scrape_fallback(self, mock_get):
        """Test scraping fallback when API fails."""
        provider = NewsSentimentProvider(api_key="fake_key")
        data = provider.fetch_sentiment("AAPL", days_back=1)
        assert len(data) > 0
        for d in data:
            assert d.source == "news_scrape" or d.source == "news_mock"

    @patch("src.data.sentiment.requests.get", side_effect=Exception("API fail"))
    @patch("src.data.sentiment.BeautifulSoup", side_effect=Exception("Scrape fail"))
    def test_news_provider_mock_fallback(self, mock_bs, mock_get):
        """Test mock fallback when both API and scraping fail."""
        provider = NewsSentimentProvider(api_key="fake_key")
        data = provider.fetch_sentiment("AAPL", days_back=1)
        assert len(data) > 0
        for d in data:
            assert d.source == "news_mock"

    def test_env_api_key_is_used(self):
        """Test that NEWSAPI_KEY from .env is loaded and used."""
        import os

        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("NEWSAPI_KEY")
        config = SentimentConfig(news_api_key=api_key)
        provider = NewsSentimentProvider(api_key=config.news_api_key)
        assert provider.api_key == api_key


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
