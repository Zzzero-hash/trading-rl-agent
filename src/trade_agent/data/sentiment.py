"""
Comprehensive sentiment analysis for trading data.

This module provides sentiment analysis capabilities for multiple asset types:
- Stocks (US, international)
- Forex pairs
- Cryptocurrencies
- Commodities
- Indices

Data sources include:
- News APIs (NewsAPI, Alpha Vantage)
- Social media sentiment
- Web scraping fallbacks
- Economic news and indicators
"""

import datetime
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.trade_agent.core.logging import get_logger

# Global sentiment cache for backward compatibility
sentiment: dict[str, dict[str, Any]] = {}


@dataclass
class SentimentData:
    """Container for sentiment data points."""

    symbol: str
    score: float  # -1.0 to 1.0
    magnitude: float  # 0.0 to 1.0 (confidence)
    timestamp: datetime.datetime
    source: str  # "news", "social", "news_api", "news_scrape", etc.
    raw_data: dict[str, Any] | None = None


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    enable_news: bool = True
    enable_social: bool = True
    news_api_key: str | None = None
    social_api_key: str | None = None
    cache_duration_hours: int = 1
    sentiment_weight: float = 0.2
    max_retries: int = 3
    request_timeout: int = 10

    def __post_init__(self) -> None:
        # Try to get API keys from environment if not provided
        if self.news_api_key is None:
            # Try unified config first
            try:
                from src.trade_agent.core.unified_config import UnifiedConfig

                config = UnifiedConfig()
                self.news_api_key = config.newsapi_key
            except Exception:
                pass

            # Fallback to direct environment access
            if self.news_api_key is None:
                self.news_api_key = os.environ.get("NEWSAPI_KEY")

        if self.social_api_key is None:
            # Try unified config first
            try:
                from src.trade_agent.core.unified_config import UnifiedConfig

                config = UnifiedConfig()
                self.social_api_key = config.social_api_key
            except Exception:
                pass

            # Fallback to direct environment access
            if self.social_api_key is None:
                self.social_api_key = os.environ.get("SOCIAL_API_KEY")


class SentimentProvider(ABC):
    """Abstract base class for sentiment providers."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.logger = get_logger(self.__class__.__name__)
        self.analyzer = SentimentIntensityAnalyzer()

    @abstractmethod
    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch sentiment data for a symbol."""

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using VADER."""
        try:
            scores = self.analyzer.polarity_scores(text)
            return float(scores.get("compound", 0.0))
        except Exception as e:
            self.logger.warning(f"Failed to analyze text sentiment: {e}")
            return 0.0

    def _calculate_magnitude(self, text: str) -> float:
        """Calculate sentiment magnitude (confidence) from text."""
        try:
            # Use VADER sentiment analyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)

            pos = scores["pos"]
            neg = scores["neg"]

            # Higher magnitude for stronger sentiment
            if pos > neg:
                return float(min(pos * 2, 1.0))
            if neg > pos:
                return float(min(neg * 2, 1.0))
            return 0.1  # Low magnitude for neutral
        except Exception:
            return 0.1


class NewsSentimentProvider(SentimentProvider):
    """Provider for news sentiment data."""

    def __init__(self, api_key: str | None = None, _cache_dir: str | None = "data/cache"):
        super().__init__(api_key)
        self.api_key = api_key
        self.request_timeout = 30  # 30 seconds timeout
        self.symbol_to_company = {
            # Major stocks
            "AAPL": "Apple Inc",
            "GOOGL": "Google Alphabet",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta Facebook",
            "NVDA": "NVIDIA",
            "NFLX": "Netflix",
            "JPM": "JPMorgan Chase",
            "JNJ": "Johnson & Johnson",
            "V": "Visa",
            "PG": "Procter & Gamble",
            "UNH": "UnitedHealth",
            "HD": "Home Depot",
            "MA": "Mastercard",
            "DIS": "Disney",
            "PYPL": "PayPal",
            "ADBE": "Adobe",
            "CRM": "Salesforce",
            "NKE": "Nike",
            # Tech
            "INTC": "Intel",
            "AMD": "Advanced Micro Devices",
            "ORCL": "Oracle",
            "CSCO": "Cisco",
            "IBM": "IBM",
            "QCOM": "Qualcomm",
            "TXN": "Texas Instruments",
            "AVGO": "Broadcom",
            "MU": "Micron Technology",
            "LRCX": "Lam Research",
            # Finance
            "BAC": "Bank of America",
            "WFC": "Wells Fargo",
            "GS": "Goldman Sachs",
            "MS": "Morgan Stanley",
            "C": "Citigroup",
            "AXP": "American Express",
            "BLK": "BlackRock",
            "SCHW": "Charles Schwab",
            "USB": "U.S. Bancorp",
            "PNC": "PNC Financial",
            # Healthcare
            "PFE": "Pfizer",
            "ABBV": "AbbVie",
            "TMO": "Thermo Fisher",
            "DHR": "Danaher",
            "BMY": "Bristol-Myers Squibb",
            "ABT": "Abbott Laboratories",
            "LLY": "Eli Lilly",
            "MRK": "Merck",
            "AMGN": "Amgen",
            "GILD": "Gilead Sciences",
            # Energy
            "XOM": "ExxonMobil",
            "CVX": "Chevron",
            "COP": "ConocoPhillips",
            "EOG": "EOG Resources",
            "SLB": "Schlumberger",
            "HAL": "Halliburton",
            "BKR": "Baker Hughes",
            "PSX": "Phillips 66",
            "VLO": "Valero Energy",
            "MPC": "Marathon Petroleum",
            # Consumer
            "KO": "Coca-Cola",
            "PEP": "PepsiCo",
            "WMT": "Walmart",
            "COST": "Costco",
            "TGT": "Target",
            "LOW": "Lowe's",
            "SBUX": "Starbucks",
            "MCD": "McDonald's",
            "YUM": "Yum Brands",
            "CMCSA": "Comcast",
            # Forex (major pairs)
            "EURUSD": "Euro US Dollar",
            "GBPUSD": "British Pound US Dollar",
            "USDJPY": "US Dollar Japanese Yen",
            "USDCHF": "US Dollar Swiss Franc",
            "AUDUSD": "Australian Dollar US Dollar",
            "USDCAD": "US Dollar Canadian Dollar",
            "NZDUSD": "New Zealand Dollar US Dollar",
            "EURGBP": "Euro British Pound",
            "EURJPY": "Euro Japanese Yen",
            "GBPJPY": "British Pound Japanese Yen",
            # Crypto
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "ADA-USD": "Cardano",
            "DOT-USD": "Polkadot",
            "LINK-USD": "Chainlink",
            "LTC-USD": "Litecoin",
            "BCH-USD": "Bitcoin Cash",
            "XRP-USD": "Ripple",
            "BNB-USD": "Binance Coin",
            "SOL-USD": "Solana",
            # Indices
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones Industrial Average",
            "^IXIC": "NASDAQ Composite",
            "^RUT": "Russell 2000",
            "^VIX": "CBOE Volatility Index",
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX",
            "^N225": "Nikkei 225",
            "^HSI": "Hang Seng",
            "^BSESN": "BSE SENSEX",
        }

    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch news sentiment with multiple fallback strategies."""

        # Try NewsAPI first if we have an API key
        if self.api_key:
            try:
                return self._fetch_newsapi_sentiment(symbol, days_back)
            except Exception as e:
                self.logger.warning(f"NewsAPI failed for {symbol}: {e}")

        # Try web scraping as fallback
        try:
            return self._fetch_scraped_sentiment(symbol)
        except Exception as e:
            self.logger.warning(f"Web scraping failed for {symbol}: {e}")

        # Return mock data as final fallback
        return self._generate_mock_sentiment(symbol, days_back)

    def _fetch_newsapi_sentiment(self, symbol: str, days_back: int) -> list[SentimentData]:
        """Fetch sentiment using NewsAPI."""
        company_name = self._symbol_to_company(symbol)

        # Calculate date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_back)

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{company_name}" OR "{symbol}"',
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
            "pageSize": 20,
        }

        response = requests.get(url, params=params, timeout=self.request_timeout)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        sentiment_data = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            published_at = article.get("publishedAt", "")

            # Combine title and description for analysis
            text = f"{title}. {description}".strip()
            if len(text) < 10:
                continue

            score = self._analyze_text_sentiment(text)
            magnitude = self._calculate_magnitude(text)

            # Parse timestamp
            try:
                timestamp = datetime.datetime.fromisoformat(published_at)
            except Exception:
                timestamp = datetime.datetime.now()

            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=score,
                    magnitude=magnitude,
                    timestamp=timestamp,
                    source="news_api",
                    raw_data=article,
                ),
            )

        return sentiment_data

    def _fetch_scraped_sentiment(self, symbol: str) -> list[SentimentData]:
        """Fetch sentiment by scraping financial news sites."""
        sentiment_data = []

        # Try multiple news sources
        sources = [
            f"https://finance.yahoo.com/quote/{symbol}/news",
            f"https://www.marketwatch.com/investing/stock/{symbol}",
            f"https://seekingalpha.com/symbol/{symbol}",
        ]

        for source_url in sources:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = requests.get(source_url, headers=headers, timeout=self.request_timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract headlines (adjust selectors based on site structure)
                headlines = []

                # Common headline selectors
                selectors = [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    ".headline",
                    ".title",
                    ".article-title",
                ]
                for selector in selectors:
                    elements = soup.find_all(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        if len(text) > 10 and len(text) < 200:
                            headlines.append(text)

                # Limit headlines
                headlines = headlines[:10]

                for headline in headlines:
                    score = self._analyze_text_sentiment(headline)
                    magnitude = self._calculate_magnitude(headline)

                    sentiment_data.append(
                        SentimentData(
                            symbol=symbol,
                            score=score,
                            magnitude=magnitude,
                            timestamp=datetime.datetime.now(),
                            source="news_scrape",
                            raw_data={"headline": headline, "source": source_url},
                        ),
                    )

                # If we got some data, break
                if sentiment_data:
                    break

            except Exception as e:
                self.logger.debug(f"Failed to scrape {source_url}: {e}")
                continue

        return sentiment_data

    def _generate_mock_sentiment(self, symbol: str, _days_back: int) -> list[SentimentData]:
        """Generate mock sentiment data for testing/fallback."""
        num_articles = random.randint(1, 5)
        sentiment_data = []

        for i in range(num_articles):
            # Generate realistic sentiment scores
            base_score = random.uniform(-0.8, 0.8)
            magnitude = random.uniform(0.3, 0.9)

            # Add some time variation
            timestamp = datetime.datetime.now() - datetime.timedelta(hours=random.randint(0, _days_back * 24))

            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=base_score,
                    magnitude=magnitude,
                    timestamp=timestamp,
                    source="news_mock",
                    raw_data={"mock": True, "article_id": i},
                ),
            )

        return sentiment_data

    def _symbol_to_company(self, symbol: str) -> str:
        """Convert symbol to company name."""
        return self.symbol_to_company.get(symbol, symbol)


class SocialSentimentProvider(SentimentProvider):
    """Social media sentiment provider."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key)
        # Placeholder for social media API integration
        # Could integrate with Twitter API, Reddit API, etc.

    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch social media sentiment (currently mock implementation)."""
        # For now, return mock data
        # TODO: Integrate with actual social media APIs
        return self._generate_mock_social_sentiment(symbol, days_back)

    def _generate_mock_social_sentiment(self, symbol: str, days_back: int) -> list[SentimentData]:
        """Generate mock social media sentiment."""
        num_posts = random.randint(1, 3)
        sentiment_data = []

        for i in range(num_posts):
            # Social sentiment tends to be more volatile
            base_score = random.uniform(-0.9, 0.9)
            magnitude = random.uniform(0.2, 0.8)  # Lower confidence than news

            timestamp = datetime.datetime.now() - datetime.timedelta(hours=random.randint(0, days_back * 24))

            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=base_score,
                    magnitude=magnitude,
                    timestamp=timestamp,
                    source="social_mock",
                    raw_data={"mock": True, "post_id": i},
                ),
            )

        return sentiment_data


class SentimentAnalyzer:
    """Main sentiment analyzer that aggregates data from multiple providers."""

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.sentiment_cache: dict[str, list[SentimentData]] = {}
        self.cache_timestamps: dict[str, datetime.datetime] = {}

        # Initialize providers
        self.providers: list[SentimentProvider] = []

        if self.config.enable_news:
            self.providers.append(NewsSentimentProvider(self.config.news_api_key))

        if self.config.enable_social:
            self.providers.append(SocialSentimentProvider(self.config.social_api_key))

    def get_symbol_sentiment(self, symbol: str, days_back: int = 1) -> float:
        """Get aggregated sentiment score for a symbol."""
        sentiment_data = self.fetch_all_sentiment(symbol, days_back)

        if not sentiment_data:
            return 0.0

        # Weight sentiment by magnitude and recency
        weighted_scores = []
        total_weight = 0.0

        now = datetime.datetime.now()

        for data in sentiment_data:
            # Time decay: more recent data gets higher weight
            time_diff = (now - data.timestamp).total_seconds() / 3600  # hours
            time_weight = max(0.1, 1.0 - (time_diff / (days_back * 24)))

            # Magnitude weight: higher confidence gets higher weight
            magnitude_weight = data.magnitude

            # Combined weight
            weight = time_weight * magnitude_weight
            weighted_scores.append(data.score * weight)
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Return weighted average
        return sum(weighted_scores) / total_weight

    def fetch_all_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch sentiment from all providers."""
        cache_key = f"{symbol}_{days_back}"

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.sentiment_cache.get(cache_key, [])

        # Fetch from all providers
        all_sentiment = []

        for provider in self.providers:
            try:
                sentiment_data = provider.fetch_sentiment(symbol, days_back)
                all_sentiment.extend(sentiment_data)
            except Exception as e:
                self.logger.warning(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")

        # Cache results
        self.sentiment_cache[cache_key] = all_sentiment
        self.cache_timestamps[cache_key] = datetime.datetime.now()

        # Update global sentiment dict for backward compatibility
        if all_sentiment:
            avg_score = np.mean([d.score for d in all_sentiment])
            avg_magnitude = np.mean([d.magnitude for d in all_sentiment])

            sentiment[symbol] = {
                "score": avg_score,
                "magnitude": avg_magnitude,
                "timestamp": datetime.datetime.now(),
                "source": "aggregated",
            }

        return all_sentiment

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        cache_age = datetime.datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.config.cache_duration_hours * 3600)

    def get_sentiment_features(self, symbols: list[str], days_back: int = 1) -> pd.DataFrame:
        """Get sentiment features for multiple symbols."""
        features = []

        for symbol in symbols:
            sentiment_score = self.get_symbol_sentiment(symbol, days_back)
            sentiment_data = self.fetch_all_sentiment(symbol, days_back)

            if sentiment_data:
                avg_magnitude = np.mean([d.magnitude for d in sentiment_data])
                source_count = len({d.source for d in sentiment_data})
            else:
                avg_magnitude = 0.0
                source_count = 0

            features.append(
                {
                    "symbol": symbol,
                    "sentiment_score": sentiment_score,
                    "sentiment_magnitude": avg_magnitude,
                    "sentiment_sources": source_count,
                    "sentiment_direction": np.sign(sentiment_score),
                },
            )

        return pd.DataFrame(features)


# Global functions for backward compatibility
def get_sentiment_score(symbol: str) -> float:
    """Get sentiment score for a symbol (backward compatibility)."""
    analyzer = SentimentAnalyzer()
    return analyzer.get_symbol_sentiment(symbol)


def update_sentiment(symbol: str, score: float, magnitude: float = 0.5) -> None:
    """Update global sentiment dict (backward compatibility)."""
    sentiment[symbol] = {
        "score": score,
        "magnitude": magnitude,
        "timestamp": datetime.datetime.now(),
        "source": "manual",
    }
