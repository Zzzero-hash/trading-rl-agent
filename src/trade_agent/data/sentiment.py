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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import ray
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trade_agent.core.logging import get_logger
from trade_agent.data.market_symbols import COMPREHENSIVE_SYMBOLS
from trade_agent.utils.ray_utils import parallel_execute

# Global sentiment cache for backward compatibility
sentiment: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"score": 0.0, "magnitude": 0.0, "timestamp": datetime.datetime.now(), "source": "default"}
)
# Global console for rich output
console = Console()
# Setup logger
logger = get_logger(__name__)
# Constants
UTC = datetime.UTC


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
                from trade_agent.core.unified_config import UnifiedConfig

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
                from trade_agent.core.unified_config import UnifiedConfig

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
            return float(scores["compound"])
        except Exception:
            return 0.1

    def _calculate_magnitude(self, text: str) -> float:
        """Calculate sentiment magnitude (confidence)."""
        try:
            scores = self.analyzer.polarity_scores(text)
            return abs(float(scores["compound"]))
        except Exception:
            return 0.1


class NewsSentimentProvider(SentimentProvider):
    """Provider for news sentiment data."""

    def __init__(self, api_key: str | None = None, _cache_dir: str | None = "data/cache"):
        super().__init__(api_key)
        self.api_key = api_key
        self.request_timeout = 30  # 30 seconds timeout

        # Company name mapping for sentiment analysis
        # This maps symbols to their official company names for news searches
        self.symbol_to_company = {
            # Major stocks - using official company names
            "AAPL": "Apple Inc",
            "GOOGL": "Alphabet Inc",
            "GOOG": "Alphabet Inc",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com Inc",
            "TSLA": "Tesla Inc",
            "META": "Meta Platforms Inc",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc",
            "JPM": "JPMorgan Chase & Co",
            "JNJ": "Johnson & Johnson",
            "V": "Visa Inc",
            "PG": "Procter & Gamble Co",
            "UNH": "UnitedHealth Group Inc",
            "HD": "Home Depot Inc",
            "MA": "Mastercard Inc",
            "DIS": "Walt Disney Co",
            "PYPL": "PayPal Holdings Inc",
            "ADBE": "Adobe Inc",
            "CRM": "Salesforce Inc",
            "NKE": "Nike Inc",

            # Tech sector
            "INTC": "Intel Corporation",
            "AMD": "Advanced Micro Devices Inc",
            "ORCL": "Oracle Corporation",
            "CSCO": "Cisco Systems Inc",
            "IBM": "International Business Machines Corp",
            "QCOM": "Qualcomm Inc",
            "TXN": "Texas Instruments Inc",
            "AVGO": "Broadcom Inc",
            "MU": "Micron Technology Inc",
            "LRCX": "Lam Research Corporation",

            # Financial sector
            "BAC": "Bank of America Corp",
            "WFC": "Wells Fargo & Co",
            "GS": "Goldman Sachs Group Inc",
            "MS": "Morgan Stanley",
            "C": "Citigroup Inc",
            "AXP": "American Express Co",
            "BLK": "BlackRock Inc",
            "SCHW": "Charles Schwab Corp",
            "USB": "U.S. Bancorp",
            "PNC": "PNC Financial Services Group Inc",

            # Healthcare sector
            "PFE": "Pfizer Inc",
            "ABBV": "AbbVie Inc",
            "TMO": "Thermo Fisher Scientific Inc",
            "DHR": "Danaher Corporation",
            "BMY": "Bristol-Myers Squibb Co",
            "ABT": "Abbott Laboratories",
            "LLY": "Eli Lilly and Co",
            "MRK": "Merck & Co Inc",
            "AMGN": "Amgen Inc",
            "GILD": "Gilead Sciences Inc",

            # Energy sector
            "XOM": "Exxon Mobil Corporation",
            "CVX": "Chevron Corporation",
            "COP": "ConocoPhillips",
            "EOG": "EOG Resources Inc",
            "SLB": "Schlumberger Ltd",
            "HAL": "Halliburton Co",
            "BKR": "Baker Hughes Co",
            "PSX": "Phillips 66",
            "VLO": "Valero Energy Corp",
            "MPC": "Marathon Petroleum Corp",

            # Consumer sector
            "KO": "Coca-Cola Co",
            "PEP": "PepsiCo Inc",
            "WMT": "Walmart Inc",
            "COST": "Costco Wholesale Corp",
            "TGT": "Target Corp",
            "LOW": "Lowe's Companies Inc",
            "SBUX": "Starbucks Corp",
            "MCD": "McDonald's Corp",
            "YUM": "Yum Brands Inc",
            "CMCSA": "Comcast Corp",

            # Forex pairs - using currency names
            "EURUSD=X": "Euro US Dollar",
            "GBPUSD=X": "British Pound US Dollar",
            "USDJPY=X": "US Dollar Japanese Yen",
            "USDCHF=X": "US Dollar Swiss Franc",
            "AUDUSD=X": "Australian Dollar US Dollar",
            "USDCAD=X": "US Dollar Canadian Dollar",
            "NZDUSD=X": "New Zealand Dollar US Dollar",
            "EURGBP=X": "Euro British Pound",
            "EURJPY=X": "Euro Japanese Yen",
            "GBPJPY=X": "British Pound Japanese Yen",

            # Cryptocurrencies
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

            # Major indices
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones Industrial Average",
            "^IXIC": "NASDAQ Composite",
            "^RUT": "Russell 2000",
            "^VIX": "CBOE Volatility Index",
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX",
            "^N225": "Nikkei 225",
            "^HSI": "Hang Seng Index",
            "^BSESN": "BSE SENSEX",
            "^AXJO": "S&P/ASX 200",
            "^TNX": "10-Year Treasury Note",
            "^TYX": "30-Year Treasury Bond",
            "^IRX": "13-Week Treasury Bill"
        }

    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> list[SentimentData]:
        """Fetch news sentiment with market-derived fallback priority."""

        # Try NewsAPI first if we have an API key
        if self.api_key:
            try:
                return self._fetch_newsapi_sentiment(symbol, days_back)
            except Exception as e:
                self.logger.warning(f"NewsAPI failed for {symbol}: {e}")

        # Try web scraping as secondary fallback
        try:
            scraped_data = self._fetch_scraped_sentiment(symbol)
            if scraped_data:
                return scraped_data
        except Exception as e:
            self.logger.warning(f"Web scraping failed for {symbol}: {e}")

        # Use market-derived sentiment as primary fallback
        try:
            market_data = self._get_market_data_for_sentiment(symbol, days_back)
            if not market_data.empty:
                self.logger.info(f"Using market-derived sentiment for {symbol}")
                return self._derive_sentiment_from_market_data(symbol, market_data)
        except Exception as e:
            self.logger.warning(f"Market-derived sentiment failed for {symbol}: {e}")

        # Return enhanced mock data as final fallback
        self.logger.info(f"Using enhanced mock sentiment for {symbol}")
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

    def _generate_mock_sentiment(self, symbol: str, days_back: int) -> list[SentimentData]:
        """Generate realistic mock sentiment data for historical periods."""
        import random
        from datetime import datetime, timedelta

        # Generate more realistic sentiment based on market patterns
        num_articles = random.randint(3, 8)  # More articles for historical data
        sentiment_data: list[SentimentData] = []

        # Create a base sentiment trend that follows market patterns
        base_sentiment = self._generate_historical_sentiment_trend(symbol, days_back)

        # Use consistent timezone handling
        utc_now = datetime.now(UTC)

        for i in range(num_articles):
            # Add noise to the base sentiment
            noise = random.uniform(-0.3, 0.3)
            day_offset = random.randint(0, days_back)

            # Get sentiment for this specific day
            base_score = base_sentiment[day_offset] if day_offset < len(base_sentiment) else random.uniform(-0.5, 0.5)

            final_score = max(-1.0, min(1.0, base_score + noise))
            magnitude = random.uniform(0.4, 0.9)  # Higher confidence for historical data

            # Create realistic timestamp with consistent timezone handling
            timestamp = utc_now - timedelta(days=day_offset, hours=random.randint(0, 23))
            # Make timestamp naive for consistency
            timestamp = timestamp.replace(tzinfo=None)

            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=final_score,
                    magnitude=magnitude,
                    timestamp=timestamp,
                    source="historical_mock",
                    raw_data={
                        "mock": True,
                        "article_id": i,
                        "historical_period": True,
                        "days_back": day_offset
                    },
                ),
            )

        return sentiment_data

    def _generate_historical_sentiment_trend(self, _symbol: str, days_back: int) -> list[float]:
        """Generate a realistic historical sentiment trend based on market patterns."""
        import random

        import numpy as np

        # Create a trend that follows typical market sentiment patterns
        trend_length = min(days_back, 365)  # Cap at 1 year

        # Start with a random base sentiment
        base_sentiment = random.uniform(-0.2, 0.2)

        # Add cyclical patterns (weekly, monthly)
        weekly_cycle = np.sin(np.linspace(0, 2 * np.pi * (trend_length // 7), trend_length)) * 0.1
        monthly_cycle = np.sin(np.linspace(0, 2 * np.pi * (trend_length // 30), trend_length)) * 0.05

        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.02, trend_length))

        # Add occasional "events" (earnings, news, etc.)
        events = np.zeros(trend_length)
        num_events = random.randint(2, 5)
        for _ in range(num_events):
            event_day = random.randint(0, trend_length - 1)
            event_impact = random.uniform(-0.4, 0.4)
            # Event impact decays over time
            for i in range(min(7, trend_length - event_day)):
                if event_day + i < trend_length:
                    events[event_day + i] += event_impact * np.exp(-i * 0.3)

        # Combine all components
        sentiment_trend = base_sentiment + weekly_cycle + monthly_cycle + random_walk + events

        # Normalize to [-1, 1] range
        sentiment_trend = np.clip(sentiment_trend, -1.0, 1.0)

        return [float(x) for x in sentiment_trend.tolist()]

    def _derive_sentiment_from_market_data(self, symbol: str, market_data: pd.DataFrame) -> list[SentimentData]:
        """Derive sentiment from market data as a proxy for historical sentiment."""
        sentiment_data: list[SentimentData] = []

        if market_data.empty or "close" not in market_data.columns:
            return sentiment_data

        # Calculate market-based sentiment indicators
        market_data = market_data.copy()

        # Price momentum (short-term)
        market_data["returns"] = market_data["close"].pct_change()
        market_data["momentum_5d"] = market_data["returns"].rolling(5).mean()
        market_data["momentum_20d"] = market_data["returns"].rolling(20).mean()

        # Volatility-based sentiment
        market_data["volatility"] = market_data["returns"].rolling(20).std()

        # Volume-based sentiment
        if "volume" in market_data.columns:
            market_data["volume_ma"] = market_data["volume"].rolling(20).mean()
            market_data["volume_ratio"] = market_data["volume"] / market_data["volume_ma"]
        else:
            market_data["volume_ratio"] = 1.0

        # High-Low spread sentiment
        if "high" in market_data.columns and "low" in market_data.columns:
            market_data["hl_spread"] = (market_data["high"] - market_data["low"]) / market_data["close"]
        else:
            market_data["hl_spread"] = 0.02  # Default 2% spread

        for idx, row in market_data.iterrows():
            if pd.isna(row["returns"]):
                continue

            # Combine multiple indicators into sentiment score
            momentum_score = (row["momentum_5d"] * 0.6 + row["momentum_20d"] * 0.4) * 10

            # Volatility sentiment (lower volatility = more positive sentiment)
            volatility_score = -row["volatility"] * 5 if not pd.isna(row["volatility"]) else 0

            # Volume sentiment (higher volume = stronger sentiment)
            volume_score = (row["volume_ratio"] - 1) * 0.2

            # Spread sentiment (tighter spreads = more positive sentiment)
            spread_score = -row["hl_spread"] * 10

            # Combine all scores
            combined_score = momentum_score + volatility_score + volume_score + spread_score

            # Normalize to [-1, 1] range
            sentiment_score = max(-1.0, min(1.0, combined_score))

            # Calculate confidence based on data quality
            confidence = 0.5 + 0.3 * (1 - abs(sentiment_score))  # Higher confidence for extreme values

            # Handle timestamp conversion consistently
            if hasattr(idx, "to_pydatetime"):
                timestamp = idx.to_pydatetime()
            elif hasattr(idx, "timestamp"):
                timestamp = pd.Timestamp(idx)
            else:
                timestamp = pd.Timestamp(idx)

            # Ensure timestamp is timezone-naive for consistency
            if hasattr(timestamp, "tz_localize"):
                timestamp = timestamp.tz_localize(None)
            elif hasattr(timestamp, "replace"):
                timestamp = timestamp.replace(tzinfo=None)

            sentiment_data.append(
                SentimentData(
                    symbol=symbol,
                    score=sentiment_score,
                    magnitude=confidence,
                    timestamp=timestamp,
                    source="market_derived",
                    raw_data={
                        "momentum_5d": row.get("momentum_5d", 0),
                        "momentum_20d": row.get("momentum_20d", 0),
                        "volatility": row.get("volatility", 0),
                        "volume_ratio": row.get("volume_ratio", 1),
                        "hl_spread": row.get("hl_spread", 0),
                        "derived_from_market": True
                    },
                ),
            )

        return sentiment_data

    def _get_market_data_for_sentiment(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Fetch market data for sentiment derivation."""
        try:
            from datetime import datetime, timedelta

            import yfinance as yf

            # Calculate date range with consistent timezone handling
            utc_now = datetime.now(UTC)
            end_date = utc_now.replace(tzinfo=None)  # Make naive for yfinance
            start_date = end_date - timedelta(days=days_back + 30)  # Extra days for rolling calculations

            # Fetch market data
            ticker = yf.Ticker(symbol)
            market_data = ticker.history(start=start_date, end=end_date, interval="1d")

            if market_data.empty:
                self.logger.warning(f"No market data available for {symbol}")
                return pd.DataFrame()

            # Ensure we have the required columns
            required_columns = ["Close", "Volume"]
            if not all(col in market_data.columns for col in required_columns):
                self.logger.warning(f"Missing required columns for {symbol}: {market_data.columns.tolist()}")
                return pd.DataFrame()

            # Rename columns to match our sentiment derivation method
            market_data = market_data.rename(columns={
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume"
            })

            # Add missing columns if needed
            if "high" not in market_data.columns:
                market_data["high"] = market_data["close"]
            if "low" not in market_data.columns:
                market_data["low"] = market_data["close"]
            if "open" not in market_data.columns:
                market_data["open"] = market_data["close"]

            # Ensure timestamps are timezone-naive for consistency
            if hasattr(market_data.index, "tz_localize"):
                market_data.index = market_data.index.tz_localize(None)

            self.logger.debug(f"Fetched {len(market_data)} days of market data for {symbol}")
            return market_data

        except Exception as e:
            self.logger.warning(f"Failed to fetch market data for {symbol}: {e}")
            return pd.DataFrame()

    def _build_company_mapping(self) -> dict[str, str]:
        """Build company mapping from comprehensive symbols."""
        mapping = {}

        # Add all symbols from market_symbols with default company names
        for asset_type, symbols in COMPREHENSIVE_SYMBOLS.items():
            for symbol in symbols:
                if symbol not in mapping:
                    # Use the symbol as default company name if not in our curated list
                    mapping[symbol] = self.symbol_to_company.get(symbol, symbol)

        return mapping

    def _symbol_to_company(self, symbol: str) -> str:
        """Convert symbol to company name."""
        # First check our curated mapping
        if symbol in self.symbol_to_company:
            return self.symbol_to_company[symbol]

        # For symbols not in our curated list, try to generate a reasonable name
        if symbol.endswith("=X"):
            # Forex pairs
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            return f"{base_currency} {quote_currency} Exchange Rate"
        elif symbol.endswith("=F"):
            # Futures
            return f"{symbol[:-2]} Futures"
        elif symbol.startswith("^"):
            # Indices
            index_name = symbol[1:]
            return f"{index_name} Index"
        elif symbol.endswith("-USD"):
            # Crypto
            crypto_name = symbol[:-4]
            return f"{crypto_name} Cryptocurrency"
        else:
            # Default to symbol
            return symbol


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


@ray.remote
def _analyze_symbol_sentiment(
    symbol: str, days_back: int, config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Analyze sentiment for a single symbol as a Ray remote task."""
    try:
        from trade_agent.data.sentiment import SentimentAnalyzer, SentimentConfig

        analyzer = SentimentAnalyzer(SentimentConfig(**config))
        score = analyzer.get_symbol_sentiment_with_market_fallback(symbol, days_back)

        return symbol, {
            "score": score,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "success",
        }
    except Exception as e:
        return symbol, {
            "score": 0.0,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }


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

        # Use timezone-naive datetime for consistency
        now = datetime.datetime.now().replace(tzinfo=None)

        for data in sentiment_data:
            # Ensure timestamp is timezone-naive for comparison
            timestamp = data.timestamp
            if hasattr(timestamp, "tz_localize"):
                timestamp = timestamp.tz_localize(None)
            elif hasattr(timestamp, "replace"):
                timestamp = timestamp.replace(tzinfo=None)

            # Time decay: more recent data gets higher weight
            time_diff = (now - timestamp).total_seconds() / 3600  # hours
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

        # Use timezone-naive datetime for consistency
        now = datetime.datetime.now().replace(tzinfo=None)
        cache_timestamp = self.cache_timestamps[cache_key]

        # Ensure cache timestamp is timezone-naive for comparison
        if hasattr(cache_timestamp, "tz_localize"):
            cache_timestamp = cache_timestamp.tz_localize(None)
        elif hasattr(cache_timestamp, "replace"):
            cache_timestamp = cache_timestamp.replace(tzinfo=None)

        cache_age = now - cache_timestamp
        return cache_age.total_seconds() < (self.config.cache_duration_hours * 3600)

    def get_symbol_sentiment_with_market_fallback(self, symbol: str, days_back: int = 1, market_data: pd.DataFrame | None = None) -> float:
        """Get sentiment score with market-derived fallback."""
        # Try to get news sentiment first
        sentiment_data = self.fetch_all_sentiment(symbol, days_back)

        # If we have news sentiment, use it
        if sentiment_data and any(d.source not in ["historical_mock", "market_derived"] for d in sentiment_data):
            return self.get_symbol_sentiment(symbol, days_back)

        # If no news sentiment and we have market data, derive sentiment from market data
        if market_data is not None and not market_data.empty:
            try:
                from .sentiment import NewsSentimentProvider
                provider = NewsSentimentProvider()
                market_sentiment = provider._derive_sentiment_from_market_data(symbol, market_data)
                if market_sentiment:
                    # Return weighted average of market-derived sentiment
                    scores = [d.score * d.magnitude for d in market_sentiment]
                    weights = [d.magnitude for d in market_sentiment]
                    if weights and sum(weights) > 0:
                        return sum(scores) / sum(weights)
            except Exception as e:
                self.logger.warning(f"Market-derived sentiment failed for {symbol}: {e}")

        # Fallback to regular sentiment (which will use enhanced mock)
        return self.get_symbol_sentiment(symbol, days_back)

    def get_sentiment_features(self, symbols: list[str], days_back: int = 1) -> pd.DataFrame:
        """Get sentiment features for multiple symbols with robust fallback to 0."""
        features = []

        for symbol in symbols:
            try:
                # Get sentiment score with fallback to 0.0
                sentiment_score = self.get_symbol_sentiment(symbol, days_back)

                # Get sentiment data with fallback to empty list
                try:
                    sentiment_data = self.fetch_all_sentiment(symbol, days_back)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch sentiment data for {symbol}: {e}")
                    sentiment_data = []

                # Calculate features with fallbacks
                if sentiment_data:
                    try:
                        avg_magnitude = np.mean([d.magnitude for d in sentiment_data])
                        source_count = len({d.source for d in sentiment_data})
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate sentiment features for {symbol}: {e}")
                        avg_magnitude = 0.0
                        source_count = 0
                else:
                    avg_magnitude = 0.0
                    source_count = 0

                # Ensure sentiment_score is a valid float
                if not isinstance(sentiment_score, int | float) or np.isnan(sentiment_score):
                    sentiment_score = 0.0

                # Calculate sentiment direction with fallback
                try:
                    sentiment_direction = 1 if sentiment_score > 0.01 else (-1 if sentiment_score < -0.01 else 0)
                except Exception:
                    sentiment_direction = 0

                features.append(
                    {
                        "symbol": symbol,
                        "sentiment_score": sentiment_score,
                        "sentiment_magnitude": avg_magnitude,
                        "sentiment_sources": source_count,
                        "sentiment_direction": sentiment_direction,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to get sentiment features for {symbol}: {e}")
                features.append(
                    {
                        "symbol": symbol,
                        "sentiment_score": 0.0,
                        "sentiment_magnitude": 0.0,
                        "sentiment_sources": 0,
                        "sentiment_direction": 0,
                    }
                )

        return pd.DataFrame(features)

    def get_sentiment_features_parallel(
        self,
        symbols: list[str],
        days_back: int = 1,
        max_workers: int = 4
    ) -> pd.DataFrame:
        """Get sentiment features for multiple symbols in parallel."""

        def _get_features_for_symbol(symbol: str) -> pd.DataFrame:
            return self.get_sentiment_features([symbol], days_back)

        results = parallel_execute(
            _get_features_for_symbol,
            symbols,
            max_workers=max_workers
        )

        all_features = [features for features in results if not features.empty]

        if not all_features:
            return pd.DataFrame()

        return pd.concat(all_features, ignore_index=True)


# Global functions for backward compatibility
def get_sentiment_score(symbol: str) -> float:
    """Get the current sentiment score for a symbol."""
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
