"""Alternative data feature calculations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.trade_agent.core.logging import get_logger


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    enable_news: bool = True
    enable_social: bool = True
    provider: str = "yahoo"


class SentimentAnalyzer:
    """Analyzes sentiment from news and social media sources."""

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = get_logger(self.__class__.__name__)

    def get_symbol_sentiment(self, symbol: str) -> float:
        """Get aggregated sentiment score for a symbol."""
        scores = []
        if self.config.enable_news:
            try:
                news_headlines = self._scrape_yahoo_finance(symbol, max_headlines=20)
                for headline in news_headlines:
                    scores.append(self._analyze_text(headline))
            except Exception as e:
                self.logger.warning(f"Could not fetch news sentiment for {symbol}: {e}")

        # Placeholder for social sentiment
        if self.config.enable_social:
            self.logger.debug(f"Social sentiment for {symbol} is not yet implemented.")

        # Log summary of sentiment collection
        if scores:
            self.logger.debug(f"Collected {len(scores)} sentiment scores for {symbol}, avg: {np.mean(scores):.3f}")
        else:
            self.logger.debug(f"No sentiment scores collected for {symbol}, using default")

        return float(np.mean(scores)) if scores else 0.0

    def _scrape_yahoo_finance(self, symbol: str, max_headlines: int = 20) -> list[str]:
        """Scrape news headlines from Yahoo Finance."""
        # For forex pairs, use general forex news instead of specific pair news
        if symbol.endswith("=X"):
            # Extract base currency for forex pairs
            base_currency = symbol[:3]
            # Try general forex news first
            urls_to_try = [
                f"https://finance.yahoo.com/quote/{base_currency}USD=X/news",
                "https://finance.yahoo.com/news/forex",
                f"https://finance.yahoo.com/quote/USD{base_currency}=X/news",
            ]
        else:
            urls_to_try = [f"https://finance.yahoo.com/quote/{symbol}/news"]

        for url in urls_to_try:
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                headlines = {
                    item.get_text(strip=True)
                    for item in soup.find_all("h3")
                    if item.get_text(strip=True) and len(item.get_text(strip=True)) > 10
                }
                if headlines:
                    return list(headlines)[:max_headlines]
            except requests.RequestException:
                self.logger.debug(f"Failed to fetch news from {url}")
                continue

        # If all URLs fail, return mock headlines for forex pairs
        if symbol.endswith("=X"):
            base_currency = symbol[:3]
            quote_currency = symbol[3:6] if len(symbol) >= 6 else "USD"
            return [
                f"{base_currency}/{quote_currency} shows mixed signals",
                f"Forex market volatility affects {base_currency} pairs",
                f"Central bank decisions impact {base_currency} strength",
                f"Economic data influences {base_currency} trading",
                f"Market sentiment shifts for {base_currency} currency",
            ][:max_headlines]

        return []

    def _analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text string."""
        scores = self.analyzer.polarity_scores(text)
        return float(scores.get("compound", 0.0))


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data features."""

    # Sentiment features
    sentiment_column: str = "news_sentiment"
    enable_news_sentiment: bool = True
    enable_social_sentiment: bool = True
    sentiment_lookback_days: int = 7

    # Economic indicators
    enable_economic_indicators: bool = True
    economic_indicators: list[str] | None = None

    # Market microstructure
    enable_microstructure: bool = True
    microstructure_features: list[str] | None = None

    # Feature robustness
    handle_missing_data: bool = True
    fill_method: str = "forward"  # forward, backward, interpolate, zero

    def __post_init__(self) -> None:
        if self.economic_indicators is None:
            self.economic_indicators = ["vix", "treasury_yield", "dollar_index"]
        if self.microstructure_features is None:
            self.microstructure_features = [
                "bid_ask_spread",
                "order_imbalance",
                "volume_profile",
            ]


class AlternativeDataFeatures:
    """Enhanced alternative data feature engineering with sentiment and economic indicators."""

    def __init__(self, config: AlternativeDataConfig | None = None) -> None:
        self.config = config or AlternativeDataConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Initialize sentiment analyzer if enabled
        self.sentiment_analyzer: SentimentAnalyzer | None = None
        if self.config.enable_news_sentiment or self.config.enable_social_sentiment:
            sentiment_config = SentimentConfig(
                enable_news=self.config.enable_news_sentiment,
                enable_social=self.config.enable_social_sentiment,
            )
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)

    def add_alternative_features(
        self,
        df: pd.DataFrame,
        sentiment: pd.Series | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Add comprehensive alternative data features to the DataFrame."""
        result = df.copy()

        # Add sentiment features
        if self.sentiment_analyzer is not None and symbol is not None:
            result = self._add_sentiment_features(result, symbol)
        elif sentiment is not None:
            result = self._add_provided_sentiment(result, sentiment)
        else:
            result = self._add_default_sentiment(result)

        # Add economic indicators
        if self.config.enable_economic_indicators:
            result = self._add_economic_indicators(result)

        # Add market microstructure features
        if self.config.enable_microstructure:
            result = self._add_microstructure_features(result)

        # Handle missing data
        if self.config.handle_missing_data:
            result = self._handle_missing_data(result)

        return result

    def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment features using the sentiment analyzer."""
        try:
            # Get sentiment scores
            if self.sentiment_analyzer is not None:
                sentiment_score = self.sentiment_analyzer.get_symbol_sentiment(symbol)
            else:
                sentiment_score = 0.0

            # Add sentiment features
            df[f"{self.config.sentiment_column}_score"] = sentiment_score
            df[f"{self.config.sentiment_column}_magnitude"] = abs(sentiment_score)
            df[f"{self.config.sentiment_column}_direction"] = np.sign(sentiment_score)

            # Add rolling sentiment features
            df[f"{self.config.sentiment_column}_ma_5"] = (
                df[f"{self.config.sentiment_column}_score"].rolling(5, min_periods=1).mean()
            )
            df[f"{self.config.sentiment_column}_std_5"] = (
                df[f"{self.config.sentiment_column}_score"].rolling(5, min_periods=1).std()
            )

        except Exception as e:
            self.logger.warning(f"Failed to add sentiment features for {symbol}: {e}")
            # Add default sentiment features
            df = self._add_default_sentiment(df)

        return df

    def _add_provided_sentiment(self, df: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
        """Add provided sentiment data to the DataFrame."""
        df[self.config.sentiment_column] = sentiment.reset_index(drop=True).iloc[: len(df)]

        # Ensure the sentiment column exists and has the right length
        if len(df[self.config.sentiment_column]) < len(df):
            # Pad with the last value
            last_value = df[self.config.sentiment_column].iloc[-1] if len(df[self.config.sentiment_column]) > 0 else 0.0
            df[self.config.sentiment_column] = df[self.config.sentiment_column].fillna(last_value)

        return df

    def _add_default_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default sentiment features when no sentiment data is available."""
        # Add neutral sentiment features
        df[self.config.sentiment_column] = 0.0
        df[f"{self.config.sentiment_column}_magnitude"] = 0.0
        df[f"{self.config.sentiment_column}_direction"] = 0.0
        df[f"{self.config.sentiment_column}_ma_5"] = 0.0
        df[f"{self.config.sentiment_column}_std_5"] = 0.0

        return df

    def _add_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic indicator features (mock implementation for now)."""
        if self.config.economic_indicators is not None:
            for indicator in self.config.economic_indicators:
                if indicator == "vix":
                    # Mock VIX-like volatility index
                    if "close" in df.columns:
                        returns = df["close"].pct_change().fillna(0)
                        df["vix"] = returns.rolling(20, min_periods=1).std() * np.sqrt(252) * 100
                    else:
                        df["vix"] = 20.0  # Default VIX value

                elif indicator == "treasury_yield":
                    # Mock treasury yield (inverse relationship with market stress)
                    df["treasury_yield"] = 2.5 + np.random.normal(0, 0.1, len(df))

                elif indicator == "dollar_index":
                    # Mock dollar index
                    df["dollar_index"] = 100.0 + np.random.normal(0, 1.0, len(df))

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        if self.config.microstructure_features is not None:
            for feature in self.config.microstructure_features:
                if feature == "bid_ask_spread":
                    # Mock bid-ask spread based on volatility
                    if "close" in df.columns:
                        volatility = df["close"].pct_change().rolling(10, min_periods=1).std()
                        df["bid_ask_spread"] = volatility * 0.1  # 10% of volatility
                    else:
                        df["bid_ask_spread"] = 0.001  # Default spread

                elif feature == "order_imbalance":
                    # Mock order imbalance
                    df["order_imbalance"] = np.random.normal(0, 0.1, len(df))

                elif feature == "volume_profile":
                    # Volume profile based on time of day (if timestamp available)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        hour = df["timestamp"].dt.hour
                        # Higher volume during market hours
                        df["volume_profile"] = np.where((hour >= 9) & (hour <= 16), 1.0, 0.5)
                    else:
                        df["volume_profile"] = 1.0

        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in alternative features."""
        # Get alternative feature columns
        alt_cols = [
            col
            for col in df.columns
            if any(
                indicator in col
                for indicator in [
                    self.config.sentiment_column,
                    "vix",
                    "treasury_yield",
                    "dollar_index",
                    "bid_ask_spread",
                    "order_imbalance",
                    "volume_profile",
                ]
            )
        ]

        for col in alt_cols:
            if col in df.columns and df[col].isnull().any():
                if self.config.fill_method == "forward":
                    df[col] = df[col].ffill()
                elif self.config.fill_method == "backward":
                    df[col] = df[col].bfill()
                elif self.config.fill_method == "interpolate":
                    df[col] = df[col].interpolate(method="linear")
                elif self.config.fill_method == "zero":
                    df[col] = df[col].fillna(0.0)

                # Final fallback to zero if any NaN remains
                df[col] = df[col].fillna(0.0)

        return df

    def get_feature_names(self) -> list[str]:
        """Return names of all alternative data features that can be generated."""
        names = []

        # Sentiment features
        names.extend(
            [
                self.config.sentiment_column,
                f"{self.config.sentiment_column}_magnitude",
                f"{self.config.sentiment_column}_direction",
                f"{self.config.sentiment_column}_ma_5",
                f"{self.config.sentiment_column}_std_5",
            ],
        )

        # Economic indicators
        if self.config.enable_economic_indicators and self.config.economic_indicators is not None:
            names.extend(self.config.economic_indicators)

        # Microstructure features
        if self.config.enable_microstructure and self.config.microstructure_features is not None:
            names.extend(self.config.microstructure_features)

        return names
