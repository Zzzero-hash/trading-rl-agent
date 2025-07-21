"""
Live Data Feed - Real-time market data ingestion and processing.

This module provides real-time data feeds for live trading with:
- Multiple data source support (yfinance, alpaca, etc.)
- Feature engineering for real-time data
- Data caching and optimization
- Error handling and retry logic
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from trade_agent.features.alternative_data import SentimentAnalyzer
from trade_agent.features.technical_indicators import TechnicalIndicators


class LiveDataFeed:
    """Real-time market data feed for live trading."""

    def __init__(self, symbols: list[str], data_source: str = "yfinance"):
        self.symbols = symbols
        self.data_source = data_source
        self.logger = logging.getLogger(__name__)

        # Data caching
        self.price_cache: dict[str, pd.DataFrame] = {}
        self.feature_cache: dict[str, np.ndarray] = {}
        self.last_update: dict[str, datetime] = {}

        # Feature engineering
        self.feature_engine = FeatureEngine()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Connection state
        self.connected = False
        self.update_interval = 60  # seconds

    async def connect(self) -> None:
        """Connect to data source."""
        try:
            if self.data_source == "yfinance":
                # yfinance doesn't require explicit connection
                self.connected = True
                self.logger.info("Connected to Yahoo Finance data feed")
            else:
                raise ValueError(f"Unsupported data source: {self.data_source}")

        except Exception:
            self.logger.exception("Failed to connect to data source")
            raise

    async def disconnect(self) -> None:
        """Disconnect from data source."""
        self.connected = False
        self.logger.info("Disconnected from data feed")

    async def get_latest_data(self) -> dict[str, float]:
        """Get latest price data for all symbols."""
        if not self.connected:
            raise RuntimeError("Data feed not connected")

        latest_prices = {}

        for symbol in self.symbols:
            try:
                price = await self._get_symbol_price(symbol)
                if price is not None:
                    latest_prices[symbol] = price

            except Exception:
                self.logger.exception(f"Error getting price for {symbol}")

        return latest_prices

    async def get_features(self, symbol: str, window_size: int = 50) -> np.ndarray | None:
        """Get engineered features for a symbol."""
        if not self.connected:
            raise RuntimeError("Data feed not connected")

        try:
            # Check if we have recent features in cache
            if symbol in self.feature_cache:
                last_update = self.last_update.get(symbol)
                if last_update and (datetime.now() - last_update).seconds < self.update_interval:
                    return self.feature_cache[symbol]

            # Get historical data for feature engineering
            historical_data = await self._get_historical_data(symbol, window_size)

            if historical_data is None or len(historical_data) < window_size:
                # Return default features if insufficient data
                return np.zeros(window_size * 10)  # Assuming 10 features per timestep

            # Generate features
            features = self.feature_engine.generate_features(historical_data)

            # Add sentiment features
            sentiment_features = await self._get_sentiment_features(symbol)
            if sentiment_features is not None:
                features = np.concatenate([features, sentiment_features])

            # Cache features
            self.feature_cache[symbol] = features
            self.last_update[symbol] = datetime.now()

            return features

        except Exception:
            self.logger.exception(f"Error generating features for {symbol}")
            return np.zeros(window_size * 10)

    async def _get_symbol_price(self, symbol: str) -> float | None:
        """Get current price for a single symbol."""
        try:
            if self.data_source == "yfinance":
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get("regularMarketPrice", info.get("previousClose"))
                return float(price) if price is not None else None
            raise ValueError(f"Unsupported data source: {self.data_source}")

        except Exception:
            self.logger.exception(f"Error getting price for {symbol}")
            return None

    async def _get_historical_data(self, symbol: str, window_size: int) -> pd.DataFrame | None:
        """Get historical data for feature engineering."""
        try:
            if self.data_source == "yfinance":
                ticker = yf.Ticker(symbol)

                # Get recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=window_size * 2)  # Get extra data

                hist = ticker.history(start=start_date, end=end_date, interval="1d")

                if hist.empty:
                    return None

                # Ensure we have enough data
                if len(hist) < window_size:
                    return None

                # Return the most recent window_size rows
                return hist.tail(window_size)

            raise ValueError(f"Unsupported data source: {self.data_source}")

        except Exception:
            self.logger.exception(f"Error getting historical data for {symbol}")
            return None

    async def _get_sentiment_features(self, symbol: str) -> np.ndarray | None:
        """Get sentiment features for a symbol."""
        try:
            # Get news sentiment
            news_sentiment = self.sentiment_analyzer.get_symbol_sentiment(symbol)

            # Placeholder for social sentiment
            social_sentiment = 0.0

            # Combine sentiment features
            return np.array(
                [
                    news_sentiment,
                    social_sentiment,
                ],
            )

        except Exception:
            self.logger.exception(f"Error getting sentiment for {symbol}")
            return None

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get basic information about a symbol."""
        try:
            if self.data_source == "yfinance":
                ticker = yf.Ticker(symbol)
                info = ticker.info

                return {
                    "symbol": symbol,
                    "name": info.get("longName", symbol),
                    "sector": info.get("sector", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                }
            return {"symbol": symbol, "name": symbol}

        except Exception:
            self.logger.exception(f"Error getting info for {symbol}")
            return {"symbol": symbol, "name": symbol}


class FeatureEngine:
    """Engine for generating features from market data."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()

    def generate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Generate features from market data."""
        try:
            # Ensure we have required columns
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in data.columns for col in required_columns):
                self.logger.warning("Missing required columns for feature generation")
                return np.zeros((len(data), 10))

            # Generate technical indicators
            features = self.technical_indicators.calculate_all_indicators(data)

            # Convert to numpy array
            if isinstance(features, pd.DataFrame):
                features = features.values

            # Flatten features
            return features.flatten()

        except Exception:
            self.logger.exception("Error generating features")
            return np.zeros((len(data), 10))

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [
            "sma_5",
            "sma_10",
            "sma_20",
            "ema_5",
            "ema_10",
            "ema_20",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "adx",
            "cci",
            "stoch_k",
            "stoch_d",
            "volume_sma",
            "volume_ratio",
            "price_change",
            "price_change_pct",
            "high_low_ratio",
            "close_open_ratio",
        ]


class DataQualityChecker:
    """Check data quality and validity."""

    @staticmethod
    def check_price_data(data: pd.DataFrame) -> dict[str, Any]:
        """Check for anomalies in price data."""
        results = {
            "missing_data": data.isnull().sum().sum(),
            "zeros": (data == 0).sum().sum(),
        }

        # Check for price anomalies
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if col in data.columns and len(data) > 1:
                # Check for extreme price changes (>50% in one day)
                price_changes = data[col].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()
                if extreme_changes > 0:
                    results[f"extreme_changes_{col}"] = extreme_changes

        return results
