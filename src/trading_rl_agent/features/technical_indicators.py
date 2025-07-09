"""
Technical indicators module using industry-standard libraries.

Provides comprehensive technical analysis indicators using pandas-ta for robust
feature engineering in trading strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""

    # Moving averages
    sma_periods: Optional[list[int]] = None
    ema_periods: Optional[list[int]] = None

    # Momentum indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volatility indicators
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # Volume indicators
    obv_enabled: bool = True
    vwap_enabled: bool = True

    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 50, 200]


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator using multiple libraries.

    Supports both talib (C-based, fast) and pandas-ta (Python-based, flexible)
    for maximum compatibility and performance.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.logger = get_logger(self.__class__.__name__)

        try:
            import pandas_ta  # noqa: F401
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "pandas-ta is required for TechnicalIndicators. "
                "Please install it with `pip install pandas-ta`."
            ) from exc

        self.logger.info("Using pandas-ta for technical indicators")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive set of technical indicators.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with original data plus technical indicators
        """
        result_df = df.copy()

        try:
            # Moving averages
            result_df = self._add_moving_averages(result_df)

            # Momentum indicators
            result_df = self._add_momentum_indicators(result_df)

            # Volatility indicators
            result_df = self._add_volatility_indicators(result_df)

            # Volume indicators
            result_df = self._add_volume_indicators(result_df)

            # Pattern recognition
            result_df = self._add_pattern_recognition(result_df)

            self.logger.info(
                f"Calculated {len(result_df.columns) - len(df.columns)} indicators"
            )
            return result_df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        for period in self.config.sma_periods:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)

        for period in self.config.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI
        df["rsi"] = ta.rsi(df["close"], length=self.config.rsi_period)

        # MACD
        macd_df = ta.macd(
            df["close"],
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            talib=False,
        )
        df["macd"] = macd_df[
            f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"
        ]
        df["macd_signal"] = macd_df[
            f"MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"
        ]
        df["macd_histogram"] = macd_df[
            f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"
        ]

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        bb = ta.bbands(
            df["close"], length=self.config.bb_period, std=self.config.bb_std
        )
        df["bb_upper"] = bb[f"BBU_{self.config.bb_period}_{self.config.bb_std}"]
        df["bb_middle"] = bb[f"BBM_{self.config.bb_period}_{self.config.bb_std}"]
        df["bb_lower"] = bb[f"BBL_{self.config.bb_period}_{self.config.bb_std}"]

        # ATR (Average True Range)
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"], length=self.config.atr_period
        )

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if "volume" not in df.columns:
            self.logger.warning("Volume data not available, skipping volume indicators")
            return df

        # On-Balance Volume (OBV)
        if self.config.obv_enabled:
            df["obv"] = ta.obv(df["close"], df["volume"])

        # VWAP (Volume Weighted Average Price)
        if self.config.vwap_enabled:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        return df

    def _add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition using pandas-ta."""

        pattern_names = [
            "doji",
            "hammer",
            "engulfing",
            "harami",
            "morning_star",
            "evening_star",
        ]

        for name in pattern_names:
            try:
                res = ta.cdl_pattern(
                    df["open"], df["high"], df["low"], df["close"], name=name
                )
                if res is not None:
                    df[f"pattern_{name}"] = res.iloc[:, 0]
            except Exception as e:  # pragma: no cover - non-critical
                self.logger.warning(f"Failed to calculate pattern {name}: {e}")

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names that would be generated."""
        features = []

        # Moving averages
        for period in self.config.sma_periods:
            features.append(f"sma_{period}")
        for period in self.config.ema_periods:
            features.append(f"ema_{period}")

        # Momentum
        features.extend(["rsi", "macd", "macd_signal", "macd_histogram"])

        # Volatility
        features.extend(["bb_upper", "bb_middle", "bb_lower", "atr"])

        # Volume
        if self.config.obv_enabled:
            features.append("obv")
        if self.config.vwap_enabled:
            features.append("vwap")

        # Pattern recognition features
        pattern_names = [
            "doji",
            "hammer",
            "engulfing",
            "harami",
            "morning_star",
            "evening_star",
        ]
        features.extend([f"pattern_{name}" for name in pattern_names])

        return features
