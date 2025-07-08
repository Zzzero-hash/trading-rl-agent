"""
Technical indicators module using industry-standard libraries.

Provides comprehensive technical analysis indicators using TA-Lib and pandas-ta
for robust feature engineering in trading strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

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

    Supports both TA-Lib (C-based, fast) and pandas-ta (Python-based, flexible)
    for maximum compatibility and performance.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.logger = get_logger(self.__class__.__name__)

        if not TALIB_AVAILABLE and not PANDAS_TA_AVAILABLE:
            raise ImportError(
                "Neither TA-Lib nor pandas-ta is available. "
                "Please install at least one: pip install TA-Lib pandas-ta"
            )

        self.use_talib = TALIB_AVAILABLE
        self.logger.info(
            f"Using TA-Lib: {TALIB_AVAILABLE}, pandas-ta: {PANDAS_TA_AVAILABLE}"
        )

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

            # Pattern recognition (if TA-Lib available)
            if self.use_talib:
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
            if self.use_talib:
                df[f"sma_{period}"] = talib.SMA(df["close"], timeperiod=period)
            else:
                df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

        for period in self.config.ema_periods:
            if self.use_talib:
                df[f"ema_{period}"] = talib.EMA(df["close"], timeperiod=period)
            else:
                df[f"ema_{period}"] = df["close"].ewm(span=period).mean()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI
        if self.use_talib:
            df["rsi"] = talib.RSI(df["close"], timeperiod=self.config.rsi_period)
        else:
            delta = df["close"].diff()
            gain = (
                (delta.where(delta > 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        if self.use_talib:
            macd, macd_signal, macd_hist = talib.MACD(
                df["close"],
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal,
            )
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_histogram"] = macd_hist
        else:
            ema_fast = df["close"].ewm(span=self.config.macd_fast).mean()
            ema_slow = df["close"].ewm(span=self.config.macd_slow).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        if self.use_talib:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df["close"],
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std,
                nbdevdn=self.config.bb_std,
            )
            df["bb_upper"] = bb_upper
            df["bb_middle"] = bb_middle
            df["bb_lower"] = bb_lower
        else:
            sma = df["close"].rolling(window=self.config.bb_period).mean()
            std = df["close"].rolling(window=self.config.bb_period).std()
            df["bb_upper"] = sma + (std * self.config.bb_std)
            df["bb_middle"] = sma
            df["bb_lower"] = sma - (std * self.config.bb_std)

        # ATR (Average True Range)
        if self.use_talib:
            df["atr"] = talib.ATR(
                df["high"], df["low"], df["close"], timeperiod=self.config.atr_period
            )
        else:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            df["atr"] = tr.rolling(window=self.config.atr_period).mean()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if "volume" not in df.columns:
            self.logger.warning("Volume data not available, skipping volume indicators")
            return df

        # On-Balance Volume (OBV)
        if self.config.obv_enabled:
            if self.use_talib:
                df["obv"] = talib.OBV(df["close"], df["volume"])
            else:
                obv = [0]
                for i in range(1, len(df)):
                    if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                        obv.append(obv[-1] + df["volume"].iloc[i])
                    elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                        obv.append(obv[-1] - df["volume"].iloc[i])
                    else:
                        obv.append(obv[-1])
                df["obv"] = obv

        # VWAP (Volume Weighted Average Price)
        if self.config.vwap_enabled:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        return df

    def _add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition (TA-Lib only)."""
        if not self.use_talib:
            return df

        # Key candlestick patterns
        patterns = {
            "doji": talib.CDLDOJI,
            "hammer": talib.CDLHAMMER,
            "engulfing": talib.CDLENGULFING,
            "harami": talib.CDLHARAMI,
            "morning_star": talib.CDLMORNINGSTAR,
            "evening_star": talib.CDLEVENINGSTAR,
        }

        for pattern_name, pattern_func in patterns.items():
            try:
                df[f"pattern_{pattern_name}"] = pattern_func(
                    df["open"], df["high"], df["low"], df["close"]
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate pattern {pattern_name}: {e}")

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

        # Patterns (if TA-Lib available)
        if self.use_talib:
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
