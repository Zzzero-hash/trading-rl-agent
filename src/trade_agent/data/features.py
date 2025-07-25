#!/usr/bin/env python3
"""
Unified Feature Engineering for Trading RL Agent

This module provides a single, consistent feature engineering function that ensures
exactly 78 features are generated, matching the DataStandardizer expectations.
"""


import numpy as np
import pandas as pd
import pandas_ta as ta

from trade_agent.core.logging import get_logger


def generate_unified_features(
    df: pd.DataFrame,
    ma_windows: list | None = None,
    rsi_window: int = 14,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Generate exactly 78 features as expected by the DataStandardizer.

    This function ensures consistent feature engineering between training and inference.

    Args:
        df: DataFrame with OHLCV data
        ma_windows: Moving average windows (default: [5, 10, 20, 50])
        rsi_window: RSI window (default: 14)
        vol_window: Volatility window (default: 20)
        advanced_candles: Whether to include advanced candlestick patterns (default: True)

    Returns:
        DataFrame with exactly 78 features in the correct order
    """
    logger = get_logger("UnifiedFeatures")

    if ma_windows is None:
        ma_windows = [5, 10, 20, 50]

    df = df.copy()

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Clean and validate data
    df = _clean_input_data(df)

    # Generate all 78 features in the exact order expected by DataStandardizer
    result_df = pd.DataFrame(index=df.index)

    # 1. Price features (5)
    result_df["open"] = df["open"]
    result_df["high"] = df["high"]
    result_df["low"] = df["low"]
    result_df["close"] = df["close"]
    result_df["volume"] = df["volume"]

    # 2. Technical indicators (20)
    result_df = _add_technical_indicators(result_df, df, ma_windows, rsi_window, vol_window)

    # 3. Candlestick patterns (18)
    result_df = _add_candlestick_patterns(result_df, df)

    # 4. Candlestick characteristics (9)
    result_df = _add_candlestick_characteristics(result_df, df)

    # 5. Rolling candlestick features (15)
    result_df = _add_rolling_candlestick_features(result_df)

    # 6. Sentiment features (2)
    result_df = _add_sentiment_features(result_df, df)

    # 7. Time features (4)
    result_df = _add_time_features(result_df, df)

    # 8. Market regime features (5)
    result_df = _add_market_regime_features(result_df, df)

    # Final validation
    expected_features = 78
    actual_features = len([col for col in result_df.columns if col not in ["timestamp", "symbol", "data_source"]])

    if actual_features != expected_features:
        logger.error(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
        logger.error(f"Extra features: {[col for col in result_df.columns if col not in _get_expected_feature_names()]}")
        raise ValueError(f"Feature count mismatch: expected {expected_features}, got {actual_features}")

    # Ensure no NaN or inf values
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].fillna(0.0)
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], 0.0)

    logger.info(f"Generated {actual_features} features successfully")
    return result_df


def _clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate input data with robust handling."""
    logger = get_logger("UnifiedFeatures")

    original_length = len(df)

    # Check for minimum data requirements
    if len(df) < 5:
        logger.warning(f"Insufficient data: only {len(df)} rows available (minimum 5 recommended)")

    # Fill missing values with more robust strategies
    price_cols = ["open", "high", "low", "close"]

    # Check for zero or negative prices
    for col in price_cols:
        zero_count = (df[col] <= 0).sum()
        if zero_count > 0:
            logger.warning(f"Found {zero_count} zero/negative values in {col}")

    # More robust price cleaning
    for col in price_cols:
        # Replace zeros and negatives with NaN first
        df[col] = df[col].replace([0, -np.inf, np.inf], np.nan)

        # Forward fill, then backward fill
        df[col] = df[col].ffill().bfill()

        # If still NaN, use a reasonable default (1.0 for prices)
        if df[col].isna().any():
            logger.warning(f"Still have NaN values in {col} after cleaning, using default value 1.0")
            df[col] = df[col].fillna(1.0)

    # Volume cleaning
    df["volume"] = df["volume"].replace([-np.inf, np.inf], np.nan)
    df["volume"] = df["volume"].ffill().bfill().fillna(0.0)

    # Ensure price consistency (high >= low, high >= open, high >= close, etc.)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    # Log cleaning results
    cleaned_length = len(df)
    if cleaned_length != original_length:
        logger.info(f"Data cleaning: {original_length} -> {cleaned_length} rows")

    return df


def _add_technical_indicators(
    result_df: pd.DataFrame,
    df: pd.DataFrame,
    ma_windows: list,
    rsi_window: int,
    vol_window: int
) -> pd.DataFrame:
    """Add exactly 20 technical indicators with robust NaN handling."""
    logger = get_logger("UnifiedFeatures")

    # Log return - more robust calculation
    close_shifted = df["close"].shift(1)
    valid_mask = (df["close"] > 0) & (close_shifted > 0) & (close_shifted.notna())
    result_df["log_return"] = np.where(valid_mask, np.log(df["close"] / close_shifted), 0.0)
    result_df["log_return"] = result_df["log_return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Moving averages - ensure minimum periods
    for w in ma_windows:
        if len(df) >= w:
            result_df[f"sma_{w}"] = df["close"].rolling(w, min_periods=1).mean().fillna(df["close"].mean())
        else:
            # If not enough data, use available data
            result_df[f"sma_{w}"] = df["close"].mean()

    # RSI - more robust with fallback
    try:
        if len(df) >= rsi_window:
            rsi_result = ta.rsi(df["close"], length=rsi_window)
            result_df[f"rsi_{rsi_window}"] = rsi_result.fillna(50.0)
        else:
            result_df[f"rsi_{rsi_window}"] = 50.0
    except Exception as e:
        logger.warning(f"RSI calculation failed: {e}, using default value 50.0")
        result_df[f"rsi_{rsi_window}"] = 50.0

    # Volatility - more robust calculation
    if len(df) >= vol_window:
        result_df[f"vol_{vol_window}"] = (
            result_df["log_return"].rolling(vol_window, min_periods=1).std(ddof=0).fillna(0) * np.sqrt(vol_window)
        )
    else:
        result_df[f"vol_{vol_window}"] = 0.0

    # EMA 20 - more robust
    try:
        if len(df) >= 20:
            ema_result = ta.ema(df["close"], length=20)
            result_df["ema_20"] = ema_result.fillna(df["close"])
        else:
            result_df["ema_20"] = df["close"]
    except Exception as e:
        logger.warning(f"EMA calculation failed: {e}, using close price")
        result_df["ema_20"] = df["close"]

    # MACD - more robust with fallbacks
    try:
        if len(df) >= 26:  # MACD needs at least 26 periods
            macd = ta.macd(df["close"])
            result_df["macd_line"] = macd["MACD_12_26_9"].fillna(0)
            result_df["macd_signal"] = macd["MACDs_12_26_9"].fillna(0)
            result_df["macd_hist"] = macd["MACDh_12_26_9"].fillna(0)
        else:
            result_df["macd_line"] = 0.0
            result_df["macd_signal"] = 0.0
            result_df["macd_hist"] = 0.0
    except Exception as e:
        logger.warning(f"MACD calculation failed: {e}, using default values")
        result_df["macd_line"] = 0.0
        result_df["macd_signal"] = 0.0
        result_df["macd_hist"] = 0.0

    # ATR - more robust
    try:
        if len(df) >= 14:
            atr_result = ta.atr(df["high"], df["low"], df["close"], length=14)
            result_df["atr_14"] = atr_result.fillna(0)
        else:
            result_df["atr_14"] = 0.0
    except Exception as e:
        logger.warning(f"ATR calculation failed: {e}, using default value 0.0")
        result_df["atr_14"] = 0.0

    # Bollinger Bands - more robust
    try:
        if len(df) >= 20:
            bb = ta.bbands(df["close"], length=20)
            result_df["bb_mavg_20"] = bb["BBM_20_2.0"].fillna(df["close"])
            result_df["bb_upper_20"] = bb["BBU_20_2.0"].fillna(df["close"])
            result_df["bb_lower_20"] = bb["BBL_20_2.0"].fillna(df["close"])
        else:
            result_df["bb_mavg_20"] = df["close"]
            result_df["bb_upper_20"] = df["close"]
            result_df["bb_lower_20"] = df["close"]
    except Exception as e:
        logger.warning(f"Bollinger Bands calculation failed: {e}, using close price")
        result_df["bb_mavg_20"] = df["close"]
        result_df["bb_upper_20"] = df["close"]
        result_df["bb_lower_20"] = df["close"]

    # Stochastic - more robust
    try:
        if len(df) >= 14:
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            result_df["stoch_k"] = stoch["STOCHk_14_3_3"].fillna(50.0)
            result_df["stoch_d"] = stoch["STOCHd_14_3_3"].fillna(50.0)
        else:
            result_df["stoch_k"] = 50.0
            result_df["stoch_d"] = 50.0
    except Exception as e:
        logger.warning(f"Stochastic calculation failed: {e}, using default values")
        result_df["stoch_k"] = 50.0
        result_df["stoch_d"] = 50.0

    # ADX - more robust
    try:
        if len(df) >= 14:
            adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
            # ADX returns a DataFrame, we need to extract the ADX column
            if isinstance(adx_result, pd.DataFrame) and "ADX_14" in adx_result.columns:
                result_df["adx_14"] = adx_result["ADX_14"].fillna(25.0)
            else:
                result_df["adx_14"] = 25.0
        else:
            result_df["adx_14"] = 25.0
    except Exception as e:
        logger.warning(f"ADX calculation failed: {e}, using default value 25.0")
        result_df["adx_14"] = 25.0

    # Williams %R - more robust
    try:
        if len(df) >= 14:
            wr_result = ta.willr(df["high"], df["low"], df["close"], length=14)
            result_df["wr_14"] = wr_result.fillna(-50.0)
        else:
            result_df["wr_14"] = -50.0
    except Exception as e:
        logger.warning(f"Williams %R calculation failed: {e}, using default value -50.0")
        result_df["wr_14"] = -50.0

    # OBV - more robust
    try:
        obv_result = ta.obv(df["close"], df["volume"])
        result_df["obv"] = obv_result.fillna(0)
    except Exception as e:
        logger.warning(f"OBV calculation failed: {e}, using default value 0.0")
        result_df["obv"] = 0.0

    return result_df


def _add_candlestick_patterns(result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 18 candlestick patterns."""

    # Basic patterns
    result_df["doji"] = detect_doji(df)
    result_df["hammer"] = detect_hammer(df)
    result_df["hanging_man"] = _detect_hanging_man(df)
    result_df["bullish_engulfing"] = _detect_bullish_engulfing(df)
    result_df["bearish_engulfing"] = _detect_bearish_engulfing(df)
    result_df["shooting_star"] = detect_shooting_star(df)
    result_df["morning_star"] = detect_morning_star(df)
    result_df["evening_star"] = detect_evening_star(df)
    result_df["inside_bar"] = _detect_inside_bar(df)
    result_df["outside_bar"] = _detect_outside_bar(df)
    result_df["tweezer_top"] = _detect_tweezer_top(df)
    result_df["tweezer_bottom"] = _detect_tweezer_bottom(df)
    result_df["three_white_soldiers"] = _detect_three_white_soldiers(df)
    result_df["three_black_crows"] = _detect_three_black_crows(df)
    result_df["bullish_harami"] = _detect_bullish_harami(df)
    result_df["bearish_harami"] = _detect_bearish_harami(df)
    result_df["dark_cloud_cover"] = _detect_dark_cloud_cover(df)
    result_df["piercing_line"] = _detect_piercing_line(df)

    # Ensure all candlestick pattern columns are integers (not boolean)
    pattern_columns = [
        "doji", "hammer", "hanging_man", "bullish_engulfing", "bearish_engulfing",
        "shooting_star", "morning_star", "evening_star", "inside_bar", "outside_bar",
        "tweezer_top", "tweezer_bottom", "three_white_soldiers", "three_black_crows",
        "bullish_harami", "bearish_harami", "dark_cloud_cover", "piercing_line"
    ]

    for col in pattern_columns:
        if col in result_df.columns and (result_df[col].dtype == bool or result_df[col].dtype == "boolean"):
            # Convert any boolean dtypes to int to prevent quantile calculation errors
            result_df[col] = result_df[col].astype(int)

    return result_df


def _add_candlestick_characteristics(result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 9 candlestick characteristics."""

    # Basic characteristics
    result_df["body_size"] = (df["close"] - df["open"]).abs()
    result_df["range_size"] = df["high"] - df["low"]
    result_df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    result_df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Relative characteristics
    result_df["rel_body_size"] = result_df["body_size"] / result_df["range_size"].replace(0, 1)
    result_df["rel_upper_shadow"] = result_df["upper_shadow"] / result_df["range_size"].replace(0, 1)
    result_df["rel_lower_shadow"] = result_df["lower_shadow"] / result_df["range_size"].replace(0, 1)

    # Body position and type
    result_df["body_position"] = (df["close"] + df["open"]) / 2 / result_df["range_size"].replace(0, 1)
    result_df["body_type"] = np.where(df["close"] > df["open"], 1, np.where(df["close"] < df["open"], -1, 0))

    # Clean up
    result_df["rel_body_size"] = result_df["rel_body_size"].fillna(0)
    result_df["rel_upper_shadow"] = result_df["rel_upper_shadow"].fillna(0)
    result_df["rel_lower_shadow"] = result_df["rel_lower_shadow"].fillna(0)
    result_df["body_position"] = result_df["body_position"].fillna(0.5)

    return result_df


def _add_rolling_candlestick_features(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 15 rolling candlestick features."""

    # Ensure required columns exist
    required_cols = ["rel_body_size", "upper_shadow", "lower_shadow", "body_position", "body_size"]
    for col in required_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0

    # 5-period averages
    result_df["avg_rel_body_5"] = result_df["rel_body_size"].rolling(5, min_periods=1).mean().fillna(0)
    result_df["avg_upper_shadow_5"] = result_df["upper_shadow"].rolling(5, min_periods=1).mean().fillna(0)
    result_df["avg_lower_shadow_5"] = result_df["lower_shadow"].rolling(5, min_periods=1).mean().fillna(0)
    result_df["avg_body_pos_5"] = result_df["body_position"].rolling(5, min_periods=1).mean().fillna(0.5)
    result_df["body_momentum_5"] = result_df["body_size"].diff(5).rolling(5, min_periods=1).mean().fillna(0)

    # 10-period averages
    result_df["avg_rel_body_10"] = result_df["rel_body_size"].rolling(10, min_periods=1).mean().fillna(0)
    result_df["avg_upper_shadow_10"] = result_df["upper_shadow"].rolling(10, min_periods=1).mean().fillna(0)
    result_df["avg_lower_shadow_10"] = result_df["lower_shadow"].rolling(10, min_periods=1).mean().fillna(0)
    result_df["avg_body_pos_10"] = result_df["body_position"].rolling(10, min_periods=1).mean().fillna(0.5)
    result_df["body_momentum_10"] = result_df["body_size"].diff(10).rolling(10, min_periods=1).mean().fillna(0)

    # 20-period averages
    result_df["avg_rel_body_20"] = result_df["rel_body_size"].rolling(20, min_periods=1).mean().fillna(0)
    result_df["avg_upper_shadow_20"] = result_df["upper_shadow"].rolling(20, min_periods=1).mean().fillna(0)
    result_df["avg_lower_shadow_20"] = result_df["lower_shadow"].rolling(20, min_periods=1).mean().fillna(0)
    result_df["avg_body_pos_20"] = result_df["body_position"].rolling(20, min_periods=1).mean().fillna(0.5)
    result_df["body_momentum_20"] = result_df["body_size"].diff(20).rolling(20, min_periods=1).mean().fillna(0)

    return result_df


def _add_sentiment_features(result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 2 sentiment features."""

    # Default sentiment values
    result_df["sentiment"] = 0.0
    result_df["sentiment_magnitude"] = 0.0

    # If sentiment column exists, use it
    if "sentiment" in df.columns:
        result_df["sentiment"] = df["sentiment"].fillna(0.0)
        result_df["sentiment_magnitude"] = result_df["sentiment"].abs()

    return result_df


def _add_time_features(result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 4 time features."""

    # Default time features
    result_df["hour"] = 0
    result_df["day_of_week"] = 0
    result_df["month"] = 1
    result_df["quarter"] = 1

    # If timestamp exists, extract time features
    if "timestamp" in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            result_df["hour"] = df["timestamp"].dt.hour.fillna(0)
            result_df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0)
            result_df["month"] = df["timestamp"].dt.month.fillna(1)
            result_df["quarter"] = df["timestamp"].dt.quarter.fillna(1)
        except Exception:
            pass  # Keep default values

    return result_df


def _add_market_regime_features(result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add exactly 5 market regime features."""

    # Price change percentage
    result_df["price_change_pct"] = df["close"].pct_change().fillna(0)

    # High-low percentage
    result_df["high_low_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, 1)

    # Volume features
    result_df["volume_ma_20"] = df["volume"].rolling(20, min_periods=1).mean().fillna(0)
    result_df["volume_ratio"] = df["volume"] / result_df["volume_ma_20"].replace(0, 1)
    result_df["volume_change"] = df["volume"].pct_change().fillna(0)

    # Clean up
    result_df["high_low_pct"] = result_df["high_low_pct"].fillna(0)
    result_df["volume_ratio"] = result_df["volume_ratio"].fillna(1)

    return result_df


def _get_expected_feature_names() -> list[str]:
    """Get the exact list of 78 expected feature names."""
    return [
        # Price features (5)
        "open", "high", "low", "close", "volume",

        # Technical indicators (20)
        "log_return", "sma_5", "sma_10", "sma_20", "sma_50",
        "rsi_14", "vol_20", "ema_20", "macd_line", "macd_signal",
        "macd_hist", "atr_14", "bb_mavg_20", "bb_upper_20", "bb_lower_20",
        "stoch_k", "stoch_d", "adx_14", "wr_14", "obv",

        # Candlestick patterns (18)
        "doji", "hammer", "hanging_man", "bullish_engulfing", "bearish_engulfing",
        "shooting_star", "morning_star", "evening_star", "inside_bar", "outside_bar",
        "tweezer_top", "tweezer_bottom", "three_white_soldiers", "three_black_crows",
        "bullish_harami", "bearish_harami", "dark_cloud_cover", "piercing_line",

        # Candlestick characteristics (9)
        "body_size", "range_size", "rel_body_size", "upper_shadow", "lower_shadow",
        "rel_upper_shadow", "rel_lower_shadow", "body_position", "body_type",

        # Rolling candlestick features (15)
        "avg_rel_body_5", "avg_upper_shadow_5", "avg_lower_shadow_5", "avg_body_pos_5", "body_momentum_5",
        "avg_rel_body_10", "avg_upper_shadow_10", "avg_lower_shadow_10", "avg_body_pos_10", "body_momentum_10",
        "avg_rel_body_20", "avg_upper_shadow_20", "avg_lower_shadow_20", "avg_body_pos_20", "body_momentum_20",

        # Sentiment features (2)
        "sentiment", "sentiment_magnitude",

        # Time features (4)
        "hour", "day_of_week", "month", "quarter",

        # Market regime features (5)
        "price_change_pct", "high_low_pct", "volume_ma_20", "volume_ratio", "volume_change"
    ]


# Pattern detection functions
def detect_doji(df: pd.DataFrame) -> pd.Series:
    """Detect Doji candlestick pattern."""
    return (df["close"] - df["open"]).abs() <= (df["high"] - df["low"]) * 0.1


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect Engulfing candlestick pattern."""
    return (
        (df["close"].shift(1) > df["open"].shift(1)) & (df["open"] < df["close"]) & (df["close"] > df["open"].shift(1))
    ) | (
        (df["close"].shift(1) < df["open"].shift(1)) & (df["open"] > df["close"]) & (df["close"] < df["open"].shift(1))
    )


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Detect Evening Star candlestick pattern."""
    return (
        (df["close"].shift(2) > df["open"].shift(2))
        & (df["open"].shift(1) > df["close"].shift(2))
        & (df["open"] > df["close"])
        & (df["close"] < df["open"].shift(1))
    )


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect Hammer candlestick pattern."""
    return (
        (df["high"] - df["low"] > 3 * (df["open"] - df["close"]))
        & ((df["close"] - df["low"]) / (0.001 + df["high"] - df["low"]) > 0.6)
        & ((df["open"] - df["low"]) / (0.001 + df["high"] - df["low"]) > 0.6)
    )


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Detect Morning Star candlestick pattern."""
    return (
        (df["close"].shift(2) < df["open"].shift(2))
        & (df["open"].shift(1) < df["close"].shift(2))
        & (df["open"] < df["close"])
        & (df["close"] > df["open"].shift(1))
    )


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Detect Shooting Star candlestick pattern."""
    return (
        (df["high"] - df["low"] > 3 * (df["open"] - df["close"]))
        & ((df["high"] - df["close"]) / (0.001 + df["high"] - df["low"]) > 0.6)
        & ((df["high"] - df["open"]) / (0.001 + df["high"] - df["low"]) > 0.6)
    )


def add_sentiment(df: pd.DataFrame, sentiment_col: str = "sentiment") -> pd.DataFrame:
    """Add a sentiment column to the dataframe."""
    df[sentiment_col] = 0.0
    return df


def _detect_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Detect Hanging Man pattern."""
    body = (df["close"] - df["open"]).abs()
    lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
    upper_shadow = df["high"] - np.maximum(df["open"], df["close"])

    hanging_man = (
        (lower_shadow >= 2 * body) &
        (upper_shadow <= body * 0.1) &
        (body <= (df["high"] - df["low"]) * 0.3)
    )
    return hanging_man.fillna(False).astype(int)


def _detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect Bullish Engulfing pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    bullish_engulfing = (
        (prev_close < prev_open) &  # Previous candle is bearish
        (df["close"] > df["open"]) &  # Current candle is bullish
        (df["open"] < prev_close) &  # Current open below previous close
        (df["close"] > prev_open)  # Current close above previous open
    )
    return bullish_engulfing.fillna(False).astype(int)


def _detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect Bearish Engulfing pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    bearish_engulfing = (
        (prev_close > prev_open) &  # Previous candle is bullish
        (df["close"] < df["open"]) &  # Current candle is bearish
        (df["open"] > prev_close) &  # Current open above previous close
        (df["close"] < prev_open)  # Current close below previous open
    )
    return bearish_engulfing.fillna(False).astype(int)


def _detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Detect Inside Bar pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    inside_bar = (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
    return inside_bar.fillna(False).astype(int)


def _detect_outside_bar(df: pd.DataFrame) -> pd.Series:
    """Detect Outside Bar pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    outside_bar = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    return outside_bar.fillna(False).astype(int)


def _detect_tweezer_top(df: pd.DataFrame) -> pd.Series:
    """Detect Tweezer Top pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Two candles with same high after uptrend
    high1 = df["high"].shift(1)
    high2 = df["high"]
    body1 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    body2 = (df["close"] - df["open"]).abs()

    tweezer_top = (
        (abs(high1 - high2) <= (high1 * 0.001)) &  # Same high
        (body1 > (df["high"].shift(1) - df["low"].shift(1)) * 0.3) &  # First candle has body
        (body2 > (df["high"] - df["low"]) * 0.3)  # Second candle has body
    )
    return tweezer_top.fillna(False).astype(int)


def _detect_tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """Detect Tweezer Bottom pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Two candles with same low after downtrend
    low1 = df["low"].shift(1)
    low2 = df["low"]
    body1 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    body2 = (df["close"] - df["open"]).abs()

    tweezer_bottom = (
        (abs(low1 - low2) <= (low1 * 0.001)) &  # Same low
        (body1 > (df["high"].shift(1) - df["low"].shift(1)) * 0.3) &  # First candle has body
        (body2 > (df["high"] - df["low"]) * 0.3)  # Second candle has body
    )
    return tweezer_bottom.fillna(False).astype(int)


def _detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Detect Three White Soldiers pattern."""
    if len(df) < 3:
        return pd.Series([0] * len(df), index=df.index)

    # Three consecutive bullish candles with higher highs
    first_bullish = (df["close"].shift(2) > df["open"].shift(2))
    second_bullish = (df["close"].shift(1) > df["open"].shift(1))
    third_bullish = (df["close"] > df["open"])

    higher_highs = (df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))

    three_white_soldiers = first_bullish & second_bullish & third_bullish & higher_highs
    return three_white_soldiers.fillna(False).astype(int)


def _detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Detect Three Black Crows pattern."""
    if len(df) < 3:
        return pd.Series([0] * len(df), index=df.index)

    # Three consecutive bearish candles with lower lows
    first_bearish = (df["close"].shift(2) < df["open"].shift(2))
    second_bearish = (df["close"].shift(1) < df["open"].shift(1))
    third_bearish = (df["close"] < df["open"])

    lower_lows = (df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))

    three_black_crows = first_bearish & second_bearish & third_bearish & lower_lows
    return three_black_crows.fillna(False).astype(int)


def _detect_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bullish Harami pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Previous bearish candle, current bullish candle inside previous body
    prev_bearish = (df["close"].shift(1) < df["open"].shift(1))
    current_bullish = (df["close"] > df["open"])
    inside_body = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))

    bullish_harami = prev_bearish & current_bullish & inside_body
    return bullish_harami.fillna(False).astype(int)


def _detect_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bearish Harami pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Previous bullish candle, current bearish candle inside previous body
    prev_bullish = (df["close"].shift(1) > df["open"].shift(1))
    current_bearish = (df["close"] < df["open"])
    inside_body = (df["close"] > df["close"].shift(1)) & (df["open"] < df["open"].shift(1))

    bearish_harami = prev_bullish & current_bearish & inside_body
    return bearish_harami.fillna(False).astype(int)


def _detect_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Detect Dark Cloud Cover pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Previous bullish candle, current bearish candle opening above previous high
    prev_bullish = (df["close"].shift(1) > df["open"].shift(1))
    current_bearish = (df["close"] < df["open"])
    open_above_high = df["open"] > df["high"].shift(1)

    dark_cloud_cover = prev_bullish & current_bearish & open_above_high
    return dark_cloud_cover.fillna(False).astype(int)


def _detect_piercing_line(df: pd.DataFrame) -> pd.Series:
    """Detect Piercing Line pattern."""
    if len(df) < 2:
        return pd.Series([0] * len(df), index=df.index)

    # Previous bearish candle, current bullish candle opening below previous low
    prev_bearish = (df["close"].shift(1) < df["open"].shift(1))
    current_bullish = (df["close"] > df["open"])
    open_below_low = df["open"] < df["low"].shift(1)

    piercing_line = prev_bearish & current_bullish & open_below_low
    return piercing_line.fillna(False).astype(int)


def compute_adx(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute the Average Directional Movement Index (ADX)."""
    df[f"adx_{timeperiod}"] = ta.adx(df["high"], df["low"], df["close"], length=timeperiod)["ADX_14"]
    return df


def compute_macd(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Compute the Moving Average Convergence Divergence (MACD)."""
    macd = ta.macd(df[price_col])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    return df


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the On-Balance Volume (OBV)."""
    df["obv"] = ta.obv(df["close"], df["volume"])
    return df


def compute_stochastic(
    df: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3
) -> pd.DataFrame:
    """Compute the Stochastic Oscillator."""
    stoch = ta.stoch(
        df["high"],
        df["low"],
        df["close"],
        k=fastk_period,
        d=slowd_period,
        smooth_k=slowk_period,
    )
    df["stoch_k"] = stoch[f"STOCHk_{fastk_period}_{slowd_period}_{slowk_period}"]
    df["stoch_d"] = stoch[f"STOCHd_{fastk_period}_{slowd_period}_{slowk_period}"]
    return df


def compute_williams_r(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute the Williams %R."""
    df[f"wr_{timeperiod}"] = ta.willr(df["high"], df["low"], df["close"], length=timeperiod)
    return df


def compute_bollinger_bands(df: pd.DataFrame, price_col: str = "close", timeperiod: int = 20) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    df[[f"bb_lower_{timeperiod}", f"bb_mavg_{timeperiod}", f"bb_upper_{timeperiod}"]] = ta.bbands(
        df[price_col], length=timeperiod
    )
    return df


def compute_atr(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute the Average True Range (ATR)."""
    df[f"atr_{timeperiod}"] = ta.atr(df["high"], df["low"], df["close"], length=timeperiod)
    return df


def compute_ema(df: pd.DataFrame, price_col: str = "close", timeperiod: int = 20) -> pd.DataFrame:
    """Compute the Exponential Moving Average (EMA)."""
    df[f"ema_{timeperiod}"] = ta.ema(df[price_col], length=timeperiod)
    return df


# Legacy compatibility - keep the FeatureEngineer class for backward compatibility
class FeatureEngineer:
    """Feature engineering class for financial time series data."""

    def __init__(self, config: dict | None = None):
        """Initialize the feature engineer."""
        self.config = config or {}
        self.ma_windows = self.config.get("ma_windows", [5, 10, 20, 50])
        self.rsi_window = self.config.get("rsi_window", 14)
        self.vol_window = self.config.get("vol_window", 20)
        self.advanced_candles = self.config.get("advanced_candles", True)

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available features for the given dataframe."""
        if df.empty:
            return df

        return generate_unified_features(
            df.copy(),
            ma_windows=self.ma_windows,
            rsi_window=self.rsi_window,
            vol_window=self.vol_window,
        )

    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators only."""
        if df.empty:
            return df

        return generate_unified_features(
            df.copy(),
            ma_windows=self.ma_windows,
            rsi_window=self.rsi_window,
            vol_window=self.vol_window,
        )

    def calculate_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick pattern features only."""
        if df.empty:
            return df

        # For candlestick-only features, we still need the full pipeline
        # but we can filter the output to only include candlestick features
        result_df = generate_features(
            df.copy(),
            ma_windows=self.ma_windows,
            rsi_window=self.rsi_window,
            vol_window=self.vol_window,
        )

        # Extract only candlestick-related features
        candlestick_features = [
            "doji", "hammer", "hanging_man", "bullish_engulfing", "bearish_engulfing",
            "shooting_star", "morning_star", "evening_star", "inside_bar", "outside_bar",
            "tweezer_top", "tweezer_bottom", "three_white_soldiers", "three_black_crows",
            "bullish_harami", "bearish_harami", "dark_cloud_cover", "piercing_line",
            "body_size", "range_size", "rel_body_size", "upper_shadow", "lower_shadow",
            "rel_upper_shadow", "rel_lower_shadow", "body_position", "body_type"
        ]

        available_features = [f for f in candlestick_features if f in result_df.columns]
        return result_df[available_features]


# Legacy compatibility aliases
generate_features = generate_unified_features
