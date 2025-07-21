"""
Feature generation for financial time series data.

This module provides functions for generating technical indicators and features
from financial time series data.
"""

# Monkey patch for pandas_ta numpy compatibility
import numpy as np

if not hasattr(np, "NaN"):
    np.nan = np.nan

import pandas as pd
import pandas_ta as ta


def add_sentiment(df: pd.DataFrame, sentiment_col: str = "sentiment") -> pd.DataFrame:
    """
    Stub for sentiment feature (defaults to zero).
    """
    df[sentiment_col] = 0.0
    return df


def compute_ema(
    df: pd.DataFrame,
    price_col: str = "close",
    timeperiod: int = 20,
) -> pd.DataFrame:
    """Compute Exponential Moving Average (EMA) using pandas-ta."""
    df[f"ema_{timeperiod}"] = ta.ema(df[price_col], length=timeperiod)
    return df


def compute_macd(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Compute MACD line, signal, and histogram using pandas-ta."""
    macd = ta.macd(df[price_col])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    return df


def compute_atr(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR) using pandas-ta."""
    df[f"atr_{timeperiod}"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=timeperiod,
    ).fillna(0.0)
    return df


def compute_bollinger_bands(
    df: pd.DataFrame,
    price_col: str = "close",
    timeperiod: int = 20,
) -> pd.DataFrame:
    """Compute Bollinger Bands using pandas-ta."""
    bb = ta.bbands(df[price_col], length=timeperiod)
    df[f"bb_lower_{timeperiod}"] = bb.iloc[:, 0]
    df[f"bb_mavg_{timeperiod}"] = bb.iloc[:, 1]
    df[f"bb_upper_{timeperiod}"] = bb.iloc[:, 2]
    return df


def compute_stochastic(
    df: pd.DataFrame,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pd.DataFrame:
    """Compute Stochastic Oscillator using pandas-ta."""
    stoch = ta.stoch(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        k=fastk_period,
        d=slowd_period,
        smooth_k=slowk_period,
    )
    df["stoch_k"] = stoch.iloc[:, 0]
    df["stoch_d"] = stoch.iloc[:, 1]
    return df


def compute_adx(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) using pandas-ta."""
    adx = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=timeperiod)
    df[f"adx_{timeperiod}"] = adx.iloc[:, 0].fillna(0.0)
    return df


def compute_williams_r(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """Compute Williams %R using pandas-ta."""
    df[f"wr_{timeperiod}"] = ta.willr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=timeperiod,
    )
    return df


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute On-Balance Volume (OBV) using pandas-ta."""
    df["obv"] = ta.obv(close=df["close"], volume=df["volume"])
    return df


def detect_doji(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Doji candlestick pattern: open ≈ close."""
    df["doji"] = (np.isclose(df["open"].astype(float), df["close"].astype(float))).astype(int)
    return df


def detect_hammer(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Hammer candlestick pattern: small body, long lower shadow in downtrend."""
    # Body size
    body = (df["close"] - df["open"]).abs().astype(float)
    # Shadows
    lower_shadow = (np.minimum(df["open"], df["close"]) - df["low"]).astype(float)
    # upper_shadow = (df["high"] - np.maximum(df["open"], df["close"])).astype(float)  # Not used currently
    # Hammer: lower shadow at least twice body
    df["hammer"] = (lower_shadow >= 2 * body).astype(int)
    return df


def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Bullish and Bearish Engulfing patterns using pandas logic."""
    df_copy = df.copy()

    # Calculate bullish engulfing pattern
    # Current candle is bullish (close > open) and completely engulfs previous bearish candle
    bullish_engulfing = (
        (df_copy["close"] > df_copy["open"])  # Current is bullish
        & (df_copy["close"].shift(1) < df_copy["open"].shift(1))  # Previous is bearish
        & (df_copy["open"] < df_copy["close"].shift(1))  # Current open < previous close
        & (df_copy["close"] > df_copy["open"].shift(1))  # Current close > previous open
    )

    # Calculate bearish engulfing pattern
    # Current candle is bearish (close < open) and completely engulfs previous bullish candle
    bearish_engulfing = (
        (df_copy["close"] < df_copy["open"])  # Current is bearish
        & (df_copy["close"].shift(1) > df_copy["open"].shift(1))  # Previous is bullish
        & (df_copy["open"] > df_copy["close"].shift(1))  # Current open > previous close
        & (df_copy["close"] < df_copy["open"].shift(1))  # Current close < previous open
    )

    df_copy["bullish_engulfing"] = bullish_engulfing.fillna(False).astype(int)
    df_copy["bearish_engulfing"] = bearish_engulfing.fillna(False).astype(int)

    return df_copy


def detect_shooting_star(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Shooting Star candlestick pattern using pandas logic."""
    df_copy = df.copy()

    # Shooting star characteristics:
    # 1. Small real body at lower end of the trading range
    # 2. Long upper shadow (at least 2x the body size)
    # 3. Little or no lower shadow
    # 4. Appears after uptrend (bearish reversal signal)

    body = abs(df_copy["close"] - df_copy["open"])
    upper_shadow = df_copy["high"] - df_copy[["open", "close"]].max(axis=1)
    lower_shadow = df_copy[["open", "close"]].min(axis=1) - df_copy["low"]

    shooting_star = (
        (upper_shadow >= 2 * body)  # Long upper shadow
        & (lower_shadow <= body * 0.1)  # Very small lower shadow
        & (body <= (df_copy["high"] - df_copy["low"]) * 0.3)
    )  # Small body relative to range

    df_copy["shooting_star"] = shooting_star.fillna(False).astype(int)
    return df_copy


def detect_morning_star(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Morning Star pattern using vectorized operations."""
    df_copy = df.copy()

    if len(df_copy) < 3:
        df_copy["morning_star"] = 0
        return df_copy

    # Morning star is a 3-candle bullish reversal pattern:
    # 1. First candle: bearish (close < open)
    # 2. Second candle: small body (star)
    # 3. Third candle: bullish (close > open), closes above first candle's midpoint

    # Vectorized calculations
    first_bearish = df_copy["close"].shift(2) < df_copy["open"].shift(2)

    # Second candle is small relative to first candle
    first_body_size = (df_copy["close"].shift(2) - df_copy["open"].shift(2)).abs()
    second_body_size = (df_copy["close"].shift(1) - df_copy["open"].shift(1)).abs()
    second_small = second_body_size < first_body_size * 0.3

    # Third candle is bullish and closes above first candle's midpoint
    third_bullish = df_copy["close"] > df_copy["open"]
    first_midpoint = (df_copy["open"].shift(2) + df_copy["close"].shift(2)) / 2
    third_recovery = df_copy["close"] > first_midpoint

    morning_star = first_bearish & second_small & third_bullish & third_recovery
    df_copy["morning_star"] = morning_star.fillna(False).astype(int)

    return df_copy


def detect_evening_star(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Evening Star pattern using vectorized operations."""
    df_copy = df.copy()

    if len(df_copy) < 3:
        df_copy["evening_star"] = 0
        return df_copy

    # Evening star is a 3-candle bearish reversal pattern:
    # 1. First candle: bullish (close > open)
    # 2. Second candle: small body (star)
    # 3. Third candle: bearish (close < open), closes below first candle's midpoint

    # Vectorized calculations
    first_bullish = df_copy["close"].shift(2) > df_copy["open"].shift(2)

    # Second candle is small relative to first candle
    first_body_size = (df_copy["close"].shift(2) - df_copy["open"].shift(2)).abs()
    second_body_size = (df_copy["close"].shift(1) - df_copy["open"].shift(1)).abs()
    second_small = second_body_size < first_body_size * 0.3

    # Third candle is bearish and closes below first candle's midpoint
    third_bearish = df_copy["close"] < df_copy["open"]
    first_midpoint = (df_copy["open"].shift(2) + df_copy["close"].shift(2)) / 2
    third_decline = df_copy["close"] < first_midpoint

    evening_star = first_bullish & second_small & third_bearish & third_decline
    df_copy["evening_star"] = evening_star.fillna(False).astype(int)

    return df_copy


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all candlestick pattern features.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with candlestick pattern columns
    """
    # Use basic patterns: assign Doji, then apply other patterns
    df = df.copy()
    df = detect_doji(df)
    df = detect_hammer(df)
    df = detect_engulfing(df)
    df = detect_shooting_star(df)
    df = detect_morning_star(df)
    return detect_evening_star(df)


def generate_features(
    df: pd.DataFrame,
    ma_windows: list | None = None,
    rsi_window: int = 14,
    vol_window: int = 20,
    advanced_candles: bool = True,
) -> pd.DataFrame:
    """
    Apply a sequence of feature transformations to the DataFrame.
    Enhanced to generate all features expected by the DataStandardizer.
    """
    # Robust error handling for missing/empty columns and insufficient data
    if ma_windows is None:
        ma_windows = [5, 10, 20, 50]  # Added 50 for standardizer compatibility
    required_cols = ["open", "high", "low", "close", "volume"]
    df = df.copy()

    # Fill missing columns with zeros if not present
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Fill None/NaN values with forward/backward fill, then zeros
    df[required_cols] = df[required_cols].ffill().bfill().fillna(0.0)

    # Ensure no zero values in price columns for log calculations
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        df[col] = df[col].replace(0, np.nan).ffill().bfill()
        # If still NaN, use a small positive value
        if df[col].isna().any():
            df[col] = df[col].fillna(1.0)

    for col in required_cols:
        if df[col].isnull().all() or len(df[col].dropna()) == 0:
            df[col] = 1.0 if col in price_cols else 0.0

    # Calculate minimum required data length
    min_required_length = max(
        [*ma_windows, rsi_window, vol_window, 26, 20, 14, 9, 3, 100],
    )  # Added 100 for long-term features

    if len(df) < min_required_length:
        import warnings

        warning_msg = (
            f"Insufficient data for full feature engineering "
            f"(need at least {min_required_length} rows, got {len(df)}). "
            f"Some features may be NaN."
        )
        warnings.warn(warning_msg, stacklevel=2)
        if len(df) < 26:
            ma_windows = [w for w in ma_windows if w <= len(df) // 2]
            if not ma_windows:
                ma_windows = [min(3, len(df) - 1)] if len(df) > 1 else []
            rsi_window = min(rsi_window, len(df) // 2) if len(df) > 2 else 3
            vol_window = min(vol_window, len(df) // 2) if len(df) > 2 else 3

    # Safe log return calculation
    close_shifted = df["close"].shift(1)
    valid_mask = (df["close"] > 0) & (close_shifted > 0) & (close_shifted.notna())
    df["log_return"] = np.where(valid_mask, np.log(df["close"] / close_shifted), 0.0)
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Moving averages with proper NaN handling
    for w in ma_windows:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=1).mean().fillna(0)

    # RSI with error handling
    try:
        df[f"rsi_{rsi_window}"] = ta.rsi(df["close"], length=rsi_window).fillna(50.0)
    except Exception:
        df[f"rsi_{rsi_window}"] = 50.0

    # Volatility with proper handling
    df[f"vol_{vol_window}"] = df["log_return"].rolling(vol_window, min_periods=1).std(ddof=0).fillna(0) * np.sqrt(
        vol_window,
    )

    # Add sentiment features
    df = add_sentiment(df)
    df["sentiment_magnitude"] = df["sentiment"].abs()  # Add sentiment magnitude

    # Additional technical indicators with robust error handling
    try:
        df = compute_ema(df, price_col="close", timeperiod=12)
        df = compute_ema(df, price_col="close", timeperiod=26)
        df = compute_ema(df, price_col="close", timeperiod=20)
        df = compute_macd(df, price_col="close")
        df["macd"] = df["macd_line"].fillna(0)
        df = compute_atr(df, timeperiod=14)
        df["atr"] = df["atr_14"].fillna(0)
        df = compute_bollinger_bands(df, price_col="close", timeperiod=20)
        df["bb_upper"] = df["bb_upper_20"].fillna(df["close"])
        df["bb_lower"] = df["bb_lower_20"].fillna(df["close"])
        df = compute_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3)
        if len(df) >= 28:
            df = compute_adx(df, timeperiod=14)
        df = compute_williams_r(df, timeperiod=14)
        df = compute_obv(df)

        # Add additional pandas_ta indicators that might be expected
        try:
            # Parabolic SAR - handle the DataFrame return properly
            psar_result = ta.psar(df["high"], df["low"], df["close"])
            if isinstance(psar_result, pd.DataFrame):
                # Take the first column if it's a DataFrame
                df["psar"] = psar_result.iloc[:, 0] if len(psar_result.columns) > 0 else 0.0
            else:
                df["psar"] = psar_result

            # Commodity Channel Index
            df["cci"] = ta.cci(df["high"], df["low"], df["close"])

            # Money Flow Index
            df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])

            # Rate of Change
            df["roc"] = ta.roc(df["close"])

            # True Strength Index - handle DataFrame return
            tsi_result = ta.tsi(df["close"])
            if isinstance(tsi_result, pd.DataFrame):
                # Take the first column if it's a DataFrame
                df["tsi"] = tsi_result.iloc[:, 0] if len(tsi_result.columns) > 0 else 0.0
            else:
                df["tsi"] = tsi_result

        except Exception as e:
            print(f"Warning: Additional pandas_ta indicators failed: {e}")
            # Set default values for failed indicators
            df["psar"] = 0.0
            df["cci"] = 0.0
            df["mfi"] = 0.0
            df["roc"] = 0.0
            df["tsi"] = 0.0

    except Exception as e:
        print(f"Warning: Error in technical indicators: {e}")

    # Enhanced candlestick patterns - add missing patterns
    try:
        df = compute_candle_features(df)
        # Add missing candlestick patterns that standardizer expects
        df = add_missing_candlestick_patterns(df)
    except Exception as e:
        print(f"Warning: Error in candlestick patterns: {e}")

    # Add candlestick characteristics
    df = add_candlestick_characteristics(df)

    # Add rolling candlestick features
    df = add_rolling_candlestick_features(df)

    # Add time-based features
    df = add_time_features(df)

    # Add market regime features
    df = add_market_regime_features(df)

    # Final cleanup - fill any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Convert to float64 for consistency
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    # Ensure no inf/-inf values remain
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0)

    # Drop initial rows if needed (but keep at least 1 row)
    windows = [*ma_windows, rsi_window, vol_window, 20, 26, 9, 14, 20, 14, 14, 14, 100]
    max_pattern_window = 3 if advanced_candles else 2
    windows.append(max_pattern_window)
    max_core_window = max(windows) if windows else 0

    if len(df) > max_core_window:
        rows_to_drop = min(max_core_window, len(df) - 1)
        df = df.iloc[rows_to_drop:].reset_index(drop=True)

    # Final validation - ensure no NaN or inf values
    assert not df[numeric_cols].isna().any().any(), "NaN values found after feature engineering"
    assert not df[numeric_cols].isin([np.inf, -np.inf]).any().any(), "Inf values found after feature engineering"

    return df


def add_missing_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing candlestick patterns using manual implementation."""
    df = df.copy()

    # Implement missing patterns manually
    try:
        # Hanging Man pattern
        df["hanging_man"] = detect_hanging_man(df)

        # Inside Bar pattern
        df["inside_bar"] = detect_inside_bar(df)

        # Outside Bar pattern
        df["outside_bar"] = detect_outside_bar(df)

        # Tweezer patterns
        df["tweezer_top"] = detect_tweezer_top(df)
        df["tweezer_bottom"] = detect_tweezer_bottom(df)

        # Three White Soldiers pattern
        df["three_white_soldiers"] = detect_three_white_soldiers(df)

        # Three Black Crows pattern
        df["three_black_crows"] = detect_three_black_crows(df)

        # Harami patterns
        df["bullish_harami"] = detect_bullish_harami(df)
        df["bearish_harami"] = detect_bearish_harami(df)

        # Dark Cloud Cover pattern
        df["dark_cloud_cover"] = detect_dark_cloud_cover(df)

        # Piercing Line pattern
        df["piercing_line"] = detect_piercing_line(df)

    except Exception as e:
        print(f"Warning: Failed to compute candlestick patterns: {e}")
        # Set default values for all patterns
        pattern_names = [
            "hanging_man",
            "inside_bar",
            "outside_bar",
            "tweezer_top",
            "tweezer_bottom",
            "three_white_soldiers",
            "three_black_crows",
            "bullish_harami",
            "bearish_harami",
            "dark_cloud_cover",
            "piercing_line",
        ]
        for pattern in pattern_names:
            df[pattern] = 0

    return df


def detect_hanging_man(df: pd.DataFrame) -> pd.Series:
    """Detect Hanging Man pattern: small body, long lower shadow in uptrend."""
    body = (df["close"] - df["open"]).abs()
    lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
    upper_shadow = df["high"] - np.maximum(df["open"], df["close"])

    # Hanging man: small body, long lower shadow, small upper shadow
    hanging_man = (
        (lower_shadow >= 2 * body)  # Long lower shadow
        & (upper_shadow <= body * 0.1)  # Very small upper shadow
        & (body <= (df["high"] - df["low"]) * 0.3)  # Small body relative to range
    )

    return hanging_man.fillna(False).astype(int)


def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Detect Inside Bar pattern: current bar completely inside previous bar."""
    inside_bar = (df["high"] <= df["high"].shift(1)) & (
        df["low"] >= df["low"].shift(1)
    )  # Current high <= previous high  # Current low >= previous low
    return inside_bar.fillna(False).astype(int)


def detect_outside_bar(df: pd.DataFrame) -> pd.Series:
    """Detect Outside Bar pattern: current bar completely engulfs previous bar."""
    outside_bar = (df["high"] > df["high"].shift(1)) & (
        df["low"] < df["low"].shift(1)
    )  # Current high > previous high  # Current low < previous low
    return outside_bar.fillna(False).astype(int)


def detect_tweezer_top(df: pd.DataFrame) -> pd.Series:
    """Detect Tweezer Top pattern: two candles with same high after uptrend."""
    if len(df) < 2:
        return pd.Series([False] * len(df), index=df.index).astype(int)

    # Vectorized calculation
    high1 = df["high"].shift(1)
    high2 = df["high"]
    body1 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    body2 = (df["close"] - df["open"]).abs()

    # Similar highs (within 0.1% tolerance)
    similar_highs = (high1 - high2).abs() / high1 < 0.001

    # Both candles should have small bodies
    small_bodies = (body1 < high1 * 0.01) & (body2 < high2 * 0.01)

    tweezer_top = similar_highs & small_bodies
    return tweezer_top.fillna(False).astype(int)


def detect_tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    """Detect Tweezer Bottom pattern: two candles with same low after downtrend."""
    if len(df) < 2:
        return pd.Series([False] * len(df), index=df.index).astype(int)

    # Vectorized calculation
    low1 = df["low"].shift(1)
    low2 = df["low"]
    body1 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    body2 = (df["close"] - df["open"]).abs()

    # Similar lows (within 0.1% tolerance)
    similar_lows = (low1 - low2).abs() / low1 < 0.001

    # Both candles should have small bodies
    small_bodies = (body1 < low1 * 0.01) & (body2 < low2 * 0.01)

    tweezer_bottom = similar_lows & small_bodies
    return tweezer_bottom.fillna(False).astype(int)


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Detect Three White Soldiers pattern: three consecutive bullish candles."""
    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index).astype(int)

    # Vectorized calculation
    bullish1 = df["close"].shift(2) > df["open"].shift(2)
    bullish2 = df["close"].shift(1) > df["open"].shift(1)
    bullish3 = df["close"] > df["open"]

    # Each candle should open within previous candle's body
    open2_in_body1 = (df["open"].shift(1) >= df["open"].shift(2)) & (df["open"].shift(1) <= df["close"].shift(2))
    open3_in_body2 = (df["open"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))

    three_white_soldiers = bullish1 & bullish2 & bullish3 & open2_in_body1 & open3_in_body2
    return three_white_soldiers.fillna(False).astype(int)


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Detect Three Black Crows pattern: three consecutive bearish candles."""
    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index).astype(int)

    # Vectorized calculation
    bearish1 = df["close"].shift(2) < df["open"].shift(2)
    bearish2 = df["close"].shift(1) < df["open"].shift(1)
    bearish3 = df["close"] < df["open"]

    # Each candle should open within previous candle's body
    open2_in_body1 = (df["open"].shift(1) >= df["open"].shift(2)) & (df["open"].shift(1) <= df["close"].shift(2))
    open3_in_body2 = (df["open"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))

    three_black_crows = bearish1 & bearish2 & bearish3 & open2_in_body1 & open3_in_body2
    return three_black_crows.fillna(False).astype(int)


def detect_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bullish Harami pattern: small bullish candle inside large bearish candle."""
    bullish_harami = (
        (df["close"].shift(1) < df["open"].shift(1))  # Previous candle is bearish
        & (df["close"] > df["open"])  # Current candle is bullish
        & (df["open"] > df["close"].shift(1))  # Current open > previous close
        & (df["close"] < df["open"].shift(1))  # Current close < previous open
        & (abs(df["close"] - df["open"]) < abs(df["close"].shift(1) - df["open"].shift(1)) * 0.5)  # Small body
    )
    return bullish_harami.fillna(False).astype(int)


def detect_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Detect Bearish Harami pattern: small bearish candle inside large bullish candle."""
    bearish_harami = (
        (df["close"].shift(1) > df["open"].shift(1))  # Previous candle is bullish
        & (df["close"] < df["open"])  # Current candle is bearish
        & (df["open"] < df["close"].shift(1))  # Current open < previous close
        & (df["close"] > df["open"].shift(1))  # Current close > previous open
        & (abs(df["close"] - df["open"]) < abs(df["close"].shift(1) - df["open"].shift(1)) * 0.5)  # Small body
    )
    return bearish_harami.fillna(False).astype(int)


def detect_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Detect Dark Cloud Cover pattern: bearish candle opens above previous high, closes below midpoint."""
    dark_cloud_cover = (
        (df["close"].shift(1) > df["open"].shift(1))  # Previous candle is bullish
        & (df["close"] < df["open"])  # Current candle is bearish
        & (df["open"] > df["high"].shift(1))  # Current open > previous high
        & (df["close"] < (df["open"].shift(1) + df["close"].shift(1)) / 2)  # Close below midpoint
    )
    return dark_cloud_cover.fillna(False).astype(int)


def detect_piercing_line(df: pd.DataFrame) -> pd.Series:
    """Detect Piercing Line pattern: bullish candle opens below previous low, closes above midpoint."""
    piercing_line = (
        (df["close"].shift(1) < df["open"].shift(1))  # Previous candle is bearish
        & (df["close"] > df["open"])  # Current candle is bullish
        & (df["open"] < df["low"].shift(1))  # Current open < previous low
        & (df["close"] > (df["open"].shift(1) + df["close"].shift(1)) / 2)  # Close above midpoint
    )
    return piercing_line.fillna(False).astype(int)


def add_candlestick_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick characteristic features using manual calculation."""
    df = df.copy()

    # Manual calculation of candlestick characteristics
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["range_size"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Relative body size
    df["rel_body_size"] = df["body_size"] / df["range_size"].replace(0, 1)

    # Relative shadows
    df["rel_upper_shadow"] = df["upper_shadow"] / df["range_size"].replace(0, 1)
    df["rel_lower_shadow"] = df["lower_shadow"] / df["range_size"].replace(0, 1)

    # Body position (0 = bottom, 1 = top)
    df["body_position"] = (df["close"] + df["open"]) / 2 / df["range_size"].replace(0, 1)

    # Handle any remaining NaN values
    df["rel_body_size"] = df["rel_body_size"].fillna(0)
    df["rel_upper_shadow"] = df["rel_upper_shadow"].fillna(0)
    df["rel_lower_shadow"] = df["rel_lower_shadow"].fillna(0)
    df["body_position"] = df["body_position"].fillna(0.5)

    # Body type (1 = bullish, -1 = bearish, 0 = doji)
    df["body_type"] = np.where(df["close"] > df["open"], 1, np.where(df["close"] < df["open"], -1, 0))

    return df


def add_rolling_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages of candlestick features."""
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        "rel_body_size",
        "upper_shadow",
        "lower_shadow",
        "body_position",
        "body_size",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    for window in [5, 10, 20]:
        # Average relative body size
        df[f"avg_rel_body_{window}"] = df["rel_body_size"].rolling(window, min_periods=1).mean().fillna(0)
        # Average upper shadow
        df[f"avg_upper_shadow_{window}"] = df["upper_shadow"].rolling(window, min_periods=1).mean().fillna(0)
        # Average lower shadow
        df[f"avg_lower_shadow_{window}"] = df["lower_shadow"].rolling(window, min_periods=1).mean().fillna(0)
        # Average body position
        df[f"avg_body_pos_{window}"] = df["body_position"].rolling(window, min_periods=1).mean().fillna(0.5)
        # Body momentum (change in body size)
        df[f"body_momentum_{window}"] = df["body_size"].diff(window).rolling(window, min_periods=1).mean().fillna(0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()
    if "timestamp" in df.columns:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Extract time features with error handling
            df["hour"] = df["timestamp"].dt.hour.fillna(0)
            df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0)
            df["month"] = df["timestamp"].dt.month.fillna(1)
            df["quarter"] = df["timestamp"].dt.quarter.fillna(1)
        except Exception as e:
            print(f"Warning: Error processing timestamp features: {e}")
            # Create dummy time features if timestamp processing fails
            df["hour"] = 0
            df["day_of_week"] = 0
            df["month"] = 1
            df["quarter"] = 1
    else:
        # Create dummy time features if no timestamp
        df["hour"] = 0
        df["day_of_week"] = 0
        df["month"] = 1
        df["quarter"] = 1
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features using manual calculation."""
    df = df.copy()

    # Ensure required columns exist
    if "close" not in df.columns:
        df["close"] = 1.0
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Price change percentage
    df["price_change_pct"] = df["close"].pct_change().fillna(0)

    # High-low percentage
    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, 1)

    # Volume features using manual calculation
    df["volume_ma_20"] = df["volume"].rolling(20, min_periods=1).mean().fillna(0)
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"].replace(0, 1)
    df["volume_change"] = df["volume"].pct_change().fillna(0)

    # Handle any remaining NaN values
    df["price_change_pct"] = df["price_change_pct"].fillna(0)
    df["high_low_pct"] = df["high_low_pct"].fillna(0)
    df["volume_ratio"] = df["volume_ratio"].fillna(1)
    df["volume_change"] = df["volume_change"].fillna(0)

    return df


class FeatureEngineer:
    """Feature engineering class for financial time series data."""

    def __init__(self, config: dict | None = None):
        """Initialize the feature engineer.

        Args:
            config: Configuration dictionary for feature engineering
        """
        self.config = config or {}
        self.ma_windows = self.config.get("ma_windows", [5, 10, 20, 50, 200])
        self.rsi_window = self.config.get("rsi_window", 14)
        self.vol_window = self.config.get("vol_window", 20)
        self.advanced_candles = self.config.get("advanced_candles", True)

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available features for the given dataframe.

        Args:
            df: Input dataframe with OHLCV data

        Returns:
            DataFrame with all calculated features
        """
        if df.empty:
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Basic technical indicators
        result_df = generate_features(
            result_df,
            ma_windows=self.ma_windows,
            rsi_window=self.rsi_window,
            vol_window=self.vol_window,
            advanced_candles=self.advanced_candles,
        )

        # Add candlestick patterns
        result_df = add_missing_candlestick_patterns(result_df)

        # Add time features
        result_df = add_time_features(result_df)

        # Add market regime features
        result_df = add_market_regime_features(result_df)

        # Add rolling candlestick features
        result_df = add_rolling_candlestick_features(result_df)

        # Add candlestick characteristics
        return add_candlestick_characteristics(result_df)

    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators only.

        Args:
            df: Input dataframe with OHLCV data

        Returns:
            DataFrame with basic features
        """
        if df.empty:
            return df

        return generate_features(
            df.copy(),
            ma_windows=self.ma_windows,
            rsi_window=self.rsi_window,
            vol_window=self.vol_window,
            advanced_candles=False,
        )

    def calculate_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick pattern features only.

        Args:
            df: Input dataframe with OHLCV data

        Returns:
            DataFrame with candlestick features
        """
        if df.empty:
            return df

        result_df = df.copy()
        result_df = add_missing_candlestick_patterns(result_df)
        result_df = add_candlestick_characteristics(result_df)
        return add_rolling_candlestick_features(result_df)
