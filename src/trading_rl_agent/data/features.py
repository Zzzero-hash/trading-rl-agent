"""
Feature engineering utilities for trading data pipelines.
"""

import numpy as np
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
        & (body <= (df_copy["high"] - df_copy["low"]) * 0.3)  # Small body relative to range
    )

    df_copy["shooting_star"] = shooting_star.fillna(False).astype(int)
    return df_copy


def detect_morning_star(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Morning Star pattern using pandas logic."""
    df_copy = df.copy()

    # Morning star is a 3-candle bullish reversal pattern:
    # 1. First candle: bearish (close < open)
    # 2. Second candle: small body (star), gaps down
    # 3. Third candle: bullish (close > open), closes above first candle's midpoint

    morning_star = [False] * len(df_copy)

    for i in range(2, len(df_copy)):
        # First candle is bearish
        first_bearish = df_copy["close"].iloc[i - 2] < df_copy["open"].iloc[i - 2]

        # Second candle is small and gaps down
        second_small = (
            abs(df_copy["close"].iloc[i - 1] - df_copy["open"].iloc[i - 1])
            < abs(df_copy["close"].iloc[i - 2] - df_copy["open"].iloc[i - 2]) * 0.3
        )
        # second_gap_down = df_copy["high"].iloc[i - 1] < df_copy["low"].iloc[i - 2]  # Not used currently

        # Third candle is bullish and closes above first candle's midpoint
        third_bullish = df_copy["close"].iloc[i] > df_copy["open"].iloc[i]
        first_midpoint = (df_copy["open"].iloc[i - 2] + df_copy["close"].iloc[i - 2]) / 2
        third_recovery = df_copy["close"].iloc[i] > first_midpoint

        if first_bearish and second_small and third_bullish and third_recovery:
            morning_star[i] = True

    df_copy["morning_star"] = morning_star
    df_copy["morning_star"] = df_copy["morning_star"].astype(int)
    return df_copy


def detect_evening_star(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Evening Star pattern using pandas logic."""
    df_copy = df.copy()

    # Evening star is a 3-candle bearish reversal pattern:
    # 1. First candle: bullish (close > open)
    # 2. Second candle: small body (star), gaps up
    # 3. Third candle: bearish (close < open), closes below first candle's midpoint

    evening_star = [False] * len(df_copy)

    for i in range(2, len(df_copy)):
        # First candle is bullish
        first_bullish = df_copy["close"].iloc[i - 2] > df_copy["open"].iloc[i - 2]

        # Second candle is small and gaps up
        second_small = (
            abs(df_copy["close"].iloc[i - 1] - df_copy["open"].iloc[i - 1])
            < abs(df_copy["close"].iloc[i - 2] - df_copy["open"].iloc[i - 2]) * 0.3
        )
        # second_gap_up = df_copy["low"].iloc[i - 1] > df_copy["high"].iloc[i - 2]  # Not used currently

        # Third candle is bearish and closes below first candle's midpoint
        third_bearish = df_copy["close"].iloc[i] < df_copy["open"].iloc[i]
        first_midpoint = (df_copy["open"].iloc[i - 2] + df_copy["close"].iloc[i - 2]) / 2
        third_decline = df_copy["close"].iloc[i] < first_midpoint

        if first_bullish and second_small and third_bearish and third_decline:
            evening_star[i] = True

    df_copy["evening_star"] = evening_star
    df_copy["evening_star"] = df_copy["evening_star"].astype(int)
    return df_copy


def compute_candle_features(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """
    Compute all candlestick pattern features.

    Args:
        df: DataFrame with OHLC data
        advanced: If True, use advanced patterns from candle_patterns.py

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
    Robustly handles missing/None/NaN values in required columns.
    """
    # Robust error handling for missing/empty columns and insufficient data
    if ma_windows is None:
        ma_windows = [5, 10, 20]
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
    min_required_length = max([*ma_windows, rsi_window, vol_window, 26, 20, 14, 9, 3])

    if len(df) < min_required_length:
        import warnings

        warnings.warn(
            f"Insufficient data for full feature engineering (need at least {min_required_length} rows, got {len(df)}). Some features may be NaN.",
            stacklevel=2,
        )
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
    # Replace any remaining inf/-inf with 0
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Moving averages with proper NaN handling
    for w in ma_windows:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=1).mean().fillna(0)

    # RSI with error handling
    try:
        df[f"rsi_{rsi_window}"] = ta.rsi(df["close"], length=rsi_window).fillna(50.0)  # Default to neutral 50
    except Exception:
        df[f"rsi_{rsi_window}"] = 50.0

    # Volatility with proper handling
    df[f"vol_{vol_window}"] = df["log_return"].rolling(vol_window, min_periods=1).std(ddof=0).fillna(0) * np.sqrt(
        vol_window
    )

    # Add sentiment
    df = add_sentiment(df)

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
    except Exception as e:
        print(f"Warning: Error in technical indicators: {e}")

    # Candlestick patterns
    try:
        df = compute_candle_features(df, advanced=advanced_candles)
    except Exception as e:
        print(f"Warning: Error in candlestick patterns: {e}")

    # Final cleanup - fill any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Convert to float64 for consistency
    df[numeric_cols] = df[numeric_cols].astype(np.float64)

    # Safe normalization - avoid division by zero
    for col in numeric_cols:
        max_abs = df[col].abs().max()
        if pd.notna(max_abs) and max_abs > 1e-8:  # Avoid very small denominators
            df[col] = df[col] / max_abs
        else:
            df[col] = 0.0

    # Ensure no inf/-inf values remain
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0)

    # Drop initial rows if needed (but keep at least 1 row)
    windows = [*ma_windows, rsi_window, vol_window, 20, 26, 9, 14, 20, 14, 14, 14]
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
