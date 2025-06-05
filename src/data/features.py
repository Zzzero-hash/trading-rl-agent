"""
Feature engineering utilities for trading data pipelines.
"""
import pandas as pd
import numpy as np


def compute_log_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Compute log returns from price column.
    """
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def compute_simple_moving_average(df: pd.DataFrame, price_col: str = 'close', window: int = 20) -> pd.DataFrame:
    """
    Compute simple moving average for given window.
    """
    df[f'sma_{window}'] = df[price_col].rolling(window).mean()
    return df


def compute_rsi(df: pd.DataFrame, price_col: str = 'close', window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).
    """
    delta = df[price_col].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()

    rs = roll_up / roll_down
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    return df


def compute_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility based on log returns.
    """
    # Use population standard deviation (ddof=0) so vol = sqrt(mean(x^2)) * sqrt(window)
    df[f'vol_{window}'] = df['log_return'].rolling(window).std(ddof=0) * np.sqrt(window)
    return df


def add_sentiment(df: pd.DataFrame, sentiment_col: str = 'sentiment') -> pd.DataFrame:
    """
    Stub for sentiment feature (defaults to zero).
    """
    df[sentiment_col] = 0.0
    return df


def compute_ema(df: pd.DataFrame, price_col: str = 'close', timeperiod: int = 20) -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA).
    """
    # Use pandas built-in EMA implementation
    ema = df[price_col].ewm(span=timeperiod, adjust=False).mean()
    df[f'ema_{timeperiod}'] = ema
    # Force NaN values for warmup period
    df.loc[:timeperiod-2, f'ema_{timeperiod}'] = np.nan
    return df


def compute_macd(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Compute MACD, MACD signal, and MACD histogram.
    """
    slow, fast, signal_period = 26, 12, 9
    # Calculate MACD line: fast EMA - slow EMA
    fast_ema = df[price_col].ewm(span=fast, adjust=False).mean()
    slow_ema = df[price_col].ewm(span=slow, adjust=False).mean()
    df['macd_line'] = fast_ema - slow_ema
    # Calculate signal line: EMA of MACD line with signal_period
    df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    # Calculate MACD histogram: MACD line - signal line
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    # Enforce warmup period
    df.loc[:slow-1, 'macd_line'] = np.nan
    df.loc[:slow-1, 'macd_signal'] = np.nan
    df.loc[:slow-1, 'macd_hist'] = np.nan
    return df


def compute_atr(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) as rolling mean of high-low differences.
    """
    # Use simple high-low rolling mean for ATR
    tr = df['high'] - df['low']
    atr = tr.rolling(window=timeperiod).mean()
    df[f'atr_{timeperiod}'] = atr
    # Enforce warm-up: first timeperiod entries should be NaN
    df.loc[:timeperiod-1, f'atr_{timeperiod}'] = np.nan
    return df


def compute_bollinger_bands(df: pd.DataFrame, price_col: str = 'close', timeperiod: int = 20) -> pd.DataFrame:
    """
    Compute Bollinger Bands (upper, middle, lower).
    """
    # Calculate middle band (SMA)
    middle = df[price_col].rolling(window=timeperiod).mean()
    df[f'bb_mavg_{timeperiod}'] = middle
    
    # Calculate standard deviation
    stddev = df[price_col].rolling(window=timeperiod).std()
    
    # Calculate upper and lower bands
    df[f'bb_upper_{timeperiod}'] = middle + (2 * stddev)
    df[f'bb_lower_{timeperiod}'] = middle - (2 * stddev)
    return df


def compute_stochastic(df: pd.DataFrame, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator (%K and %D).
    """
    # Calculate raw %K (fast %K)
    roll_high = df['high'].rolling(window=fastk_period).max()
    roll_low = df['low'].rolling(window=fastk_period).min()
    fastk = 100 * ((df['close'] - roll_low) / (roll_high - roll_low))
    
    # Calculate slow %K
    slowk = fastk.rolling(window=slowk_period).mean()
    df['stoch_k'] = slowk
    
    # Calculate slow %D
    df['stoch_d'] = slowk.rolling(window=slowd_period).mean()
    return df


def compute_adx(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Compute a simplified directional movement indicator.
    """
    # Simple directional indicator (normalized high-low range)
    high_low_range = df['high'] - df['low']
    avg_range = high_low_range.rolling(window=timeperiod).mean()
    max_range = high_low_range.rolling(window=timeperiod).max()
    df[f'adx_{timeperiod}'] = 100 * (avg_range / max_range)
    return df


def compute_williams_r(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Compute Williams %R.
    """
    roll_high = df['high'].rolling(window=timeperiod).max()
    roll_low = df['low'].rolling(window=timeperiod).min()
    wr = -100 * ((roll_high - df['close']) / (roll_high - roll_low))
    df[f'wr_{timeperiod}'] = wr
    return df


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute On-Balance Volume (OBV).
    """
    # Determine price change direction
    price_change = df['close'].diff()
    # Create signals: 1 for up, -1 for down, 0 for no change
    signals = pd.Series(0, index=df.index)
    signals.loc[price_change > 0] = 1
    signals.loc[price_change < 0] = -1
    # Multiply signal by volume and compute cumulative sum
    df['obv'] = (signals * df['volume']).fillna(0).cumsum()
    return df


def detect_doji(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Detect Doji candlestick pattern (open and close are nearly equal).
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum percentage difference between open and close to qualify as a doji
        
    Returns:
        DataFrame with new 'doji' column (1 for doji, 0 otherwise)
    """
    body = abs(df['close'] - df['open'])
    range_size = df['high'] - df['low']
    
    # Avoid division by zero
    range_size = range_size.replace(0, np.nan)
    
    # Calculate body-to-range ratio
    body_ratio = body / range_size
    
    # Doji has very small body compared to range
    df['doji'] = ((body_ratio < threshold) & (range_size > 0)).astype(int)
    
    return df


def detect_hammer(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Detect Hammer and Hanging Man candlestick patterns.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum body size as ratio of total range to qualify
        
    Returns:
        DataFrame with 'hammer' and 'hanging_man' columns
    """
    body = (df['close'] - df['open']).abs()
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']

    # Avoid division by zero
    total_range = total_range.replace(0, np.nan)

    body_ratio = body / total_range
    lower_shadow_ratio = lower_shadow / total_range

    hammer_shape = (body_ratio < threshold) & (lower_shadow_ratio > 0.6)

    df['hammer'] = (
        hammer_shape & (df['close'].shift(1) < df['open'].shift(1))
    ).astype(int)

    df['hanging_man'] = (
        hammer_shape & (df['close'].shift(1) > df['open'].shift(1))
    ).astype(int)
    
    return df


def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Bullish and Bearish Engulfing candlestick patterns.
    
    Returns:
        DataFrame with 'bullish_engulfing' and 'bearish_engulfing' columns
    """
    # Previous candle's body
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_body_size = abs(prev_close - prev_open)
    
    # Current candle's body
    curr_body_size = abs(df['close'] - df['open'])
    
    # Bullish engulfing: current candle opens below previous close and closes above previous open
    df['bullish_engulfing'] = (
        (df['open'] < prev_close) & 
        (df['close'] > prev_open) & 
        (curr_body_size > prev_body_size) &
        (prev_close < prev_open)  # Previous candle was bearish
    ).astype(int)
    
    # Bearish engulfing: current candle opens above previous close and closes below previous open
    df['bearish_engulfing'] = (
        (df['open'] > prev_close) & 
        (df['close'] < prev_open) & 
        (curr_body_size > prev_body_size) &
        (prev_close > prev_open)  # Previous candle was bullish
    ).astype(int)
    
    return df


def detect_shooting_star(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Detect Shooting Star candlestick pattern.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum body size as ratio of total range to qualify
        
    Returns:
        DataFrame with 'shooting_star' column
    """
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Avoid division by zero
    total_range = total_range.replace(0, np.nan)
    
    # Calculate ratios
    body_ratio = body / total_range
    upper_shadow_ratio = upper_shadow / total_range
    
    # Shooting star criteria:
    # 1. Small body (less than threshold of total range)
    # 2. Long upper shadow (at least 2x the body)
    # 3. Very small or no lower shadow
    # 4. Appears in uptrend
    df['shooting_star'] = (
        (body_ratio < threshold) & 
        (upper_shadow_ratio > 0.6) & 
        (lower_shadow < 0.1 * total_range) &
        (upper_shadow > 2 * body) &
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle was bullish
        (df['close'].shift(2) < df['close'].shift(1))    # Prior trend was upward
    ).astype(int)
    
    return df


def detect_morning_star(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Morning Star candlestick pattern.
    
    Returns:
        DataFrame with 'morning_star' column
    """
    # First day: long bearish candle
    first_bearish = (df['open'].shift(2) > df['close'].shift(2)) & (abs(df['open'].shift(2) - df['close'].shift(2)) > 0.5 * (df['high'].shift(2) - df['low'].shift(2)))
    
    # Second day: small body with gap down
    second_small = abs(df['open'].shift(1) - df['close'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1))
    gap_down = df['high'].shift(1) < df['close'].shift(2)
    
    # Third day: bullish candle closing into first candle body
    third_bullish = df['close'] > df['open']
    closes_into_first = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
    
    df['morning_star'] = (first_bearish & second_small & gap_down & third_bullish & closes_into_first).astype(int)
    
    return df


def detect_evening_star(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Evening Star candlestick pattern.
    
    Returns:
        DataFrame with 'evening_star' column
    """
    # First day: long bullish candle
    first_bullish = (df['open'].shift(2) < df['close'].shift(2)) & (abs(df['open'].shift(2) - df['close'].shift(2)) > 0.5 * (df['high'].shift(2) - df['low'].shift(2)))
    
    # Second day: small body with gap up
    second_small = abs(df['open'].shift(1) - df['close'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1))
    gap_up = df['low'].shift(1) > df['close'].shift(2)
    
    # Third day: bearish candle closing into first candle body
    third_bearish = df['close'] < df['open']
    closes_into_first = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
    
    df['evening_star'] = (first_bullish & second_small & gap_up & third_bearish & closes_into_first).astype(int)
    
    return df


def compute_candle_features(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """
    Compute all candlestick pattern features.
    
    Args:
        df: DataFrame with OHLC data
        advanced: If True, use advanced patterns from candle_patterns.py
        
    Returns:
        DataFrame with candlestick pattern columns
    """
    if advanced:
        # Use advanced patterns from candle_patterns.py
        from src.data.candle_patterns import compute_all_candle_patterns
        df = compute_all_candle_patterns(df)
    else:
        # Use basic patterns
        df = detect_doji(df)
        df = detect_hammer(df)
        df = detect_engulfing(df)
        df = detect_shooting_star(df)
        df = detect_morning_star(df)
        df = detect_evening_star(df)
    return df


def generate_features(
    df: pd.DataFrame,
    ma_windows: list = [5, 10, 20],
    rsi_window: int = 14,
    vol_window: int = 20,
    advanced_candles: bool = True
) -> pd.DataFrame:
    """
    Apply a sequence of feature transformations to the DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        ma_windows: List of moving average window sizes
        rsi_window: Window size for RSI calculation
        vol_window: Window size for volatility calculation
        advanced_candles: If True, use advanced candlestick patterns
        
    Returns:
        DataFrame with all technical indicators and features applied
    """
    df = df.copy()
    df = compute_log_returns(df)

    for w in ma_windows:
        df = compute_simple_moving_average(df, window=w)

    df = compute_rsi(df, window=rsi_window)
    df = compute_rolling_volatility(df, window=vol_window)
    df = add_sentiment(df)

    # Additional technical indicators
    df = compute_ema(df, price_col='close', timeperiod=20)
    df = compute_macd(df, price_col='close')
    df = compute_atr(df, timeperiod=14)
    df = compute_bollinger_bands(df, price_col='close', timeperiod=20)
    df = compute_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df = compute_adx(df, timeperiod=14)
    df = compute_williams_r(df, timeperiod=14)
    df = compute_obv(df)
    
    # Candlestick patterns
    df = compute_candle_features(df, advanced=advanced_candles)

    # Drop initial rows based on the largest warm-up window across all indicators
    # Core MA/RSI/Vol windows
    windows = ma_windows + [rsi_window, vol_window]
    # Technical indicators warm-up periods: EMA, MACD slow & signal, ATR, Bollinger Bands, Stochastic, ADX, Williams %R
    windows += [20, 26, 9, 14, 20, 14, 14, 14]
    
    # If using advanced candles, account for patterns that use up to 3 previous candles
    max_pattern_window = 3 if advanced_candles else 2
    windows.append(max_pattern_window)
    
    max_core_window = max(windows)
    df = df.iloc[max_core_window:].reset_index(drop=True)
    return df
