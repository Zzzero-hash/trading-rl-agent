"""
Advanced candlestick pattern detection for trading data.

This module provides functions to detect various candlestick patterns
and create statistics and features based on candlestick characteristics.
"""
import pandas as pd
import numpy as np


def detect_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Inside Bar pattern (current bar's high and low are within previous bar's range).
    
    Returns:
        DataFrame with 'inside_bar' column
    """
    df['inside_bar'] = (
        (df['high'] <= df['high'].shift(1)) &
        (df['low'] >= df['low'].shift(1))
    ).astype(int)
    
    return df


def detect_outside_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Outside Bar pattern (current bar's high and low are outside previous bar's range).
    
    Returns:
        DataFrame with 'outside_bar' column
    """
    df['outside_bar'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['low'] < df['low'].shift(1))
    ).astype(int)
    
    return df


def detect_tweezer_top(df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
    """
    Detect Tweezer Top pattern.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Maximum percentage difference in highs to qualify
        
    Returns:
        DataFrame with 'tweezer_top' column
    """
    # Calculate high price difference percentage
    high_diff_pct = abs(df['high'] - df['high'].shift(1)) / df['high'].shift(1)
    
    # Tweezer top criteria:
    # 1. First candle is bullish (close > open)
    # 2. Second candle is bearish (close < open)
    # 3. Both candles have similar highs
    # 4. Occurs in an uptrend
    df['tweezer_top'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &   # First candle bullish
        (df['close'] < df['open']) &                     # Second candle bearish
        (high_diff_pct <= tolerance) &                   # Similar highs
        (df['close'].shift(2) < df['close'].shift(1))    # Uptrend
    ).astype(int)
    
    return df


def detect_tweezer_bottom(df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
    """
    Detect Tweezer Bottom pattern.
    
    Args:
        df: DataFrame with OHLC data
        tolerance: Maximum percentage difference in lows to qualify
        
    Returns:
        DataFrame with 'tweezer_bottom' column
    """
    # Calculate low price difference percentage
    low_diff_pct = abs(df['low'] - df['low'].shift(1)) / df['low'].shift(1)
    
    # Tweezer bottom criteria:
    # 1. First candle is bearish (close < open)
    # 2. Second candle is bullish (close > open)
    # 3. Both candles have similar lows
    # 4. Occurs in a downtrend
    df['tweezer_bottom'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &   # First candle bearish
        (df['close'] > df['open']) &                     # Second candle bullish
        (low_diff_pct <= tolerance) &                     # Similar lows
        (df['close'].shift(2) > df['close'].shift(1))    # Downtrend
    ).astype(int)
    
    return df


def detect_three_white_soldiers(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Detect Three White Soldiers pattern.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum upper shadow ratio to qualify
        
    Returns:
        DataFrame with 'three_white_soldiers' column
    """
    # Check for three consecutive bullish candles
    bullish_1 = df['close'].shift(2) > df['open'].shift(2)
    bullish_2 = df['close'].shift(1) > df['open'].shift(1)
    bullish_3 = df['close'] > df['open']
    
    # Each candle should open within previous candle's body
    open_in_range_2 = (df['open'].shift(1) > df['open'].shift(2)) & (df['open'].shift(1) < df['close'].shift(2))
    open_in_range_3 = (df['open'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    
    # Each candle should close higher than the previous
    higher_close_2 = df['close'].shift(1) > df['close'].shift(2)
    higher_close_3 = df['close'] > df['close'].shift(1)
    
    # Each candle should have small upper shadows
    body_1 = abs(df['close'].shift(2) - df['open'].shift(2))
    body_2 = abs(df['close'].shift(1) - df['open'].shift(1))
    body_3 = abs(df['close'] - df['open'])
    
    upper_shadow_1 = df['high'].shift(2) - df['close'].shift(2)
    upper_shadow_2 = df['high'].shift(1) - df['close'].shift(1)
    upper_shadow_3 = df['high'] - df['close']
    
    small_upper_shadow_1 = upper_shadow_1 < threshold * body_1
    small_upper_shadow_2 = upper_shadow_2 < threshold * body_2
    small_upper_shadow_3 = upper_shadow_3 < threshold * body_3
    
    # Combine all conditions
    df['three_white_soldiers'] = (
        bullish_1 & bullish_2 & bullish_3 &
        open_in_range_2 & open_in_range_3 &
        higher_close_2 & higher_close_3 &
        small_upper_shadow_1 & small_upper_shadow_2 & small_upper_shadow_3
    ).astype(int)
    
    return df


def detect_three_black_crows(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Detect Three Black Crows pattern.
    
    Args:
        df: DataFrame with OHLC data
        threshold: Maximum lower shadow ratio to qualify
        
    Returns:
        DataFrame with 'three_black_crows' column
    """
    # Check for three consecutive bearish candles
    bearish_1 = df['close'].shift(2) < df['open'].shift(2)
    bearish_2 = df['close'].shift(1) < df['open'].shift(1)
    bearish_3 = df['close'] < df['open']
    
    # Each candle should open within previous candle's body
    open_in_range_2 = (df['open'].shift(1) < df['open'].shift(2)) & (df['open'].shift(1) > df['close'].shift(2))
    open_in_range_3 = (df['open'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    
    # Each candle should close lower than the previous
    lower_close_2 = df['close'].shift(1) < df['close'].shift(2)
    lower_close_3 = df['close'] < df['close'].shift(1)
    
    # Each candle should have small lower shadows
    body_1 = abs(df['close'].shift(2) - df['open'].shift(2))
    body_2 = abs(df['close'].shift(1) - df['open'].shift(1))
    body_3 = abs(df['close'] - df['open'])
    
    lower_shadow_1 = df['close'].shift(2) - df['low'].shift(2)
    lower_shadow_2 = df['close'].shift(1) - df['low'].shift(1)
    lower_shadow_3 = df['close'] - df['low']
    
    small_lower_shadow_1 = lower_shadow_1 < threshold * body_1
    small_lower_shadow_2 = lower_shadow_2 < threshold * body_2
    small_lower_shadow_3 = lower_shadow_3 < threshold * body_3
    
    # Combine all conditions
    df['three_black_crows'] = (
        bearish_1 & bearish_2 & bearish_3 &
        open_in_range_2 & open_in_range_3 &
        lower_close_2 & lower_close_3 &
        small_lower_shadow_1 & small_lower_shadow_2 & small_lower_shadow_3
    ).astype(int)
    
    return df


def detect_harami(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Bullish and Bearish Harami patterns.
    
    Returns:
        DataFrame with 'bullish_harami' and 'bearish_harami' columns
    """
    # Previous candle's body
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_body_size = abs(prev_close - prev_open)
    
    # Current candle's body
    curr_body_size = abs(df['close'] - df['open'])
    
    # Bullish harami: previous bearish candle engulfs current bullish candle
    df['bullish_harami'] = (
        (prev_close < prev_open) &                # Previous candle bearish
        (df['close'] > df['open']) &              # Current candle bullish
        (df['open'] > prev_close) &               # Current open > previous close
        (df['open'] < prev_open) &                # Current open < previous open
        (df['close'] > prev_close) &              # Current close > previous close
        (df['close'] < prev_open) &               # Current close < previous open
        (curr_body_size < prev_body_size * 0.6)   # Current body significantly smaller
    ).astype(int)
    
    # Bearish harami: previous bullish candle engulfs current bearish candle
    df['bearish_harami'] = (
        (prev_close > prev_open) &                # Previous candle bullish
        (df['close'] < df['open']) &              # Current candle bearish
        (df['open'] < prev_close) &               # Current open < previous close
        (df['open'] > prev_open) &                # Current open > previous open
        (df['close'] < prev_close) &              # Current close < previous close
        (df['close'] > prev_open) &               # Current close > previous open
        (curr_body_size < prev_body_size * 0.6)   # Current body significantly smaller
    ).astype(int)
    
    return df


def detect_dark_cloud_cover(df: pd.DataFrame, penetration: float = 0.5) -> pd.DataFrame:
    """
    Detect Dark Cloud Cover pattern.
    
    Args:
        df: DataFrame with OHLC data
        penetration: Minimum penetration into previous candle's body
        
    Returns:
        DataFrame with 'dark_cloud_cover' column
    """
    # Previous candle bullish
    prev_bullish = df['close'].shift(1) > df['open'].shift(1)
    
    # Current candle bearish
    curr_bearish = df['close'] < df['open']
    
    # Current candle opens above previous high
    gap_up = df['open'] > df['high'].shift(1)
    
    # Current candle closes below midpoint of previous candle's body
    midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
    deep_penetration = df['close'] < midpoint
    
    # Penetration ratio
    prev_body = df['close'].shift(1) - df['open'].shift(1)
    penetration_ratio = (df['open'] - df['close']) / prev_body
    
    # Combine conditions
    df['dark_cloud_cover'] = (
        prev_bullish &
        curr_bearish &
        gap_up &
        deep_penetration &
        (penetration_ratio >= penetration)
    ).astype(int)
    
    return df


def detect_piercing_line(df: pd.DataFrame, penetration: float = 0.5) -> pd.DataFrame:
    """
    Detect Piercing Line pattern.
    
    Args:
        df: DataFrame with OHLC data
        penetration: Minimum penetration into previous candle's body
        
    Returns:
        DataFrame with 'piercing_line' column
    """
    # Previous candle bearish
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)
    
    # Current candle bullish
    curr_bullish = df['close'] > df['open']
    
    # Current candle opens below previous low
    gap_down = df['open'] < df['low'].shift(1)
    
    # Current candle closes above midpoint of previous candle's body
    midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
    deep_penetration = df['close'] > midpoint
    
    # Penetration ratio
    prev_body = df['open'].shift(1) - df['close'].shift(1)
    penetration_ratio = (df['close'] - df['open']) / prev_body
    
    # Combine conditions
    df['piercing_line'] = (
        prev_bearish &
        curr_bullish &
        gap_down &
        deep_penetration &
        (penetration_ratio >= penetration)
    ).astype(int)
    
    return df


def compute_candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical features based on candlestick characteristics.
    
    Returns:
        DataFrame with additional candle statistic columns
    """
    # Body size (absolute and relative to range)
    df['body_size'] = abs(df['close'] - df['open'])
    df['range_size'] = df['high'] - df['low']
    df['rel_body_size'] = df['body_size'] / df['range_size']
    
    # Upper and lower shadow sizes
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Relative shadow sizes
    df['rel_upper_shadow'] = df['upper_shadow'] / df['range_size']
    df['rel_lower_shadow'] = df['lower_shadow'] / df['range_size']
    
    # Body position within range (0 = at bottom, 1 = at top)
    min_price = df[['open', 'close']].min(axis=1)
    max_price = df[['open', 'close']].max(axis=1)
    df['body_position'] = (((min_price - df['low']) + (max_price - df['low'])) / 2) / df['range_size']
    
    # Body type (1 = bullish, -1 = bearish, 0 = doji)
    df['body_type'] = np.sign(df['close'] - df['open'])
    
    # Moving averages of body features
    windows = [5, 10, 20]
    for w in windows:
        # Average relative body size over window
        df[f'avg_rel_body_{w}'] = df['rel_body_size'].rolling(window=w).mean()
        
        # Average upper shadow ratio
        df[f'avg_upper_shadow_{w}'] = df['rel_upper_shadow'].rolling(window=w).mean()
        
        # Average lower shadow ratio
        df[f'avg_lower_shadow_{w}'] = df['rel_lower_shadow'].rolling(window=w).mean()
        
        # Body position trend
        df[f'avg_body_pos_{w}'] = df['body_position'].rolling(window=w).mean()
        
        # Bullish/bearish momentum (avg of body_type)
        df[f'body_momentum_{w}'] = df['body_type'].rolling(window=w).mean()
    
    return df


def compute_all_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all candlestick patterns and statistics.
    
    Returns:
        DataFrame with all candlestick pattern columns
    """
    # Standard patterns (from features.py)
    from src.data.features import (
        detect_doji,
        detect_hammer,
        detect_engulfing,
        detect_shooting_star,
        detect_morning_star,
        detect_evening_star,
    )
    
    # Standard patterns
    df = detect_doji(df)
    df = detect_hammer(df)
    df = detect_engulfing(df)
    df = detect_shooting_star(df)
    df = detect_morning_star(df)
    df = detect_evening_star(df)
    
    # Additional patterns
    df = detect_inside_bar(df)
    df = detect_outside_bar(df)
    df = detect_tweezer_top(df)
    df = detect_tweezer_bottom(df)
    df = detect_three_white_soldiers(df)
    df = detect_three_black_crows(df)
    df = detect_harami(df)
    df = detect_dark_cloud_cover(df)
    df = detect_piercing_line(df)
      # Statistical features
    df = compute_candle_stats(df)
    
    return df
