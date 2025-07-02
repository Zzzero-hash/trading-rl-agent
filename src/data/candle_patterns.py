"""
Advanced candlestick pattern detection for trading data.

This module provides functions to detect various candlestick patterns
and create statistics and features based on candlestick characteristics.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_candle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical features based on candlestick characteristics.

    Returns:
        DataFrame with additional candle statistic columns
    """
    # Body size (absolute and relative to range)
    df["body_size"] = abs(df["close"] - df["open"])
    df["range_size"] = df["high"] - df["low"]
    df["rel_body_size"] = df["body_size"] / df["range_size"]

    # Upper and lower shadow sizes
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    # Relative shadow sizes
    df["rel_upper_shadow"] = df["upper_shadow"] / df["range_size"]
    df["rel_lower_shadow"] = df["lower_shadow"] / df["range_size"]

    # Body position within range (0 = at bottom, 1 = at top)
    min_price = df[["open", "close"]].min(axis=1)
    max_price = df[["open", "close"]].max(axis=1)
    df["body_position"] = (
        ((min_price - df["low"]) + (max_price - df["low"])) / 2
    ) / df["range_size"]

    # Body type (1 = bullish, -1 = bearish, 0 = doji)
    df["body_type"] = np.sign(df["close"] - df["open"])

    # Moving averages of body features
    windows = [5, 10, 20]
    for w in windows:
        # Average relative body size over window
        df[f"avg_rel_body_{w}"] = df["rel_body_size"].rolling(window=w).mean()

        # Average upper shadow ratio
        df[f"avg_upper_shadow_{w}"] = df["rel_upper_shadow"].rolling(window=w).mean()

        # Average lower shadow ratio
        df[f"avg_lower_shadow_{w}"] = df["rel_lower_shadow"].rolling(window=w).mean()

        # Body position trend
        df[f"avg_body_pos_{w}"] = df["body_position"].rolling(window=w).mean()

        # Bullish/bearish momentum (avg of body_type)
        df[f"body_momentum_{w}"] = df["body_type"].rolling(window=w).mean()

    return df


def compute_all_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all candlestick patterns using pandas_ta and statistical features.

    Returns:
        DataFrame with all candlestick pattern columns
    """
    df = df.copy()
    # Append all TA-Lib supported candlestick pattern columns (CDL_*)
    df.ta.cdl_pattern(append=True)
    # Rename pattern columns to lowercase snake_case
    pattern_cols = [col for col in df.columns if col.startswith("CDL_")]
    rename_map = {col: col.lower() for col in pattern_cols}
    df = df.rename(columns=rename_map)
    # Compute statistical candle features
    df = compute_candle_stats(df)

    return df
