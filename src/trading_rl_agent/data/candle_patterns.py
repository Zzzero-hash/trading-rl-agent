"""
Advanced candlestick pattern detection for trading data.

This module provides functions to detect various candlestick patterns
and create statistics and features based on candlestick characteristics.
"""

import numpy as np
import pandas as pd


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
    df["body_position"] = (((min_price - df["low"]) + (max_price - df["low"])) / 2) / df["range_size"]

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
    Compute all candlestick patterns using pure pandas implementations.

    Returns:
        DataFrame with all candlestick pattern columns
    """
    df = df.copy()

    # Apply individual pattern detection functions instead of pandas_ta
    df = detect_inside_bar(df)
    df = detect_outside_bar(df)
    df = detect_tweezer_top(df)
    df = detect_tweezer_bottom(df)
    df = detect_three_white_soldiers(df)
    df = detect_three_black_crows(df)
    df = detect_piercing_line(df)
    df = detect_dark_cloud_cover(df)
    df = detect_harami(df)

    # Compute statistical candle features
    return compute_candle_stats(df)


# Individual pattern detection functions
def detect_inside_bar(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["inside_bar"] = (
        ((df2["high"] < df2["high"].shift(1)) & (df2["low"] > df2["low"].shift(1))).fillna(False).astype(int)
    )
    return df2


def detect_outside_bar(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["outside_bar"] = (
        ((df2["high"] > df2["high"].shift(1)) & (df2["low"] < df2["low"].shift(1))).fillna(False).astype(int)
    )
    return df2


def detect_tweezer_top(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["tweezer_top"] = (df2["high"] == df2["high"].shift(1)).fillna(False).astype(int)
    return df2


def detect_tweezer_bottom(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["tweezer_bottom"] = (df2["low"] == df2["low"].shift(1)).fillna(False).astype(int)
    return df2


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    res = [0] * len(df2)
    for i in range(2, len(df2)):
        o0, c0 = df2["open"].iloc[i - 2], df2["close"].iloc[i - 2]
        o1, c1 = df2["open"].iloc[i - 1], df2["close"].iloc[i - 1]
        o2, c2 = df2["open"].iloc[i], df2["close"].iloc[i]
        if c0 > o0 and c1 > o1 and c2 > o2 and o1 > o0 and o2 > o1 and c1 > c0 and c2 > c1:
            res[i] = 1
    df2["three_white_soldiers"] = res
    return df2


def detect_three_black_crows(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    res = [0] * len(df2)
    for i in range(2, len(df2)):
        o0, c0 = df2["open"].iloc[i - 2], df2["close"].iloc[i - 2]
        o1, c1 = df2["open"].iloc[i - 1], df2["close"].iloc[i - 1]
        o2, c2 = df2["open"].iloc[i], df2["close"].iloc[i]
        if c0 < o0 and c1 < o1 and c2 < o2 and o1 < o0 and o2 < o1 and c1 < c0 and c2 < c1:
            res[i] = 1
    df2["three_black_crows"] = res
    return df2


def detect_piercing_line(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    res = [0] * len(df2)
    for i in range(1, len(df2)):
        o1, c1 = df2["open"].iloc[i - 1], df2["close"].iloc[i - 1]
        o2, c2 = df2["open"].iloc[i], df2["close"].iloc[i]
        if c2 > o2 and o2 < df2["low"].iloc[i - 1] and c2 > (o1 + c1) / 2:
            res[i] = 1
    df2["piercing_line"] = res
    return df2


def detect_dark_cloud_cover(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    res = [0] * len(df2)
    for i in range(1, len(df2)):
        o1, c1 = df2["open"].iloc[i - 1], df2["close"].iloc[i - 1]
        o2, c2 = df2["open"].iloc[i], df2["close"].iloc[i]
        if c2 < o2 and o2 > df2["high"].iloc[i - 1] and c2 < (o1 + c1) / 2:
            res[i] = 1
    df2["dark_cloud_cover"] = res
    return df2


def detect_harami(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    bullish = [0] * len(df2)
    bearish = [0] * len(df2)
    for i in range(1, len(df2)):
        o1, c1 = df2["open"].iloc[i - 1], df2["close"].iloc[i - 1]
        o2, c2 = df2["open"].iloc[i], df2["close"].iloc[i]
        # Bullish harami: first bearish, second bullish inside body
        if c1 < o1 and c2 > o2 > c1 and c2 < o1:
            bullish[i] = 1
        # Bearish harami: first bullish, second bearish inside body
        if c1 > o1 and c2 < o2 < c1 and c2 > o1:
            bearish[i] = 1
    df2["bullish_harami"] = bullish
    df2["bearish_harami"] = bearish
    return df2
