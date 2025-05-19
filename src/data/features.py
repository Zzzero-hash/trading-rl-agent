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
    df[f'vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(window)
    return df


def add_sentiment(df: pd.DataFrame, sentiment_col: str = 'sentiment') -> pd.DataFrame:
    """
    Stub for sentiment feature (defaults to zero).
    """
    df[sentiment_col] = 0.0
    return df


def generate_features(
    df: pd.DataFrame,
    ma_windows: list = [5, 10, 20],
    rsi_window: int = 14,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Apply a sequence of feature transformations to the DataFrame.
    """
    df = df.copy()
    df = compute_log_returns(df)

    for w in ma_windows:
        df = compute_simple_moving_average(df, window=w)

    df = compute_rsi(df, window=rsi_window)
    df = compute_rolling_volatility(df, window=vol_window)
    df = add_sentiment(df)

    # Drop initial NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df
