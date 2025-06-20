"""Data ingestion and feature engineering pipeline for trading data.

This module provides utilities to load historical market data from different
sources, compute common technical indicators, and split the resulting dataset
into train/validation/test segments.  It is designed to be configuration driven
and easily extended.  Example usage:

>>> from src.data_pipeline import PipelineConfig, load_data, generate_features, split_by_date
>>> cfg = PipelineConfig(sma_windows=[3], momentum_windows=[3], rsi_window=14, vol_window=5)
>>> df = load_data({"type": "csv", "path": "prices.csv"})
>>> features = generate_features(df, cfg)
>>> train, val, test = split_by_date(features, '2020-01-01', '2020-06-01')

The functions rely primarily on pandas for computations but can scale to larger
than memory datasets by leveraging Ray Datasets when ``use_ray`` is set in the
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import ray.data as rdata
except Exception:  # pragma: no cover - Ray is optional
    rdata = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for feature generation."""

    sma_windows: list[int] = field(default_factory=lambda: [5, 10])
    momentum_windows: list[int] = field(default_factory=list)
    rsi_window: int = 14
    vol_window: int = 20
    use_ray: bool = False


def load_data(source_cfg: dict[str, Any]) -> pd.DataFrame:
    """Load market data from a CSV file or database.

    Parameters
    ----------
    source_cfg : dict
        Configuration with at least a ``type`` key. Supported types are
        ``"csv"`` and ``"database"``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the historical data with a ``timestamp`` column.
    """
    src_type = source_cfg.get("type", "csv")
    logger.info("Loading data of type %s", src_type)

    if src_type == "csv":
        path = source_cfg["path"]
        df = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    elif src_type == "database":
        import sqlite3  # lightweight default

        conn = sqlite3.connect(source_cfg["connection"])
        query = source_cfg.get("query", "SELECT * FROM prices")
        df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
        conn.close()
        logger.info("Loaded %d rows from database", len(df))
        return df

    raise ValueError(f"Unsupported data source type: {src_type}")


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns column ``log_return``.

    log_return_t = log(close_t / close_{t-1})
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add simple moving average feature.

    SMA_t = mean(close_{t-window+1:t})
    """
    df = df.copy()
    df[f"sma_{window}"] = df["close"].rolling(window).mean()
    return df


def compute_momentum(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add momentum indicator as difference from ``window`` days ago."""
    df = df.copy()
    df[f"mom_{window}"] = df["close"].diff(window)
    return df


def compute_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) feature."""
    df = df.copy()
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean()
    rs = roll_up / roll_down
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


def compute_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling volatility based on ``log_return``."""
    df = df.copy()
    df[f"vol_{window}"] = df["log_return"].rolling(window).std(ddof=0) * np.sqrt(window)
    return df


# ---------------------------------------------------------------------------


def generate_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Generate features as specified in ``cfg``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input OHLCV data sorted by ``timestamp``.
    cfg : PipelineConfig
        Configuration specifying which indicators to compute.

    Returns
    -------
    pandas.DataFrame
        DataFrame with feature columns appended.
    """
    if cfg.use_ray and rdata is not None:
        logger.info("Using Ray Datasets for feature computation")
        ds = rdata.from_pandas(df)
        for w in cfg.sma_windows:
            ds = ds.map_batches(lambda d, w=w: compute_sma(d, w))
        for w in cfg.momentum_windows:
            ds = ds.map_batches(lambda d, w=w: compute_momentum(d, w))
        ds = ds.map_batches(lambda d: compute_log_returns(d))
        ds = ds.map_batches(lambda d: compute_rsi(d, cfg.rsi_window))
        ds = ds.map_batches(lambda d: compute_volatility(d, cfg.vol_window))
        df = ds.to_pandas()
    else:
        df = df.copy()
        df = compute_log_returns(df)
        for w in cfg.sma_windows:
            df = compute_sma(df, w)
        for w in cfg.momentum_windows:
            df = compute_momentum(df, w)
        df = compute_rsi(df, cfg.rsi_window)
        df = compute_volatility(df, cfg.vol_window)

    return df


def split_by_date(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation/test by date.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with a ``timestamp`` column.
    train_end : str
        Last date (exclusive) for the training set.
    val_end : str
        Last date (exclusive) for the validation set. The remainder is test.

    Returns
    -------
    tuple of DataFrames
        ``(train_df, val_df, test_df)`` split chronologically.
    """
    df = df.sort_values("timestamp")
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)

    train = df[df["timestamp"] < train_end_ts]
    val = df[(df["timestamp"] >= train_end_ts) & (df["timestamp"] < val_end_ts)]
    test = df[df["timestamp"] >= val_end_ts]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


__all__ = [
    "PipelineConfig",
    "load_data",
    "generate_features",
    "split_by_date",
]
