"""Data ingestion and feature engineering pipeline for trading data.

This module provides utilities to load historical market data from different
sources, compute common technical indicators, and split the resulting dataset
into train/validation/test segments.  It is designed to be configuration driven
and easily extended.  Example usage:

>>> from trading_rl_agent.data_pipeline import PipelineConfig, load_data, generate_features, split_by_date
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
import pandas_ta as ta

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

        def apply_basic_features(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            batch["log_return"] = np.log(batch["close"] / batch["close"].shift(1))
            for w in cfg.sma_windows:
                batch[f"sma_{w}"] = batch["close"].rolling(w).mean()
            for w in cfg.momentum_windows:
                batch[f"mom_{w}"] = batch["close"].diff(w)
            batch[f"rsi_{cfg.rsi_window}"] = ta.rsi(
                batch["close"].astype(float), length=cfg.rsi_window
            )
            batch[f"vol_{cfg.vol_window}"] = batch["log_return"].rolling(
                cfg.vol_window
            ).std(ddof=0) * np.sqrt(cfg.vol_window)
            return batch

        ds = ds.map_batches(apply_basic_features)
        df = ds.to_pandas()
    else:
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        for w in cfg.sma_windows:
            df[f"sma_{w}"] = df["close"].rolling(w).mean()
        for w in cfg.momentum_windows:
            df[f"mom_{w}"] = df["close"].diff(w)
        df[f"rsi_{cfg.rsi_window}"] = ta.rsi(
            df["close"].astype(float), length=cfg.rsi_window
        )
        df[f"vol_{cfg.vol_window}"] = df["log_return"].rolling(cfg.vol_window).std(
            ddof=0
        ) * np.sqrt(cfg.vol_window)

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
