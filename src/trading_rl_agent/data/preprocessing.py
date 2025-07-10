"""
Data preprocessing utilities for trading RL agent.
Provides functions for data standardization, sequence creation, and normalization.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def create_sequences(
    data: np.ndarray | pd.DataFrame,
    sequence_length: int,
    target_column: str | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data for model training.

    Args:
        data: Input time series data
        sequence_length: Length of each sequence
        target_column: Name of target column (for DataFrame)
        stride: Step size between sequences

    Returns:
        Tuple of (sequences, targets)
    """
    if isinstance(data, pd.DataFrame):
        if target_column and target_column in data.columns:
            features = data.drop(columns=[target_column]).to_numpy()
            targets = data[target_column].to_numpy()
        else:
            features = data.to_numpy()
            targets = data.to_numpy()[:, -1]  # Use last column as target
    else:
        features = np.asarray(data)
        targets = data[:, -1] if data.ndim > 1 else data

    features = np.asarray(features)
    targets = np.asarray(targets)

    if len(features) <= sequence_length:
        seq_shape = (0, sequence_length, features.shape[1]) if features.ndim > 1 else (0, sequence_length)
        return np.empty(seq_shape), np.empty((0, 1))

    windows = sliding_window_view(features, sequence_length, axis=0)[:-1]
    if features.ndim > 1:
        windows = windows.transpose(0, 2, 1)

    sequences_arr = windows[::stride]
    targets_arr = targets[sequence_length:][::stride].reshape(-1, 1)
    return sequences_arr, targets_arr


def preprocess_trading_data(
    data: pd.DataFrame,
    sequence_length: int = 20,
    target_column: str = "close",
    normalize_method: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler]:
    """Complete preprocessing pipeline for trading data."""

    df = data.copy()

    # Convert datetime columns to numeric for scaling
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col]).view("int64")

    if normalize_method == "minmax":
        scaler = MinMaxScaler()
    elif normalize_method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")

    normalized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )

    sequences, targets = create_sequences(normalized, sequence_length, target_column)

    return sequences, targets, scaler
