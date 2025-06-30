"""
Data preprocessing utilities for trading RL agent.
Provides functions for data standardization, sequence creation, and normalization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def create_sequences(
    data: Union[np.ndarray, pd.DataFrame],
    sequence_length: int,
    target_column: Optional[str] = None,
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
            features = data.drop(columns=[target_column]).values
            targets = data[target_column].values
        else:
            features = data.values
            targets = data.values[:, -1]  # Use last column as target
    else:
        features = data
        targets = data[:, -1] if data.ndim > 1 else data

    sequences = []
    sequence_targets = []

    # Create sequences and next-step targets
    for i in range(0, len(features) - sequence_length, stride):
        seq = features[i : i + sequence_length]
        # target is the next value after the sequence
        target = targets[i + sequence_length]
        sequences.append(seq)
        sequence_targets.append(target)

    sequences_arr = np.array(sequences)
    targets_arr = np.array(sequence_targets).reshape(-1, 1)
    return sequences_arr, targets_arr




def preprocess_trading_data(
    data: pd.DataFrame,
    sequence_length: int = 20,
    target_column: str = "close",
    normalize_method: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
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
        scaler.fit_transform(df), columns=df.columns, index=df.index
    )

    sequences, targets = create_sequences(normalized, sequence_length, target_column)

    return sequences, targets, scaler
