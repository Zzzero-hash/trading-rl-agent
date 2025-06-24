"""
Data preprocessing utilities for trading RL agent.
Provides functions for data standardization, sequence creation, and normalization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardize_data(
    data: Union[np.ndarray, pd.DataFrame],
    scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None,
    fit_scaler: bool = True,
) -> tuple[Union[np.ndarray, pd.DataFrame], Union[StandardScaler, MinMaxScaler]]:
    """
    Standardize data using StandardScaler or provided scaler.

    Args:
        data: Input data to standardize
        scaler: Pre-fitted scaler to use (optional)
        fit_scaler: Whether to fit the scaler on the data

    Returns:
        Tuple of (standardized_data, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()

    if isinstance(data, pd.DataFrame):
        if fit_scaler:
            standardized = scaler.fit_transform(data)
        else:
            standardized = scaler.transform(data)
        return (
            pd.DataFrame(standardized, columns=data.columns, index=data.index),
            scaler,
        )
    else:
        if fit_scaler:
            standardized = scaler.fit_transform(data)
        else:
            standardized = scaler.transform(data)
        return standardized, scaler


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

    for i in range(0, len(features) - sequence_length + 1, stride):
        seq = features[i : i + sequence_length]
        target = targets[i + sequence_length - 1]
        sequences.append(seq)
        sequence_targets.append(target)

    return np.array(sequences), np.array(sequence_targets)


def normalize_data(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "minmax",
    feature_range: tuple[float, float] = (0, 1),
) -> tuple[Union[np.ndarray, pd.DataFrame], Union[StandardScaler, MinMaxScaler]]:
    """
    Normalize data using specified method.

    Args:
        data: Input data to normalize
        method: Normalization method ('minmax' or 'standard')
        feature_range: Range for MinMaxScaler

    Returns:
        Tuple of (normalized_data, scaler)
    """
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return standardize_data(data, scaler, fit_scaler=True)


def preprocess_trading_data(
    data: pd.DataFrame,
    sequence_length: int = 20,
    target_column: str = "close",
    normalize_method: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Complete preprocessing pipeline for trading data.

    Args:
        data: Trading data DataFrame
        sequence_length: Length of sequences to create
        target_column: Name of target column
        normalize_method: Normalization method

    Returns:
        Tuple of (sequences, targets, scaler)
    """
    # Normalize data
    normalized_data, scaler = normalize_data(data, method=normalize_method)

    # Create sequences
    sequences, targets = create_sequences(
        normalized_data, sequence_length, target_column
    )

    return sequences, targets, scaler
