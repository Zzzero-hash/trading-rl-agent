"""
Data preprocessing utilities for trading RL agent.
Provides functions for data standardization, sequence creation, and normalization.
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def validate_data(df: pd.DataFrame) -> None:
    """Validate trading data schema and integrity.

    Raises:
        ValueError: If validation fails
    """
    # Expected columns
    expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Type checks
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("timestamp must be datetime type")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")

    # Range validation
    if (df["high"] < df["low"]).any():
        raise ValueError("high < low in some rows")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError("close out of low-high range in some rows")
    if (df["open"] < 0).any() or (df["close"] < 0).any():
        raise ValueError("Negative prices")
    if (df["volume"] < 0).any():
        raise ValueError("Negative volume")

    # Duplicates
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps")

    # Sorted
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps not sorted")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean trading data: handle duplicates, NaNs, outliers.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"])

    # Handle NaNs: forward fill, then interpolate remaining
    df = df.ffill()
    df = df.interpolate(method="linear")

    # Handle outliers: clip to 3 std deviations for prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower = max(0, mean - 3 * std)  # prices can't be negative
        upper = mean + 3 * std
        df[col] = df[col].clip(lower, upper)

    # Ensure no remaining NaNs
    if df.isnull().any().any():
        raise ValueError("Remaining NaNs after cleaning")

    return df


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

    # Validate and clean
    validate_data(data)
    df = clean_data(data)

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


def prepare_data_for_trial(params: dict) -> tuple[DataLoader, DataLoader, int]:
    """
    Prepare data for Optuna trial optimization.

    Args:
        params: Dictionary containing trial parameters

    Returns:
        Tuple of (train_loader, val_loader, n_features)
    """
    # Import here to avoid circular imports

    # Create a simple dataset for optimization trials
    # In a real scenario, this would use the actual dataset
    sequence_length = params.get("sequence_length", 30)
    batch_size = params.get("batch_size", 32)

    # Generate synthetic data for optimization trials
    np.random.seed(42)
    n_samples = 1000
    n_features = 20  # Number of features

    # Create synthetic sequences and targets
    sequences = np.random.randn(n_samples, sequence_length, n_features)
    targets = np.random.randn(n_samples, 1)

    # Split data
    split_idx = int(0.8 * n_samples)
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]

    # Convert to tensors
    train_sequences_tensor = torch.FloatTensor(train_sequences)
    train_targets_tensor = torch.FloatTensor(train_targets)
    val_sequences_tensor = torch.FloatTensor(val_sequences)
    val_targets_tensor = torch.FloatTensor(val_targets)

    # Create data loaders
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    val_dataset = TensorDataset(val_sequences_tensor, val_targets_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_features
