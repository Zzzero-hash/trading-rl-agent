"""
Data preprocessing utilities for trading RL agent.
Provides functions for data standardization, sequence creation, and normalization.
Enhanced with comprehensive data pipeline components for production CLI.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


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
        # Use median and MAD for more robust outlier detection
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))  # Median Absolute Deviation
        # Convert MAD to approximate std (MAD ≈ 0.6745 for normal distribution)
        std_approx = mad / 0.6745
        lower = max(0, median - 3 * std_approx)  # prices can't be negative
        upper = median + 3 * std_approx
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


class DataSplitter:
    """
    Intelligent dataset splitting with time-aware and random splitting options.

    Supports proper data leakage prevention for time series data and
    provides balanced splits for classification tasks.
    """

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        time_aware: bool = True,
        date_column: str | None = None,
        stratify_column: str | None = None,
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = 1.0 - train_ratio - validation_ratio
        self.time_aware = time_aware
        self.date_column = date_column
        self.stratify_column = stratify_column

        # Validation
        if self.test_ratio < 0:
            raise ValueError("train_ratio + validation_ratio cannot exceed 1.0")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split(self) -> dict[str, Path]:
        """
        Split the dataset into train/validation/test sets.

        Returns:
            Dictionary mapping split names to file paths
        """
        logger.info(f"Loading dataset from {self.input_file}")

        # Load data
        if self.input_file.suffix.lower() == ".csv":
            df = pd.read_csv(self.input_file)
        elif self.input_file.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(self.input_file)
        else:
            raise ValueError(f"Unsupported file format: {self.input_file.suffix}")

        logger.info(f"Dataset shape: {df.shape}")

        # Determine date column if not specified
        if self.time_aware and self.date_column is None:
            self.date_column = self._detect_date_column(df)

        # Perform splitting
        splits = self._time_aware_split(df) if self.time_aware and self.date_column else self._random_split(df)

        # Save splits
        split_paths = {}
        for split_name, split_df in splits.items():
            output_file = self.output_dir / f"{split_name}.{self.input_file.suffix.lstrip('.')}"

            if self.input_file.suffix.lower() == ".csv":
                split_df.to_csv(output_file, index=False)
            else:
                split_df.to_parquet(output_file, index=False)

            split_paths[split_name] = output_file
            logger.info(f"Saved {split_name} split: {split_df.shape} -> {output_file}")

        return split_paths

    def _detect_date_column(self, df: pd.DataFrame) -> str | None:
        """Detect the date column in the dataframe."""
        date_candidates = ["date", "timestamp", "time", "datetime", "Date", "Timestamp"]

        for col in date_candidates:
            if col in df.columns:
                return col

        # Check for datetime-like columns
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "date" in col.lower():
                return col

        logger.warning("No date column detected. Using index for time-aware splitting.")
        return None

    def _time_aware_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data in chronological order to prevent data leakage."""
        if self.date_column and self.date_column in df.columns:
            # Sort by date column
            df_sorted = df.sort_values(self.date_column).reset_index(drop=True)
        else:
            # Assume index is chronological
            df_sorted = df.reset_index(drop=True)

        n_samples = len(df_sorted)

        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))

        splits = {
            "train": df_sorted.iloc[:train_end].copy(),
            "validation": df_sorted.iloc[train_end:val_end].copy(),
            "test": df_sorted.iloc[val_end:].copy()
        }

        # Remove empty splits
        splits = {k: v for k, v in splits.items() if len(v) > 0}

        return splits

    def _random_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split data randomly with optional stratification."""
        from sklearn.model_selection import train_test_split

        # First split: train vs (val + test)
        if self.stratify_column and self.stratify_column in df.columns:
            stratify_data = df[self.stratify_column]
        else:
            stratify_data = None

        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - self.train_ratio),
            random_state=42,
            stratify=stratify_data
        )

        # Second split: validation vs test
        if self.validation_ratio > 0:
            val_test_ratio = self.validation_ratio / (self.validation_ratio + self.test_ratio)

            stratify_temp = temp_df[self.stratify_column] if stratify_data is not None else None

            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_test_ratio),
                random_state=42,
                stratify=stratify_temp
            )

            return {
                "train": train_df,
                "validation": val_df,
                "test": test_df
            }
        else:
            return {
                "train": train_df,
                "test": temp_df
            }


class DataValidator:
    """
    Data quality validation utilities.
    """

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform comprehensive data quality validation.

        Args:
            df: Dataframe to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        }

        # Check for infinite values in numeric columns
        numeric_cols = validation_results["numeric_columns"]
        if numeric_cols:
            infinite_values = {}
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    infinite_values[col] = inf_count
            validation_results["infinite_values"] = infinite_values

        # Data quality score (0-100)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_cells = validation_results["duplicate_rows"] * df.shape[1]
        infinite_cells = sum(validation_results.get("infinite_values", {}).values())

        quality_score = max(0, 100 - (
            (missing_cells + duplicate_cells + infinite_cells) / total_cells * 100
        ))
        validation_results["quality_score"] = round(quality_score, 2)

        return validation_results
