"""
Test data utilities for handling dynamic dataset discovery and generation.

Provides elegant solutions for test data management:
1. Dynamic discovery of available datasets
2. Synthetic test data generation
3. Automatic cleanup after tests
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TestDataManager:
    """Manages test data with dynamic discovery and synthetic generation."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.temp_files: list[Path] = []

    def find_available_datasets(
        self, pattern: str = "*training_data*.csv"
    ) -> list[str]:
        """Dynamically find available dataset files."""
        available_files = []

        if self.data_dir.exists():
            # Look for existing data files
            for file_path in self.data_dir.glob(pattern):
                if file_path.is_file() and file_path.stat().st_size > 0:
                    available_files.append(str(file_path))

        # Also check for any CSV files in data directory
        if not available_files and self.data_dir.exists():
            for file_path in self.data_dir.glob("*.csv"):
                if file_path.is_file() and file_path.stat().st_size > 0:
                    available_files.append(str(file_path))

        return available_files

    def generate_synthetic_dataset(
        self, n_days: int = 100, start_price: float = 100.0, volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate synthetic trading data for testing."""
        np.random.seed(42)  # Reproducible for tests

        # Generate dates
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        # Generate price data with random walk
        returns = np.random.normal(0, volatility, n_days)
        prices = [start_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create realistic OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility_factor = np.random.uniform(0.98, 1.02)
            high = close_price * volatility_factor
            low = close_price * np.random.uniform(0.98, 1.0)
            open_price = prices[i - 1] if i > 0 else close_price

            # Generate volume
            volume = np.random.randint(1000000, 10000000)

            data.append(
                {
                    "timestamp": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)

        # Ensure required columns are present
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                if col == "timestamp":
                    df[col] = dates
                else:
                    df[col] = df["close"] if "close" in df.columns else 100.0

        return df

    def get_or_create_test_dataset(
        self, required_columns: Optional[list[str]] = None
    ) -> str:
        """Get existing dataset or create synthetic one for testing."""

        # First, try to find existing datasets
        available_files = self.find_available_datasets()

        if available_files:
            # Use the first available file
            dataset_path = available_files[0]
            logger.info(f"Using existing dataset: {dataset_path}")

            # Validate the dataset has required columns
            if required_columns:
                try:
                    df = pd.read_csv(dataset_path)
                    missing_cols = set(required_columns) - set(df.columns)
                    if missing_cols:
                        logger.warning(
                            f"Dataset missing columns {missing_cols}, generating synthetic data"
                        )
                        return self._create_temp_dataset(required_columns)
                except Exception as e:
                    logger.warning(
                        f"Error reading {dataset_path}: {e}, generating synthetic data"
                    )
                    return self._create_temp_dataset(required_columns)

            return dataset_path
        else:
            # Generate synthetic dataset
            logger.info("No existing datasets found, generating synthetic test data")
            return self._create_temp_dataset(required_columns)

    def _create_temp_dataset(self, required_columns: Optional[list[str]] = None) -> str:
        """Create temporary synthetic dataset."""
        df = self.generate_synthetic_dataset()

        # Ensure required columns exist
        if required_columns:
            for col in required_columns:
                if col not in df.columns:
                    # Add missing columns with default values
                    if "price" in col.lower() or col in [
                        "open",
                        "high",
                        "low",
                        "close",
                    ]:
                        df[col] = df["close"] if "close" in df.columns else 100.0
                    elif "volume" in col.lower():
                        df[col] = 1000000
                    elif "timestamp" in col.lower() or "date" in col.lower():
                        df[col] = pd.date_range(
                            start="2023-01-01", periods=len(df), freq="D"
                        )
                    else:
                        df[col] = 0.0

        # Create temporary file
        self.data_dir.mkdir(exist_ok=True)
        temp_file = self.data_dir / f"test_data_{np.random.randint(10000, 99999)}.csv"
        df.to_csv(temp_file, index=False)

        self.temp_files.append(temp_file)
        logger.info(f"Created temporary test dataset: {temp_file}")

        return str(temp_file)

    def cleanup(self):
        """Clean up temporary test files."""
        for temp_file in self.temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_file}: {e}")
        self.temp_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@contextmanager
def test_dataset_context(
    data_dir: str = "data", required_columns: Optional[list[str]] = None
):
    """Context manager for test dataset with automatic cleanup."""
    manager = TestDataManager(data_dir)
    try:
        dataset_path = manager.get_or_create_test_dataset(required_columns)
        yield dataset_path
    finally:
        manager.cleanup()


def get_dynamic_test_config(
    base_config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Get test configuration with dynamically discovered or generated dataset."""

    manager = TestDataManager()
    # Required columns for trading environment
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    dataset_path = manager.get_or_create_test_dataset(required_columns)

    config = {
        "dataset_paths": [dataset_path],
        "window_size": 10,
        "initial_balance": 10000,
        "transaction_cost": 0.001,
        "include_features": False,
        "continuous_actions": True,
        "_test_manager": manager,  # Store manager for cleanup
    }

    if base_config:
        config.update(base_config)

    return config


# Legacy functions for backward compatibility
def generate_synthetic_test_data(path=None, days=30, num_assets=3):
    """Legacy function - maintained for backward compatibility."""
    if path is None:
        path = Path("data/synthetic_test_data.csv")
    else:
        path = Path(path)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    data = {
        "timestamp": [d.strftime("%Y-%m-%d") for d in dates for _ in range(num_assets)],
        "symbol": [f"SYM{i}" for i in range(num_assets)] * days,
        "open": np.random.uniform(100, 200, days * num_assets),
        "high": np.random.uniform(100, 200, days * num_assets),
        "low": np.random.uniform(90, 199, days * num_assets),
        "close": np.random.uniform(100, 200, days * num_assets),
        "volume": np.random.randint(1000, 10000, days * num_assets),
        "label": np.random.choice([0, 1, 2], days * num_assets),
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def cleanup_synthetic_test_data(path="data/synthetic_test_data.csv"):
    """Legacy function - maintained for backward compatibility."""
    p = Path(path)
    if p.exists():
        p.unlink()
