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


class DataManager:
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

        return pd.DataFrame(data)

    def create_test_dataset(
        self, file_path: Optional[str] = None, cleanup_on_exit: bool = True
    ) -> str:
        """Create a test dataset, either from existing data or synthetic."""
        # First try to find existing datasets
        available_datasets = self.find_available_datasets()

        if available_datasets:
            logger.info(f"Using existing dataset: {available_datasets[0]}")
            return available_datasets[0]

        # Generate synthetic data if no existing data found
        logger.info("No existing datasets found, generating synthetic data")

        if file_path is None:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="test_trading_data_"
            )
            file_path = temp_file.name
            temp_file.close()

        # Generate and save synthetic data
        df = self.generate_synthetic_dataset()
        df.to_csv(file_path, index=False)

        # Track for cleanup
        if cleanup_on_exit:
            self.temp_files.append(Path(file_path))

        logger.info(f"Created synthetic test dataset: {file_path}")
        return file_path

    def cleanup(self):
        """Clean up any temporary files created during testing."""
        for temp_file in self.temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_file}: {e}")

        self.temp_files.clear()


@contextmanager
def managed_test_data(data_dir: str = "data", **kwargs):
    """Context manager for automatic test data management and cleanup."""
    manager = DataManager(data_dir)
    try:
        dataset_path = manager.create_test_dataset(**kwargs)
        yield dataset_path
    finally:
        manager.cleanup()


def get_dynamic_test_config(
    base_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Get a test configuration with dynamically discovered or generated test data."""
    manager = DataManager()

    # Create test dataset
    dataset_path = manager.create_test_dataset()

    # Default configuration
    config = {
        "dataset_paths": [dataset_path],
        "window_size": 10,
        "initial_balance": 10000,
        "transaction_cost": 0.001,
        "_test_manager": manager,  # Store manager for cleanup
    }

    # Override with base config if provided
    if base_config:
        config.update(base_config)

    return config
