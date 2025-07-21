#!/usr/bin/env python3
"""
Test Data Management Script

This script provides robust test data management for consistent test execution:

1. Generates synthetic test data with fixed seeds
2. Manages test data lifecycle (create, validate, cleanup)
3. Ensures test data isolation between test runs
4. Provides data validation and integrity checks
5. Supports multiple data formats and sources
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestDataManager:
    """Manages test data with isolation and reproducibility."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path("test_data")
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.dirs = {
            "synthetic": self.base_dir / "synthetic",
            "fixtures": self.base_dir / "fixtures",
            "temp": self.base_dir / "temp",
            "cache": self.base_dir / "cache",
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

        # Track created files for cleanup
        self.created_files: list[Path] = []
        self.temp_dirs: list[Path] = []

    def generate_synthetic_market_data(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-31",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic market data with fixed seed for reproducibility."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        # Set fixed seed for reproducibility
        np.random.seed(seed)

        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=None)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=None)
        dates = pd.date_range(start, end, freq="D")

        data = []
        base_prices = {
            "AAPL": 150,
            "GOOGL": 2800,
            "MSFT": 300,
            "TSLA": 250,
            "AMZN": 3500,
        }

        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            price = base_price

            for date in dates:
                # Generate realistic OHLCV data
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                price = max(0.01, price * (1 + daily_return))

                # Generate OHLC from close price
                volatility = np.random.uniform(0.01, 0.03)
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = price * (1 + np.random.normal(0, 0.005))

                # Generate volume
                volume = int(np.random.uniform(1000000, 10000000))

                # Technical indicators
                data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": round(open_price, 2),
                        "high": round(high, 2),
                        "low": round(low, 2),
                        "close": round(price, 2),
                        "volume": volume,
                        "adj_close": round(price, 2),
                        "returns": daily_return,
                        "volatility": volatility,
                        "rsi": np.random.uniform(20, 80),
                        "macd": np.random.normal(0, 1),
                        "bollinger_upper": price * 1.02,
                        "bollinger_lower": price * 0.98,
                    }
                )

        return pd.DataFrame(data)

    def generate_finrl_format_data(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-31",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data in FinRL format for trading environment tests."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]

        np.random.seed(seed)

        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=None)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=None)
        dates = pd.date_range(start, end, freq="D")

        data = []
        base_prices = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300}

        for symbol in symbols:
            base_price = base_prices[symbol]
            price = base_price

            for date in dates:
                change = np.random.normal(0, 0.02) * price
                price = max(1, price + change)

                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = price + np.random.normal(0, 0.005) * price
                volume = int(np.random.uniform(1000000, 5000000))

                data.append(
                    {
                        "date": date,
                        "tic": symbol,
                        "open": round(open_price, 2),
                        "high": round(high, 2),
                        "low": round(low, 2),
                        "close": round(price, 2),
                        "volume": volume,
                        "macd": np.random.normal(0, 1),
                        "rsi_30": np.random.uniform(20, 80),
                        "cci_30": np.random.normal(0, 100),
                        "dx_30": np.random.uniform(10, 40),
                    }
                )

        return pd.DataFrame(data)

    def create_test_dataset(self, name: str, data_type: str = "market", **kwargs) -> Path:
        """Create a named test dataset and save to file."""
        if data_type == "market":
            df = self.generate_synthetic_market_data(**kwargs)
        elif data_type == "finrl":
            df = self.generate_finrl_format_data(**kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Save to file
        file_path = self.dirs["synthetic"] / f"{name}.csv"
        df.to_csv(file_path, index=False)
        self.created_files.append(file_path)

        logger.info(f"Created test dataset: {file_path} ({len(df)} rows)")
        return file_path

    def create_temp_dataset(self, **kwargs) -> Path:
        """Create a temporary dataset for single test use."""
        df = self.generate_synthetic_market_data(**kwargs)

        # Create in temp directory
        file_path = self.dirs["temp"] / f"temp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_path, index=False)
        self.created_files.append(file_path)

        return file_path

    def validate_dataset(self, file_path: Path) -> dict[str, Any]:
        """Validate a dataset for integrity and completeness."""
        results = {"valid": True, "issues": [], "warnings": [], "stats": {}}

        try:
            df = pd.read_csv(file_path)

            # Basic validation
            if len(df) == 0:
                results["valid"] = False
                results["issues"].append("Dataset is empty")

            # Check required columns
            required_columns = [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                results["warnings"].append(f"Missing columns: {missing_columns}")

            # Check data types
            if "date" in df.columns:
                try:
                    pd.to_datetime(df["date"])
                except Exception:
                    results["issues"].append("Invalid date format")

            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                results["warnings"].append(f"Missing values found: {missing_counts.to_dict()}")

            # Check price consistency
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                price_issues = df[
                    (df["high"] < df["low"])
                    | (df["open"] > df["high"])
                    | (df["open"] < df["low"])
                    | (df["close"] > df["high"])
                    | (df["close"] < df["low"])
                ]
                if len(price_issues) > 0:
                    results["issues"].append(f"Price consistency issues: {len(price_issues)} rows")

            # Generate statistics
            results["stats"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
                "date_range": (f"{df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "N/A"),
            }

        except Exception as e:
            results["valid"] = False
            results["issues"].append(f"Error reading dataset: {e!s}")

        return results

    def cleanup(self):
        """Clean up all created test files and directories."""
        logger.info("Cleaning up test data...")

        # Remove created files
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed file: {file_path}")

        # Remove temp directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Removed temp directory: {temp_dir}")

        # Clear lists
        self.created_files.clear()
        self.temp_dirs.clear()

        logger.info("Test data cleanup complete")

    def get_test_datasets(self) -> dict[str, Path]:
        """Get all available test datasets."""
        datasets = {}

        # Synthetic datasets
        for file_path in self.dirs["synthetic"].glob("*.csv"):
            datasets[file_path.stem] = file_path

        # Fixture datasets
        for file_path in self.dirs["fixtures"].glob("*.csv"):
            datasets[f"fixture_{file_path.stem}"] = file_path

        return datasets

    def create_test_environment(self) -> dict[str, Any]:
        """Create a complete test environment with all necessary data."""
        logger.info("Creating complete test environment...")

        env_data = {
            "market_data": self.create_test_dataset("market_data_2024", "market"),
            "finrl_data": self.create_test_dataset("finrl_data_2024", "finrl"),
            "small_dataset": self.create_test_dataset(
                "small_dataset",
                "market",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-10",
            ),
            "large_dataset": self.create_test_dataset(
                "large_dataset",
                "market",
                symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"],
                start_date="2023-01-01",
                end_date="2024-12-31",
            ),
        }

        # Validate all datasets
        validation_results = {}
        for name, file_path in env_data.items():
            validation_results[name] = self.validate_dataset(file_path)

        return {
            "datasets": env_data,
            "validation": validation_results,
            "base_dir": str(self.base_dir),
        }


def main():
    """Main entry point for test data management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage test data for Trading RL Agent")
    parser.add_argument(
        "--action",
        choices=["create", "validate", "cleanup", "list"],
        default="create",
        help="Action to perform",
    )
    parser.add_argument("--base-dir", help="Base directory for test data")
    parser.add_argument("--dataset-name", help="Name for the dataset")
    parser.add_argument(
        "--data-type",
        choices=["market", "finrl"],
        default="market",
        help="Type of data to generate",
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to include")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date")
    parser.add_argument("--end-date", default="2024-01-31", help="End date")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    manager = TestDataManager(args.base_dir)

    try:
        if args.action == "create":
            if args.dataset_name:
                file_path = manager.create_test_dataset(
                    args.dataset_name,
                    args.data_type,
                    symbols=args.symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    seed=args.seed,
                )
                print(f"Created dataset: {file_path}")
            else:
                env = manager.create_test_environment()
                print(f"Created test environment in: {env['base_dir']}")
                print("Datasets created:")
                for name, path in env["datasets"].items():
                    print(f"  {name}: {path}")

        elif args.action == "validate":
            if args.dataset_name:
                file_path = manager.dirs["synthetic"] / f"{args.dataset_name}.csv"
                if file_path.exists():
                    results = manager.validate_dataset(file_path)
                    print(f"Validation results for {file_path}:")
                    print(json.dumps(results, indent=2))
                else:
                    print(f"Dataset not found: {file_path}")
            else:
                datasets = manager.get_test_datasets()
                for name, path in datasets.items():
                    results = manager.validate_dataset(path)
                    print(f"{name}: {'✓' if results['valid'] else '✗'}")

        elif args.action == "list":
            datasets = manager.get_test_datasets()
            print("Available test datasets:")
            for name, path in datasets.items():
                print(f"  {name}: {path}")

        elif args.action == "cleanup":
            manager.cleanup()
            print("Test data cleanup complete")

    finally:
        if args.action != "cleanup":
            # Don't cleanup if cleanup was the requested action
            pass


if __name__ == "__main__":
    main()
