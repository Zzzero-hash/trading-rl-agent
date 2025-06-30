#!/usr/bin/env python3
"""
Production-ready advanced dataset builder for trading RL agent.

This script creates a comprehensive dataset that combines:
1. Real market data from yFinance (stocks, forex, crypto)
2. Synthetic data using advanced mathematical models
3. State-of-the-art feature engineering
4. Sentiment analysis integration
5. Production-ready format for live data compatibility

Compatible with live data integration and RL training pipeline.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf

from src.data.features import generate_features
from src.data.sentiment import get_sentiment_score
from src.data.synthetic import fetch_synthetic_data, generate_gbm_prices

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class AdvancedDatasetBuilder:
    """Production-ready advanced dataset builder for trading RL systems."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize the dataset builder with configuration."""
        self.config = config or self._get_default_config()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def _get_default_config(self) -> dict:
        """Get default configuration for dataset building."""
        return {
            "symbols": {
                "stocks": [
                    "AAPL",
                    "GOOGL",
                    "MSFT",
                    "AMZN",
                    "TSLA",
                    "META",
                    "NVDA",
                    "JPM",
                    "BAC",
                    "XOM",
                ],
                "forex": [
                    "EURUSD=X",
                    "GBPUSD=X",
                    "USDJPY=X",
                    "USDCHF=X",
                    "USDCAD=X",
                    "AUDUSD=X",
                    "NZDUSD=X",
                ],
                "crypto": ["BTC-USD", "ETH-USD"],
            },
            "date_range": {
                "start": "2020-01-01",
                "end": datetime.now().strftime("%Y-%m-%d"),
            },
            "feature_engineering": {
                "ma_windows": [5, 10, 20, 50],
                "rsi_window": 14,
                "vol_window": 20,
                "advanced_candles": True,
            },
            "synthetic_data": {
                "enabled": True,
                "n_days": 1000,
                "n_symbols": 5,
                "mu": 0.0002,
                "sigma": 0.01,
            },
            "targets": {"forward_periods": 5, "profit_threshold": 0.02},
            "output": {
                "sample_data_path": "data/sample_data.csv",
                "advanced_dataset_path": None,  # Will be generated with timestamp
                "metadata_path": None,  # Will be generated with timestamp
            },
        }

    def download_real_market_data(self) -> list[pd.DataFrame]:
        """Download real market data from various sources."""
        print("📥 Downloading real market data...")

        datasets = []
        all_symbols = []

        # Combine all symbols
        for category, symbols in self.config["symbols"].items():
            all_symbols.extend([(symbol, category) for symbol in symbols])

        with tqdm(total=len(all_symbols), desc="Market data") as pbar:
            for symbol, category in all_symbols:
                try:
                    # Download data
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=self.config["date_range"]["start"],
                        end=self.config["date_range"]["end"],
                    )

                    if df.empty:
                        pbar.set_postfix({"status": f"❌ {symbol}: No data"})
                        pbar.update(1)
                        continue

                    # Standardize column names
                    df.columns = [col.lower() for col in df.columns]
                    df = df[["open", "high", "low", "close", "volume"]].copy()

                    # Add metadata
                    df["timestamp"] = (
                        df.index.tz_localize(None) if df.index.tz else df.index
                    )
                    df["symbol"] = symbol.replace("=X", "").replace("-USD", "USD")
                    df["source"] = "real"
                    df["asset_class"] = category

                    df.reset_index(drop=True, inplace=True)
                    datasets.append(df)

                    pbar.set_postfix({"status": f"✅ {symbol}: {len(df)} rows"})

                except Exception as e:
                    pbar.set_postfix({"status": f"❌ {symbol}: {str(e)[:20]}"})

                pbar.update(1)

        print(f"✅ Downloaded data for {len(datasets)} symbols")
        return datasets

    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data using advanced models."""
        if not self.config["synthetic_data"]["enabled"]:
            return pd.DataFrame()

        print("🔧 Generating synthetic market data...")

        synthetic_datasets = []

        for i in range(self.config["synthetic_data"]["n_symbols"]):
            # Generate synthetic OHLCV data
            synthetic_df = generate_gbm_prices(
                n_days=self.config["synthetic_data"]["n_days"],
                mu=self.config["synthetic_data"]["mu"],
                sigma=self.config["synthetic_data"]["sigma"],
                s0=np.random.uniform(50, 200),
            )

            # Add metadata
            synthetic_df["symbol"] = f"SYN_{i+1:03d}"
            synthetic_df["source"] = "synthetic"
            synthetic_df["asset_class"] = "synthetic"

            synthetic_datasets.append(synthetic_df)

        combined_synthetic = pd.concat(synthetic_datasets, ignore_index=True)
        print(f"✅ Generated {len(combined_synthetic):,} synthetic data points")
        return combined_synthetic

    def apply_feature_engineering(self, datasets: list[pd.DataFrame]) -> pd.DataFrame:
        """Apply comprehensive feature engineering to all datasets."""
        print("🔧 Applying advanced feature engineering...")

        enhanced_datasets = []

        with tqdm(total=len(datasets), desc="Feature engineering") as pbar:
            for dataset in datasets:
                if dataset.empty:
                    pbar.update(1)
                    continue

                symbol = (
                    dataset["symbol"].iloc[0]
                    if "symbol" in dataset.columns
                    else "UNKNOWN"
                )

                try:
                    # Apply feature engineering using existing infrastructure
                    enhanced_data = generate_features(
                        dataset.copy(), **self.config["feature_engineering"]
                    )

                    # Add sentiment analysis for real symbols
                    if dataset["source"].iloc[0] == "real":
                        try:
                            sentiment_score = get_sentiment_score(symbol, days_back=1)
                            enhanced_data["sentiment"] = sentiment_score
                            enhanced_data["sentiment_magnitude"] = abs(sentiment_score)
                        except Exception:
                            enhanced_data["sentiment"] = 0.0
                            enhanced_data["sentiment_magnitude"] = 0.0
                    else:
                        enhanced_data["sentiment"] = 0.0
                        enhanced_data["sentiment_magnitude"] = 0.0

                    # Add time-based features
                    if "timestamp" in enhanced_data.columns:
                        ts = pd.to_datetime(enhanced_data["timestamp"])
                        enhanced_data["hour"] = ts.dt.hour
                        enhanced_data["day_of_week"] = ts.dt.dayofweek
                        enhanced_data["month"] = ts.dt.month
                        enhanced_data["quarter"] = ts.dt.quarter

                    # Add advanced price features
                    enhanced_data["price_change_pct"] = (
                        (enhanced_data["close"] - enhanced_data["open"])
                        / enhanced_data["open"]
                    ) * 100
                    enhanced_data["high_low_pct"] = (
                        (enhanced_data["high"] - enhanced_data["low"])
                        / enhanced_data["low"]
                    ) * 100
                    enhanced_data["body_size"] = abs(
                        enhanced_data["close"] - enhanced_data["open"]
                    )
                    enhanced_data["upper_shadow"] = enhanced_data["high"] - np.maximum(
                        enhanced_data["open"], enhanced_data["close"]
                    )
                    enhanced_data["lower_shadow"] = (
                        np.minimum(enhanced_data["open"], enhanced_data["close"])
                        - enhanced_data["low"]
                    )

                    # Add volume features
                    enhanced_data["volume_ma_20"] = (
                        enhanced_data["volume"].rolling(20).mean()
                    )
                    enhanced_data["volume_ratio"] = (
                        enhanced_data["volume"] / enhanced_data["volume_ma_20"]
                    )
                    enhanced_data["volume_change"] = enhanced_data[
                        "volume"
                    ].pct_change()

                    enhanced_datasets.append(enhanced_data)
                    pbar.set_postfix(
                        {"symbol": symbol, "features": len(enhanced_data.columns)}
                    )

                except Exception as e:
                    pbar.set_postfix({"symbol": symbol, "error": str(e)[:20]})

                pbar.update(1)

        if enhanced_datasets:
            combined_enhanced = pd.concat(enhanced_datasets, ignore_index=True)
            print(f"✅ Feature engineering complete: {combined_enhanced.shape}")
            return combined_enhanced
        else:
            print("❌ No datasets were successfully enhanced")
            return pd.DataFrame()

    def create_trading_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trading targets for RL training."""
        print("🎯 Creating trading targets...")

        def add_targets_by_symbol(symbol_data):
            """Add targets for a single symbol's data."""
            targets = []

            for i in range(len(symbol_data)):
                if i + self.config["targets"]["forward_periods"] >= len(symbol_data):
                    targets.append(0)  # Hold for last few rows
                    continue

                current_price = symbol_data.iloc[i]["close"]
                future_prices = symbol_data.iloc[
                    i + 1 : i + self.config["targets"]["forward_periods"] + 1
                ]["close"]

                if future_prices.empty:
                    targets.append(0)
                    continue

                max_future_price = future_prices.max()
                min_future_price = future_prices.min()

                # Calculate potential profit/loss
                buy_profit = (max_future_price - current_price) / current_price
                sell_profit = (current_price - min_future_price) / current_price

                # Determine optimal action
                threshold = self.config["targets"]["profit_threshold"]
                if buy_profit > threshold and buy_profit > sell_profit:
                    targets.append(1)  # Buy
                elif sell_profit > threshold and sell_profit > buy_profit:
                    targets.append(2)  # Sell
                else:
                    targets.append(0)  # Hold

            return targets

        # Add targets by symbol to maintain data integrity
        final_datasets = []
        for symbol in df["symbol"].unique():
            symbol_data = df[df["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp").reset_index(drop=True)
            symbol_data["target"] = add_targets_by_symbol(symbol_data)
            final_datasets.append(symbol_data)

        result = pd.concat(final_datasets, ignore_index=True)

        # Target distribution
        target_counts = result["target"].value_counts().sort_index()
        target_labels = {0: "Hold", 1: "Buy", 2: "Sell"}
        print("Target distribution:")
        for target_val, count in target_counts.items():
            percentage = count / len(result) * 100
            target_int = int(target_val) if isinstance(target_val, (int, float)) else 0
            label = target_labels.get(target_int, f"Unknown_{target_val}")
            print(f"  {label}: {count:,} ({percentage:.1f}%)")

        return result

    def create_training_compatible_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a dataset compatible with TraderEnv (numeric columns only)."""
        print("🔄 Creating training-compatible dataset...")

        # Select only numeric columns for training
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove metadata columns but keep essential ones
        exclude_cols = ["target"]  # We'll add target back at the end
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Create training dataset
        training_data = df[feature_cols + ["target"]].copy()

        # Handle any remaining NaN values
        training_data = training_data.fillna(training_data.median())

        print(f"Training dataset shape: {training_data.shape}")
        print(f"Features: {len(feature_cols)}")

        return training_data

    def save_datasets(
        self, df: pd.DataFrame, training_df: pd.DataFrame
    ) -> dict[str, str]:
        """Save datasets and metadata."""
        print("💾 Saving datasets...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Update config paths
        if not self.config["output"]["advanced_dataset_path"]:
            self.config["output"][
                "advanced_dataset_path"
            ] = f"data/advanced_trading_dataset_{timestamp}.csv"
        if not self.config["output"]["metadata_path"]:
            self.config["output"][
                "metadata_path"
            ] = f"data/dataset_metadata_{timestamp}.json"

        # Save training-compatible dataset as sample_data.csv
        training_df.to_csv(self.config["output"]["sample_data_path"], index=False)

        # Save full advanced dataset
        df.to_csv(self.config["output"]["advanced_dataset_path"], index=False)

        # Create metadata
        metadata = {
            "dataset_version": timestamp,
            "total_records": len(df),
            "training_records": len(training_df),
            "features": len(df.columns),
            "training_features": len(training_df.columns) - 1,  # Exclude target
            "symbols": list(df["symbol"].unique()) if "symbol" in df.columns else [],
            "sources": list(df["source"].unique()) if "source" in df.columns else [],
            "date_range": {
                "start": (
                    str(df["timestamp"].min()) if "timestamp" in df.columns else None
                ),
                "end": (
                    str(df["timestamp"].max()) if "timestamp" in df.columns else None
                ),
            },
            "target_distribution": (
                df["target"].value_counts().to_dict() if "target" in df.columns else {}
            ),
            "data_completeness": float(
                100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            ),
            "config": self.config,
            "compatible_with_live_data": True,
            "feature_engineering_pipeline": "src.data.features.generate_features",
            "files": {
                "training_data": self.config["output"]["sample_data_path"],
                "full_dataset": self.config["output"]["advanced_dataset_path"],
                "metadata": self.config["output"]["metadata_path"],
            },
        }

        # Save metadata
        with open(self.config["output"]["metadata_path"], "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        paths = {
            "training_data": self.config["output"]["sample_data_path"],
            "full_dataset": self.config["output"]["advanced_dataset_path"],
            "metadata": self.config["output"]["metadata_path"],
        }

        print("✅ Datasets saved:")
        for name, path in paths.items():
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name}: {path} ({size:.1f} MB)")

        return paths

    def build_dataset(self) -> dict[str, str]:
        """Build the complete advanced dataset."""
        print("🚀 Building advanced trading dataset...")
        print("=" * 60)

        # Download real market data
        real_datasets = self.download_real_market_data()

        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data()

        # Combine real and synthetic data
        all_datasets = real_datasets.copy()
        if not synthetic_data.empty:
            all_datasets.append(synthetic_data)

        if not all_datasets:
            raise ValueError("No data available for processing")

        # Apply feature engineering
        enhanced_dataset = self.apply_feature_engineering(all_datasets)

        if enhanced_dataset.empty:
            raise ValueError("Feature engineering failed")

        # Create trading targets
        final_dataset = self.create_trading_targets(enhanced_dataset)

        # Create training-compatible version
        training_dataset = self.create_training_compatible_dataset(final_dataset)

        # Save datasets
        file_paths = self.save_datasets(final_dataset, training_dataset)

        print("=" * 60)
        print("🎉 Advanced dataset building complete!")
        print(
            f"📊 Final dataset: {final_dataset.shape[0]:,} records, {final_dataset.shape[1]} features"
        )
        print(
            f"🎯 Training dataset: {training_dataset.shape[0]:,} records, {training_dataset.shape[1]-1} features"
        )
        print("✅ Ready for RL training and live data integration")

        return file_paths


def main():
    """Main function to build the advanced dataset."""
    try:
        builder = AdvancedDatasetBuilder()
        file_paths = builder.build_dataset()

        # Test compatibility with training environment
        print("\n🧪 Testing compatibility with training environment...")
        try:
            from src.envs.trader_env import TraderEnv

            env = TraderEnv(
                [file_paths["training_data"]], window_size=10, initial_balance=10000
            )
            obs, info = env.reset()
            print("✅ Training environment test passed!")
            print(f"   Observation shape: {obs.shape}")
            print(f"   Action space: {env.action_space}")
        except Exception as e:
            print(f"❌ Training environment test failed: {e}")

        return file_paths

    except Exception as e:
        print(f"❌ Dataset building failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
