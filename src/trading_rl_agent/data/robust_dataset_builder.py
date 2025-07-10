"""
Robust Dataset Builder for CNN+LSTM Training

This module provides a comprehensive, reproducible, and real-time replicatable
dataset generation pipeline for training CNN+LSTM models in trading environments.

Features:
- Multi-source data integration (real-time APIs + synthetic)
- Advanced feature engineering with temporal patterns
- CNN+LSTM optimized sequence generation
- Reproducible pipeline with versioning
- Real-time streaming compatibility
- Data quality validation and monitoring
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from trading_rl_agent.data.features import generate_features
from trading_rl_agent.data.synthetic import fetch_synthetic_data

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for robust dataset generation."""

    # Data sources
    symbols: list[str]
    start_date: str
    end_date: str
    timeframe: str = "1d"  # 1d, 1h, 5m, etc.

    # Dataset composition
    real_data_ratio: float = 0.7  # 70% real, 30% synthetic
    min_samples_per_symbol: int = 1000

    # CNN+LSTM specific
    sequence_length: int = 60  # Lookback window
    prediction_horizon: int = 1  # Steps ahead to predict
    overlap_ratio: float = 0.8  # Overlap between sequences

    # Feature engineering
    technical_indicators: bool = True
    sentiment_features: bool = True
    market_regime_features: bool = True

    # Data quality
    outlier_threshold: float = 3.0  # Standard deviations
    missing_value_threshold: float = 0.05  # Max 5% missing

    # Output
    output_dir: str = "data/robust_dataset"
    version_tag: str | None = None
    save_metadata: bool = True


class RobustDatasetBuilder:
    """Main class for building robust, reproducible datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.version = config.version_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / self.version
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = RobustScaler()
        self.feature_columns = []
        self.metadata = {}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def build_dataset(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Main method to build the complete dataset."""

        logger.info(f"🚀 Starting robust dataset generation (version: {self.version})")

        # Step 1: Collect raw data
        raw_data = self._collect_raw_data()

        # Step 2: Validate and clean data
        clean_data = self._validate_and_clean(raw_data)

        # Step 3: Engineer features
        featured_data = self._engineer_features(clean_data)

        # Step 4: Create sequences for CNN+LSTM
        sequences, targets = self._create_sequences(featured_data)

        # Step 5: Scale features
        scaled_sequences = self._scale_features(sequences)

        # Step 6: Validate final dataset
        self._validate_final_dataset(scaled_sequences, targets)

        # Step 7: Save dataset and metadata
        dataset_info = self._save_dataset(scaled_sequences, targets)

        logger.info("✅ Dataset generation completed successfully!")
        return scaled_sequences, targets, dataset_info

    def _collect_raw_data(self) -> pd.DataFrame:
        """Collect data from multiple sources."""

        logger.info("📊 Collecting raw market data...")

        all_data = []
        real_data_count = 0
        synthetic_data_count = 0

        for symbol in self.config.symbols:
            try:
                # Try to fetch real data first
                if self.config.real_data_ratio > 0:
                    real_df = self._fetch_real_data(symbol)
                    if not real_df.empty:
                        real_df["symbol"] = symbol
                        real_df["data_source"] = "real"
                        all_data.append(real_df)
                        real_data_count += len(real_df)
                        logger.info(f"  ✓ {symbol}: {len(real_df)} real samples")

                # Add synthetic data to reach minimum samples
                samples_needed = max(
                    0,
                    self.config.min_samples_per_symbol - len(real_df if "real_df" in locals() else []),
                )

                if samples_needed > 0 or self.config.real_data_ratio < 1.0:
                    synthetic_samples = int(
                        samples_needed / (1 - max(0.1, self.config.real_data_ratio)),
                    )
                    synthetic_df = self._fetch_synthetic_data(symbol, synthetic_samples)
                    synthetic_df["symbol"] = symbol
                    synthetic_df["data_source"] = "synthetic"
                    all_data.append(synthetic_df)
                    synthetic_data_count += len(synthetic_df)
                    logger.info(f"  ✓ {symbol}: {len(synthetic_df)} synthetic samples")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to fetch data for {symbol}: {e}")
                # Fallback to synthetic data only
                synthetic_df = self._fetch_synthetic_data(
                    symbol,
                    self.config.min_samples_per_symbol,
                )
                synthetic_df["symbol"] = symbol
                synthetic_df["data_source"] = "synthetic"
                all_data.append(synthetic_df)
                synthetic_data_count += len(synthetic_df)

        if not all_data:
            raise ValueError("No data could be collected for any symbol")

        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by timestamp for time series consistency
        combined_df = combined_df.sort_values(["symbol", "timestamp"]).reset_index(
            drop=True,
        )

        self.metadata["raw_data"] = {
            "total_samples": len(combined_df),
            "real_samples": real_data_count,
            "synthetic_samples": synthetic_data_count,
            "symbols": self.config.symbols,
            "timeframe": self.config.timeframe,
        }

        logger.info(
            f"📈 Collected {len(combined_df)} total samples ({real_data_count} real, {synthetic_data_count} synthetic)",
        )

        return combined_df

    def _fetch_real_data(self, symbol: str) -> pd.DataFrame:
        """Fetch real market data using yfinance."""

        try:
            ticker = yf.Ticker(symbol)

            # Convert timeframe for yfinance
            interval_map = {
                "1d": "1d",
                "1h": "1h",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
            }
            interval = interval_map.get(self.config.timeframe, "1d")

            df = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval=interval,
            )

            if df.empty:
                return pd.DataFrame()

            # Standardize column names
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
            )

            # Reset index to get timestamp as column
            df = df.reset_index()
            df = df.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})

            # Keep only OHLCV columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            return df[required_cols]

        except Exception as e:
            logger.warning(f"Failed to fetch real data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_synthetic_data(self, symbol: str, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data for a symbol."""

        timeframe_map = {
            "1d": "day",
            "1h": "hour",
            "5m": "minute",
            "15m": "minute",
            "30m": "minute",
        }

        timeframe = timeframe_map.get(self.config.timeframe, "day")

        # Generate with realistic volatility based on symbol type
        volatility = 0.02  # Default
        if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "DOGE"]):
            volatility = 0.05  # Higher for crypto
        elif any(forex in symbol.upper() for forex in ["USD", "EUR", "GBP", "JPY"]):
            volatility = 0.01  # Lower for forex

        return fetch_synthetic_data(
            n_samples=n_samples,
            timeframe=timeframe,
            volatility=volatility,
            symbol=symbol,
        )

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and clean issues."""

        logger.info("🔍 Validating and cleaning data...")

        initial_count = len(df)

        # 1. Remove rows with missing OHLCV data
        required_cols = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=required_cols)

        # 2. Validate OHLC relationships
        valid_ohlc = (
            (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["high"] >= df["low"])
        )
        df = df[valid_ohlc]

        # 3. Remove extreme outliers
        for col in ["open", "high", "low", "close"]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= self.config.outlier_threshold]

        # 4. Ensure positive prices and volumes
        df = df[(df[required_cols] > 0).all(axis=1)]

        # 5. Remove duplicates
        df = df.drop_duplicates(subset=["timestamp", "symbol"])

        # 6. Sort by symbol and timestamp
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        cleaned_count = len(df)
        removal_rate = (initial_count - cleaned_count) / initial_count

        if removal_rate > self.config.missing_value_threshold:
            warnings.warn(f"High data removal rate: {removal_rate:.2%}", stacklevel=2)

        self.metadata["data_cleaning"] = {
            "initial_samples": initial_count,
            "cleaned_samples": cleaned_count,
            "removal_rate": removal_rate,
        }

        logger.info(
            f"  ✓ Cleaned data: {cleaned_count} samples ({removal_rate:.2%} removed)",
        )

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features optimized for CNN+LSTM models."""

        logger.info("🔧 Engineering features for CNN+LSTM...")

        featured_dfs = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].copy()

            # Generate technical features using existing pipeline
            if self.config.technical_indicators:
                symbol_df = generate_features(
                    symbol_df,
                    ma_windows=[5, 10, 20, 50],
                    rsi_window=14,
                    vol_window=20,
                    advanced_candles=True,
                )

            # Add temporal features (important for LSTM)
            symbol_df = self._add_temporal_features(symbol_df)

            # Add market regime features
            if self.config.market_regime_features:
                symbol_df = self._add_market_regime_features(symbol_df)

            # Add volatility regime features
            symbol_df = self._add_volatility_features(symbol_df)

            featured_dfs.append(symbol_df)

        featured_df = pd.concat(featured_dfs, ignore_index=True)

        # Select numeric features only (required for CNN+LSTM)
        numeric_cols = featured_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ["timestamp"]]

        self.feature_columns = feature_cols

        self.metadata["feature_engineering"] = {
            "total_features": len(feature_cols),
            "feature_types": {
                "technical_indicators": sum(
                    1 for col in feature_cols if any(ind in col for ind in ["sma", "ema", "rsi", "macd", "bb"])
                ),
                "temporal_features": sum(
                    1 for col in feature_cols if any(temp in col for temp in ["hour", "day", "month", "quarter"])
                ),
                "volatility_features": sum(1 for col in feature_cols if "vol" in col),
                "returns_features": sum(1 for col in feature_cols if "return" in col),
            },
        }

        logger.info(f"  ✓ Generated {len(feature_cols)} features")

        return featured_df[["timestamp", "symbol", *feature_cols]]

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features that help LSTM understand time patterns."""

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Cyclical time features (better for neural networks)
        df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.month / 12)

        # Quarter and year features
        df["quarter"] = df["timestamp"].dt.quarter
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["timestamp"].dt.is_quarter_end.astype(int)

        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""

        df = df.copy()

        # Trend regime (based on moving averages)
        if "close" in df.columns:
            df["ma_short"] = df["close"].rolling(10).mean()
            df["ma_long"] = df["close"].rolling(50).mean()
            df["trend_regime"] = (df["ma_short"] > df["ma_long"]).astype(int)

            # Volatility regime
            df["volatility_regime"] = (
                df["close"].pct_change().rolling(20).std() > df["close"].pct_change().rolling(100).std()
            ).astype(int)

            # Momentum regime
            df["momentum_regime"] = (df["close"].pct_change(5) > 0).astype(int)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility features."""

        df = df.copy()

        if "close" in df.columns:
            returns = df["close"].pct_change()

            # Realized volatility (multiple windows)
            for window in [5, 10, 20]:
                df[f"realized_vol_{window}"] = returns.rolling(window).std() * np.sqrt(
                    252,
                )

            # Volatility of volatility
            df["vol_of_vol"] = df["realized_vol_20"].rolling(10).std()

            # Intraday volatility (if OHLC available)
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                df["intraday_vol"] = np.log(df["high"] / df["low"])
                df["overnight_vol"] = np.log(df["open"] / df["close"].shift(1))

        return df

    def _create_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences optimized for CNN+LSTM training."""

        logger.info("📊 Creating sequences for CNN+LSTM...")

        sequences = []
        targets = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].copy()

            # Remove non-numeric columns for feature matrix
            feature_cols = [col for col in self.feature_columns if col in symbol_df.columns]
            features = symbol_df[feature_cols].values

            # Create target variable (future return)
            if "close" in symbol_df.columns:
                returns = (
                    symbol_df["close"].pct_change(self.config.prediction_horizon).shift(-self.config.prediction_horizon)
                )
                target_values = returns.values
            else:
                # Fallback target if close is not available
                target_values = np.random.normal(0, 0.01, len(features))

            # Generate sequences with overlap
            step_size = max(
                1,
                int(self.config.sequence_length * (1 - self.config.overlap_ratio)),
            )

            for i in range(
                0,
                len(features) - self.config.sequence_length - self.config.prediction_horizon + 1,
                step_size,
            ):
                # Extract sequence
                seq = features[i : i + self.config.sequence_length]
                target = target_values[i + self.config.sequence_length - 1]

                # Skip if target is NaN
                if not np.isnan(target):
                    sequences.append(seq)
                    targets.append(target)

        sequences_array = np.array(sequences, dtype=np.float32)
        targets_array = np.array(targets, dtype=np.float32)

        self.metadata["sequences"] = {
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "overlap_ratio": self.config.overlap_ratio,
            "total_sequences": len(sequences_array),
            "features_per_timestep": (sequences_array.shape[2] if len(sequences_array.shape) == 3 else 0),
        }

        logger.info(
            f"  ✓ Created {len(sequences_array)} sequences of shape {sequences_array.shape}",
        )

        return sequences_array, targets_array

    def _scale_features(self, sequences: np.ndarray) -> np.ndarray:
        """Scale features using robust scaling."""

        logger.info("📏 Scaling features...")

        # Reshape for scaling (combine all timesteps)
        original_shape = sequences.shape
        reshaped = sequences.reshape(-1, sequences.shape[-1])

        # Fit and transform
        scaled = self.scaler.fit_transform(reshaped)

        # Reshape back to sequences
        scaled_sequences = scaled.reshape(original_shape)

        # Save scaler for real-time use
        scaler_path = self.output_dir / "feature_scaler.pkl"
        with Path(scaler_path).open("wb") as f:
            pickle.dump(self.scaler, f)

        logger.info(f"  ✓ Features scaled and scaler saved to {scaler_path}")

        return scaled_sequences.astype(np.float32)

    def _validate_final_dataset(self, sequences: np.ndarray, targets: np.ndarray):
        """Validate the final dataset quality."""

        logger.info("✅ Validating final dataset...")

        # Check for NaN or infinite values
        if np.any(np.isnan(sequences)) or np.any(np.isinf(sequences)):
            raise ValueError("Dataset contains NaN or infinite values in sequences")

        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            raise ValueError("Dataset contains NaN or infinite values in targets")

        # Check shapes
        if len(sequences) != len(targets):
            raise ValueError(
                f"Sequence count {len(sequences)} != target count {len(targets)}",
            )

        # Check for reasonable value ranges
        seq_std = np.std(sequences)
        if seq_std < 1e-6:
            warnings.warn(
                "Very low variance in sequences - check scaling",
                stacklevel=2,
            )

        # Log quality metrics
        self.metadata["data_quality"] = {
            "sequences_shape": sequences.shape,
            "targets_shape": targets.shape,
            "sequences_mean": float(np.mean(sequences)),
            "sequences_std": float(np.std(sequences)),
            "targets_mean": float(np.mean(targets)),
            "targets_std": float(np.std(targets)),
            "nan_count": int(np.sum(np.isnan(sequences)) + np.sum(np.isnan(targets))),
            "inf_count": int(np.sum(np.isinf(sequences)) + np.sum(np.isinf(targets))),
        }

        logger.info("  ✓ Dataset validation passed")

    def _save_dataset(self, sequences: np.ndarray, targets: np.ndarray) -> dict:
        """Save the dataset and metadata."""

        logger.info("💾 Saving dataset...")

        # Save sequences and targets
        sequences_path = self.output_dir / "sequences.npy"
        targets_path = self.output_dir / "targets.npy"

        np.save(sequences_path, sequences)
        np.save(targets_path, targets)

        # Save feature columns
        features_path = self.output_dir / "feature_columns.json"
        with Path(features_path).open("w") as f:
            json.dump(self.feature_columns, f, indent=2)

        # Save complete metadata
        self.metadata["dataset_info"] = {
            "version": self.version,
            "created_at": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "files": {
                "sequences": str(sequences_path),
                "targets": str(targets_path),
                "scaler": str(self.output_dir / "feature_scaler.pkl"),
                "features": str(features_path),
                "metadata": str(self.output_dir / "metadata.json"),
            },
        }

        metadata_path = self.output_dir / "metadata.json"
        with Path(metadata_path).open("w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Create summary
        summary = {
            "version": self.version,
            "sequences_shape": sequences.shape,
            "targets_shape": targets.shape,
            "total_features": len(self.feature_columns),
            "output_directory": str(self.output_dir),
        }

        logger.info(f"  ✓ Dataset saved to {self.output_dir}")
        logger.info(f"  ✓ Sequences: {sequences.shape}, Targets: {targets.shape}")

        return summary


class RealTimeDatasetLoader:
    """Loader for real-time data processing with the same pipeline."""

    def __init__(self, dataset_version_dir: str):
        self.dataset_dir = Path(dataset_version_dir)

        # Load metadata and scaler
        with (self.dataset_dir / "metadata.json").open("r") as f:
            self.metadata = json.load(f)

        with (self.dataset_dir / "feature_columns.json").open("r") as f:
            self.feature_columns = json.load(f)

        with (self.dataset_dir / "feature_scaler.pkl").open("rb") as f:
            self.scaler = pickle.load(f)

        self.config = DatasetConfig(**self.metadata["dataset_info"]["config"])

    def process_realtime_data(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Process real-time data using the same pipeline as training."""

        # Apply same feature engineering
        builder = RobustDatasetBuilder(self.config)
        featured_df = builder._engineer_features(raw_df)

        # Extract features in same order
        features = featured_df[self.feature_columns].values

        # Scale using fitted scaler
        scaled_features = self.scaler.transform(features)

        # Create sequences if enough data
        if len(scaled_features) >= self.config.sequence_length:
            sequence = scaled_features[-self.config.sequence_length :]
            return sequence.reshape(1, self.config.sequence_length, -1)
        # Pad with zeros if insufficient data
        padded = np.zeros((self.config.sequence_length, len(self.feature_columns)))
        padded[-len(scaled_features) :] = scaled_features
        return padded.reshape(1, self.config.sequence_length, -1)


def create_example_config() -> DatasetConfig:
    """Create an example configuration for dataset building."""

    return DatasetConfig(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        timeframe="1d",
        real_data_ratio=0.8,
        min_samples_per_symbol=1000,
        sequence_length=60,
        prediction_horizon=1,
        overlap_ratio=0.8,
        technical_indicators=True,
        sentiment_features=False,  # Disable for initial testing
        market_regime_features=True,
        output_dir="data/robust_dataset",
    )


if __name__ == "__main__":
    # Example usage
    config = create_example_config()
    builder = RobustDatasetBuilder(config)

    sequences, targets, info = builder.build_dataset()
    print(f"Dataset created: {info}")
