#!/usr/bin/env python3
"""
Data Standardizer for Trading RL Agent

This module ensures consistent feature engineering between training and live inference.
It maintains a standardized feature set, handles missing data consistently, and provides
the same preprocessing pipeline for both training and production.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.trading_rl_agent.core.logging import get_logger


@dataclass
class FeatureConfig:
    """Configuration for standardized features."""

    # Core price features (always required)
    price_features: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])

    # Technical indicators (standard set)
    technical_indicators: list[str] = field(
        default_factory=lambda: [
            "log_return",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "rsi_14",
            "vol_20",
            "ema_20",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "bb_mavg_20",
            "bb_upper_20",
            "bb_lower_20",
            "stoch_k",
            "stoch_d",
            "adx_14",
            "wr_14",
            "obv",
        ]
    )

    # Candlestick patterns (binary features)
    candlestick_patterns: list[str] = field(
        default_factory=lambda: [
            "doji",
            "hammer",
            "hanging_man",
            "bullish_engulfing",
            "bearish_engulfing",
            "shooting_star",
            "morning_star",
            "evening_star",
            "inside_bar",
            "outside_bar",
            "tweezer_top",
            "tweezer_bottom",
            "three_white_soldiers",
            "three_black_crows",
            "bullish_harami",
            "bearish_harami",
            "dark_cloud_cover",
            "piercing_line",
        ]
    )

    # Candlestick characteristics
    candlestick_features: list[str] = field(
        default_factory=lambda: [
            "body_size",
            "range_size",
            "rel_body_size",
            "upper_shadow",
            "lower_shadow",
            "rel_upper_shadow",
            "rel_lower_shadow",
            "body_position",
            "body_type",
        ]
    )

    # Rolling averages of candlestick features
    rolling_candlestick_features: list[str] = field(
        default_factory=lambda: [
            "avg_rel_body_5",
            "avg_upper_shadow_5",
            "avg_lower_shadow_5",
            "avg_body_pos_5",
            "body_momentum_5",
            "avg_rel_body_10",
            "avg_upper_shadow_10",
            "avg_lower_shadow_10",
            "avg_body_pos_10",
            "body_momentum_10",
            "avg_rel_body_20",
            "avg_upper_shadow_20",
            "avg_lower_shadow_20",
            "avg_body_pos_20",
            "body_momentum_20",
        ]
    )

    # Sentiment features
    sentiment_features: list[str] = field(default_factory=lambda: ["sentiment", "sentiment_magnitude"])

    # Time-based features
    time_features: list[str] = field(default_factory=lambda: ["hour", "day_of_week", "month", "quarter"])

    # Market regime features
    market_regime_features: list[str] = field(
        default_factory=lambda: ["price_change_pct", "high_low_pct", "volume_ma_20", "volume_ratio", "volume_change"]
    )

    # Target variable
    target_feature: str = "target"

    def get_all_features(self) -> list[str]:
        """Get all feature names in the correct order."""
        all_features = []
        all_features.extend(self.price_features)
        all_features.extend(self.technical_indicators)
        all_features.extend(self.candlestick_patterns)
        all_features.extend(self.candlestick_features)
        all_features.extend(self.rolling_candlestick_features)
        all_features.extend(self.sentiment_features)
        all_features.extend(self.time_features)
        all_features.extend(self.market_regime_features)
        return all_features

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_all_features())


@dataclass
class DataStandardizer:
    """Standardizes data processing for consistent training and inference."""

    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    scaler: Any | None = None
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    missing_value_strategies: dict[str, str] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: get_logger("DataStandardizer"))

    def __post_init__(self) -> None:
        """Initialize default missing value strategies."""
        if not self.missing_value_strategies:
            self._set_default_missing_strategies()

    def _set_default_missing_strategies(self) -> None:
        """Set default strategies for handling missing values."""
        # Price features - forward fill, then backward fill
        for feature in self.feature_config.price_features:
            self.missing_value_strategies[feature] = "forward_backward"

        # Technical indicators - forward fill
        for feature in self.feature_config.technical_indicators:
            self.missing_value_strategies[feature] = "forward"

        # Candlestick patterns - fill with 0
        for feature in self.feature_config.candlestick_patterns:
            self.missing_value_strategies[feature] = "zero"

        # Candlestick features - forward fill
        for feature in self.feature_config.candlestick_features:
            self.missing_value_strategies[feature] = "forward"

        # Rolling features - forward fill
        for feature in self.feature_config.rolling_candlestick_features:
            self.missing_value_strategies[feature] = "forward"

        # Sentiment features - fill with 0
        for feature in self.feature_config.sentiment_features:
            self.missing_value_strategies[feature] = "zero"

        # Time features - no missing values expected
        for feature in self.feature_config.time_features:
            self.missing_value_strategies[feature] = "zero"

        # Market regime features - forward fill
        for feature in self.feature_config.market_regime_features:
            self.missing_value_strategies[feature] = "forward"

    def fit(self, df: pd.DataFrame) -> "DataStandardizer":
        """Fit the standardizer on training data."""
        self.logger.info("Fitting DataStandardizer...")

        # Calculate feature statistics
        self._calculate_feature_stats(df)

        # Fit scaler if needed
        self._fit_scaler(df)

        self.logger.info(f"DataStandardizer fitted with {len(self.feature_config.get_all_features())} features")
        return self

    def _calculate_feature_stats(self, df: pd.DataFrame) -> None:
        """Calculate statistics for each feature."""
        self.feature_stats = {}

        for feature in self.feature_config.get_all_features():
            if feature in df.columns:
                data = df[feature].dropna()
                if len(data) > 0:
                    self.feature_stats[feature] = {
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "median": float(data.median()),
                        "q25": float(data.quantile(0.25)),
                        "q75": float(data.quantile(0.75)),
                    }
                else:
                    self.feature_stats[feature] = {
                        "mean": 0.0,
                        "std": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "median": 0.0,
                        "q25": 0.0,
                        "q75": 1.0,
                    }
            else:
                # Feature not in dataset, use defaults
                self.feature_stats[feature] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "median": 0.0,
                    "q25": 0.0,
                    "q75": 1.0,
                }

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit scaler for normalization."""
        try:
            from sklearn.preprocessing import RobustScaler

            # Get features that exist in the dataset
            available_features = [f for f in self.feature_config.get_all_features() if f in df.columns]

            if available_features:
                # Use RobustScaler for outlier-resistant scaling
                self.scaler = RobustScaler()
                self.scaler.fit(df[available_features].fillna(0))
                self.logger.info(f"Scaler fitted on {len(available_features)} features")
            else:
                self.logger.warning("No features available for scaling")

        except ImportError:
            self.logger.warning("sklearn not available, skipping scaler fitting")

    def transform(
        self, df: pd.DataFrame, is_training: bool = False, chunk_size: int = 5000, show_progress: bool = True
    ) -> pd.DataFrame:
        """Transform data to standardized format, with chunked processing and progress reporting."""
        self.logger.info(f"Transforming data (training={is_training}) with chunk_size={chunk_size}...")
        n_rows = len(df)
        if n_rows <= chunk_size:
            # Small enough, process all at once
            return self._transform_chunk(df, is_training)
        # Process in chunks
        result_chunks = []
        iterator = range(0, n_rows, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, total=(n_rows + chunk_size - 1) // chunk_size, desc="Standardizing", unit="chunk")
        for start in iterator:
            end = min(start + chunk_size, n_rows)
            chunk = df.iloc[start:end].copy()
            chunk_result = self._transform_chunk(chunk, is_training)
            result_chunks.append(chunk_result)
        result = pd.concat(result_chunks, axis=0, ignore_index=True)
        self.logger.info(f"Transformation complete. Output shape: {result.shape}")
        return result

    def _transform_chunk(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transform a single chunk of data (internal use)."""
        result = df.copy()
        result = self._ensure_features_exist(result)
        result = self._handle_missing_values(result)
        result = self._clean_invalid_values(result)
        if self.scaler is not None:
            result = self._apply_scaling(result)
        result = self._ensure_feature_order(result)
        self._validate_output(result)
        return result

    def _ensure_features_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features exist in the dataframe."""
        required_features = self.feature_config.get_all_features()

        for feature in required_features:
            if feature not in df.columns:
                self.logger.warning(f"Missing feature '{feature}', creating with default values")

                if feature in self.feature_config.candlestick_patterns:
                    # Binary features default to 0
                    df[feature] = 0
                elif feature in self.feature_config.sentiment_features:
                    # Sentiment features default to 0
                    df[feature] = 0.0
                elif feature in self.feature_config.time_features:
                    # Time features default to 0
                    df[feature] = 0
                else:
                    # Other features default to 0
                    df[feature] = 0.0

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategies."""
        for feature, strategy in self.missing_value_strategies.items():
            if feature in df.columns and df[feature].isnull().any():
                if strategy == "forward":
                    df[feature] = df[feature].ffill()
                elif strategy == "backward":
                    df[feature] = df[feature].bfill()
                elif strategy == "forward_backward":
                    df[feature] = df[feature].ffill().bfill()
                elif strategy == "zero":
                    df[feature] = df[feature].fillna(0)
                elif strategy == "mean":
                    mean_val = self.feature_stats.get(feature, {}).get("mean", 0)
                    df[feature] = df[feature].fillna(mean_val)
                elif strategy == "median":
                    median_val = self.feature_stats.get(feature, {}).get("median", 0)
                    df[feature] = df[feature].fillna(median_val)

                # Final fallback to 0 if any NaN remains
                df[feature] = df[feature].fillna(0)

        return df

    def _clean_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean invalid values (negative prices, infinite values, etc.)."""
        for feature in df.columns:
            if feature in df.columns:
                # Handle infinite values
                df[feature] = df[feature].replace([np.inf, -np.inf], 0)

                # Handle negative values in price-related features
                if any(keyword in feature.lower() for keyword in ["price", "close", "open", "high", "low"]):
                    df[feature] = df[feature].clip(lower=0)

                # Handle negative values in volume
                if "volume" in feature.lower():
                    df[feature] = df[feature].clip(lower=0)

                # Handle negative values in technical indicators that should be positive
                if feature in ["rsi_14", "stoch_k", "stoch_d", "adx_14", "atr_14"]:
                    df[feature] = df[feature].clip(lower=0)

                # Handle values outside expected ranges for binary features
                if feature in self.feature_config.candlestick_patterns:
                    df[feature] = df[feature].clip(0, 1)

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to features."""
        if self.scaler is None:
            return df

        try:
            # Get features that the scaler was trained on
            available_features = [f for f in self.feature_config.get_all_features() if f in df.columns]

            if available_features:
                # Apply scaling
                scaled_values = self.scaler.transform(df[available_features])
                df[available_features] = scaled_values

        except Exception as e:
            self.logger.warning(f"Scaling failed: {e}")

        return df

    def _ensure_feature_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features are in the correct order."""
        required_features = self.feature_config.get_all_features()

        # Add any missing features with default values
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0.0

        # Reorder columns to match required order
        existing_features = [f for f in required_features if f in df.columns]
        other_features = [f for f in df.columns if f not in required_features]

        # Return dataframe with correct column order
        return df[existing_features + other_features]

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Validate the output dataframe."""
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Output contains {nan_count} NaN values")

        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Output contains {inf_count} infinite values")

        # Check feature count
        expected_features = len(self.feature_config.get_all_features())
        actual_features = len([f for f in self.feature_config.get_all_features() if f in df.columns])

        if actual_features != expected_features:
            self.logger.warning(f"Feature count mismatch: expected {expected_features}, got {actual_features}")

    def get_feature_names(self) -> list[str]:
        """Get the standardized feature names."""
        return self.feature_config.get_all_features()

    def get_feature_count(self) -> int:
        """Get the number of features."""
        return self.feature_config.get_feature_count()

    def save(self, filepath: str) -> None:
        """Save the standardizer to disk."""
        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        with open(filepath_path, "wb") as f:
            pickle.dump(self, f)

        # Also save as JSON for inspection
        json_filepath = filepath_path.with_suffix(".json")
        save_dict = {
            "feature_config": {
                "price_features": self.feature_config.price_features,
                "technical_indicators": self.feature_config.technical_indicators,
                "candlestick_patterns": self.feature_config.candlestick_patterns,
                "candlestick_features": self.feature_config.candlestick_features,
                "rolling_candlestick_features": self.feature_config.rolling_candlestick_features,
                "sentiment_features": self.feature_config.sentiment_features,
                "time_features": self.feature_config.time_features,
                "market_regime_features": self.feature_config.market_regime_features,
                "target_feature": self.feature_config.target_feature,
            },
            "feature_stats": self.feature_stats,
            "missing_value_strategies": self.missing_value_strategies,
            "feature_count": self.get_feature_count(),
            "all_features": self.get_feature_names(),
        }

        with open(json_filepath, "w") as f:
            json.dump(save_dict, f, indent=2)

        self.logger.info(f"DataStandardizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DataStandardizer":
        """Load the standardizer from disk."""
        filepath_path = Path(filepath)

        with open(filepath_path, "rb") as f:
            standardizer = pickle.load(f)  # nosec B301 - This is loading our own trusted standardizer

        if not isinstance(standardizer, cls):
            raise ValueError(f"Loaded object is not a {cls.__name__}")

        standardizer.logger.info(f"DataStandardizer loaded from {filepath_path}")
        return standardizer

    def create_live_data_template(self) -> pd.DataFrame:
        """Create a template dataframe for live data with correct structure."""
        template_data = {}

        for feature in self.get_feature_names():
            if feature in self.feature_config.candlestick_patterns:
                template_data[feature] = [0]  # Binary features
            elif feature in self.feature_config.sentiment_features:
                template_data[feature] = [0]  # Sentiment features (use int for consistency)
            elif feature in self.feature_config.time_features:
                template_data[feature] = [0]  # Time features
            else:
                template_data[feature] = [0]  # Numeric features (use int for consistency)

        return pd.DataFrame(template_data)


class LiveDataProcessor:
    """Process live data using the same standardization as training."""

    def __init__(self, standardizer: DataStandardizer):
        self.standardizer = standardizer
        self.logger = get_logger("LiveDataProcessor")

    def process_single_row(self, data: dict[str, Any]) -> pd.DataFrame:
        """Process a single row of live data."""
        # Convert to dataframe
        df = pd.DataFrame([data])

        # Apply standardization
        processed_df = self.standardizer.transform(df, is_training=False)

        # Return only the features needed for prediction
        feature_names = self.standardizer.get_feature_names()
        return processed_df[feature_names]

    def process_batch(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of live data."""
        # Convert to dataframe
        df = pd.DataFrame(data)

        # Apply standardization
        processed_df = self.standardizer.transform(df, is_training=False)

        # Return only the features needed for prediction
        feature_names = self.standardizer.get_feature_names()
        return processed_df[feature_names]

    def get_feature_names(self) -> list[str]:
        """Get the feature names for the model."""
        return self.standardizer.get_feature_names()

    def get_feature_count(self) -> int:
        """Get the number of features."""
        return self.standardizer.get_feature_count()


def create_standardized_dataset(
    df: pd.DataFrame, save_path: str | None = None, feature_config: FeatureConfig | None = None
) -> tuple[pd.DataFrame, DataStandardizer]:
    """Create a standardized dataset with consistent features."""

    # Create standardizer
    standardizer = DataStandardizer(feature_config=feature_config or FeatureConfig())

    # Fit and transform the data
    standardizer.fit(df)
    standardized_df = standardizer.transform(df, is_training=True)

    # Save if path provided
    if save_path:
        standardizer.save(save_path)

    return standardized_df, standardizer


def load_standardized_dataset(data_path: str, standardizer_path: str) -> tuple[pd.DataFrame, DataStandardizer]:
    """Load and standardize a dataset using a pre-trained standardizer."""

    # Load data
    df = pd.read_csv(data_path)

    # Load standardizer
    standardizer = DataStandardizer.load(standardizer_path)

    # Transform data
    standardized_df = standardizer.transform(df, is_training=False)

    return standardized_df, standardizer


def create_live_inference_processor(standardizer_path: str = "outputs/data_standardizer.pkl") -> LiveDataProcessor:
    """Create a live inference processor for real-time trading."""
    try:
        standardizer = DataStandardizer.load(standardizer_path)
        return LiveDataProcessor(standardizer)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Standardizer not found at {standardizer_path}. Please run dataset building first."
        ) from err
    except Exception as err:
        raise RuntimeError(f"Failed to create live inference processor: {err}") from err


def process_live_data(data: dict[str, Any], standardizer_path: str = "outputs/data_standardizer.pkl") -> np.ndarray:
    """Process live data for inference using the trained standardizer."""
    processor = create_live_inference_processor(standardizer_path)
    processed_df = processor.process_single_row(data)
    return processed_df.values
