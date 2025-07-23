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
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from tqdm import tqdm

from config import FeatureConfig
from trade_agent.core.logging import get_logger

logger = logging.getLogger(__name__)


@dataclass
class DataStandardizer:
    """
    A class to standardize data using different methods, handle missing values,
    and manage feature configurations.
    """

    method: Literal["robust", "standard"] = "robust"
    features: list[str] | None = None
    feature_config: FeatureConfig | None = None
    is_fitted: bool = False
    stats_: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if self.feature_config:
            self.features = self.feature_config.get_all_features()

    def fit(self, df: pd.DataFrame) -> "DataStandardizer":
        """
        Fit the standardizer by calculating the required statistics from the dataframe.

        For 'robust' scaling, it calculates quantiles. For 'standard' scaling,
        it calculates mean and std deviation.
        """
        logger.info("Fitting DataStandardizer...")

        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()

        if self.features is None:
            self.features = [
                col
                for col in df.columns
                if col not in ["timestamp", "symbol"]
            ]

        # Convert boolean columns to integers (0 or 1) for statistical analysis
        for col in self.features:
            if col in df_processed.columns:
                # Check for boolean dtype explicitly
                if is_bool_dtype(df_processed[col]):
                    logger.debug(f"Converting boolean column '{col}' to integer.")
                    df_processed[col] = df_processed[col].astype(int)
                # Also check if the column contains only boolean values (True/False)
                elif df_processed[col].dtype == "object":
                    unique_values = set(df_processed[col].dropna().unique())
                    if unique_values.issubset({True, False}):
                        logger.debug(f"Converting object column '{col}' with boolean values to integer.")
                        df_processed[col] = df_processed[col].astype(int)
                # Handle boolean values stored as object dtype
                elif str(df_processed[col].dtype) == "bool":
                    logger.debug(f"Converting bool dtype column '{col}' to integer.")
                    df_processed[col] = df_processed[col].astype(int)

        self._calculate_feature_stats(df_processed)
        self.is_fitted = True
        logger.info("DataStandardizer fitted successfully.")
        return self

    def _calculate_feature_stats(self, df: pd.DataFrame) -> None:
        """
        Calculate and store statistics for each feature.
        """
        self.stats_ = {}
        if self.features is None:
            return

        for feature in self.features:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' not found in DataFrame. Skipping.")
                continue

            # Ensure the column is numeric before processing
            if not is_numeric_dtype(df[feature]):
                logger.warning(
                    f"Feature '{feature}' is not numeric. Skipping stats calculation."
                )
                continue

            data = df[feature].dropna()

            if data.empty:
                logger.warning(f"No data for feature '{feature}' after dropping NaNs. Skipping.")
                stats = {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 1.0,
                    "q25": 0.0,
                    "q75": 0.0,
                }
            else:
                stats = {
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "q25": float(data.quantile(0.25)),
                    "q75": float(data.quantile(0.75)),
                }
            self.stats_[feature] = stats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("Standardizer is not fitted yet. Call 'fit' first.")

        if self.features is None:
            return df

        df_transformed = df.copy()

        # Convert boolean columns to int before transformation
        for feature in self.features:
            if feature in df_transformed.columns and is_bool_dtype(
                df_transformed[feature]
            ):
                df_transformed[feature] = df_transformed[feature].astype(int)

        # Use tqdm for progress tracking if the dataframe is large
        num_chunks = 100
        chunk_size = len(df_transformed) // num_chunks + 1

        # Process in chunks to handle large datasets
        for i in tqdm(
            range(0, len(df_transformed), chunk_size),
            desc="Standardizing data",
            leave=False,
        ):
            chunk = df_transformed.iloc[i : i + chunk_size].copy()
            for feature in self.features:
                if feature not in chunk.columns or feature not in self.stats_:
                    continue

                stats = self.stats_[feature]
                col = chunk[feature]

                if self.method == "robust":
                    q25 = stats["q25"]
                    q75 = stats["q75"]
                    iqr = q75 - q25
                    if iqr > 1e-8:
                        chunk[feature] = (col - q25) / iqr
                    else:
                        chunk[feature] = col - col.mean() # Fallback for zero IQR
                elif self.method == "standard":
                    mean = stats["mean"]
                    std = stats["std"]
                    if std > 1e-8:
                        chunk[feature] = (col - mean) / std
                    else:
                        chunk[feature] = col - mean # Fallback for zero std

            df_transformed.iloc[i : i + chunk_size] = chunk

        # Handle missing values after transformation
        df_transformed.fillna(0, inplace=True)
        df_transformed.replace([np.inf, -np.inf], 0, inplace=True)

        return df_transformed

    def get_feature_names(self) -> list[str]:
        """Get the list of feature names."""
        return self.features if self.features is not None else []

    def get_feature_count(self) -> int:
        """Get the number of features."""
        return len(self.features) if self.features is not None else 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the standardizer to the data and transform it in one step.

        This is a convenience method that combines 'fit' and 'transform'.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> None:
        """Save the standardizer to disk."""
        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle
        with open(filepath_path, "wb") as f:
            pickle.dump(self, f)

        # Also save as JSON for inspection
        json_filepath = filepath_path.with_suffix(".json")

        # Handle case where feature_config might be None
        feature_config_dict = {}
        if self.feature_config is not None:
            feature_config_dict = {
                "price_features": self.feature_config.price_features,
                "technical_indicators": self.feature_config.technical_indicators,
                "candlestick_patterns": self.feature_config.candlestick_patterns,
                "candlestick_features": self.feature_config.candlestick_features,
                "rolling_candlestick_features": self.feature_config.rolling_candlestick_features,
                "sentiment_features": self.feature_config.sentiment_features,
                "time_features": self.feature_config.time_features,
                "market_regime_features": self.feature_config.market_regime_features,
                "target_feature": self.feature_config.target_feature,
            }

        save_dict = {
            "feature_config": feature_config_dict,
            "feature_stats": self.stats_,
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
            if self.feature_config is not None:
                if feature in self.feature_config.candlestick_patterns:
                    template_data[feature] = [0]  # Binary features
                elif feature in self.feature_config.sentiment_features:
                    template_data[feature] = [0]  # Sentiment features (use int for consistency)
                elif feature in self.feature_config.time_features:
                    template_data[feature] = [0]  # Time features
                else:
                    template_data[feature] = [0]  # Numeric features (use int for consistency)
            else:
                template_data[feature] = [0]  # Default to numeric features

        return pd.DataFrame(template_data)


class LiveDataProcessor:
    """Process live data using the same standardization as trading."""

    def __init__(self, standardizer: DataStandardizer):
        self.standardizer = standardizer
        self.logger = get_logger("LiveDataProcessor")

    def process_single_row(self, data: dict[str, Any]) -> pd.DataFrame:
        """Process a single row of live data."""
        # Convert to dataframe
        df = pd.DataFrame([data])

        # Apply standardization
        processed_df = self.standardizer.transform(df)

        # Return only the features needed for prediction
        feature_names = self.standardizer.get_feature_names()
        # Ensure we always return a DataFrame by selecting columns as a list
        return processed_df.loc[:, feature_names]

    def process_batch(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of live data."""
        # Convert to dataframe
        df = pd.DataFrame(data)

        # Apply standardization
        processed_df = self.standardizer.transform(df)

        # Return only the features needed for prediction
        feature_names = self.standardizer.get_feature_names()
        # Ensure we always return a DataFrame by selecting columns as a list
        return processed_df.loc[:, feature_names]

    def get_feature_names(self) -> list[str]:
        """Get the feature names for the model."""
        return self.standardizer.get_feature_names()

    def get_feature_count(self) -> int:
        """Get the number of features."""
        return self.standardizer.get_feature_count()


def create_standardized_dataset(
    df: pd.DataFrame,
    save_path: str | None = None,
    feature_config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, DataStandardizer]:
    """Create a standardized dataset with consistent features."""

    # Create standardizer
    standardizer = DataStandardizer(feature_config=feature_config or FeatureConfig())

    # Fit and transform the data
    standardizer.fit(df)
    standardized_df = standardizer.transform(df)

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
    standardized_df = standardizer.transform(df)

    return standardized_df, standardizer


def create_live_inference_processor(
    standardizer_path: str = "data/processed/data_standardizer.pkl",
) -> LiveDataProcessor:
    """Create a live inference processor for real-time trading."""
    try:
        standardizer = DataStandardizer.load(standardizer_path)
        return LiveDataProcessor(standardizer)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Standardizer not found at {standardizer_path}. Please run dataset building first.",
        ) from err
    except Exception as err:
        raise RuntimeError(f"Failed to create live inference processor: {err}") from err


def process_live_data(data: dict[str, Any], standardizer_path: str = "data/processed/data_standardizer.pkl") -> np.ndarray:
    """Process live data for inference using the trained standardizer."""
    processor = create_live_inference_processor(standardizer_path)
    processed_df = processor.process_single_row(data)
    return processed_df.values
