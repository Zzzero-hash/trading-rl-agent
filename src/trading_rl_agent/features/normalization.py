"""
Comprehensive normalization and scaling utilities for trading features.

This module provides robust normalization capabilities including:
- Per-symbol normalization for multi-asset datasets
- Multiple scaling methods (MinMax, Standard, Robust, etc.)
- Handling of missing data and outliers
- Feature-specific normalization strategies
- Persistence and loading of fitted scalers
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.trading_rl_agent.core.logging import get_logger


@dataclass
class NormalizationConfig:
    """Configuration for feature normalization."""

    # Scaling method
    method: str = "robust"  # robust, standard, minmax, quantile

    # Per-symbol normalization
    per_symbol: bool = True

    # Feature-specific settings
    price_features: str = "robust"  # How to scale price-based features
    volume_features: str = "log"  # How to scale volume features
    indicator_features: str = "robust"  # How to scale technical indicators
    temporal_features: str = "none"  # How to scale temporal features (usually none)

    # Robustness settings
    handle_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    handle_missing: bool = True
    missing_strategy: str = "median"  # mean, median, zero, drop

    # Persistence
    save_scalers: bool = True
    scaler_dir: str = "models/scalers"

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_methods = ["robust", "standard", "minmax", "quantile"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")

        valid_strategies = ["mean", "median", "zero", "drop"]
        if self.missing_strategy not in valid_strategies:
            raise ValueError(f"Invalid missing_strategy: {self.missing_strategy}")


class FeatureNormalizer:
    """
    Comprehensive feature normalizer with per-symbol support.

    This class provides robust normalization capabilities for trading features,
    including per-symbol normalization for multi-asset datasets and feature-specific
    scaling strategies.
    """

    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Storage for fitted scalers
        self.scalers: dict[str, Any] = {}
        self.feature_columns: list[str] = []
        self.symbol_columns: list[str] = []

        # Statistics for feature-specific normalization
        self.feature_stats: dict[str, dict[str, float]] = {}

        # Create scaler directory
        if self.config.save_scalers:
            Path(self.config.scaler_dir).mkdir(parents=True, exist_ok=True)

    def fit(self, df: pd.DataFrame, symbol_column: str = "symbol") -> "FeatureNormalizer":
        """
        Fit the normalizer on the training data.

        Args:
            df: DataFrame with features to normalize
            symbol_column: Name of the symbol column for per-symbol normalization

        Returns:
            Self for chaining
        """
        self.logger.info(f"Fitting normalizer with method: {self.config.method}")

        # Store feature columns
        self.feature_columns = [col for col in df.columns if col != symbol_column]
        self.symbol_columns = df[symbol_column].unique().tolist() if symbol_column in df.columns else []

        if self.config.per_symbol and symbol_column in df.columns:
            self._fit_per_symbol(df, symbol_column)
        else:
            self._fit_global(df)

        # Compute feature statistics for feature-specific normalization
        self._compute_feature_statistics(df)

        self.logger.info(f"Fitted normalizer for {len(self.feature_columns)} features")
        return self

    def transform(self, df: pd.DataFrame, symbol_column: str = "symbol") -> pd.DataFrame:
        """
        Transform the data using fitted scalers.

        Args:
            df: DataFrame to normalize
            symbol_column: Name of the symbol column

        Returns:
            Normalized DataFrame
        """
        self.logger.info("Transforming data with fitted normalizer")

        result = df.copy()

        if self.config.per_symbol and symbol_column in df.columns:
            result = self._transform_per_symbol(result, symbol_column)
        else:
            result = self._transform_global(result)

        # Apply feature-specific normalization
        result = self._apply_feature_specific_normalization(result)

        # Handle any remaining missing values
        if self.config.handle_missing:
            result = self._handle_missing_values(result)

        self.logger.info("Data transformation completed")
        return result

    def fit_transform(self, df: pd.DataFrame, symbol_column: str = "symbol") -> pd.DataFrame:
        """
        Fit the normalizer and transform the data.

        Args:
            df: DataFrame to fit and transform
            symbol_column: Name of the symbol column

        Returns:
            Normalized DataFrame
        """
        return self.fit(df, symbol_column).transform(df, symbol_column)

    def _fit_per_symbol(self, df: pd.DataFrame, symbol_column: str) -> None:
        """Fit scalers for each symbol separately."""
        for symbol in df[symbol_column].unique():
            symbol_data = df[df[symbol_column] == symbol][self.feature_columns]

            # Handle missing values before fitting
            if self.config.handle_missing:
                symbol_data = self._handle_missing_values(symbol_data)

            # Handle outliers before fitting
            if self.config.handle_outliers:
                symbol_data = self._handle_outliers(symbol_data)

            # Create and fit scaler
            scaler = self._create_scaler()
            scaler.fit(symbol_data)

            self.scalers[symbol] = scaler
            self.logger.debug(f"Fitted scaler for symbol: {symbol}")

    def _fit_global(self, df: pd.DataFrame) -> None:
        """Fit a single scaler for all data."""
        feature_data = df[self.feature_columns]

        # Handle missing values before fitting
        if self.config.handle_missing:
            feature_data = self._handle_missing_values(feature_data)

        # Handle outliers before fitting
        if self.config.handle_outliers:
            feature_data = self._handle_outliers(feature_data)

        # Create and fit scaler
        scaler = self._create_scaler()
        scaler.fit(feature_data)

        self.scalers["global"] = scaler
        self.logger.debug("Fitted global scaler")

    def _transform_per_symbol(self, df: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
        """Transform data using per-symbol scalers."""
        result = df.copy()

        for symbol in df[symbol_column].unique():
            if symbol in self.scalers:
                symbol_mask = df[symbol_column] == symbol
                symbol_data = df.loc[symbol_mask, self.feature_columns]

                # Transform
                transformed_data = self.scalers[symbol].transform(symbol_data)
                result.loc[symbol_mask, self.feature_columns] = transformed_data
            else:
                self.logger.warning(f"No fitted scaler found for symbol: {symbol}")

        return result

    def _transform_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using global scaler."""
        if "global" not in self.scalers:
            raise ValueError("No global scaler fitted. Call fit() first.")

        result = df.copy()
        feature_data = df[self.feature_columns]

        # Transform
        transformed_data = self.scalers["global"].transform(feature_data)
        result[self.feature_columns] = transformed_data

        return result

    def _create_scaler(self) -> Any:
        """Create a scaler based on the configuration."""
        if self.config.method == "robust":
            return RobustScaler()
        if self.config.method == "standard":
            return StandardScaler()
        if self.config.method == "minmax":
            return MinMaxScaler()
        if self.config.method == "quantile":
            return QuantileTransformer(output_distribution="normal")
        raise ValueError(f"Unknown scaling method: {self.config.method}")

    def _compute_feature_statistics(self, df: pd.DataFrame) -> None:
        """Compute statistics for feature-specific normalization."""
        for col in self.feature_columns:
            if col in df.columns:
                self.feature_stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                }

    def _apply_feature_specific_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature-specific normalization strategies."""
        result = df.copy()

        for col in self.feature_columns:
            if col not in df.columns:
                continue

            # Determine feature type and apply appropriate normalization
            if self._is_price_feature(col):
                result = self._normalize_price_feature(result, col)
            elif self._is_volume_feature(col):
                result = self._normalize_volume_feature(result, col)
            elif self._is_temporal_feature(col):
                result = self._normalize_temporal_feature(result, col)
            elif self._is_indicator_feature(col):
                result = self._normalize_indicator_feature(result, col)

        return result

    def _is_price_feature(self, col: str) -> bool:
        """Check if a column is a price feature."""
        price_keywords = ["open", "high", "low", "close", "price", "bid", "ask"]
        return any(keyword in col.lower() for keyword in price_keywords)

    def _is_volume_feature(self, col: str) -> bool:
        """Check if a column is a volume feature."""
        volume_keywords = ["volume", "vol", "trade", "quantity"]
        return any(keyword in col.lower() for keyword in volume_keywords)

    def _is_temporal_feature(self, col: str) -> bool:
        """Check if a column is a temporal feature."""
        temporal_keywords = ["hour", "day", "month", "quarter", "year", "sin", "cos"]
        return any(keyword in col.lower() for keyword in temporal_keywords)

    def _is_indicator_feature(self, col: str) -> bool:
        """Check if a column is a technical indicator feature."""
        indicator_keywords = ["sma", "ema", "rsi", "macd", "bb", "atr", "stoch"]
        return any(keyword in col.lower() for keyword in indicator_keywords)

    def _normalize_price_feature(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize price features."""
        if self.config.price_features == "log":
            # Log normalization for price features
            df[col] = np.log1p(df[col].abs())
        elif self.config.price_features == "returns":
            # Convert to returns
            df[col] = df[col].pct_change().fillna(0)
        # Other methods are handled by the main scaler

        return df

    def _normalize_volume_feature(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize volume features."""
        if self.config.volume_features == "log":
            # Log normalization for volume features
            df[col] = np.log1p(df[col].abs())
        elif self.config.volume_features == "sqrt":
            # Square root normalization
            df[col] = np.sqrt(df[col].abs())

        return df

    def _normalize_temporal_feature(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize temporal features."""
        if self.config.temporal_features == "none":
            # Temporal features are usually already normalized (e.g., sine/cosine)
            pass
        elif self.config.temporal_features == "minmax":
            # Min-max scaling for temporal features
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)

        return df

    def _normalize_indicator_feature(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize technical indicator features."""
        if self.config.indicator_features == "robust":
            # Robust scaling for indicators
            median = df[col].median()
            mad = df[col].mad()  # Median Absolute Deviation
            if mad > 0:
                df[col] = (df[col] - median) / mad

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        result = df.copy()

        for col in result.columns:
            if result[col].isnull().any():
                if self.config.missing_strategy == "mean":
                    result[col] = result[col].fillna(result[col].mean())
                elif self.config.missing_strategy == "median":
                    result[col] = result[col].fillna(result[col].median())
                elif self.config.missing_strategy == "zero":
                    result[col] = result[col].fillna(0.0)
                elif self.config.missing_strategy == "drop":
                    result = result.dropna(subset=[col])

        return result

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        result = df.copy()

        for col in result.columns:
            if result[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = result[col].mean()
                std = result[col].std()

                if std > 0:
                    lower_bound = mean - self.config.outlier_threshold * std
                    upper_bound = mean + self.config.outlier_threshold * std

                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

        return result

    def save(self, filepath: str) -> None:
        """Save the fitted normalizer to disk."""
        if not self.config.save_scalers:
            return

        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "scalers": self.scalers,
                    "feature_columns": self.feature_columns,
                    "symbol_columns": self.symbol_columns,
                    "feature_stats": self.feature_stats,
                },
                f,
            )

        self.logger.info(f"Saved normalizer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeatureNormalizer":
        """Load a fitted normalizer from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)  # nosec B301

        normalizer = cls(data["config"])
        normalizer.scalers = data["scalers"]
        normalizer.feature_columns = data["feature_columns"]
        normalizer.symbol_columns = data["symbol_columns"]
        normalizer.feature_stats = data["feature_stats"]

        return normalizer

    def get_feature_names(self) -> list[str]:
        """Get the list of feature names."""
        return self.feature_columns.copy()

    def get_scaler_info(self) -> dict[str, Any]:
        """Get information about the fitted scalers."""
        return {
            "method": self.config.method,
            "per_symbol": self.config.per_symbol,
            "n_symbols": len(self.symbol_columns),
            "n_features": len(self.feature_columns),
            "symbols": self.symbol_columns,
            "features": self.feature_columns,
        }
