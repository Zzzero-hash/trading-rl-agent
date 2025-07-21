"""Feature pipeline composing multiple feature generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.trading_rl_agent.core.logging import get_logger

from .alternative_data import AlternativeDataFeatures
from .cross_asset import CrossAssetFeatures
from .market_microstructure import MarketMicrostructure
from .normalization import FeatureNormalizer, NormalizationConfig
from .technical_indicators import TechnicalIndicators

if TYPE_CHECKING:
    import pandas as pd


class FeaturePipeline:
    """Compose various feature generators into a single pipeline with robust normalization."""

    def __init__(
        self,
        technical: TechnicalIndicators | None = None,
        microstructure: MarketMicrostructure | None = None,
        cross_asset: CrossAssetFeatures | None = None,
        alternative: AlternativeDataFeatures | None = None,
        normalizer: FeatureNormalizer | None = None,
        normalization_config: NormalizationConfig | None = None,
    ) -> None:
        self.technical = technical or TechnicalIndicators()
        self.microstructure = microstructure or MarketMicrostructure()
        self.cross_asset = cross_asset or CrossAssetFeatures()
        self.alternative = alternative or AlternativeDataFeatures()

        # Initialize normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        elif normalization_config is not None:
            self.normalizer = FeatureNormalizer(normalization_config)
        else:
            # Default normalization config
            default_config = NormalizationConfig(
                method="robust",
                per_symbol=True,
                handle_outliers=True,
                handle_missing=True,
            )
            self.normalizer = FeatureNormalizer(default_config)

        self.logger = get_logger(self.__class__.__name__)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
        symbol_column: str = "symbol",
    ) -> FeaturePipeline:
        """
        Fit the feature pipeline including normalization.

        Args:
            df: Training DataFrame
            cross_df: Cross-asset data
            sentiment: Sentiment data
            symbol_column: Name of the symbol column

        Returns:
            Self for chaining
        """
        self.logger.info("Fitting feature pipeline...")

        # Apply feature engineering
        featured_df = self._apply_feature_engineering(df, cross_df, sentiment)

        # Fit normalizer
        self.normalizer.fit(featured_df, symbol_column)

        self.is_fitted = True
        self.logger.info("Feature pipeline fitted successfully")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
        symbol_column: str = "symbol",
    ) -> pd.DataFrame:
        """
        Apply all feature generators and normalization sequentially.

        Args:
            df: Input DataFrame
            cross_df: Cross-asset data
            sentiment: Sentiment data
            symbol_column: Name of the symbol column

        Returns:
            Transformed DataFrame with features and normalization
        """
        if not self.is_fitted:
            self.logger.warning("Pipeline not fitted. Call fit() first or use fit_transform().")
            return self.fit_transform(df, cross_df, sentiment, symbol_column)

        # Apply feature engineering
        featured_df = self._apply_feature_engineering(df, cross_df, sentiment)

        # Apply normalization
        return self.normalizer.transform(featured_df, symbol_column)

    def fit_transform(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
        symbol_column: str = "symbol",
    ) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data.

        Args:
            df: Input DataFrame
            cross_df: Cross-asset data
            sentiment: Sentiment data
            symbol_column: Name of the symbol column

        Returns:
            Transformed DataFrame
        """
        return self.fit(df, cross_df, sentiment, symbol_column).transform(df, cross_df, sentiment, symbol_column)

    def _apply_feature_engineering(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply all feature generators sequentially."""
        result = df.copy()

        # Apply technical indicators
        self.logger.debug("Applying technical indicators...")
        result = self.technical.calculate_all_indicators(result)

        # Apply microstructure features
        self.logger.debug("Applying microstructure features...")
        result = self.microstructure.add_microstructure_features(result)

        # Apply cross-asset features
        if cross_df is not None:
            self.logger.debug("Applying cross-asset features...")
            result = self.cross_asset.add_cross_asset_features(result, cross_df)

        # Apply alternative data features
        self.logger.debug("Applying alternative data features...")
        # Extract symbol from DataFrame if available
        symbol = None
        if "symbol" in result.columns:
            symbol = result["symbol"].iloc[0] if len(result) > 0 else None

        return self.alternative.add_alternative_features(result, sentiment, symbol)

    def get_feature_names(self) -> list[str]:
        """Return names of all features that can be generated."""
        names: list[str] = []
        names.extend(self.technical.get_feature_names())
        names.extend(self.microstructure.get_feature_names())
        names.extend(self.cross_asset.get_feature_names())
        names.extend(self.alternative.get_feature_names())
        return names

    def get_normalizer_info(self) -> dict:
        """Get information about the fitted normalizer."""
        if hasattr(self, "normalizer"):
            return self.normalizer.get_scaler_info()
        return {}

    def save_pipeline(self, filepath: str) -> None:
        """Save the fitted pipeline to disk."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        # Save normalizer
        normalizer_path = filepath.replace(".pkl", "_normalizer.pkl")
        self.normalizer.save(normalizer_path)

        # Save pipeline configuration
        import pickle

        pipeline_data = {
            "feature_names": self.get_feature_names(),
            "is_fitted": self.is_fitted,
            "normalizer_path": normalizer_path,
        }

        with open(filepath, "wb") as f:
            pickle.dump(pipeline_data, f)

        self.logger.info(f"Pipeline saved to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: str) -> FeaturePipeline:
        """Load a fitted pipeline from disk."""
        import pickle

        with open(filepath, "rb") as f:
            pipeline_data = pickle.load(f)  # nosec B301

        # Load normalizer
        normalizer = FeatureNormalizer.load(pipeline_data["normalizer_path"])

        # Create pipeline
        pipeline = cls(normalizer=normalizer)
        pipeline.is_fitted = pipeline_data["is_fitted"]

        return pipeline
