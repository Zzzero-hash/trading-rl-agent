"""Feature pipeline composing multiple feature generators."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..core.logging import get_logger
from .alternative_data import AlternativeDataFeatures
from .cross_asset import CrossAssetFeatures
from .market_microstructure import MarketMicrostructure
from .technical_indicators import TechnicalIndicators


class FeaturePipeline:
    """Compose various feature generators into a single pipeline."""

    def __init__(
        self,
        technical: TechnicalIndicators | None = None,
        microstructure: MarketMicrostructure | None = None,
        cross_asset: CrossAssetFeatures | None = None,
        alternative: AlternativeDataFeatures | None = None,
    ) -> None:
        self.technical = technical or TechnicalIndicators()
        self.microstructure = microstructure or MarketMicrostructure()
        self.cross_asset = cross_asset or CrossAssetFeatures()
        self.alternative = alternative or AlternativeDataFeatures()
        self.logger = get_logger(self.__class__.__name__)

    def transform(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply all feature generators sequentially. Modifies the input DataFrame in place."""
        self.technical.calculate_all_indicators(df)
        self.microstructure.add_microstructure_features(df)
        if cross_df is not None:
            self.cross_asset.add_cross_asset_features(df, cross_df)
        self.alternative.add_alternative_features(df, sentiment)
        return df

    def get_feature_names(self) -> list[str]:
        """Return names of all features that can be generated."""
        names: list[str] = []
        names.extend(self.technical.get_feature_names())
        names.extend(self.microstructure.get_feature_names())
        names.extend(self.cross_asset.get_feature_names())
        names.extend(self.alternative.get_feature_names())
        return names
