"""
Feature engineering pipeline for trading system.

This module provides comprehensive feature engineering capabilities including:
- Technical indicators powered by pandas-ta
- Market microstructure features
- Cross-asset correlation features
- Alternative data integration
- Real-time feature calculation
"""

from .alternative_data import AlternativeDataFeatures
from .cross_asset import CrossAssetFeatures
from .market_microstructure import MarketMicrostructure
from .pipeline import FeaturePipeline
from .technical_indicators import TechnicalIndicators

__all__ = [
    "AlternativeDataFeatures",
    "CrossAssetFeatures",
    "FeaturePipeline",
    "MarketMicrostructure",
    "TechnicalIndicators",
]
