"""
Feature engineering pipeline for trading system.

This module provides comprehensive feature engineering capabilities including:
- Technical indicators (using TA-Lib)
- Market microstructure features
- Cross-asset correlation features
- Alternative data integration
- Real-time feature calculation
"""

from .technical_indicators import TechnicalIndicators
from .market_microstructure import MarketMicrostructure
from .cross_asset import CrossAssetFeatures
from .alternative_data import AlternativeDataFeatures
from .pipeline import FeaturePipeline

__all__ = [
    "TechnicalIndicators",
    "MarketMicrostructure",
    "CrossAssetFeatures",
    "AlternativeDataFeatures",
    "FeaturePipeline",
]
