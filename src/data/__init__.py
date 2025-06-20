"""
Data processing and ingestion modules for trading data.

This package provides various data sources and processing utilities
for financial market data, including historical data, live feeds,
sentiment analysis, and synthetic data generation.
"""

# Import main classes that are actually implemented
from .sentiment import SentimentAnalyzer, SentimentData, SentimentConfig
from .forex_sentiment import ForexSentimentData

# Note: Some modules are stubs or have different class names
# Update imports as modules are implemented

__all__ = [
    "SentimentAnalyzer",
    "SentimentData",
    "SentimentConfig",
    "ForexSentimentData",
]
