"""
Data processing and ingestion modules for trading data.

This package provides various data sources and processing utilities
for financial market data, including historical data, live feeds,
sentiment analysis, and synthetic data generation.
"""

# Import main classes that are actually implemented
# Alpaca integration
from .alpaca_integration import (
    AlpacaConfig,
    AlpacaConnectionError,
    AlpacaDataError,
    AlpacaError,
    AlpacaIntegration,
    AlpacaOrderError,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioPosition,
    create_alpaca_config_from_env,
)
from .forex_sentiment import ForexSentimentData
from .sentiment import (
    NewsSentimentProvider,
    SentimentAnalyzer,
    SentimentConfig,
    SentimentData,
    SocialSentimentProvider,
    get_sentiment_score,
    update_sentiment,
)

# Note: Some modules are stubs or have different class names
# Update imports as modules are implemented

__all__ = [
    "AlpacaConfig",
    "AlpacaConnectionError",
    "AlpacaDataError",
    "AlpacaError",
    "AlpacaIntegration",
    "AlpacaOrderError",
    "ForexSentimentData",
    "MarketData",
    "NewsSentimentProvider",
    "OrderRequest",
    "OrderSide",
    "OrderType",
    "PortfolioPosition",
    "SentimentAnalyzer",
    "SentimentConfig",
    "SentimentData",
    "SocialSentimentProvider",
    "create_alpaca_config_from_env",
    "get_sentiment_score",
    "update_sentiment",
]
