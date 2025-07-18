"""
Data processing and ingestion modules for trading data.

This package provides various data sources and processing utilities
for financial market data, including historical data, live feeds,
sentiment analysis, and synthetic data generation.
"""

# Import main classes that are actually implemented
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

# Alpaca integration
from .alpaca_integration import (
    AlpacaIntegration,
    AlpacaConfig,
    OrderRequest,
    OrderType,
    OrderSide,
    MarketData,
    PortfolioPosition,
    AlpacaError,
    AlpacaConnectionError,
    AlpacaOrderError,
    AlpacaDataError,
    create_alpaca_config_from_env,
)

# Note: Some modules are stubs or have different class names
# Update imports as modules are implemented

__all__ = [
    "ForexSentimentData",
    "NewsSentimentProvider",
    "SentimentAnalyzer",
    "SentimentConfig",
    "SentimentData",
    "SocialSentimentProvider",
    "get_sentiment_score",
    "update_sentiment",
    # Alpaca integration
    "AlpacaIntegration",
    "AlpacaConfig",
    "OrderRequest",
    "OrderType",
    "OrderSide",
    "MarketData",
    "PortfolioPosition",
    "AlpacaError",
    "AlpacaConnectionError",
    "AlpacaOrderError",
    "AlpacaDataError",
    "create_alpaca_config_from_env",
]
