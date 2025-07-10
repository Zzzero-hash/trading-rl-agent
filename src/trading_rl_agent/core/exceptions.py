"""
Custom exceptions for the trading system.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""


class DataValidationError(TradingSystemError):
    """Raised when data validation fails."""


class ModelError(TradingSystemError):
    """Raised when model operations fail."""


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""


class MarketDataError(TradingSystemError):
    """Raised when market data operations fail."""


class RiskManagementError(TradingSystemError):
    """Raised when risk management constraints are violated."""


class ExecutionError(TradingSystemError):
    """Raised when trade execution fails."""
