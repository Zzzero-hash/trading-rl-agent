"""
Custom exceptions for the trading system.
"""


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""

    pass


class DataValidationError(TradingSystemError):
    """Raised when data validation fails."""

    pass


class ModelError(TradingSystemError):
    """Raised when model operations fail."""

    pass


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid."""

    pass


class MarketDataError(TradingSystemError):
    """Raised when market data operations fail."""

    pass


class RiskManagementError(TradingSystemError):
    """Raised when risk management constraints are violated."""

    pass


class ExecutionError(TradingSystemError):
    """Raised when trade execution fails."""

    pass
