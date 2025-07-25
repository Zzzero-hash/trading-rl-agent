"""
Configuration validation and consistency checks.

This module provides validation utilities to ensure configuration consistency
across different components of the trading system.
"""

from typing import Any

from .logging import get_logger
from .unified_config import UnifiedConfig

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""


class ConfigValidator:
    """Validates configuration consistency and completeness."""

    def __init__(self, config: UnifiedConfig):
        """Initialize validator with configuration."""
        self.config = config
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all validations pass, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Run individual validation methods
        self._validate_data_config()
        self._validate_model_config()
        self._validate_live_trading_config()
        self._validate_api_credentials()
        self._validate_consistency()

        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration validation error: {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration validation warning: {warning}")

        return len(self.errors) == 0

    def _validate_data_config(self) -> None:
        """Validate data configuration settings."""
        data = self.config.data

        # Check symbol list
        if not data.symbols:
            self.errors.append("No symbols specified in data configuration")

        # Check date ranges
        if data.start_date >= data.end_date:
            self.errors.append("Data start_date must be before end_date")

        # Check sequence length vs feature window
        if data.sequence_length <= 0:
            self.errors.append("Data sequence_length must be positive")

        # Check data paths
        if not data.data_path:
            self.warnings.append("Data path not specified, using default")

        # Check cache settings
        if data.cache_ttl_hours <= 0:
            self.warnings.append("Cache TTL is 0 or negative, caching effectively disabled")

    def _validate_model_config(self) -> None:
        """Validate model configuration settings."""
        model = self.config.model

        # Check learning rate
        if model.learning_rate <= 0 or model.learning_rate > 1:
            self.errors.append("Model learning_rate must be between 0 and 1")

        # Check batch size
        if model.batch_size <= 0:
            self.errors.append("Model batch_size must be positive")

        # Check architecture consistency
        if model.type == "cnn_lstm":
            if len(model.cnn_filters) != len(model.cnn_kernel_sizes):
                self.errors.append("CNN filters and kernel sizes must have same length")

            if model.lstm_units <= 0:
                self.errors.append("LSTM units must be positive")

        # Check dropout rates
        if not (0 <= model.cnn_dropout <= 1):
            self.errors.append("CNN dropout must be between 0 and 1")

        if not (0 <= model.lstm_dropout <= 1):
            self.errors.append("LSTM dropout must be between 0 and 1")

        # Check training parameters
        if model.epochs <= 0:
            self.errors.append("Model epochs must be positive")

        if model.early_stopping_patience <= 0:
            self.warnings.append("Early stopping patience is 0 or negative")

    def _validate_live_trading_config(self) -> None:
        """Validate live trading configuration settings."""
        live = self.config.live

        # Check position sizing
        if not (0 < live.max_position_size <= 1):
            self.errors.append("Max position size must be between 0 and 1")

        if live.max_leverage < 1:
            self.errors.append("Max leverage must be >= 1")

        # Check risk parameters
        if not (0 < live.max_drawdown <= 1):
            self.errors.append("Max drawdown must be between 0 and 1")

        if not (0 < live.stop_loss_pct <= 1):
            self.errors.append("Stop loss percentage must be between 0 and 1")

        if not (0 < live.take_profit_pct <= 1):
            self.errors.append("Take profit percentage must be between 0 and 1")

        # Check execution parameters
        if live.order_timeout <= 0:
            self.errors.append("Order timeout must be positive")

        if live.max_slippage < 0:
            self.errors.append("Max slippage cannot be negative")

        # Check capital
        if live.initial_capital <= 0:
            self.errors.append("Initial capital must be positive")

        # Warn about paper trading in production
        if self.config.environment == "production" and live.paper_trading:
            self.warnings.append("Paper trading enabled in production environment")

    def _validate_api_credentials(self) -> None:
        """Validate API credentials availability."""
        # Check Alpaca credentials if using Alpaca
        if self.config.live.exchange.lower() == "alpaca":
            if not self.config.alpaca_api_key:
                self.errors.append("Alpaca API key required but not provided")
            if not self.config.alpaca_secret_key:
                self.errors.append("Alpaca secret key required but not provided")

        # Check news API if sentiment features enabled
        if self.config.data.sentiment_features and not self.config.newsapi_key:
            self.warnings.append("Sentiment features enabled but NewsAPI key not provided")

    def _validate_consistency(self) -> None:
        """Validate consistency between different configuration sections."""
        # Check symbol consistency between data and live trading
        data_symbols = set(self.config.data.symbols)
        live_symbols = set(self.config.live.symbols)

        if not data_symbols.intersection(live_symbols):
            self.warnings.append("No common symbols between data and live trading configs")

        # Check date consistency
        if self.config.backtest.start_date < self.config.data.start_date:
            self.warnings.append("Backtest start date is before data collection start date")

        # Check infrastructure vs model requirements
        if self.config.model.device == "cuda" and not self.config.infrastructure.gpu_enabled:
            self.errors.append("Model requires CUDA but GPU is not enabled in infrastructure")

        # Check distributed computing consistency
        if self.config.infrastructure.distributed and not self.config.infrastructure.ray_address:
            self.warnings.append("Distributed computing enabled but no Ray address specified")

        # Check memory requirements
        if self.config.model.batch_size > 128 and self.config.infrastructure.memory_limit == "8GB":
            self.warnings.append("Large batch size with limited memory may cause issues")


def validate_config(config: UnifiedConfig) -> bool:
    """
    Validate a configuration object.

    Args:
        config: Configuration to validate

    Returns:
        True if validation passes, False otherwise

    Raises:
        ConfigValidationError: If critical validation errors are found
    """
    validator = ConfigValidator(config)
    is_valid = validator.validate_all()

    if not is_valid and validator.errors:
        error_msg = f"Configuration validation failed with {len(validator.errors)} errors"
        raise ConfigValidationError(error_msg)

    return is_valid


def check_config_completeness(config: UnifiedConfig) -> dict[str, Any]:
    """
    Check configuration completeness and return a report.

    Args:
        config: Configuration to check

    Returns:
        Dictionary with completeness report
    """
    report: dict[str, Any] = {
        "complete": True,
        "missing_optional": [],
        "missing_required": [],
        "environment_vars_missing": []
    }

    # Check required API keys based on configuration
    if config.live.exchange.lower() == "alpaca":
        if not config.alpaca_api_key:
            report["missing_required"].append("alpaca_api_key")
            report["environment_vars_missing"].append("TRADING_RL_AGENT_ALPACA_API_KEY")
        if not config.alpaca_secret_key:
            report["missing_required"].append("alpaca_secret_key")
            report["environment_vars_missing"].append("TRADING_RL_AGENT_ALPACA_SECRET_KEY")

    # Check optional features
    if config.data.sentiment_features and not config.newsapi_key:
        report["missing_optional"].append("newsapi_key")
        report["environment_vars_missing"].append("TRADING_RL_AGENT_NEWSAPI_KEY")

    if config.monitoring.mlflow_enabled:
        # MLflow doesn't require API keys but might need specific configuration
        pass

    # Update completeness status
    report["complete"] = len(report["missing_required"]) == 0

    return report
