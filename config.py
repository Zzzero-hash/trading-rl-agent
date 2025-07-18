"""
Configuration system for Trading RL Agent.

This module provides a comprehensive configuration system using Pydantic models
for type-safe configuration management with support for YAML files, environment
variables, and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Configuration for data sources and processing."""

    primary_source: str = Field(default="yfinance", description="Primary data source")
    backup_source: str = Field(default="yfinance", description="Backup data source")
    real_time_enabled: bool = Field(default=False, description="Enable real-time data feeds")
    update_frequency: int = Field(default=60, description="Data update frequency in seconds")
    symbols: list[str] = Field(default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], description="Trading symbols")
    start_date: str = Field(default="2023-01-01", description="Start date for historical data")
    end_date: str = Field(default="2024-01-01", description="End date for historical data")
    timeframe: str = Field(default="1d", description="Data timeframe")
    feature_window: int = Field(default=50, description="Window size for feature calculation")
    technical_indicators: bool = Field(default=True, description="Enable technical indicators")
    sentiment_features: bool = Field(default=True, description="Enable sentiment analysis features")
    data_path: str = Field(default="data/", description="Data storage directory")
    cache_dir: str = Field(default="data/cache", description="Cache directory")
    cache_ttl_hours: int = Field(default=24, description="Cache time-to-live in hours")


class ModelConfig(BaseModel):
    """Configuration for model architecture and training."""

    type: str = Field(default="cnn_lstm", description="Model type")
    algorithm: str = Field(default="sac", description="RL algorithm")
    cnn_filters: list[int] = Field(default=[64, 128, 256], description="CNN filter sizes")
    cnn_kernel_sizes: list[int] = Field(default=[3, 3, 3], description="CNN kernel sizes")
    cnn_dropout: float = Field(default=0.2, description="CNN dropout rate")
    lstm_units: int = Field(default=128, description="Number of LSTM units")
    lstm_layers: int = Field(default=2, description="Number of LSTM layers")
    lstm_dropout: float = Field(default=0.2, description="LSTM dropout rate")
    dense_units: list[int] = Field(default=[64, 32], description="Dense layer units")
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    epochs: int = Field(default=100, description="Number of training epochs")
    total_timesteps: int = Field(default=1000000, description="Total RL training timesteps")
    model_save_path: str = Field(default="models/", description="Model save directory")
    checkpoint_dir: str = Field(default="models/checkpoints", description="Checkpoint directory")
    device: str = Field(default="auto", description="Device selection")


class AgentConfig(BaseModel):
    """Configuration for RL agent settings."""

    agent_type: str = Field(default="sac", description="Agent type")
    ensemble_size: int = Field(default=1, description="Number of agents in ensemble")
    eval_frequency: int = Field(default=10000, description="Evaluation frequency in timesteps")
    save_frequency: int = Field(default=50000, description="Model save frequency in timesteps")


class RiskConfig(BaseModel):
    """Configuration for risk management."""

    max_position_size: float = Field(default=0.1, description="Maximum position size")
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    max_drawdown: float = Field(default=0.15, description="Maximum drawdown")
    var_confidence_level: float = Field(default=0.05, description="VaR confidence level")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")


class ExecutionConfig(BaseModel):
    """Configuration for trade execution."""

    broker: str = Field(default="alpaca", description="Trading broker")
    paper_trading: bool = Field(default=True, description="Enable paper trading")
    order_timeout: int = Field(default=60, description="Order timeout in seconds")
    max_slippage: float = Field(default=0.001, description="Maximum slippage")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    execution_frequency: int = Field(default=5, description="Execution frequency in seconds")
    market_hours_only: bool = Field(default=True, description="Trade only during market hours")


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and logging."""

    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/trading_system.log", description="Log file path")
    structured_logging: bool = Field(default=True, description="Enable structured logging")
    mlflow_enabled: bool = Field(default=True, description="Enable MLflow tracking")
    tensorboard_enabled: bool = Field(default=True, description="Enable TensorBoard logging")
    metrics_frequency: int = Field(default=300, description="Metrics collection frequency in seconds")
    alerts_enabled: bool = Field(default=True, description="Enable system alerts")


class InfrastructureConfig(BaseModel):
    """Configuration for infrastructure settings."""

    distributed: bool = Field(default=False, description="Enable distributed computing")
    num_workers: int = Field(default=4, description="Number of worker processes")
    gpu_enabled: bool = Field(default=True, description="Enable GPU support")
    ray_address: str | None = Field(default=None, description="Ray cluster address")
    use_gpu: bool = Field(default=False, description="Use GPU for training")
    max_workers: int = Field(default=4, description="Maximum number of workers")
    memory_limit: str = Field(default="8GB", description="Memory limit per worker")


class BacktestConfig(BaseModel):
    """Configuration for backtesting settings."""

    start_date: str = Field(default="2024-01-01", description="Start date for backtesting")
    end_date: str = Field(default="2024-12-31", description="End date for backtesting")
    symbols: list[str] = Field(default=["AAPL", "GOOGL", "MSFT"], description="Symbols for backtesting")
    initial_capital: float = Field(default=100000.0, description="Initial capital for backtesting")
    commission_rate: float = Field(default=0.001, description="Commission rate for backtesting")
    slippage_rate: float = Field(default=0.0001, description="Slippage rate for backtesting")
    max_position_size: float = Field(default=0.1, description="Maximum position size")
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")
    output_dir: str = Field(default="backtest_results", description="Output directory for backtest results")
    save_trades: bool = Field(default=True, description="Save detailed trade information")


class Settings(BaseSettings):
    """Main settings class for the Trading RL Agent."""

    model_config = SettingsConfigDict(
        env_prefix="TRADING_RL_AGENT_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Environment settings
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Component configurations
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # API keys (loaded from environment variables)
    alpaca_api_key: str | None = Field(default=None, description="Alpaca API key")
    alpaca_secret_key: str | None = Field(default=None, description="Alpaca secret key")
    alpaca_base_url: str | None = Field(default=None, description="Alpaca base URL")
    alphavantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")
    newsapi_key: str | None = Field(default=None, description="News API key")
    social_api_key: str | None = Field(default=None, description="Social media API key")

    @field_validator("alpaca_api_key", "alpaca_secret_key", "alphavantage_api_key", "newsapi_key", "social_api_key")
    @classmethod
    def validate_api_keys(cls, v: str | None) -> str | None:
        """Validate API keys - empty strings become None."""
        if v == "":
            return None
        return v

    def get_api_credentials(self, exchange: str) -> dict[str, str | None]:
        """Get API credentials for a specific exchange."""
        if exchange.lower() == "alpaca":
            return {
                "api_key": self.alpaca_api_key,
                "secret_key": self.alpaca_secret_key,
                "base_url": self.alpaca_base_url,
            }
        if exchange.lower() == "alphavantage":
            return {"api_key": self.alphavantage_api_key}
        raise ValueError(f"Unknown exchange: {exchange}")


# Global settings cache
_settings_cache: Settings | None = None


def load_settings(config_path: str | Path | None = None, env_file: str | Path | None = None) -> Settings:
    """
    Load settings from configuration file and environment variables.

    Args:
        config_path: Path to YAML configuration file
        env_file: Path to .env file

    Returns:
        Settings object with loaded configuration
    """
    global _settings_cache

    # Load YAML configuration if provided
    config_data: dict[str, Any] = {}
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e

    # Load environment file if provided
    env_file_path = None
    if env_file:
        env_file_path = Path(env_file)
        if not env_file_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file_path}")

    # Create settings with environment file
    settings_kwargs = {}
    if env_file_path:
        settings_kwargs["_env_file"] = str(env_file_path)

    # Create settings object
    settings = Settings(**settings_kwargs)

    # Override with YAML config
    if config_data:
        # Update nested configurations
        for key, value in config_data.items():
            if hasattr(settings, key) and isinstance(value, dict):
                current_config = getattr(settings, key)
                if isinstance(current_config, BaseModel):
                    # Update the nested config
                    updated_config = current_config.model_copy(update=value)
                    setattr(settings, key, updated_config)
            elif hasattr(settings, key):
                setattr(settings, key, value)

    # Cache the settings
    _settings_cache = settings
    return settings


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Cached Settings object
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache


def clear_settings_cache() -> None:
    """Clear the settings cache."""
    global _settings_cache
    _settings_cache = None
    get_settings.cache_clear()


# Convenience function for backward compatibility
def load_config(config_path: str | Path | None = None) -> Settings:
    """Alias for load_settings for backward compatibility."""
    return load_settings(config_path=config_path)
