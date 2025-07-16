"""
Configuration management for Trading RL Agent.

This module provides a centralized configuration system that:
1. Loads default settings (embedded)
2. Overlays configuration from YAML files
3. Parses .env files for API keys and secrets
4. Allows environment variable overrides with TRADING_RL_AGENT_ prefix
5. Uses Pydantic for validation
6. Caches settings globally
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    # Data sources
    primary_source: str = Field(default="yfinance", description="Primary data source")
    backup_source: str = Field(default="yfinance", description="Backup data source")
    real_time_enabled: bool = Field(default=False, description="Enable real-time data")
    update_frequency: int = Field(default=60, description="Update frequency in seconds")

    # Data collection
    symbols: list[str] = Field(default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], description="Trading symbols")
    start_date: str = Field(default="2023-01-01", description="Start date")
    end_date: str = Field(default="2024-01-01", description="End date")
    timeframe: str = Field(default="1d", description="Data timeframe")

    # Feature engineering
    feature_window: int = Field(default=50, description="Feature window size")
    technical_indicators: bool = Field(default=True, description="Enable technical indicators")
    sentiment_features: bool = Field(default=True, description="Enable sentiment features")

    # Storage
    data_path: str = Field(default="data/", description="Data storage path")
    cache_dir: str = Field(default="data/cache", description="Cache directory")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")


class ModelConfig(BaseModel):
    """Model configuration."""

    # Model type
    type: str = Field(default="cnn_lstm", description="Model type")
    algorithm: str = Field(default="sac", description="RL algorithm")

    # Architecture
    cnn_filters: list[int] = Field(default=[64, 128, 256], description="CNN filters")
    cnn_kernel_sizes: list[int] = Field(default=[3, 3, 3], description="CNN kernel sizes")
    cnn_dropout: float = Field(default=0.2, description="CNN dropout rate")
    lstm_units: int = Field(default=128, description="LSTM units")
    lstm_layers: int = Field(default=2, description="LSTM layers")
    lstm_dropout: float = Field(default=0.2, description="LSTM dropout rate")
    dense_units: list[int] = Field(default=[64, 32], description="Dense units")

    # Training
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    epochs: int = Field(default=100, description="Training epochs")
    total_timesteps: int = Field(default=1000000, description="Total timesteps")

    # Persistence
    model_save_path: str = Field(default="models/", description="Model save path")
    checkpoint_dir: str = Field(default="models/checkpoints", description="Checkpoint directory")

    # Device
    device: str = Field(default="auto", description="Device: auto, cpu, cuda")


class AgentConfig(BaseModel):
    """RL agent configuration."""

    agent_type: str = Field(default="sac", description="Agent type")
    ensemble_size: int = Field(default=1, description="Ensemble size")
    eval_frequency: int = Field(default=10000, description="Evaluation frequency")
    save_frequency: int = Field(default=50000, description="Save frequency")


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size: float = Field(default=0.1, description="Max position size")
    max_leverage: float = Field(default=1.0, description="Max leverage")
    max_drawdown: float = Field(default=0.15, description="Max drawdown")
    var_confidence_level: float = Field(default=0.05, description="VaR confidence level")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    broker: str = Field(default="alpaca", description="Trading broker")
    paper_trading: bool = Field(default=True, description="Paper trading mode")
    order_timeout: int = Field(default=60, description="Order timeout in seconds")
    max_slippage: float = Field(default=0.001, description="Max slippage")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    execution_frequency: int = Field(default=5, description="Execution frequency")
    market_hours_only: bool = Field(default=True, description="Market hours only")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/trading_system.log", description="Log file")
    structured_logging: bool = Field(default=True, description="Structured logging")
    mlflow_enabled: bool = Field(default=True, description="MLflow tracking")
    tensorboard_enabled: bool = Field(default=True, description="TensorBoard")
    metrics_frequency: int = Field(default=300, description="Metrics frequency")
    alerts_enabled: bool = Field(default=True, description="Enable alerts")


class InfrastructureConfig(BaseModel):
    """Infrastructure configuration."""

    distributed: bool = Field(default=False, description="Distributed computing")
    num_workers: int = Field(default=4, description="Number of workers")
    gpu_enabled: bool = Field(default=True, description="GPU support")
    ray_address: str | None = Field(default=None, description="Ray cluster address")
    use_gpu: bool = Field(default=False, description="Use GPU")
    max_workers: int = Field(default=4, description="Max workers")
    memory_limit: str = Field(default="8GB", description="Memory limit")


class Settings(BaseSettings):
    """Main settings class with environment variable support."""

    model_config = SettingsConfigDict(env_prefix="TRADING_RL_AGENT_", case_sensitive=False, extra="ignore")

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

    # API keys and secrets (loaded from environment variables)
    alpaca_api_key: str | None = Field(default=None, description="Alpaca API key")
    alpaca_secret_key: str | None = Field(default=None, description="Alpaca secret key")
    alpaca_base_url: str | None = Field(default=None, description="Alpaca base URL")
    alphavantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")
    newsapi_key: str | None = Field(default=None, description="News API key")
    social_api_key: str | None = Field(default=None, description="Social media API key")

    @field_validator("alpaca_api_key", "alpaca_secret_key", "newsapi_key", "social_api_key")
    @classmethod
    def validate_api_keys(cls, v: str | None) -> str | None:
        """Validate API keys are not empty strings."""
        if v == "":
            return None
        return v

    def get_api_credentials(self, exchange: str) -> dict[str, str]:
        """Get API credentials for a specific exchange."""
        credentials = {}

        if exchange.lower() == "alpaca":
            if self.alpaca_api_key:
                credentials["api_key"] = self.alpaca_api_key
            if self.alpaca_secret_key:
                credentials["secret_key"] = self.alpaca_secret_key
            if self.alpaca_base_url:
                credentials["base_url"] = self.alpaca_base_url
        elif exchange.lower() == "alphavantage":
            if self.alphavantage_api_key:
                credentials["api_key"] = self.alphavantage_api_key

        return credentials


# Global cache for settings
_settings_cache: Settings | None = None


def load_settings(config_path: Path | None = None, env_file: Path | None = None) -> Settings:
    """
    Load settings with the following precedence:
    1. Default settings (embedded)
    2. YAML config file (if provided)
    3. .env file (if provided)
    4. Environment variables (TRADING_RL_AGENT_ prefix)

    Args:
        config_path: Path to YAML configuration file
        env_file: Path to .env file

    Returns:
        Settings object with all configuration loaded

    Raises:
        ValueError: If config file is invalid
        FileNotFoundError: If config file doesn't exist
    """
    global _settings_cache

    # Load .env file first if provided
    if env_file:
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv(env_file, override=True)

    # Start with default settings
    settings_dict = {}

    # Overlay YAML config if provided
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    settings_dict.update(yaml_config)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {e}") from e

    # Create settings object (environment variables will override)
    try:
        settings = Settings(**settings_dict)
        # Only cache if no specific files were provided
        if config_path is None and env_file is None:
            _settings_cache = settings
        return settings
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def get_settings() -> Settings:
    """
    Get the cached settings or load default settings.

    Returns:
        Settings object
    """
    global _settings_cache

    if _settings_cache is None:
        _settings_cache = load_settings()

    return _settings_cache


def clear_settings_cache() -> None:
    """Clear the settings cache to force reload."""
    global _settings_cache
    _settings_cache = None
