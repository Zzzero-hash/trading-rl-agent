"""
Unified Configuration Schema for Trading RL Agent.

This module provides Pydantic models for a unified configuration system that consolidates
all settings for data, model, backtest, and live trading operations.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourcesConfig(BaseModel):
    """Data sources configuration."""

    primary: str = Field(default="yfinance", description="Primary data source")
    backup: str = Field(default="yfinance", description="Backup data source")
    real_time_enabled: bool = Field(default=False, description="Enable real-time data feeds")
    update_frequency: int = Field(default=60, description="Data update frequency in seconds")


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    # Data sources
    sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)

    # Data collection
    symbols: list[str] = Field(
        default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], description="Trading symbols to collect data for"
    )
    start_date: str = Field(default="2023-01-01", description="Start date for data collection")
    end_date: str = Field(default="2024-01-01", description="End date for data collection")
    timeframe: str = Field(default="1d", description="Data timeframe (1d, 1h, 5m, 1m)")

    # Dataset composition
    real_data_ratio: float = Field(default=0.95, description="Ratio of real vs synthetic data")
    min_samples_per_symbol: int = Field(default=2500, description="Minimum samples per symbol")
    sequence_length: int = Field(default=60, description="Sequence length for time series")
    prediction_horizon: int = Field(default=1, description="Prediction horizon")
    overlap_ratio: float = Field(default=0.8, description="Overlap ratio between sequences")

    # Feature engineering
    technical_indicators: bool = Field(default=True, description="Enable technical indicators")
    sentiment_features: bool = Field(default=True, description="Enable sentiment features")
    market_regime_features: bool = Field(default=True, description="Enable market regime features")
    alternative_data: bool = Field(default=False, description="Enable alternative data sources")

    # Data quality
    outlier_threshold: float = Field(default=5.0, description="Outlier detection threshold")
    missing_value_threshold: float = Field(default=0.25, description="Missing value threshold")

    # Cache and storage
    cache_dir: str = Field(default="data/cache", description="Cache directory")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    data_path: str = Field(default="data/", description="Data storage path")
    output_dir: str = Field(default="outputs/datasets", description="Output directory")

    # Performance
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    chunk_size: int = Field(default=1000, description="Data chunk size")
    use_memory_mapping: bool = Field(default=True, description="Use memory mapping")


class ModelConfig(BaseModel):
    """Model configuration."""

    # Model type and algorithm
    type: str = Field(default="cnn_lstm", description="Model type: cnn_lstm, rl, hybrid")
    algorithm: str = Field(default="sac", description="RL algorithm: ppo, sac, td3")

    # CNN+LSTM architecture
    cnn_filters: list[int] = Field(default=[64, 128, 256], description="CNN filter sizes")
    cnn_kernel_sizes: list[int] = Field(default=[3, 3, 3], description="CNN kernel sizes")
    cnn_dropout: float = Field(default=0.2, description="CNN dropout rate")
    lstm_units: int = Field(default=128, description="LSTM units")
    lstm_layers: int = Field(default=2, description="LSTM layers")
    lstm_dropout: float = Field(default=0.2, description="LSTM dropout rate")
    dense_units: list[int] = Field(default=[64, 32], description="Dense layer units")
    output_dim: int = Field(default=1, description="Output dimension")
    activation: str = Field(default="relu", description="Activation function")
    use_attention: bool = Field(default=False, description="Use attention mechanism")

    # Training parameters
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float = Field(default=1e-5, description="Weight decay")
    optimizer: str = Field(default="adam", description="Optimizer: adam, sgd, rmsprop")
    epochs: int = Field(default=100, description="Training epochs")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")
    reduce_lr_patience: int = Field(default=5, description="LR reduction patience")
    reduce_lr_factor: float = Field(default=0.5, description="LR reduction factor")

    # RL-specific settings
    total_timesteps: int = Field(default=1000000, description="Total training timesteps")
    eval_frequency: int = Field(default=10000, description="Evaluation frequency")
    save_frequency: int = Field(default=50000, description="Save frequency")

    # Loss and metrics
    loss_function: str = Field(default="mse", description="Loss function: mse, mae, huber")
    metrics: list[str] = Field(default=["mae", "rmse", "correlation", "r2_score"], description="Evaluation metrics")

    # Model persistence
    model_save_path: str = Field(default="models/", description="Model save path")
    checkpoint_dir: str = Field(default="models/checkpoints", description="Checkpoint directory")
    save_best_only: bool = Field(default=True, description="Save only best model")
    save_frequency_epochs: int = Field(default=5, description="Save frequency in epochs")

    # Device and performance
    device: str = Field(default="auto", description="Device: auto, cpu, cuda")
    num_workers: int = Field(default=4, description="Number of workers")
    mixed_precision: bool = Field(default=True, description="Use mixed precision")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    # Test period
    start_date: str = Field(default="2024-01-01", description="Backtest start date")
    end_date: str = Field(default="2024-12-31", description="Backtest end date")

    # Instruments
    symbols: list[str] = Field(default=["AAPL", "GOOGL", "MSFT"], description="Symbols to backtest")

    # Capital and position sizing
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    slippage_rate: float = Field(default=0.0001, description="Slippage rate")

    # Risk management
    max_position_size: float = Field(default=0.1, description="Maximum position size")
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")

    # Evaluation metrics
    metrics: list[str] = Field(
        default=["total_return", "sharpe_ratio", "max_drawdown", "win_rate"], description="Backtest metrics"
    )

    # Output settings
    output_dir: str = Field(default="backtest_results", description="Output directory")
    save_trades: bool = Field(default=True, description="Save trade history")
    save_portfolio: bool = Field(default=True, description="Save portfolio history")
    generate_plots: bool = Field(default=True, description="Generate plots")
    plot_dir: str = Field(default="plots/backtest", description="Plot directory")


class LiveTradingConfig(BaseModel):
    """Live trading configuration."""

    # Exchange and broker settings
    exchange: str = Field(default="alpaca", description="Trading exchange")
    paper_trading: bool = Field(default=True, description="Enable paper trading")

    # Trading symbols
    symbols: list[str] = Field(default=["AAPL", "GOOGL", "MSFT"], description="Trading symbols")

    # Execution settings
    order_timeout: int = Field(default=60, description="Order timeout in seconds")
    max_slippage: float = Field(default=0.001, description="Maximum slippage")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    execution_frequency: int = Field(default=5, description="Execution frequency in seconds")
    market_hours_only: bool = Field(default=True, description="Trade only during market hours")

    # Risk management
    max_position_size: float = Field(default=0.1, description="Maximum position size")
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    max_drawdown: float = Field(default=0.15, description="Maximum drawdown")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")

    # VaR and risk metrics
    var_confidence_level: float = Field(default=0.05, description="VaR confidence level")
    var_time_horizon: int = Field(default=1, description="VaR time horizon")
    kelly_fraction: float = Field(default=0.25, description="Kelly criterion fraction")
    risk_per_trade: float = Field(default=0.02, description="Risk per trade")

    # Portfolio settings
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    rebalance_frequency: int = Field(default=3600, description="Rebalance frequency in seconds")
    max_positions: int = Field(default=10, description="Maximum number of positions")

    # Monitoring and alerts
    monitoring_interval: int = Field(default=60, description="Monitoring interval in seconds")
    alerts_enabled: bool = Field(default=True, description="Enable alerts")
    email_alerts: bool = Field(default=False, description="Enable email alerts")
    slack_alerts: bool = Field(default=False, description="Enable Slack alerts")


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/trading_system.log", description="Log file path")
    structured_logging: bool = Field(default=True, description="Use structured logging")

    # Experiment tracking
    experiment_name: str = Field(default="trading_rl_agent", description="Experiment name")
    tracking_uri: str = Field(default="sqlite:///mlruns.db", description="MLflow tracking URI")
    mlflow_enabled: bool = Field(default=True, description="Enable MLflow tracking")
    tensorboard_enabled: bool = Field(default=True, description="Enable TensorBoard")
    tensorboard_log_dir: str = Field(default="logs/tensorboard", description="TensorBoard log directory")

    # Performance tracking
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_frequency: int = Field(default=300, description="Metrics collection frequency")
    health_check_interval: int = Field(default=30, description="Health check interval")

    # Alerting
    alerts_enabled: bool = Field(default=True, description="Enable alerts")
    email_alerts: bool = Field(default=False, description="Enable email alerts")
    slack_alerts: bool = Field(default=False, description="Enable Slack alerts")


class InfrastructureConfig(BaseModel):
    """Infrastructure configuration."""

    # Distributed computing
    distributed: bool = Field(default=False, description="Enable distributed computing")
    ray_address: str | None = Field(default=None, description="Ray cluster address")
    num_workers: int = Field(default=4, description="Number of workers")
    gpu_enabled: bool = Field(default=True, description="Enable GPU support")

    # Storage
    model_registry_path: str = Field(default="models", description="Model registry path")
    experiment_tracking: str = Field(default="mlflow", description="Experiment tracking system")

    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    metrics_port: int = Field(default=8080, description="Metrics port")
    health_check_interval: int = Field(default=30, description="Health check interval")

    # System resources
    use_gpu: bool = Field(default=False, description="Use GPU")
    max_workers: int = Field(default=4, description="Maximum workers")
    memory_limit: str = Field(default="8GB", description="Memory limit")


class HyperoptConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    enabled: bool = Field(default=False, description="Enable hyperparameter optimization")
    n_trials: int = Field(default=50, description="Number of optimization trials")
    timeout: int = Field(default=3600, description="Optimization timeout in seconds")

    # Search space
    search_space: dict[str, Any] = Field(
        default={
            "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
        },
        description="Hyperparameter search space",
    )


class ProductionConfig(BaseModel):
    """Production configuration."""

    # Model serving
    model_format: str = Field(default="torchscript", description="Model format")
    model_version: str = Field(default="v1.0.0", description="Model version")

    # API configuration
    api_host: str = Field(default="127.0.0.1", description="API host (use 0.0.0.0 only in production)")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="API workers")

    # Monitoring
    health_check_interval: int = Field(default=30, description="Health check interval")
    metrics_export_interval: int = Field(default=60, description="Metrics export interval")


class UnifiedConfig(BaseSettings):
    """Unified configuration for Trading RL Agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Environment and system settings
    environment: str = Field(default="production", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # Component configurations
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    live: LiveTradingConfig = Field(default_factory=LiveTradingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    hyperopt: HyperoptConfig = Field(default_factory=HyperoptConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)

    # Sensitive fields that should be loaded from environment variables
    # These are marked as sensitive and will be loaded from env vars
    alpaca_api_key: str | None = Field(default=None, description="Alpaca API key")
    alpaca_secret_key: str | None = Field(default=None, description="Alpaca secret key")
    alpaca_base_url: str | None = Field(default=None, description="Alpaca base URL")
    alphavantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")
    newsapi_key: str | None = Field(default=None, description="News API key")
    social_api_key: str | None = Field(default=None, description="Social media API key")

    @field_validator("alpaca_api_key", "alpaca_secret_key", "newsapi_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v: Any) -> Any:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result: dict[str, Any] = self.model_dump(exclude_none=True)
        return result

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "UnifiedConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        # Convert to dict and exclude sensitive fields
        config_dict = self.model_dump(
            exclude={
                "alpaca_api_key",
                "alpaca_secret_key",
                "alpaca_base_url",
                "alphavantage_api_key",
                "newsapi_key",
                "social_api_key",
            }
        )

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Convenience function to load configuration
def load_config(config_path: str | Path | None = None) -> UnifiedConfig:
    """Load unified configuration from file or environment."""
    if config_path:
        return UnifiedConfig.from_yaml(config_path)
    return UnifiedConfig()
