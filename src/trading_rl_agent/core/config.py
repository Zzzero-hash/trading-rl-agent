"""
Configuration management system using Hydra.

Provides hierarchical configuration management with support for:
- Environment-specific configs
- Dynamic parameter overrides
- Configuration validation
- YAML/JSON support
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.trading_rl_agent.agents.configs import PPOConfig, SACConfig, TD3Config

from .exceptions import ConfigurationError

try:
    # Future imports for hydra support
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# Import logger with fallback
try:
    from .logging import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging if structlog is not available
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "alpaca": "alpaca",
            "yfinance": "yfinance",
            "ccxt": "ccxt",
        },
    )

    # Data storage
    data_path: str = "data/"
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds

    # Feature engineering
    feature_window: int = 50
    technical_indicators: bool = True
    alternative_data: bool = False

    # Real-time data
    real_time_enabled: bool = False
    update_frequency: int = 60  # seconds


@dataclass
class ModelConfig:
    """Model configuration."""

    # CNN+LSTM architecture
    cnn_filters: list = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3
    lstm_units: int = 256
    lstm_layers: int = 2
    dropout_rate: float = 0.2
    batch_normalization: bool = True

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10

    # Model persistence
    model_save_path: str = "models/"
    checkpoint_frequency: int = 10


@dataclass
class AgentConfig:
    """RL agent configuration."""

    # Agent type
    agent_type: str = "sac"  # sac, td3, ppo

    # Agent-specific configurations
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    td3: TD3Config = field(default_factory=TD3Config)

    # Training
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Portfolio limits
    max_position_size: float = 0.1
    max_leverage: float = 1.0
    max_drawdown: float = 0.1

    # VaR parameters
    var_confidence_level: float = 0.05
    var_time_horizon: int = 1

    # Position sizing
    kelly_fraction: float = 0.25
    risk_per_trade: float = 0.02


@dataclass
class ExecutionConfig:
    """Execution configuration."""

    # Broker settings
    broker: str = "alpaca"  # alpaca, ib, paper
    paper_trading: bool = True

    # Order execution
    order_timeout: int = 60  # seconds
    max_slippage: float = 0.001
    commission_rate: float = 0.0

    # Real-time execution
    execution_frequency: int = 5  # seconds
    market_hours_only: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading_system.log"
    structured_logging: bool = True

    # Performance tracking
    metrics_enabled: bool = True
    metrics_frequency: int = 300  # seconds

    # Alerting
    alerts_enabled: bool = True
    email_alerts: bool = False
    slack_alerts: bool = False

    # MLflow tracking
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "http://localhost:5000"


@dataclass
class InfrastructureConfig:
    """Infrastructure and deployment configuration."""

    distributed: bool = False
    num_workers: int = 4
    gpu_enabled: bool = True
    ray_address: str | None = None

    # Storage
    model_registry_path: str = "models"
    experiment_tracking: str = "mlflow"  # mlflow, wandb, tensorboard

    # Monitoring
    enable_monitoring: bool = True
    metrics_port: int = 8080
    health_check_interval: int = 30


@dataclass
class SystemConfig:
    """Complete system configuration."""

    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False

    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)

    # System resources
    use_gpu: bool = False
    max_workers: int = 4
    memory_limit: str = "8GB"


class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: SystemConfig | None = None

    def load_config(
        self,
        config_path: str | Path | None = None,
    ) -> SystemConfig:
        """
        Load configuration from file or create default.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config_path:
            self.config_path = Path(config_path)

        if self.config_path and self.config_path.exists():
            try:
                with self.config_path.open("r") as f:
                    config_dict = yaml.safe_load(f)
                self._config = self._dict_to_config(config_dict)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load config from {self.config_path}: {e}",
                ) from e
        else:
            self._config = SystemConfig()

        return self._config

    def save_config(
        self,
        config: SystemConfig,
        path: str | Path | None = None,
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            path: Output path (optional)
        """
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ConfigurationError("No path specified for saving configuration")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config_to_dict(config)
        with Path(save_path).open("w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_config(self) -> SystemConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def update_config(self, updates: dict[str, Any]) -> SystemConfig:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates

        Returns:
            Updated configuration
        """
        if self._config is None:
            self._config = self.load_config()

        # Apply updates recursively
        self._apply_updates(self._config, updates)
        return self._config

    def _dict_to_config(self, config_dict: dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig."""
        return SystemConfig(
            environment=config_dict.get("environment", "development"),
            debug=config_dict.get("debug", False),
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            agent=AgentConfig(
                agent_type=config_dict.get("agent", {}).get("agent_type", "sac"),
                ppo=PPOConfig(**config_dict.get("agent", {}).get("ppo", {})),
                sac=SACConfig(**config_dict.get("agent", {}).get("sac", {})),
                td3=TD3Config(**config_dict.get("agent", {}).get("td3", {})),
                total_timesteps=config_dict.get("agent", {}).get(
                    "total_timesteps",
                    1_000_000,
                ),
                eval_frequency=config_dict.get("agent", {}).get(
                    "eval_frequency",
                    10_000,
                ),
                save_frequency=config_dict.get("agent", {}).get(
                    "save_frequency",
                    50_000,
                ),
            ),
            risk=RiskConfig(**config_dict.get("risk", {})),
            execution=ExecutionConfig(**config_dict.get("execution", {})),
            monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
            infrastructure=InfrastructureConfig(
                **config_dict.get("infrastructure", {}),
            ),
            use_gpu=config_dict.get("use_gpu", False),
            max_workers=config_dict.get("max_workers", 4),
            memory_limit=config_dict.get("memory_limit", "8GB"),
        )

    def _config_to_dict(self, config: SystemConfig) -> dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        return {
            "environment": config.environment,
            "debug": config.debug,
            "data": config.data.__dict__,
            "model": config.model.__dict__,
            "agent": {
                "agent_type": config.agent.agent_type,
                "ppo": config.agent.ppo.__dict__,
                "sac": config.agent.sac.__dict__,
                "td3": config.agent.td3.__dict__,
                "total_timesteps": config.agent.total_timesteps,
                "eval_frequency": config.agent.eval_frequency,
                "save_frequency": config.agent.save_frequency,
            },
            "risk": config.risk.__dict__,
            "execution": config.execution.__dict__,
            "monitoring": config.monitoring.__dict__,
            "infrastructure": config.infrastructure.__dict__,
            "use_gpu": config.use_gpu,
            "max_workers": config.max_workers,
            "memory_limit": config.memory_limit,
        }

    def _apply_updates(self, config: Any, updates: dict[str, Any]) -> None:
        """Apply updates recursively."""
        for key, value in updates.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(
                    getattr(config, key),
                    "__dict__",
                ):
                    self._apply_updates(getattr(config, key), value)
                else:
                    setattr(config, key, value)
