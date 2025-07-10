"""
Configuration management for the trading system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    sources: list[str] = field(default_factory=lambda: ["yfinance"])
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    interval: str = "1d"
    features: list[str] = field(default_factory=lambda: ["close", "volume", "returns"])
    cache_dir: str = "data/cache"
    max_cache_size_gb: float = 10.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    type: str = "cnn_lstm"
    input_dim: int = 78
    sequence_length: int = 60
    prediction_horizon: int = 1
    uncertainty_estimation: bool = True
    ensemble_size: int = 3

    # CNN-LSTM specific
    cnn_filters: list[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    lstm_units: int = 256
    dropout_rate: float = 0.2
    use_attention: bool = True


@dataclass
class RLConfig:
    """Reinforcement Learning configuration."""

    algorithm: str = "SAC"
    framework: str = "ray"  # ray, stable_baselines3

    # Environment settings
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    max_position: float = 1.0
    window_size: int = 60

    # Training settings
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    n_eval_episodes: int = 10

    # Algorithm specific hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    tau: float = 0.005  # For SAC
    gamma: float = 0.99


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_drawdown: float = 0.2
    var_confidence: float = 0.05
    position_limit: float = 0.1
    concentration_limit: float = 0.3
    stop_loss: float = 0.05
    take_profit: float = 0.15
    risk_free_rate: float = 0.02


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
    """Master system configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)

    # Global settings
    random_seed: int = 42
    log_level: str = "INFO"
    environment: str = "development"  # development, staging, production


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
                with Path(self.config_path).open() as f:
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
        # Implementation would recursively build dataclass instances
        # For now, simplified version
        return SystemConfig(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            rl=RLConfig(**config_dict.get("rl", {})),
            risk=RiskConfig(**config_dict.get("risk", {})),
            infrastructure=InfrastructureConfig(
                **config_dict.get("infrastructure", {}),
            ),
        )

    def _config_to_dict(self, config: SystemConfig) -> dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        return {
            "data": config.data.__dict__,
            "model": config.model.__dict__,
            "rl": config.rl.__dict__,
            "risk": config.risk.__dict__,
            "infrastructure": config.infrastructure.__dict__,
            "random_seed": config.random_seed,
            "log_level": config.log_level,
            "environment": config.environment,
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
