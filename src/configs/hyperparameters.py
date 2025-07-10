"""Externalized Hyperparameter Configuration System.

This module provides a centralized configuration system for all RL agents
with support for YAML configs, environment variables, and dynamic loading.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PPOConfig:
    """Configuration for PPO agent with GAE and clipped surrogate."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_vf_ratio: float | None = None  # None means no clipping

    # Training parameters
    batch_size: int = 256
    minibatch_size: int = 64
    n_epochs: int = 10
    max_grad_norm: float = 0.5

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"

    # Value function parameters
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    target_kl: float = 0.01

    # Buffer parameters
    buffer_size: int = 2048

    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = True

    # Exploration
    exploration_noise: float = 0.1


@dataclass
class RainbowDQNConfig:
    """Configuration for Rainbow DQN with all modern improvements."""

    # Learning parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512])
    dueling: bool = True
    double_dqn: bool = True

    # Rainbow components
    noisy_nets: bool = True
    prioritized_replay: bool = True
    multi_step: int = 3
    distributional: bool = True

    # Distributional parameters (C51)
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    # Prioritized replay parameters
    alpha: float = 0.6  # Prioritization exponent
    beta_start: float = 0.4  # Importance sampling start
    beta_end: float = 1.0  # Importance sampling end
    beta_frames: int = 100000

    # Buffer parameters
    buffer_capacity: int = 100000
    batch_size: int = 32

    # Training parameters
    target_update_freq: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1

    # Noisy nets parameters
    sigma_init: float = 0.017


@dataclass
class EnhancedSACConfig:
    """Enhanced SAC configuration with modern features."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"

    # Entropy parameters
    automatic_entropy_tuning: bool = True
    target_entropy: float = -1.0
    alpha: float = 0.2

    # Buffer parameters
    buffer_capacity: int = 1000000
    batch_size: int = 256

    # Training parameters
    gradient_steps: int = 1
    target_update_interval: int = 1

    # Advanced features
    layer_norm: bool = False
    spectral_norm: bool = False
    dropout: float = 0.0


@dataclass
class EnhancedTD3Config:
    """Enhanced TD3 configuration for Ray deployment."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"

    # TD3 specific parameters
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1

    # Buffer parameters
    buffer_capacity: int = 1000000
    batch_size: int = 256

    # Training parameters
    gradient_steps: int = 1

    # Advanced features
    layer_norm: bool = False
    spectral_norm: bool = False
    dropout: float = 0.0


@dataclass
class EnsembleConfig:
    """Enhanced ensemble configuration with consensus methods."""

    # Agent configurations
    agents: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "ppo": {"enabled": True, "weight": 0.3},
            "sac": {"enabled": True, "weight": 0.3},
            "td3": {"enabled": True, "weight": 0.2},
            "rainbow_dqn": {"enabled": True, "weight": 0.2},
        },
    )

    # Consensus methods
    consensus_method: str = "weighted_voting"  # weighted_voting, majority_voting, confidence_weighted
    confidence_threshold: float = 0.7

    # Dynamic weighting
    dynamic_weighting: bool = True
    weight_update_frequency: int = 1000
    performance_window: int = 100
    min_weight: float = 0.05

    # Risk management
    risk_adjustment: bool = True
    risk_aversion: float = 0.1
    diversification_bonus: float = 0.05

    # Uncertainty quantification
    uncertainty_estimation: bool = True
    monte_carlo_samples: int = 10

    # Meta-learning
    meta_learning: bool = False
    adaptation_rate: float = 0.01


class ConfigManager:
    """Centralized configuration management system."""

    def __init__(self, config_dir: str | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(exist_ok=True)

    def load_config(
        self,
        config_type: str,
        config_name: str = "default",
    ) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = self.config_dir / f"{config_type}_{config_name}.yaml"

        if config_file.exists():
            with config_file.open("r") as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self._get_default_config(config_type)

    def save_config(
        self,
        config: dict[str, Any],
        config_type: str,
        config_name: str = "default",
    ) -> None:
        """Save configuration to YAML file."""
        config_file = self.config_dir / f"{config_type}_{config_name}.yaml"

        with config_file.open("w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _get_default_config(self, config_type: str) -> dict[str, Any]:
        """Get default configuration for given type."""
        config_classes = {
            "ppo": PPOConfig,
            "rainbow_dqn": RainbowDQNConfig,
            "enhanced_sac": EnhancedSACConfig,
            "enhanced_td3": EnhancedTD3Config,
            "ensemble": EnsembleConfig,
        }

        config_class = config_classes.get(config_type)
        if config_class:
            return asdict(config_class())
        raise ValueError(f"Unknown config type: {config_type}")

    def get_config_from_env(
        self,
        config_type: str,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        """Load configuration from environment variables."""
        if prefix is None:
            prefix = f"{config_type.upper()}_"

        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                # Try to convert to appropriate type
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    config[config_key] = value

        return config

    def merge_configs(self, *configs: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged


# Global config manager instance
config_manager = ConfigManager()


def get_agent_config(
    agent_type: str,
    config_name: str = "default",
    env_override: bool = True,
) -> dict[str, Any]:
    """Get agent configuration with optional environment override."""
    # Load base config
    config = config_manager.load_config(agent_type, config_name)

    # Override with environment variables if requested
    if env_override:
        env_config = config_manager.get_config_from_env(agent_type)
        config = config_manager.merge_configs(config, env_config)

    return config


def save_agent_config(
    config: dict[str, Any],
    agent_type: str,
    config_name: str = "default",
) -> None:
    """Save agent configuration."""
    config_manager.save_config(config, agent_type, config_name)


if __name__ == "__main__":
    # Example usage
    print("=== Configuration System Demo ===")

    # Create default configurations
    ppo_config = PPOConfig()
    rainbow_config = RainbowDQNConfig()
    sac_config = EnhancedSACConfig()
    td3_config = EnhancedTD3Config()
    ensemble_config = EnsembleConfig()

    print("✅ Default configurations created")

    # Save configurations
    configs_to_save = [
        (asdict(ppo_config), "ppo"),
        (asdict(rainbow_config), "rainbow_dqn"),
        (asdict(sac_config), "enhanced_sac"),
        (asdict(td3_config), "enhanced_td3"),
        (asdict(ensemble_config), "ensemble"),
    ]

    for config, config_type in configs_to_save:
        save_agent_config(config, config_type)
        print(f"✅ Saved {config_type} configuration")

    # Load and display a configuration
    loaded_config = get_agent_config("ppo")
    print(f"\n📋 Loaded PPO config: {loaded_config}")

    print("\n🎯 Configuration system ready!")
