from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnsembleConfig:
    """Configuration options for :class:`WeightedEnsembleAgent`."""

    agents: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "sac": {"enabled": True, "config": None},
            "td3": {"enabled": True, "config": None},
        },
    )
    ensemble_method: str = "weighted_average"
    weight_update_frequency: int = 1000
    diversity_penalty: float = 0.1
    min_weight: float = 0.1
    performance_window: int = 100
    risk_adjustment: bool = True
    combination_method: str = "weighted_average"
    # Optional dimension parameters for compatibility
    state_dim: int | None = None
    action_dim: int | None = None
    # Support agent_configs as parameter name
    agent_configs: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Handle agent_configs alias and validation after initialization."""
        # If agent_configs was passed, use it instead of agents
        if self.agent_configs is not None:
            self.agents = self.agent_configs

        # Validate that at least one agent is specified
        if not self.agents:
            raise ValueError("At least one agent must be specified")


@dataclass
class SACConfig:
    """Configuration options for :class:`SACAgent`."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 1000000
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    automatic_entropy_tuning: bool = True
    target_entropy: float = -1.0
    alpha: float = 0.2  # Used when automatic_entropy_tuning is False


@dataclass
class TD3Config:
    """Configuration options for :class:`TD3Agent`."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 1000000
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
