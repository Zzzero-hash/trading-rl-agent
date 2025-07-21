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
class AdvancedPPOConfig(PPOConfig):
    """Enhanced PPO configuration with advanced features."""

    # Advanced clipping parameters
    adaptive_clip_ratio: bool = True
    clip_ratio_decay: float = 0.995
    min_clip_ratio: float = 0.05
    max_clip_ratio: float = 0.3

    # Trust region parameters
    use_trust_region: bool = True
    trust_region_radius: float = 0.01
    max_kl_divergence: float = 0.01

    # Natural gradient parameters
    use_natural_gradient: bool = False
    natural_gradient_damping: float = 1e-3
    natural_gradient_max_iter: int = 10

    # Multi-objective parameters
    risk_weight: float = 0.1
    return_weight: float = 0.9
    sharpe_weight: float = 0.0
    max_drawdown_weight: float = 0.0

    # Adaptive learning rate
    adaptive_lr: bool = True
    lr_schedule: str = "cosine"  # cosine, linear, exponential
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Performance tracking
    performance_window: int = 100
    early_stopping_patience: int = 20


@dataclass
class TRPOConfig:
    """Configuration for Trust Region Policy Optimization."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Trust region parameters
    max_kl_divergence: float = 0.01
    damping_coeff: float = 0.1
    max_backtrack_iter: int = 10
    backtrack_coeff: float = 0.8
    max_cg_iter: int = 10
    cg_tolerance: float = 1e-6

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"

    # Training parameters
    batch_size: int = 256
    n_epochs: int = 10
    max_grad_norm: float = 0.5

    # Value function parameters
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Buffer parameters
    buffer_size: int = 2048

    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = True


@dataclass
class NaturalPolicyGradientConfig:
    """Configuration for Natural Policy Gradient."""

    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Natural gradient parameters
    damping_coeff: float = 1e-3
    max_cg_iter: int = 10
    cg_tolerance: float = 1e-6

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"

    # Training parameters
    batch_size: int = 256
    n_epochs: int = 10
    max_grad_norm: float = 0.5

    # Value function parameters
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # Buffer parameters
    buffer_size: int = 2048

    # Normalization
    normalize_advantages: bool = True
    normalize_returns: bool = True


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""

    # Objective weights
    return_weight: float = 0.9
    risk_weight: float = 0.1
    sharpe_weight: float = 0.0
    max_drawdown_weight: float = 0.0

    # Risk parameters
    var_alpha: float = 0.05
    max_position_size: float = 1.0
    max_leverage: float = 2.0

    # Performance tracking
    performance_window: int = 100
    rebalancing_frequency: int = 10

    # Optimization parameters
    optimization_method: str = "weighted_sum"  # weighted_sum, pareto_frontier
    pareto_points: int = 10
