"""
Enhanced TD3 Agent for Ray Deployment

Upgrades the existing TD3 agent with:
- Ray RLlib integration
- Advanced network architectures
- Distributed training support
- Better hyperparameter management
"""

import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ray imports (with fallback for when Ray is not available)
try:
    import ray
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.typing import ModelConfigDict

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    TorchModelV2 = object
    ModelConfigDict = dict

# Use absolute imports to avoid relative import issues when running tests
try:
    from configs.hyperparameters import EnhancedTD3Config, get_agent_config
except ImportError:
    from src.configs.hyperparameters import EnhancedTD3Config, get_agent_config


class EnhancedActor(nn.Module):
    """Enhanced TD3 Actor with advanced features."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build network
        layers = []
        input_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(input_dim, hidden_dim)

            # Apply spectral normalization if requested
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)

            # Add layer normalization if requested
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self.activation)

            # Add dropout if requested
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(input_dim, action_dim)
        if spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.append(output_layer)
        layers.append(nn.Tanh())  # Bounded actions [-1, 1]

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action."""
        return self.network(state)


class EnhancedCritic(nn.Module):
    """Enhanced TD3 Critic with advanced features."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = False,
        spectral_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build network
        layers = []
        input_dim = state_dim + action_dim

        for i, hidden_dim in enumerate(hidden_dims):
            linear = nn.Linear(input_dim, hidden_dim)

            # Apply spectral normalization if requested
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)

            # Add layer normalization if requested
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self.activation)

            # Add dropout if requested
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer
        output_layer = nn.Linear(input_dim, 1)
        if spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class EnhancedReplayBuffer:
    """Enhanced replay buffer with better memory management."""

    def __init__(
        self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate tensors for better performance
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        self.states[self.pos] = torch.as_tensor(state, device=self.device)
        self.actions[self.pos] = torch.as_tensor(action, device=self.device)
        self.rewards[self.pos] = torch.as_tensor(reward, device=self.device)
        self.next_states[self.pos] = torch.as_tensor(next_state, device=self.device)
        self.dones[self.pos] = torch.as_tensor(done, device=self.device)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


class EnhancedTD3Agent:
    """Enhanced TD3 Agent with Ray integration capabilities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Union[str, dict]] = None,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Load configuration
        if isinstance(config, str):
            self.config = get_agent_config("enhanced_td3", config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = asdict(EnhancedTD3Config())

        # Extract hyperparameters
        self.lr = self.config.get("learning_rate", 3e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.tau = self.config.get("tau", 0.005)
        self.policy_delay = self.config.get("policy_delay", 2)
        self.target_noise = self.config.get("target_noise", 0.2)
        self.noise_clip = self.config.get("noise_clip", 0.5)
        self.exploration_noise = self.config.get("exploration_noise", 0.1)
        self.batch_size = self.config.get("batch_size", 256)
        self.buffer_capacity = self.config.get("buffer_capacity", 1000000)
        self.gradient_steps = self.config.get("gradient_steps", 1)

        # Network configuration
        hidden_dims = self.config.get("hidden_dims", [256, 256])
        activation = self.config.get("activation", "relu")
        layer_norm = self.config.get("layer_norm", False)
        spectral_norm = self.config.get("spectral_norm", False)
        dropout = self.config.get("dropout", 0.0)

        # Initialize networks
        self.actor = EnhancedActor(
            state_dim,
            action_dim,
            hidden_dims,
            activation,
            layer_norm,
            spectral_norm,
            dropout,
        ).to(self.device)

        self.critic_1 = EnhancedCritic(
            state_dim,
            action_dim,
            hidden_dims,
            activation,
            layer_norm,
            spectral_norm,
            dropout,
        ).to(self.device)

        self.critic_2 = EnhancedCritic(
            state_dim,
            action_dim,
            hidden_dims,
            activation,
            layer_norm,
            spectral_norm,
            dropout,
        ).to(self.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = EnhancedReplayBuffer(
            self.buffer_capacity, state_dim, action_dim, device
        )

        # Training counters
        self.training_step = 0
        self.total_it = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()

        # Add exploration noise during training
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> dict[str, float]:
        """Update TD3 networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        metrics = {}

        for _ in range(self.gradient_steps):
            self.total_it += 1

            # Sample from replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample(
                self.batch_size
            )

            # Update Twin Critics
            with torch.no_grad():
                # Target policy smoothing: add clipped noise to target actions
                noise = (torch.randn_like(action) * self.target_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

                # Compute target Q-values (take minimum to reduce overestimation)
                q1_next = self.critic_1_target(next_state, next_action)
                q2_next = self.critic_2_target(next_state, next_action)
                q_next = torch.min(q1_next, q2_next)
                target_q = reward + (1 - done.float()) * self.gamma * q_next

            # Current Q-values
            q1_current = self.critic_1(state, action)
            q2_current = self.critic_2(state, action)

            # Critic losses
            critic_1_loss = F.mse_loss(q1_current, target_q)
            critic_2_loss = F.mse_loss(q2_current, target_q)

            # Update critic 1
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            # Update critic 2
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            metrics.update(
                {
                    "critic_1_loss": critic_1_loss.item(),
                    "critic_2_loss": critic_2_loss.item(),
                    "mean_q1": q1_current.mean().item(),
                    "mean_q2": q2_current.mean().item(),
                    "target_q_mean": target_q.mean().item(),
                }
            )

            # Delayed policy updates
            if self.total_it % self.policy_delay == 0:
                # Actor loss (maximize Q1)
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                self._soft_update(self.critic_1_target, self.critic_1, self.tau)
                self._soft_update(self.critic_2_target, self.critic_2, self.tau)
                self._soft_update(self.actor_target, self.actor, self.tau)

                metrics.update({"actor_loss": actor_loss.item(), "policy_update": True})
            else:
                metrics.update({"actor_loss": 0.0, "policy_update": False})

        self.training_step += 1
        metrics["training_step"] = self.training_step

        return metrics

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, filepath: str):
        """Save agent state."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_2_state_dict": self.critic_2.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_1_target_state_dict": self.critic_1_target.state_dict(),
                "critic_2_target_state_dict": self.critic_2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
                "training_step": self.training_step,
                "total_it": self.total_it,
                "config": self.config,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target_state_dict"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(
            checkpoint["critic_1_optimizer_state_dict"]
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint["critic_2_optimizer_state_dict"]
        )

        self.training_step = checkpoint.get("training_step", 0)
        self.total_it = checkpoint.get("total_it", 0)


if RAY_AVAILABLE:

    class TD3RayModel(TorchModelV2, nn.Module):
        """TD3 model wrapper for Ray RLlib."""

        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config: ModelConfigDict,
            name: str,
        ):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.obs_size = int(np.prod(obs_space.shape))
            self.action_size = int(np.prod(action_space.shape))

            # Extract TD3 config from model config
            td3_config = model_config.get("custom_model_config", {})

            # Actor network
            self.actor = EnhancedActor(
                self.obs_size,
                self.action_size,
                td3_config.get("hidden_dims", [256, 256]),
                td3_config.get("activation", "relu"),
                td3_config.get("layer_norm", False),
                td3_config.get("spectral_norm", False),
                td3_config.get("dropout", 0.0),
            )

            # Value network (for bootstrapping)
            self.value_net = EnhancedCritic(
                self.obs_size,
                self.action_size,
                td3_config.get("hidden_dims", [256, 256]),
                td3_config.get("activation", "relu"),
                td3_config.get("layer_norm", False),
                td3_config.get("spectral_norm", False),
                td3_config.get("dropout", 0.0),
            )

        def forward(self, input_dict, state, seq_lens):
            """Forward pass for policy."""
            obs = input_dict["obs_flat"]
            actions = self.actor(obs)
            return actions, state

        def value_function(self) -> torch.Tensor:
            """Return value function (not used in TD3 but required by interface)."""
            return torch.zeros(1)

else:

    class TD3RayModel:
        """Placeholder when Ray is not available."""

        pass


def get_enhanced_td3_config():
    """Get Ray Tune configuration for enhanced TD3."""
    from ray.tune import grid_search, loguniform, uniform

    return {
        "algorithm": "TD3",
        "env": "TradingEnv",
        "framework": "torch",
        "model": {
            "custom_model": "td3_ray_model",
            "custom_model_config": {
                "hidden_dims": grid_search([[256, 256], [512, 512], [256, 256, 256]]),
                "activation": grid_search(["relu", "tanh"]),
                "layer_norm": grid_search([True, False]),
                "spectral_norm": grid_search([True, False]),
                "dropout": uniform(0.0, 0.3),
            },
        },
        "learning_rate": loguniform(1e-5, 1e-3),
        "gamma": uniform(0.95, 0.999),
        "tau": uniform(0.001, 0.01),
        "policy_delay": grid_search([1, 2, 3]),
        "target_noise": uniform(0.1, 0.3),
        "noise_clip": uniform(0.3, 0.7),
        "exploration_noise": uniform(0.05, 0.2),
        "batch_size": grid_search([256, 512, 1024]),
        "buffer_size": grid_search([100000, 500000, 1000000]),
        "train_batch_size": 4000,
        "num_workers": 4,
        "num_gpus": 0.5,
        "evaluation_interval": 10,
        "evaluation_duration": 10,
        "clip_actions": True,
        "normalize_actions": True,
    }


if __name__ == "__main__":
    # Test Enhanced TD3 agent
    print("=== Enhanced TD3 Agent Test ===")

    # Create agent
    state_dim = 10
    action_dim = 3
    agent = EnhancedTD3Agent(state_dim, action_dim)

    print(
        f"✅ Created Enhanced TD3 agent with state_dim={state_dim}, action_dim={action_dim}"
    )
    print("   Features: Layer norm, Spectral norm, Dropout, Enhanced buffer")

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state, add_noise=False)

    print(f"✅ Action selection test: action={action}")

    # Test experience storage
    print("\n📈 Testing experience collection...")
    for i in range(1000):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False

        agent.store_experience(state, action, reward, next_state, done)

    print(f"✅ Stored {len(agent.replay_buffer)} experiences")

    # Test updates
    print("\n🔄 Testing updates...")
    for i in range(5):
        metrics = agent.update()
        if metrics:
            print(
                f"   Step {i+1}: C1 loss={metrics['critic_1_loss']:.4f}, "
                f"C2 loss={metrics['critic_2_loss']:.4f}, "
                f"Actor loss={metrics['actor_loss']:.4f}"
            )

    print("\n🎯 Enhanced TD3 agent test complete!")
