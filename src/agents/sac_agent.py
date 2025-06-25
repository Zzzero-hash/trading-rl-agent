"""
Soft Actor-Critic (SAC) Agent

SAC is an off-policy actor-critic method based on the maximum entropy framework.
Key features:
1. Stochastic policy with entropy regularization
2. Twin Q-networks to reduce overestimation bias
3. Automatic entropy temperature tuning
4. Stable training through maximum entropy objective

Ideal for trading applications requiring exploration-exploitation balance.
"""

from collections import deque
import copy
from dataclasses import asdict, is_dataclass
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml


class Actor(nn.Module):
    """SAC Actor Network with stochastic policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature extraction layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return action and log_prob."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        # Compute log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """SAC Q-Network (single critic)."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] | None = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


# Alias for test compatibility
class Critic(nn.Module):
    """Twin Critic Network for SAC (returns both Q1 and Q2)."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] | None = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Create two Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q-values."""
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2


class ReplayBuffer:
    """Experience replay buffer for SAC agent."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer (alias for push to match test API)."""
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),  # 1D tensor for rewards
            torch.FloatTensor(next_state),
            torch.BoolTensor(done),  # 1D tensor for dones
        )

    def __len__(self) -> int:
        """Return the current size of replay buffer."""
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic Agent implementation."""

    def __init__(
        self, state_dim: int, action_dim: int, config=None, device: str = "cpu"
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: SAC configuration (dict, SACConfig, or None for defaults)
            device: Device to run on ("cpu" or "cuda")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.config = config  # Store original config
        self.training_step = 0
        # Optional production risk manager integration
        from src.risk.production_risk_manager import ProductionRiskManager

        self.risk_manager = None
        if isinstance(config, dict) and "risk_manager_config" in config:
            self.risk_manager = ProductionRiskManager(config["risk_manager_config"])

        # Handle different config types and set defaults
        if config is None:
            # Default configuration
            self.lr = 3e-4
            self.gamma = 0.99
            self.tau = 0.01
            self.batch_size = 32
            self.automatic_entropy_tuning = True
            self.target_entropy = -float(action_dim)
            self.alpha = 0.2
            hidden_dims = [256, 256]
            buffer_capacity = 10000
        elif isinstance(config, dict):
            # Dictionary configuration
            self.lr = config.get("learning_rate", 3e-4)
            self.gamma = config.get("gamma", 0.99)
            self.tau = config.get("tau", 0.01)
            self.batch_size = config.get("batch_size", 32)
            self.automatic_entropy_tuning = config.get("automatic_entropy_tuning", True)
            self.target_entropy = config.get("target_entropy", -float(action_dim))
            self.alpha = config.get("alpha", 0.2)
            hidden_dims = config.get("hidden_dims", [256, 256])
            buffer_capacity = config.get("buffer_capacity", 10000)
        else:
            # SACConfig object
            self.lr = config.learning_rate
            self.gamma = config.gamma
            self.tau = config.tau
            self.batch_size = config.batch_size
            self.automatic_entropy_tuning = config.automatic_entropy_tuning
            self.target_entropy = config.target_entropy
            self.alpha = config.alpha
            hidden_dims = config.hidden_dims
            buffer_capacity = config.buffer_capacity

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Initialize state processor for enhanced state representation
        self.state_processor = None  # Disabled for debugging

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        # Backward compatibility aliases for tests
        self.critic = self.critic1
        self.critic_1 = self.critic1  # For test compatibility with TD3 naming
        self.critic_2 = self.critic2  # For test compatibility with TD3 naming

        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic_target = self.target_critic1  # Backward compatibility

        # Freeze target networks
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.critic_optimizer = self.critic1_optimizer  # Backward compatibility

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            if self.target_entropy == -1.0:
                self.target_entropy = -action_dim  # Default heuristic
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

    @property
    def alpha(self) -> float:
        """Get current entropy coefficient."""
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """Set entropy coefficient (only when not using automatic tuning)."""
        if not self.automatic_entropy_tuning:
            self._alpha = value

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Current state (can be raw or dict format)
            evaluate: If True, use deterministic policy (mean action)

        Returns:
            Selected action
        """
        # Process state through enhanced representation if enabled
        if self.state_processor is not None:
            processed_state = self.state_processor.process_state(state)
        else:
            # Handle dict state format without processor
            if isinstance(state, dict):
                market_features = state.get("market_features", state)
                processed_state = (
                    market_features.flatten()
                    if market_features.ndim > 1
                    else market_features
                )
            else:
                processed_state = state.flatten() if state.ndim > 1 else state

        state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)

        action_array = action.cpu().numpy().flatten()
        # Validate action with risk manager if available
        if self.risk_manager is not None:
            valid = self.risk_manager.validate_action(action_array, processed_state)
            if not valid:
                # Override action to zero (halt)
                action_array = np.zeros_like(action_array)
        return action_array

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer with enhanced state processing."""
        # Process states through enhanced representation if enabled
        if self.state_processor is not None:
            processed_state = self.state_processor.process_state(state)
            processed_next_state = self.state_processor.process_state(next_state)
        else:
            # Handle dict state format without processor
            if isinstance(state, dict):
                market_features = state.get("market_features", state)
                processed_state = (
                    market_features.flatten()
                    if market_features.ndim > 1
                    else market_features
                )
            else:
                processed_state = state.flatten() if state.ndim > 1 else state

            if isinstance(next_state, dict):
                market_features = next_state.get("market_features", next_state)
                processed_next_state = (
                    market_features.flatten()
                    if market_features.ndim > 1
                    else market_features
                )
            else:
                processed_next_state = (
                    next_state.flatten() if next_state.ndim > 1 else next_state
                )

        self.replay_buffer.push(
            processed_state, action, reward, processed_next_state, done
        )

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer. Alias for store_transition for test compatibility."""
        self.store_transition(state, action, reward, next_state, done)

    def update(self) -> dict[str, float]:
        """
        Update agent networks.

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Update critic networks
        critic_losses, q_values = self._update_critics(
            state, action, reward, next_state, done
        )

        # Update actor network
        actor_loss, alpha_loss = self._update_actor(state)

        # Update target networks
        self._soft_update_targets()

        self.training_step += 1

        metrics = {
            "critic_loss": (critic_losses[0] + critic_losses[1])
            / 2,  # Average critic loss
            "actor_loss": actor_loss,
            "alpha": self.alpha,
            "mean_q1": q_values[0],
            "mean_q2": q_values[1],
        }

        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss

        return metrics

    def _update_critics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Update critic networks and return losses and mean Q-values."""
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = (
                reward.unsqueeze(1)
                + (1 - done.float().unsqueeze(1)) * self.gamma * target_q
            )

        # Current Q-values
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Store mean Q-values for metrics
        mean_q1 = current_q1.mean().item()
        mean_q2 = current_q2.mean().item()

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return (critic1_loss.item(), critic2_loss.item()), (mean_q1, mean_q2)

    def _update_actor(self, state: torch.Tensor) -> tuple[float, Optional[float]]:
        """Update actor network and entropy coefficient."""
        # Sample actions
        action, log_prob = self.actor.sample(state)

        # Q-values for sampled actions
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q = torch.min(q1, q2)

        # Actor loss (maximize Q + entropy)
        actor_loss = (self.alpha * log_prob - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = None
        # Update entropy coefficient
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss = alpha_loss.item()

        return actor_loss.item(), alpha_loss

    def _soft_update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save agent state."""
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "training_step": self.training_step,
        }

        if self.automatic_entropy_tuning:
            state["log_alpha"] = self.log_alpha
            state["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        torch.save(state, filepath)

    def load(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(state["actor_state_dict"])
        self.critic1.load_state_dict(state["critic1_state_dict"])
        self.critic2.load_state_dict(state["critic2_state_dict"])
        self.target_critic1.load_state_dict(state["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(state["target_critic2_state_dict"])

        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer_state_dict"])

        if self.automatic_entropy_tuning and "log_alpha" in state:
            self.log_alpha = state["log_alpha"]
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer_state_dict"])

        self.training_step = state.get("training_step", 0)

    def to_yaml(self, filepath: str):
        """Export agent configuration to YAML."""
        config_dict = {
            "agent_type": "SAC",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
            "target_entropy": getattr(self, "target_entropy", None),
            "alpha": self.alpha if not self.automatic_entropy_tuning else None,
            "training_step": self.training_step,
        }

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
