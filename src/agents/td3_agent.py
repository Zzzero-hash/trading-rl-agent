"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

TD3 improvements over DDPG:
1. Twin Critic Networks - reduces overestimation bias
2. Delayed Policy Updates - update policy less frequently than critics
3. Target Policy Smoothing - add noise to target actions for regularization

Ideal for trading applications requiring stable continuous control.
"""

from collections import deque
import copy
from dataclasses import asdict, is_dataclass
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from .configs import TD3Config


class Actor(nn.Module):
    """TD3 Actor Network for deterministic policy."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] = [256, 256]
    ):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action."""
        return self.network(state)


class Critic(nn.Module):
    """TD3 Critic Network (single critic)."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] = [256, 256]
    ):
        super().__init__()

        input_dim = state_dim + action_dim

        # Single critic network
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for TD3."""

    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer (matches test API)."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer (alternative API)."""
        self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    @property
    def size(self):
        """Get current buffer size (alias for len)."""
        return len(self.buffer)


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient Agent.

    Features:
    - Twin critic networks to reduce overestimation
    - Delayed policy updates for stability
    - Target policy smoothing for regularization
    - Continuous action space for position sizing
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, TD3Config]] = None,
        state_dim: int = 10,
        action_dim: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize TD3 Agent.

        Args:
            config: Configuration (dataclass, dict, or file path)
            state_dim: State space dimension (defaults to 10 for tests)
            action_dim: Action space dimension (defaults to 3 for tests)
            device: Device to run on ("cpu" or "cuda")
        """
        # Load configuration first
        if is_dataclass(config):
            self.config = config  # Keep original dataclass for direct comparison
            self._config_dict = asdict(config)  # Store dict version for .get() calls
        else:
            self._config_dict = self._load_config(config)
            self.config = self._config_dict

        # Store dimensions
        # Log and assert state shape for debugging
        print(f"[TD3Agent] Initializing with state_dim={state_dim}")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Extract hyperparameters
        self.lr = self._config_dict.get("learning_rate", 3e-4)
        self.gamma = self._config_dict.get("gamma", 0.99)
        self.tau = self._config_dict.get("tau", 0.005)
        self.batch_size = self._config_dict.get("batch_size", 32)  # Match test default
        self.buffer_capacity = self._config_dict.get(
            "buffer_capacity", 10000
        )  # Match test default
        self.hidden_dims = self._config_dict.get(
            "hidden_dims", [64, 64]
        )  # Match test default

        # TD3 specific parameters
        self.policy_delay = self._config_dict.get("policy_delay", 2)
        self.target_noise = self._config_dict.get("target_noise", 0.2)
        self.noise_clip = self._config_dict.get("noise_clip", 0.5)
        self.exploration_noise = self._config_dict.get("exploration_noise", 0.1)

        # Device setup
        self.device = torch.device(device)

        # Initialize actor networks
        self.actor = Actor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Initialize twin critics (separate instances)
        self.critic_1 = Critic(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        # Training counters
        self.training_step = 0
        self.total_it = 0

    def _load_config(self, config: Optional[Union[str, dict, TD3Config]]) -> dict:
        """Load configuration from dataclass, file, or dict."""
        if config is None:
            return {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 32,
                "buffer_capacity": 10000,
                "hidden_dims": [64, 64],
                "policy_delay": 2,
                "target_noise": 0.2,
                "noise_clip": 0.5,
                "exploration_noise": 0.1,
            }
        elif isinstance(config, str):
            with open(config) as f:
                return yaml.safe_load(f) or {}
        elif is_dataclass(config):
            return asdict(config)
        else:
            return config or {}

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

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self) -> dict[str, float]:
        """Train the agent (alias for update to match test API)."""
        return self.update()

    def update(self) -> dict[str, float]:
        """Update TD3 networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.total_it += 1

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)

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

        metrics = {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "mean_q1": q1_current.mean().item(),
            "mean_q2": q2_current.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

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
        return metrics

    def _soft_update(self, target, source, tau):
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

        self.training_step = checkpoint["training_step"]
        self.total_it = checkpoint["total_it"]


if __name__ == "__main__":
    # Example usage
    from .configs import TD3Config

    config = TD3Config(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=32,
        buffer_capacity=10000,
        hidden_dims=[64, 64],
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
    )

    agent = TD3Agent(config, state_dim=10, action_dim=3)

    # Test action selection
    dummy_state = np.random.randn(10)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action}")

    # Test training
    for i in range(50):  # Collect some experiences
        state = np.random.randn(10).astype(np.float32)
        action = np.random.uniform(-1, 1, 3).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(10).astype(np.float32)
        done = False

        agent.store_experience(state, action, reward, next_state, done)

    # Update networks
    for i in range(5):
        metrics = agent.update()
        if metrics:
            print(
                f"Step {i+1} - Critic 1 Loss: {metrics['critic_1_loss']:.4f}, "
                f"Critic 2 Loss: {metrics['critic_2_loss']:.4f}, "
                f"Actor Loss: {metrics['actor_loss']:.4f}"
            )
