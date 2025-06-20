"""
Modern PPO Agent with Advanced Features

Implements PPO with:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function clipping
- Adaptive KL divergence
- Advanced normalization techniques
"""

from collections import deque
import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Use absolute imports to avoid relative import issues when running tests
try:
    from configs.hyperparameters import PPOConfig, get_agent_config
except ImportError:
    from src.configs.hyperparameters import PPOConfig, get_agent_config


class PPONetwork(nn.Module):
    """Shared network for PPO actor-critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "tanh",
        discrete: bool = False,
    ):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Shared feature extraction
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Actor head
        if discrete:
            self.actor_head = nn.Linear(input_dim, action_dim)
        else:
            self.actor_mean = nn.Linear(input_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic_head = nn.Linear(input_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> tuple[Any, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared_layers(state)

        if self.discrete:
            logits = self.actor_head(features)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            action_dist = torch.distributions.Normal(mean, std)

        value = self.critic_head(features)

        return action_dist, value.squeeze(-1)

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        """Get action, log probability, and value."""
        action_dist, value = self.forward(state)

        if action is None:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        # For continuous actions, sum log probs across action dimensions
        if not self.discrete and len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)

        entropy = action_dist.entropy()
        if not self.discrete and len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value


class PPORolloutBuffer:
    """Rollout buffer for PPO with GAE computation."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        discrete: bool = False,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = torch.device(device)

        # Initialize buffers
        self.states = torch.zeros((buffer_size, state_dim), device=self.device)
        if discrete:
            self.actions = torch.zeros(
                (buffer_size,), dtype=torch.long, device=self.device
            )
        else:
            self.actions = torch.zeros((buffer_size, action_dim), device=self.device)

        self.rewards = torch.zeros((buffer_size,), device=self.device)
        self.values = torch.zeros((buffer_size,), device=self.device)
        self.log_probs = torch.zeros((buffer_size,), device=self.device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=self.device)

        # GAE buffers
        self.advantages = torch.zeros((buffer_size,), device=self.device)
        self.returns = torch.zeros((buffer_size,), device=self.device)

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Store a single transition."""
        self.states[self.ptr] = torch.as_tensor(state, device=self.device)
        if self.discrete:
            self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        else:
            self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device)

        self.ptr += 1

    def finish_path(self, last_value: float = 0.0):
        """Finish a trajectory and compute GAE."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # Append last value for bootstrap
        values_with_bootstrap = torch.cat(
            [values, torch.tensor([last_value], device=self.device)]
        )

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values_with_bootstrap[t + 1]

            delta = rewards[t] + 0.99 * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + 0.99 * 0.95 * next_non_terminal * last_gae
            )

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values

        self.path_start_idx = self.ptr

    def get(self):
        """Get all stored data."""
        assert self.ptr == self.buffer_size

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns": self.returns,
            "dones": self.dones,
        }

        # Reset buffer
        self.ptr = 0
        self.path_start_idx = 0

        return data


class PPOAgent:
    """Modern PPO Agent with GAE and clipped surrogate objective."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Union[str, dict]] = None,
        discrete: bool = False,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = torch.device(device)

        # Load configuration
        if isinstance(config, str):
            self.config = get_agent_config("ppo", config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = asdict(PPOConfig())

        # Extract hyperparameters
        self.lr = self.config.get("learning_rate", 3e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.clip_ratio = self.config.get("clip_ratio", 0.2)
        self.clip_vf_ratio = self.config.get("clip_vf_ratio", None)
        self.n_epochs = self.config.get("n_epochs", 10)
        self.batch_size = self.config.get("batch_size", 256)
        self.minibatch_size = self.config.get("minibatch_size", 64)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        self.vf_coef = self.config.get("vf_coef", 0.5)
        self.ent_coef = self.config.get("ent_coef", 0.01)
        self.target_kl = self.config.get("target_kl", 0.01)
        self.buffer_size = self.config.get("buffer_size", 2048)

        # Initialize network
        self.network = PPONetwork(
            state_dim,
            action_dim,
            self.config.get("hidden_dims", [256, 256]),
            self.config.get("activation", "tanh"),
            discrete,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Initialize buffer
        self.buffer = PPORolloutBuffer(
            self.buffer_size, state_dim, action_dim, discrete, device
        )

        # Training metrics
        self.training_step = 0
        self.episode_count = 0

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Select action for given state."""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_dist, value = self.network(state_tensor)

            if deterministic:
                if self.discrete:
                    action = action_dist.probs.argmax(dim=-1)
                else:
                    action = action_dist.mean
            else:
                action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if not self.discrete and len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=-1)

        if self.discrete:
            return (
                action.cpu().numpy()[0],
                value.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
            )
        else:
            return (
                action.cpu().numpy()[0],
                value.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
            )

    def store_experience(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Store experience in buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)

    def finish_episode(self, last_value: float = 0.0):
        """Finish episode and compute advantages."""
        self.buffer.finish_path(last_value)
        self.episode_count += 1

    def update(self) -> dict[str, float]:
        """Update PPO networks."""
        if self.buffer.ptr < self.buffer_size:
            return {}

        # Get data from buffer
        data = self.buffer.get()

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl_div = 0.0
        n_updates = 0

        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Create minibatches
            indices = torch.randperm(self.buffer_size, device=self.device)

            for start in range(0, self.buffer_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_states = data["states"][mb_indices]
                mb_actions = data["actions"][mb_indices]
                mb_old_log_probs = data["log_probs"][mb_indices]
                mb_advantages = data["advantages"][mb_indices]
                mb_returns = data["returns"][mb_indices]
                mb_old_values = data["values"][mb_indices]

                # Get current network outputs
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(mb_states, mb_actions)
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )

                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * clipped_ratio
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss (optionally clipped)
                if self.clip_vf_ratio is not None:
                    clipped_values = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -self.clip_vf_ratio,
                        self.clip_vf_ratio,
                    )
                    value_loss_1 = F.mse_loss(new_values, mb_returns)
                    value_loss_2 = F.mse_loss(clipped_values, mb_returns)
                    value_loss = torch.max(value_loss_1, value_loss_2)
                else:
                    value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Compute KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - new_log_probs).mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Update metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_div += kl_div.item()
                n_updates += 1

            # Early stopping if KL divergence is too high
            if total_kl_div / n_updates > self.target_kl:
                print(
                    f"Early stopping at epoch {epoch} due to KL divergence: {total_kl_div / n_updates:.4f}"
                )
                break

        self.training_step += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy_loss": total_entropy_loss / n_updates,
            "kl_divergence": total_kl_div / n_updates,
            "n_updates": n_updates,
            "training_step": self.training_step,
        }

    def save(self, filepath: str):
        """Save agent state."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "episode_count": self.episode_count,
                "config": self.config,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)


if __name__ == "__main__":
    # Test PPO agent
    print("=== PPO Agent Test ===")

    # Create agent
    state_dim = 10
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim, discrete=False)

    print(f"✅ Created PPO agent with state_dim={state_dim}, action_dim={action_dim}")

    # Test action selection
    state = np.random.randn(state_dim)
    action, value, log_prob = agent.select_action(state)

    print(f"✅ Action selection test:")
    print(f"   Action: {action}")
    print(f"   Value: {value}")
    print(f"   Log prob: {log_prob}")

    # Simulate episode
    print("\n📈 Simulating episode...")
    for step in range(agent.buffer_size):
        state = np.random.randn(state_dim)
        action, value, log_prob = agent.select_action(state)
        reward = np.random.randn()
        done = (step + 1) % 100 == 0  # Episode ends every 100 steps

        agent.store_experience(state, action, reward, done, value, log_prob)

        if done:
            agent.finish_episode(0.0)

    print(f"✅ Filled buffer with {agent.buffer.ptr} transitions")

    # Test update
    print("\n🔄 Testing update...")
    metrics = agent.update()

    print(f"✅ Update metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")

    print("\n🎯 PPO agent test complete!")
