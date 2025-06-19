"""
Rainbow DQN Agent with All Modern Improvements

Implements Rainbow DQN with:
- Double DQN
- Dueling Networks
- Prioritized Experience Replay
- Multi-step Learning
- Distributional RL (C51)
- Noisy Networks
"""

from collections import deque, namedtuple
from dataclasses import asdict
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Use absolute imports to avoid relative import issues when running tests
try:
    from configs.hyperparameters import RainbowDQNConfig, get_agent_config
except ImportError:
    from src.configs.hyperparameters import RainbowDQNConfig, get_agent_config


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Generate new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(input, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling network architecture."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        n_atoms: int = 1,
        noisy: bool = False,
        sigma_init: float = 0.017,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.noisy = noisy

        # Shared feature extraction
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            if noisy:
                layers.append(NoisyLinear(input_dim, hidden_dim, sigma_init))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        # Value stream
        if noisy:
            self.value_stream = NoisyLinear(input_dim, n_atoms, sigma_init)
        else:
            self.value_stream = nn.Linear(input_dim, n_atoms)

        # Advantage stream
        if noisy:
            self.advantage_stream = NoisyLinear(
                input_dim, action_dim * n_atoms, sigma_init
            )
        else:
            self.advantage_stream = nn.Linear(input_dim, action_dim * n_atoms)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        features = self.feature_layers(state)

        value = self.value_stream(features)  # [batch_size, n_atoms]
        advantage = self.advantage_stream(
            features
        )  # [batch_size, action_dim * n_atoms]

        # Reshape advantage
        advantage = advantage.view(-1, self.action_dim, self.n_atoms)

        # Combine value and advantage
        if self.n_atoms == 1:
            # Standard dueling
            value = value.unsqueeze(1)  # [batch_size, 1, 1]
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values.squeeze(-1)  # [batch_size, action_dim]
        else:
            # Distributional dueling
            value = value.unsqueeze(1)  # [batch_size, 1, n_atoms]
            q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_dist  # [batch_size, action_dim, n_atoms]

    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def beta(self) -> float:
        """Calculate current beta value."""
        return min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
    ]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta()
        self.frame += 1

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1]))
        rewards = torch.FloatTensor(np.array(batch[2]))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.BoolTensor(np.array(batch[4]))
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(
        self, batch_indices: np.ndarray, batch_priorities: np.ndarray
    ):
        """Update priorities for sampled experiences."""
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)


class RainbowDQNAgent:
    """Rainbow DQN Agent with all modern improvements."""

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
            self.config = get_agent_config("rainbow_dqn", config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = asdict(RainbowDQNConfig())

        # Extract hyperparameters
        self.lr = self.config.get("learning_rate", 1e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon_start = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 10000)
        self.batch_size = self.config.get("batch_size", 32)
        self.target_update_freq = self.config.get("target_update_freq", 1000)
        self.train_freq = self.config.get("train_freq", 4)

        # Rainbow features
        self.double_dqn = self.config.get("double_dqn", True)
        self.dueling = self.config.get("dueling", True)
        self.noisy_nets = self.config.get("noisy_nets", True)
        self.prioritized_replay = self.config.get("prioritized_replay", True)
        self.multi_step = self.config.get("multi_step", 3)
        self.distributional = self.config.get("distributional", True)

        # Distributional parameters
        self.n_atoms = self.config.get("n_atoms", 51) if self.distributional else 1
        self.v_min = self.config.get("v_min", -10.0)
        self.v_max = self.config.get("v_max", 10.0)

        if self.distributional:
            self.support = torch.linspace(
                self.v_min, self.v_max, self.n_atoms, device=self.device
            )
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # Initialize networks
        self.q_network = DuelingNetwork(
            state_dim,
            action_dim,
            self.config.get("hidden_dims", [512, 512]),
            self.n_atoms,
            self.noisy_nets,
            self.config.get("sigma_init", 0.017),
        ).to(self.device)

        self.target_network = DuelingNetwork(
            state_dim,
            action_dim,
            self.config.get("hidden_dims", [512, 512]),
            self.n_atoms,
            self.noisy_nets,
            self.config.get("sigma_init", 0.017),
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Initialize replay buffer
        if self.prioritized_replay:
            self.replay_buffer: Union[PrioritizedReplayBuffer, deque] = (
                PrioritizedReplayBuffer(
                    self.config.get("buffer_capacity", 100000),
                    self.config.get("alpha", 0.6),
                    self.config.get("beta_start", 0.4),
                    self.config.get("beta_frames", 100000),
                )
            )
        else:
            self.replay_buffer = deque(
                maxlen=self.config.get("buffer_capacity", 100000)
            )

        # Multi-step buffer
        if self.multi_step > 1:
            self.multi_step_buffer = deque(maxlen=self.multi_step)

        # Training counters
        self.steps = 0
        self.training_step = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy or noisy networks."""
        if self.noisy_nets:
            # Reset noise for exploration
            if not evaluate:
                self.q_network.reset_noise()

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                if self.distributional:
                    q_values = (q_values * self.support).sum(dim=2)
                action = q_values.argmax(dim=1).item()
        else:
            # Epsilon-greedy exploration
            epsilon = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * math.exp(-1.0 * self.steps / self.epsilon_decay)

            if evaluate or random.random() > epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    if self.distributional:
                        q_values = (q_values * self.support).sum(dim=2)
                    action = q_values.argmax(dim=1).item()
            else:
                action = random.randrange(self.action_dim)

        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer."""
        if self.multi_step > 1:
            self.multi_step_buffer.append((state, action, reward, next_state, done))

            if len(self.multi_step_buffer) == self.multi_step:
                # Compute multi-step return
                multi_step_reward = 0
                for i, (_, _, r, _, d) in enumerate(self.multi_step_buffer):
                    multi_step_reward += (self.gamma**i) * r
                    if d:
                        break

                # Store multi-step transition
                first_state, first_action = self.multi_step_buffer[0][:2]
                last_state, last_done = self.multi_step_buffer[-1][3:]

                if self.prioritized_replay:
                    self.replay_buffer.push(
                        first_state,
                        first_action,
                        multi_step_reward,  # type: ignore
                        last_state,
                        last_done,
                    )
                else:
                    self.replay_buffer.append(
                        (
                            first_state,
                            first_action,
                            multi_step_reward,  # type: ignore
                            last_state,
                            last_done,
                        )
                    )
        else:
            if self.prioritized_replay:
                self.replay_buffer.push(state, action, reward, next_state, done)  # type: ignore
            else:
                self.replay_buffer.append((state, action, reward, next_state, done))  # type: ignore

    def update(self) -> dict[str, float]:
        """Update Rainbow DQN networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.steps += 1

        # Train every train_freq steps
        if self.steps % self.train_freq != 0:
            return {}

        # Sample batch
        if self.prioritized_replay:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights,
                indices,
            ) = self.replay_buffer.sample(
                self.batch_size
            )  # type: ignore
            weights = weights.to(self.device)
            indices = indices  # Keep indices for priority update
        else:
            batch = random.sample(list(self.replay_buffer), self.batch_size)  # type: ignore
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.ones(self.batch_size, device=self.device)
            indices = None

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states)

        if self.distributional:
            # Distributional RL (C51)
            current_q_dist = current_q_values[range(self.batch_size), actions]

            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN action selection
                    next_q_values = self.q_network(next_states)
                    next_q_means = (next_q_values * self.support).sum(dim=2)
                    next_actions = next_q_means.argmax(dim=1)
                    next_q_dist = self.target_network(next_states)[
                        range(self.batch_size), next_actions
                    ]
                else:
                    next_q_values = self.target_network(next_states)
                    next_q_means = (next_q_values * self.support).sum(dim=2)
                    next_actions = next_q_means.argmax(dim=1)
                    next_q_dist = next_q_values[range(self.batch_size), next_actions]

                # Compute target distribution
                target_support = rewards.unsqueeze(1) + (
                    self.gamma**self.multi_step
                ) * self.support.unsqueeze(0) * (~dones).float().unsqueeze(1)
                target_support = target_support.clamp(self.v_min, self.v_max)

                # Project onto support
                b = (target_support - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                target_q_dist = torch.zeros_like(next_q_dist)
                for i in range(self.batch_size):
                    for j in range(self.n_atoms):
                        if l[i, j] == u[i, j]:
                            target_q_dist[i, l[i, j]] += next_q_dist[i, j]
                        else:
                            target_q_dist[i, l[i, j]] += next_q_dist[i, j] * (
                                u[i, j] - b[i, j]
                            )
                            target_q_dist[i, u[i, j]] += next_q_dist[i, j] * (
                                b[i, j] - l[i, j]
                            )

            # Cross-entropy loss
            loss = -torch.sum(target_q_dist * torch.log(current_q_dist + 1e-8), dim=1)
        else:
            # Standard DQN
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN
                    next_actions = self.q_network(next_states).argmax(dim=1)
                    next_q = (
                        self.target_network(next_states)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                else:
                    next_q = self.target_network(next_states).max(1)[0]

                target_q = (
                    rewards + (self.gamma**self.multi_step) * next_q * (~dones).float()
                )

            loss = F.mse_loss(current_q, target_q, reduction="none")

        # Apply importance sampling weights
        loss = loss * weights
        loss = loss.mean()

        # Update priorities
        if self.prioritized_replay and indices is not None:
            priorities = loss.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)  # type: ignore

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset noise
        if self.noisy_nets:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_step += 1

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * math.exp(-1.0 * self.steps / self.epsilon_decay),
            "steps": self.steps,
            "training_step": self.training_step,
        }

    def save(self, filepath: str):
        """Save agent state."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps": self.steps,
                "training_step": self.training_step,
                "config": self.config,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps = checkpoint.get("steps", 0)
        self.training_step = checkpoint.get("training_step", 0)


if __name__ == "__main__":
    # Test Rainbow DQN agent
    print("=== Rainbow DQN Agent Test ===")

    # Create agent
    state_dim = 10
    action_dim = 4
    agent = RainbowDQNAgent(state_dim, action_dim)

    print(
        f"✅ Created Rainbow DQN agent with state_dim={state_dim}, action_dim={action_dim}"
    )
    print(
        f"   Features: Double DQN, Dueling, Noisy Nets, Prioritized Replay, Multi-step, Distributional"
    )

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.select_action(state)

    print(f"✅ Action selection test: action={action}")

    # Test experience storage and training
    print("\n📈 Testing experience collection...")
    for i in range(1000):
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = random.random() < 0.01  # 1% chance of episode end

        agent.store_experience(state, action, reward, next_state, done)

    print(f"✅ Stored {len(agent.replay_buffer)} experiences")

    # Test updates
    print("\n🔄 Testing updates...")
    for i in range(10):
        metrics = agent.update()
        if metrics:
            print(
                f"   Step {i+1}: loss={metrics['loss']:.4f}, epsilon={metrics['epsilon']:.4f}"
            )

    print("\n🎯 Rainbow DQN agent test complete!")
