"""
Advanced Trainer with Policy Optimization Techniques.

This module extends the basic trainer with advanced policy optimization algorithms
and provides a unified interface for training RL agents with different optimization methods.
"""


import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .advanced_policy_optimization import (
    TRPO,
    AdvancedPPO,
    AdvancedPPOConfig,
    MultiObjectiveOptimizer,
    NaturalPolicyGradient,
    NaturalPolicyGradientConfig,
    PolicyOptimizationComparison,
    TRPOConfig,
)
from .configs import MultiObjectiveConfig

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Policy network for RL agents."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(0.1),
                ],
            )
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "tanh":
            return nn.Tanh()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        return nn.Tanh()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy network."""
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for RL agents."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        super().__init__()
        self.state_dim = state_dim

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(0.1),
                ],
            )
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "tanh":
            return nn.Tanh()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        return nn.Tanh()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network."""
        return self.network(state)


class ExperienceBuffer:
    """Experience replay buffer for RL training."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: list[dict[str, Any]] = []
        self.position = 0

    def push(self, experience: dict[str, Any]) -> None:
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        batch_size = min(batch_size, len(self.buffer))

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Convert to tensors
        states = torch.stack([torch.FloatTensor(exp["state"]) for exp in batch])
        actions = torch.stack([torch.LongTensor([exp["action"]]) for exp in batch])
        rewards = torch.stack([torch.FloatTensor([exp["reward"]]) for exp in batch])
        next_states = torch.stack([torch.FloatTensor(exp["next_state"]) for exp in batch])
        dones = torch.stack([torch.FloatTensor([exp["done"]]) for exp in batch])
        old_log_probs = torch.stack([torch.FloatTensor([exp["log_prob"]]) for exp in batch])

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "old_log_probs": old_log_probs,
        }

    def __len__(self) -> int:
        return len(self.buffer)


class AdvancedTrainer:
    """Advanced trainer with multiple policy optimization algorithms."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        save_dir: str = "outputs",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        # Move to device
        self.policy_net.to(device)
        self.value_net.to(device)

        # Experience buffer
        self.buffer = ExperienceBuffer()

        # Training state
        self.current_algorithm: AdvancedPPO | TRPO | NaturalPolicyGradient | None = None
        self.training_history: list[dict[str, Any]] = []

        logger.info(f"Advanced trainer initialized on {device}")

    def create_algorithm(
        self,
        algorithm_type: str,
        config: AdvancedPPOConfig | TRPOConfig | NaturalPolicyGradientConfig,
    ) -> AdvancedPPO | TRPO | NaturalPolicyGradient:
        """Create policy optimization algorithm."""
        if algorithm_type == "advanced_ppo" and isinstance(config, AdvancedPPOConfig):
            return AdvancedPPO(self.policy_net, self.value_net, config, self.device)
        if algorithm_type == "trpo" and isinstance(config, TRPOConfig):
            return TRPO(self.policy_net, self.value_net, config, self.device)
        if algorithm_type == "natural_policy_gradient" and isinstance(config, NaturalPolicyGradientConfig):
            return NaturalPolicyGradient(self.policy_net, self.value_net, config, self.device)
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    def train_episode(
        self,
        algorithm: AdvancedPPO | TRPO | NaturalPolicyGradient,
        env: Any,
        max_steps: int = 1000,
        render: bool = False,
    ) -> dict[str, Any]:
        """Train for one episode."""
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []

        for step in range(max_steps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_probs = torch.nn.functional.softmax(self.policy_net(state_tensor), dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action.item())

            # Store experience
            experience = {
                "state": state,
                "action": action.item(),
                "reward": reward,
                "next_state": next_state,
                "done": done or truncated,
                "log_prob": log_prob.item(),
            }
            self.buffer.push(experience)

            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            state = next_state

            # Render if requested
            if render:
                env.render()

            if done or truncated:
                break

        # Update algorithm if we have enough data
        if len(self.buffer) >= algorithm.config.batch_size:
            batch = self.buffer.sample(algorithm.config.batch_size)
            metrics = algorithm.update(
                states=batch["states"],
                actions=batch["actions"].squeeze(-1),
                rewards=batch["rewards"].squeeze(-1),
                dones=batch["dones"].squeeze(-1),
                old_log_probs=batch["old_log_probs"].squeeze(-1),
            )
            episode_losses.append(metrics)

        episode_stats = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "losses": episode_losses,
        }

        self.training_history.append(episode_stats)
        return episode_stats

    def train(
        self,
        algorithm_type: str,
        config: AdvancedPPOConfig | TRPOConfig | NaturalPolicyGradientConfig,
        env: Any,
        num_episodes: int = 1000,
        eval_frequency: int = 100,
        save_frequency: int = 500,
    ) -> dict[str, Any]:
        """Train the agent using specified algorithm."""
        logger.info(f"Starting training with {algorithm_type}")

        # Create algorithm
        algorithm = self.create_algorithm(algorithm_type, config)
        self.current_algorithm = algorithm

        # Training loop
        episode_rewards = []
        episode_lengths = []
        training_metrics = []

        for episode in range(num_episodes):
            # Train episode
            episode_stats = self.train_episode(algorithm, env)

            episode_rewards.append(episode_stats["episode_reward"])
            episode_lengths.append(episode_stats["episode_length"])

            if episode_stats["losses"]:
                training_metrics.extend(episode_stats["losses"])

            # Log progress
            if (episode + 1) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                avg_reward = np.mean(recent_rewards)
                logger.info(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

            # Evaluation
            if (episode + 1) % eval_frequency == 0:
                eval_metrics = self.evaluate(env, num_episodes=10)
                logger.info(f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}")

            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                self.save_checkpoint(f"checkpoint_episode_{episode + 1}.pth")

        # Final evaluation
        final_eval = self.evaluate(env, num_episodes=50)

        training_results = {
            "algorithm": algorithm_type,
            "num_episodes": num_episodes,
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "final_avg_length": np.mean(episode_lengths[-100:]),
            "training_metrics": training_metrics,
            "final_evaluation": final_eval,
        }

        logger.info(f"Training completed. Final avg reward: {training_results['final_avg_reward']:.2f}")
        return training_results

    def evaluate(
        self,
        env: Any,
        num_episodes: int = 10,
        render: bool = False,
    ) -> dict[str, float]:
        """Evaluate the current policy."""
        self.policy_net.eval()

        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_probs = torch.nn.functional.softmax(self.policy_net(state_tensor), dim=-1)
                    action = torch.argmax(action_probs, dim=-1)

                state, reward, done, truncated, _ = env.step(action.item())

                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        self.policy_net.train()

        return {
            "avg_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "avg_length": float(np.mean(episode_lengths)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
        }

    def benchmark_algorithms(
        self,
        _env: Any,
        algorithms: list[tuple[str, AdvancedPPOConfig | TRPOConfig | NaturalPolicyGradientConfig]],
        num_episodes: int = 100,
    ) -> dict[str, Any]:
        """Benchmark multiple algorithms."""
        logger.info("Starting algorithm benchmark")

        comparison = PolicyOptimizationComparison({})
        results = {}

        for algorithm_type, config in algorithms:
            logger.info(f"Benchmarking {algorithm_type}")

            # Create fresh networks for each algorithm
            policy_net = PolicyNetwork(self.state_dim, self.action_dim)
            value_net = ValueNetwork(self.state_dim)

            # Create algorithm
            algorithm = self.create_algorithm(algorithm_type, config)

            # Generate synthetic training data for benchmark
            train_data = self._generate_synthetic_data(config.batch_size)

            # Benchmark
            result = comparison.benchmark_algorithm(
                algorithm_type,
                type(algorithm),
                policy_net,
                value_net,
                train_data,
                config,
                num_episodes,
            )

            results[algorithm_type] = result

        # Generate comparison report
        comparison_report = comparison.generate_report()
        logger.info("Algorithm benchmark completed")

        return {
            "results": results,
            "comparison": comparison.compare_algorithms(),
            "report": comparison_report,
        }

    def _generate_synthetic_data(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate synthetic training data for benchmarking."""
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randint(0, self.action_dim, (batch_size,))
        rewards = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()
        old_log_probs = torch.randn(batch_size, 1)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "old_log_probs": old_log_probs,
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "training_history": self.training_history,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = self.save_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)  # nosec

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_training_summary(self) -> dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_history:
            return {}

        rewards = [ep["episode_reward"] for ep in self.training_history]
        lengths = [ep["episode_length"] for ep in self.training_history]

        return {
            "total_episodes": len(self.training_history),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_length": np.mean(lengths),
            "best_reward": np.max(rewards),
            "worst_reward": np.min(rewards),
            "recent_avg_reward": (np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)),
        }


class MultiObjectiveTrainer(AdvancedTrainer):
    """Trainer with multi-objective optimization capabilities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        multi_obj_config: MultiObjectiveConfig,
        device: str = "cpu",
        save_dir: str = "outputs",
    ):
        super().__init__(state_dim, action_dim, device, save_dir)
        self.multi_obj_config = multi_obj_config
        self.multi_obj_optimizer = MultiObjectiveOptimizer(
            return_weight=multi_obj_config.return_weight,
            risk_weight=multi_obj_config.risk_weight,
            sharpe_weight=multi_obj_config.sharpe_weight,
            max_drawdown_weight=multi_obj_config.max_drawdown_weight,
        )

        # Performance tracking
        self.returns_history: list[float] = []
        self.risk_metrics_history: list[dict[str, float]] = []

    def train_episode(
        self,
        algorithm: AdvancedPPO | TRPO | NaturalPolicyGradient,
        env: Any,
        max_steps: int = 1000,
        render: bool = False,
    ) -> dict[str, Any]:
        """Train episode with multi-objective optimization."""
        episode_stats = super().train_episode(algorithm, env, max_steps, render)

        # Compute multi-objective metrics
        if len(self.returns_history) > 0:
            returns = np.array(self.returns_history)
            actions = np.array([ep["action"] for ep in self.training_history[-len(returns) :]])

            # Get risk metrics from recent history
            risk_metrics = self._compute_risk_metrics(returns)

            # Compute multi-objective value
            obj_value, objectives = self.multi_obj_optimizer.compute_objective(returns, actions, risk_metrics)

            episode_stats["multi_objective"] = objectives
            episode_stats["obj_value"] = obj_value

        return episode_stats

    def _compute_risk_metrics(self, returns: np.ndarray) -> dict[str, float]:
        """Compute risk metrics from returns."""
        if len(returns) < 2:
            return {"var": 0.0, "volatility": 0.0, "max_drawdown": 0.0}

        # Value at Risk
        var = np.percentile(returns, self.multi_obj_config.var_alpha * 100)

        # Volatility
        volatility = np.std(returns)

        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)

        return {
            "var": var,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
        }

    def update_returns_history(self, reward: float) -> None:
        """Update returns history for multi-objective optimization."""
        self.returns_history.append(reward)

        # Keep only recent history
        if len(self.returns_history) > self.multi_obj_config.performance_window:
            self.returns_history.pop(0)
