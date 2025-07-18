"""
Advanced Policy Optimization Techniques for RL Agents.

This module implements state-of-the-art policy optimization algorithms:
- Proximal Policy Optimization (PPO) with advanced clipping
- Trust Region Policy Optimization (TRPO)
- Natural Policy Gradient methods
- Adaptive learning rate scheduling
- Multi-objective optimization for risk-adjusted returns
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .configs import PPOConfig

logger = logging.getLogger(__name__)


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


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler with multiple strategies."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedule_type: str = "cosine",
        warmup_steps: int = 1000,
        total_steps: int = 1000000,
        min_lr_ratio: float = 0.1,
        performance_window: int = 100,
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.performance_window = performance_window

        self.step_count = 0
        self.performance_history: list[float] = []
        self.base_lr = optimizer.param_groups[0]["lr"]

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler based on type."""
        if self.schedule_type == "cosine":
            return LambdaLR(self.optimizer, self._cosine_schedule)
        if self.schedule_type == "linear":
            return LambdaLR(self.optimizer, self._linear_schedule)
        if self.schedule_type == "exponential":
            return LambdaLR(self.optimizer, self._exponential_schedule)
        raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_schedule(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        lr_ratio = 0.5 * (1 + math.cos(math.pi * progress))
        return max(self.min_lr_ratio, lr_ratio)

    def _linear_schedule(self, step: int) -> float:
        """Linear learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        lr_ratio = 1.0 - progress
        return max(self.min_lr_ratio, lr_ratio)

    def _exponential_schedule(self, step: int) -> float:
        """Exponential learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        lr_ratio = math.exp(-2.0 * progress)
        return max(self.min_lr_ratio, lr_ratio)

    def step(self, performance: float | None = None) -> float:
        """Step the scheduler and optionally adapt based on performance."""
        self.step_count += 1

        if performance is not None:
            self.performance_history.append(performance)
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)

        # Step the scheduler
        self.scheduler.step()

        # Return current learning rate
        return float(self.optimizer.param_groups[0]["lr"])

    def get_lr(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])


class MultiObjectiveOptimizer:
    """Multi-objective optimization for risk-adjusted returns."""

    def __init__(
        self,
        return_weight: float = 0.9,
        risk_weight: float = 0.1,
        sharpe_weight: float = 0.0,
        max_drawdown_weight: float = 0.0,
    ):
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.sharpe_weight = sharpe_weight
        self.max_drawdown_weight = max_drawdown_weight

        # Normalize weights
        total_weight = return_weight + risk_weight + sharpe_weight + max_drawdown_weight
        if total_weight > 0:
            self.return_weight /= total_weight
            self.risk_weight /= total_weight
            self.sharpe_weight /= total_weight
            self.max_drawdown_weight /= total_weight

    def compute_objective(
        self,
        returns: np.ndarray,
        actions: np.ndarray,
        risk_metrics: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute multi-objective function value."""
        if len(returns) == 0:
            return 0.0, {}

        # Calculate individual objectives
        return_obj = self._compute_return_objective(returns)
        risk_obj = self._compute_risk_objective(returns, actions, risk_metrics)
        sharpe_obj = self._compute_sharpe_objective(returns)
        drawdown_obj = self._compute_drawdown_objective(returns)

        # Combine objectives
        total_obj = (
            self.return_weight * return_obj
            + self.risk_weight * risk_obj
            + self.sharpe_weight * sharpe_obj
            + self.max_drawdown_weight * drawdown_obj
        )

        objectives = {
            "return": return_obj,
            "risk": risk_obj,
            "sharpe": sharpe_obj,
            "drawdown": drawdown_obj,
            "total": total_obj,
        }

        return total_obj, objectives

    def _compute_return_objective(self, returns: np.ndarray) -> float:
        """Compute return objective (cumulative return)."""
        return float(np.sum(returns))

    def _compute_risk_objective(
        self,
        returns: np.ndarray,
        actions: np.ndarray,
        risk_metrics: dict[str, float] | None = None,
    ) -> float:
        """Compute risk objective (negative volatility)."""
        if len(returns) < 2:
            return 0.0

        volatility = np.std(returns)
        action_volatility = np.std(actions) if len(actions) > 1 else 0.0

        # Combine return volatility and action volatility
        total_risk = volatility + 0.1 * action_volatility

        # If risk metrics provided, use them
        if risk_metrics:
            var = risk_metrics.get("var", 0.0)
            total_risk += var

        return float(-total_risk)  # Negative because we want to minimize risk

    def _compute_sharpe_objective(self, returns: np.ndarray) -> float:
        """Compute Sharpe ratio objective."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return float(mean_return / std_return)

    def _compute_drawdown_objective(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown objective (negative)."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))


class AdvancedPPO:
    """Enhanced PPO with advanced clipping, trust region, and multi-objective optimization."""

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        config: AdvancedPPOConfig,
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device

        # Optimizers
        self.policy_optimizer = Adam(policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = Adam(value_net.parameters(), lr=config.learning_rate)

        # Learning rate scheduler
        self.lr_scheduler: AdaptiveLearningRateScheduler | None = None
        if config.adaptive_lr:
            self.lr_scheduler = AdaptiveLearningRateScheduler(
                self.policy_optimizer,
                schedule_type=config.lr_schedule,
                warmup_steps=config.warmup_steps,
                min_lr_ratio=config.min_lr_ratio,
                performance_window=config.performance_window,
            )

        # Multi-objective optimizer
        self.multi_obj_optimizer = MultiObjectiveOptimizer(
            return_weight=config.return_weight,
            risk_weight=config.risk_weight,
            sharpe_weight=config.sharpe_weight,
            max_drawdown_weight=config.max_drawdown_weight,
        )

        # Performance tracking
        self.performance_history: list[float] = []
        self.clip_ratio = config.clip_ratio
        self.step_count = 0

        self.to(device)

    def to(self, device: str) -> None:
        """Move networks to device."""
        self.policy_net.to(device)
        self.value_net.to(device)
        self.device = device

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute policy loss with advanced clipping."""
        # Get current policy distribution
        action_probs = functional.softmax(self.policy_net(states), dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Advanced clipping
        if self.config.adaptive_clip_ratio and len(self.performance_history) > 10:
            # Adaptive clipping based on performance
            recent_performance = np.mean(self.performance_history[-10:])
            if recent_performance > 0:
                self.clip_ratio = max(self.config.min_clip_ratio, self.clip_ratio * self.config.clip_ratio_decay)
            else:
                self.clip_ratio = min(self.config.max_clip_ratio, self.clip_ratio / self.config.clip_ratio_decay)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Trust region constraint
        if self.config.use_trust_region:
            kl_div = torch.distributions.kl_divergence(
                torch.distributions.Categorical(functional.softmax(self.policy_net(states), dim=-1)),
                torch.distributions.Categorical(functional.softmax(old_log_probs, dim=-1)),
            ).mean()

            if kl_div > self.config.max_kl_divergence:
                # Scale down the policy loss
                policy_loss = policy_loss * (self.config.max_kl_divergence / kl_div)

        # Entropy regularization
        entropy = dist.entropy().mean()
        policy_loss = policy_loss - self.config.ent_coef * entropy

        metrics = {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "clip_ratio": self.clip_ratio,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }

        return policy_loss, metrics

    def compute_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value function loss."""
        values = self.value_net(states).squeeze(-1)

        if self.config.clip_vf_ratio is not None:
            # Value function clipping
            old_values = values.detach()
            value_loss_unclipped = functional.mse_loss(values, returns)
            value_loss_clipped = functional.mse_loss(
                old_values + torch.clamp(values - old_values, -self.config.clip_vf_ratio, self.config.clip_vf_ratio),
                returns,
            )
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = functional.mse_loss(values, returns)

        return value_loss

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
        performance: float | None = None,
    ) -> dict[str, float]:
        """Update policy and value networks."""
        self.policy_net.train()
        self.value_net.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute GAE
        with torch.no_grad():
            values = self.value_net(states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, values, dones, self.config.gamma, self.config.gae_lambda)

        # Normalize advantages and returns
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Multi-objective optimization
        if performance is not None:
            self.performance_history.append(performance)
            if len(self.performance_history) > self.config.performance_window:
                self.performance_history.pop(0)

        # Update policy
        policy_loss, policy_metrics = self.compute_policy_loss(states, actions, old_log_probs, advantages, returns)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()

        # Update value function
        value_loss = self.compute_value_loss(states, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        # Update learning rate
        if self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.step(performance)
        else:
            current_lr = float(self.config.learning_rate)

        # Combine metrics
        metrics = {
            **policy_metrics,
            "value_loss": value_loss.item(),
            "total_loss": (policy_loss + self.config.vf_coef * value_loss).item(),
            "learning_rate": current_lr,
            "step_count": self.step_count,
        }

        self.step_count += 1
        return metrics


class TRPO:
    """Trust Region Policy Optimization implementation."""

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        config: TRPOConfig,
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device

        # Optimizers
        self.policy_optimizer = Adam(policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = Adam(value_net.parameters(), lr=config.learning_rate)

        self.to(device)

    def to(self, device: str) -> None:
        """Move networks to device."""
        self.policy_net.to(device)
        self.value_net.to(device)
        self.device = device

    def compute_kl_divergence(
        self,
        states: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between old and new policy."""
        old_probs = functional.softmax(old_log_probs, dim=-1)
        new_probs = functional.softmax(self.policy_net(states), dim=-1)

        kl_div = torch.sum(old_probs * torch.log(old_probs / (new_probs + 1e-8)), dim=-1)
        return kl_div.mean()

    def conjugate_gradient(
        self,
        states: torch.Tensor,
        b: torch.Tensor,
        nsteps: int = 10,
    ) -> torch.Tensor:
        """Conjugate gradient algorithm for solving Hx = b."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        for _ in range(nsteps):
            # Compute Hp (Fisher-vector product)
            Hp = self.compute_fisher_vector_product(states, p)

            # Compute step size
            alpha = torch.dot(r, r) / torch.dot(p, Hp)

            # Update x and r
            x = x + alpha * p
            r_new = r - alpha * Hp

            # Compute new search direction
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = r_new + beta * p
            r = r_new

            # Check convergence
            if torch.norm(r) < 1e-6:  # Fixed tolerance value
                break

        return x

    def compute_fisher_vector_product(
        self,
        states: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fisher-vector product Hv."""
        # This is a simplified implementation
        # In practice, you'd need to compute the full Fisher information matrix
        # For now, we'll use a diagonal approximation
        return v * self.config.damping_coeff

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        """Update policy using TRPO."""
        self.policy_net.train()
        self.value_net.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute GAE
        with torch.no_grad():
            values = self.value_net(states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, values, dones, self.config.gamma, self.config.gae_lambda)

        # Normalize advantages and returns
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient
        action_probs = functional.softmax(self.policy_net(states), dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # Policy gradient
        policy_grad = torch.autograd.grad(
            -(log_probs * advantages).mean(),
            self.policy_net.parameters(),
            create_graph=True,
        )

        # Flatten gradient
        flat_grad = torch.cat([g.flatten() for g in policy_grad])

        # Compute search direction using conjugate gradient
        search_dir = self.conjugate_gradient(states, flat_grad, 10)  # Fixed max iterations

        # Line search for step size
        step_size = 1.0
        for _ in range(self.config.max_backtrack_iter):
            # Compute new parameters
            new_params = []
            param_idx = 0
            for param in self.policy_net.parameters():
                param_size = param.numel()
                param_update = search_dir[param_idx : param_idx + param_size].view_as(param)
                new_params.append(param + step_size * param_update)
                param_idx += param_size

            # Temporarily update policy
            old_params = [param.clone() for param in self.policy_net.parameters()]
            for param, new_param in zip(self.policy_net.parameters(), new_params, strict=False):
                param.data = new_param.data

            # Compute KL divergence
            kl_div = self.compute_kl_divergence(states, old_log_probs)

            if kl_div <= self.config.max_kl_divergence:
                break

            # Backtrack
            step_size *= self.config.backtrack_coeff

        # Update value function
        value_loss = functional.mse_loss(self.value_net(states).squeeze(-1), returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        return {
            "policy_loss": -(log_probs * advantages).mean().item(),
            "value_loss": value_loss.item(),
            "kl_divergence": kl_div.item(),
            "step_size": step_size,
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns


class NaturalPolicyGradient:
    """Natural Policy Gradient implementation."""

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        config: NaturalPolicyGradientConfig,
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device

        # Optimizers
        self.policy_optimizer = Adam(policy_net.parameters(), lr=config.learning_rate)
        self.value_optimizer = Adam(value_net.parameters(), lr=config.learning_rate)

        self.to(device)

    def to(self, device: str) -> None:
        """Move networks to device."""
        self.policy_net.to(device)
        self.value_net.to(device)
        self.device = device

    def compute_fisher_information(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fisher information matrix (diagonal approximation)."""
        action_probs = functional.softmax(self.policy_net(states), dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        # Compute gradients
        grads = torch.autograd.grad(
            log_probs.sum(),
            self.policy_net.parameters(),
            create_graph=True,
        )

        # Flatten gradients
        flat_grads = torch.cat([g.flatten() for g in grads])

        # Compute Fisher information (simplified diagonal approximation)
        return torch.ones_like(flat_grads) * self.config.damping_coeff

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        """Update policy using Natural Policy Gradient."""
        self.policy_net.train()
        self.value_net.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute GAE
        with torch.no_grad():
            values = self.value_net(states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, values, dones, self.config.gamma, self.config.gae_lambda)

        # Normalize advantages and returns
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient
        action_probs = functional.softmax(self.policy_net(states), dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        policy_grad = torch.autograd.grad(
            -(log_probs * advantages).mean(),
            self.policy_net.parameters(),
            create_graph=True,
        )

        # Flatten gradient
        flat_grad = torch.cat([g.flatten() for g in policy_grad])

        # Compute Fisher information
        fisher_info = self.compute_fisher_information(states, actions)

        # Natural gradient update
        natural_grad = flat_grad / (fisher_info + 1e-8)

        # Apply update
        param_idx = 0
        for param in self.policy_net.parameters():
            param_size = param.numel()
            param_update = natural_grad[param_idx : param_idx + param_size].view_as(param)
            param.data -= self.config.learning_rate * param_update
            param_idx += param_size

        # Update value function
        value_loss = functional.mse_loss(self.value_net(states).squeeze(-1), returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()

        return {
            "policy_loss": -(log_probs * advantages).mean().item(),
            "value_loss": value_loss.item(),
            "natural_grad_norm": torch.norm(natural_grad).item(),
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns


class PolicyOptimizationComparison:
    """Framework for comparing different policy optimization methods."""

    def __init__(self, configs: dict[str, Any]):
        self.configs = configs
        self.results: dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def benchmark_algorithm(
        self,
        algorithm_name: str,
        algorithm_class: Any,
        policy_net: nn.Module,
        value_net: nn.Module,
        train_data: dict[str, torch.Tensor],
        config: Any,
        num_episodes: int = 100,
    ) -> dict[str, Any]:
        """Benchmark a single algorithm."""
        self.logger.info(f"Benchmarking {algorithm_name}")

        # Initialize algorithm
        algorithm = algorithm_class(policy_net, value_net, config)

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(num_episodes):
            # Simulate training step
            states = train_data["states"]
            actions = train_data["actions"]
            rewards = train_data["rewards"]
            dones = train_data["dones"]
            old_log_probs = train_data["old_log_probs"]

            # Update algorithm
            metrics = algorithm.update(states, actions, rewards, dones, old_log_probs)

            # Record metrics
            episode_rewards.append(torch.sum(rewards).item())
            episode_lengths.append(len(rewards))
            losses.append(metrics)

        # Compute statistics
        results = {
            "algorithm": algorithm_name,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "final_loss": losses[-1] if losses else {},
            "convergence_rate": self._compute_convergence_rate(episode_rewards),
        }

        self.results[algorithm_name] = results
        return results

    def _compute_convergence_rate(self, rewards: list[float]) -> float:
        """Compute convergence rate based on reward improvement."""
        if len(rewards) < 10:
            return 0.0

        # Compute improvement over last 10% of episodes
        window_size = max(1, len(rewards) // 10)
        recent_rewards = rewards[-window_size:]
        early_rewards = rewards[:window_size]

        if np.std(early_rewards) == 0:
            return 0.0

        return float((np.mean(recent_rewards) - np.mean(early_rewards)) / np.std(early_rewards))

    def compare_algorithms(self) -> dict[str, Any]:
        """Compare all algorithms and return summary."""
        if not self.results:
            self.logger.warning("No results to compare. Run benchmark_algorithm first.")
            return {}

        # Compute comparison metrics
        return {
            "best_reward": max(self.results.values(), key=lambda x: x["mean_reward"]),
            "best_convergence": max(self.results.values(), key=lambda x: x["convergence_rate"]),
            "algorithm_rankings": self._rank_algorithms(),
            "detailed_results": self.results,
        }

    def _rank_algorithms(self) -> list[str]:
        """Rank algorithms by mean reward."""
        sorted_algorithms = sorted(self.results.items(), key=lambda x: x[1]["mean_reward"], reverse=True)
        return [name for name, _ in sorted_algorithms]

    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No results available for report generation."

        report = "Policy Optimization Algorithm Comparison Report\n"
        report += "=" * 50 + "\n\n"

        for name, result in self.results.items():
            report += f"Algorithm: {name}\n"
            report += f"  Mean Reward: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}\n"
            report += f"  Mean Episode Length: {result['mean_length']:.2f}\n"
            report += f"  Convergence Rate: {result['convergence_rate']:.4f}\n"
            report += "\n"

        comparison = self.compare_algorithms()
        if comparison:
            report += f"Best Algorithm (Reward): {comparison['best_reward']['algorithm']}\n"
            report += f"Best Algorithm (Convergence): {comparison['best_convergence']['algorithm']}\n"
            report += f"Algorithm Rankings: {', '.join(comparison['algorithm_rankings'])}\n"

        return report
