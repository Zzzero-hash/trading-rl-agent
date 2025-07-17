"""
Tests for Advanced Policy Optimization.

This module tests the advanced policy optimization algorithms and utilities.
"""

import numpy as np
import pytest
import torch

from trading_rl_agent.agents.advanced_policy_optimization import (
    TRPO,
    AdaptiveLearningRateScheduler,
    AdvancedPPO,
    MultiObjectiveOptimizer,
    NaturalPolicyGradient,
)
from trading_rl_agent.agents.advanced_trainer import AdvancedTrainer, PolicyNetwork, ValueNetwork
from trading_rl_agent.agents.configs import (
    AdvancedPPOConfig,
    MultiObjectiveConfig,
    NaturalPolicyGradientConfig,
    TRPOConfig,
)


class TestPolicyNetworks:
    """Test policy and value networks."""

    def test_policy_network(self):
        """Test policy network initialization and forward pass."""
        state_dim = 50
        action_dim = 3
        hidden_dims = [256, 128]

        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims)

        # Test forward pass
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        outputs = policy_net(states)

        assert outputs.shape == (batch_size, action_dim)
        assert not torch.isnan(outputs).any()

    def test_value_network(self):
        """Test value network initialization and forward pass."""
        state_dim = 50
        hidden_dims = [256, 128]

        value_net = ValueNetwork(state_dim, hidden_dims)

        # Test forward pass
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        outputs = value_net(states)

        assert outputs.shape == (batch_size, 1)
        assert not torch.isnan(outputs).any()


class TestAdvancedPPO:
    """Test Advanced PPO implementation."""

    def test_advanced_ppo_initialization(self):
        """Test Advanced PPO initialization."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = AdvancedPPOConfig()

        ppo = AdvancedPPO(policy_net, value_net, config)

        assert ppo.policy_net is policy_net
        assert ppo.value_net is value_net
        assert ppo.config is config

    def test_gae_computation(self):
        """Test Generalized Advantage Estimation computation."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = AdvancedPPOConfig()

        ppo = AdvancedPPO(policy_net, value_net, config)

        # Create dummy data
        batch_size = 32
        rewards = torch.randn(batch_size, 1)
        values = torch.randn(batch_size, 1)
        dones = torch.randint(0, 2, (batch_size, 1)).float()

        advantages, returns = ppo.compute_gae(rewards, values, dones)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_policy_loss_computation(self):
        """Test policy loss computation."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = AdvancedPPOConfig()

        ppo = AdvancedPPO(policy_net, value_net, config)

        # Create dummy data
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        old_log_probs = torch.randn(batch_size, 1)
        advantages = torch.randn(batch_size, 1)
        returns = torch.randn(batch_size, 1)

        loss, metrics = ppo.compute_policy_loss(states, actions, old_log_probs, advantages, returns)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert "entropy" in metrics


class TestTRPO:
    """Test TRPO implementation."""

    def test_trpo_initialization(self):
        """Test TRPO initialization."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = TRPOConfig()

        trpo = TRPO(policy_net, value_net, config)

        assert trpo.policy_net is policy_net
        assert trpo.value_net is value_net
        assert trpo.config is config

    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = TRPOConfig()

        trpo = TRPO(policy_net, value_net, config)

        # Create dummy data
        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        old_log_probs = torch.randn(batch_size, action_dim)

        kl_div = trpo.compute_kl_divergence(states, old_log_probs)

        assert isinstance(kl_div, torch.Tensor)
        assert kl_div.requires_grad
        assert kl_div.item() >= 0  # KL divergence should be non-negative


class TestNaturalPolicyGradient:
    """Test Natural Policy Gradient implementation."""

    def test_natural_policy_gradient_initialization(self):
        """Test Natural Policy Gradient initialization."""
        state_dim = 50
        action_dim = 3

        policy_net = PolicyNetwork(state_dim, action_dim)
        value_net = ValueNetwork(state_dim)
        config = NaturalPolicyGradientConfig()

        npg = NaturalPolicyGradient(policy_net, value_net, config)

        assert npg.policy_net is policy_net
        assert npg.value_net is value_net
        assert npg.config is config


class TestAdaptiveLearningRateScheduler:
    """Test adaptive learning rate scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])

        scheduler = AdaptiveLearningRateScheduler(
            optimizer,
            schedule_type="cosine",
            warmup_steps=100,
            total_steps=1000,
        )

        assert scheduler.optimizer is optimizer
        assert scheduler.schedule_type == "cosine"
        assert scheduler.warmup_steps == 100

    def test_cosine_schedule(self):
        """Test cosine learning rate schedule."""
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])

        scheduler = AdaptiveLearningRateScheduler(
            optimizer,
            schedule_type="cosine",
            warmup_steps=100,
            total_steps=1000,
        )

        # Test warmup phase
        lr = scheduler.step()
        assert lr > 0

        # Test after warmup
        for _ in range(200):
            lr = scheduler.step()

        assert lr < scheduler.base_lr  # Should have decreased


class TestMultiObjectiveOptimizer:
    """Test multi-objective optimizer."""

    def test_optimizer_initialization(self):
        """Test multi-objective optimizer initialization."""
        optimizer = MultiObjectiveOptimizer(
            return_weight=0.8,
            risk_weight=0.2,
        )

        assert optimizer.return_weight == 0.8
        assert optimizer.risk_weight == 0.2

    def test_objective_computation(self):
        """Test objective function computation."""
        optimizer = MultiObjectiveOptimizer(
            return_weight=0.8,
            risk_weight=0.2,
        )

        # Create dummy data
        returns = np.array([0.1, -0.05, 0.2, -0.1, 0.15])
        actions = np.array([1, 0, 1, 2, 1])

        obj_value, objectives = optimizer.compute_objective(returns, actions)

        assert isinstance(obj_value, float)
        assert isinstance(objectives, dict)
        assert "return" in objectives
        assert "risk" in objectives
        assert "total" in objectives


class TestAdvancedTrainer:
    """Test advanced trainer."""

    def test_trainer_initialization(self):
        """Test advanced trainer initialization."""
        trainer = AdvancedTrainer(
            state_dim=50,
            action_dim=3,
            device="cpu",
        )

        assert trainer.state_dim == 50
        assert trainer.action_dim == 3
        assert trainer.device == "cpu"
        assert len(trainer.buffer) == 0

    def test_algorithm_creation(self):
        """Test algorithm creation."""
        trainer = AdvancedTrainer(
            state_dim=50,
            action_dim=3,
            device="cpu",
        )

        # Test Advanced PPO
        config = AdvancedPPOConfig()
        algorithm = trainer.create_algorithm("advanced_ppo", config)
        assert isinstance(algorithm, AdvancedPPO)

        # Test TRPO
        config = TRPOConfig()
        algorithm = trainer.create_algorithm("trpo", config)
        assert isinstance(algorithm, TRPO)

        # Test Natural Policy Gradient
        config = NaturalPolicyGradientConfig()
        algorithm = trainer.create_algorithm("natural_policy_gradient", config)
        assert isinstance(algorithm, NaturalPolicyGradient)

    def test_invalid_algorithm(self):
        """Test invalid algorithm creation."""
        trainer = AdvancedTrainer(
            state_dim=50,
            action_dim=3,
            device="cpu",
        )

        config = AdvancedPPOConfig()

        with pytest.raises(ValueError):
            trainer.create_algorithm("invalid_algorithm", config)


class TestMultiObjectiveTrainer:
    """Test multi-objective trainer."""

    def test_trainer_initialization(self):
        """Test multi-objective trainer initialization."""
        multi_obj_config = MultiObjectiveConfig()

        trainer = MultiObjectiveTrainer(
            state_dim=50,
            action_dim=3,
            multi_obj_config=multi_obj_config,
            device="cpu",
        )

        assert trainer.multi_obj_config is multi_obj_config
        assert isinstance(trainer.multi_obj_optimizer, MultiObjectiveOptimizer)

    def test_risk_metrics_computation(self):
        """Test risk metrics computation."""
        multi_obj_config = MultiObjectiveConfig()

        trainer = MultiObjectiveTrainer(
            state_dim=50,
            action_dim=3,
            multi_obj_config=multi_obj_config,
            device="cpu",
        )

        # Create dummy returns
        returns = np.array([0.1, -0.05, 0.2, -0.1, 0.15, -0.2, 0.1])

        risk_metrics = trainer._compute_risk_metrics(returns)

        assert isinstance(risk_metrics, dict)
        assert "var" in risk_metrics
        assert "volatility" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert risk_metrics["volatility"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
