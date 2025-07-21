"""
Tests for the Multi-Agent Ensemble System.

This module tests:
- EnsembleAgent creation and configuration
- Different voting mechanisms
- Dynamic weight updates
- Agent diversity measures
- Ensemble training workflows
- Evaluation and diagnostics
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import with fallback for missing dependencies
try:
    from trading_rl_agent.agents import (
        EnsembleAgent,
        EnsembleConfig,
        EnsembleEvaluator,
        EnsembleTrainer,
    )

    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ensemble components not available: {e}")
    ENSEMBLE_AVAILABLE = False


class MockPolicy:
    """Mock policy for testing."""

    def __init__(self, name: str, action_bias: float = 0.0):
        self.name = name
        self.action_bias = action_bias

    def compute_single_action(self, _obs):
        # Return biased action for testing diversity
        action = np.array([0.5 + self.action_bias + np.random.normal(0, 0.1)])
        return action, [], {}

    def get_uncertainty(self, _obs):
        # Mock uncertainty for risk-adjusted voting
        return np.random.uniform(0, 1)


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.current_step = 0
        self.reset_called = False

    def reset(self):
        self.current_step = 0
        self.reset_called = True
        return np.array([0.1, 0.2, 0.3])

    def step(self, action):
        self.current_step += 1
        reward = np.sum(action) + np.random.normal(0, 0.1)
        done = self.current_step >= self.max_steps
        obs = np.array([0.1, 0.2, 0.3]) + np.random.normal(0, 0.01, 3)
        info = {}
        return obs, reward, done, info


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble components not available")
class TestEnsembleAgent:
    """Test EnsembleAgent functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.policies = {
            "agent1": MockPolicy("agent1", action_bias=0.1),
            "agent2": MockPolicy("agent2", action_bias=-0.1),
            "agent3": MockPolicy("agent3", action_bias=0.0),
        }
        self.weights = {"agent1": 0.4, "agent2": 0.3, "agent3": 0.3}

    def test_ensemble_creation(self):
        """Test ensemble agent creation."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        assert len(ensemble.policy_map) == 3
        assert ensemble.ensemble_method == "weighted_voting"
        assert ensemble.weights["agent1"] == pytest.approx(0.4)

    def test_weighted_voting(self):
        """Test weighted voting mechanism."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        obs = np.array([0.1, 0.2, 0.3])
        action = ensemble._weighted_voting(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_consensus_voting(self):
        """Test consensus voting mechanism."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="consensus",
            consensus_threshold=0.5,
        )

        obs = np.array([0.1, 0.2, 0.3])
        action = ensemble._consensus_voting(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_diversity_aware_voting(self):
        """Test diversity-aware voting mechanism."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="diversity_aware",
            diversity_penalty=0.1,
        )

        obs = np.array([0.1, 0.2, 0.3])
        action = ensemble._diversity_aware_voting(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_risk_adjusted_voting(self):
        """Test risk-adjusted voting mechanism."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="risk_adjusted",
        )

        obs = np.array([0.1, 0.2, 0.3])
        action = ensemble._risk_adjusted_voting(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_weight_updates(self):
        """Test dynamic weight updates."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        initial_weights = ensemble.weights.copy()

        # Update weights based on performance
        performance_metrics = {"agent1": 1.0, "agent2": 0.5, "agent3": 0.8}

        ensemble.update_weights(performance_metrics)

        # Weights should have changed
        assert ensemble.weights != initial_weights
        assert sum(ensemble.weights.values()) == pytest.approx(1.0)

    def test_diversity_calculation(self):
        """Test diversity calculation."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        # Create diverse actions
        actions = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]])

        diversity = ensemble._calculate_diversity(actions)
        assert diversity > 0
        assert isinstance(diversity, float)

    def test_agent_management(self):
        """Test adding and removing agents."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        initial_count = len(ensemble.policy_map)

        # Add new agent
        new_policy = MockPolicy("agent4", action_bias=0.2)
        ensemble.add_agent("agent4", new_policy, initial_weight=0.2)

        assert len(ensemble.policy_map) == initial_count + 1
        assert "agent4" in ensemble.weights

        # Remove agent
        ensemble.remove_agent("agent4")

        assert len(ensemble.policy_map) == initial_count
        assert "agent4" not in ensemble.weights

    def test_ensemble_diagnostics(self):
        """Test ensemble diagnostics."""
        ensemble = EnsembleAgent(
            policies=self.policies,
            weights=self.weights,
            ensemble_method="weighted_voting",
        )

        diagnostics = ensemble.get_ensemble_diagnostics()

        assert "diversity_score" in diagnostics
        assert "consensus_rate" in diagnostics
        assert "weight_stability" in diagnostics
        assert "performance_variance" in diagnostics

        for value in diagnostics.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble components not available")
class TestEnsembleEvaluator:
    """Test EnsembleEvaluator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        policies = {
            "agent1": MockPolicy("agent1", action_bias=0.1),
            "agent2": MockPolicy("agent2", action_bias=-0.1),
        }
        weights = {"agent1": 0.5, "agent2": 0.5}

        self.ensemble = EnsembleAgent(policies=policies, weights=weights, ensemble_method="weighted_voting")
        self.evaluator = EnsembleEvaluator(self.ensemble)
        self.env = MockEnvironment(max_steps=5)

    def test_ensemble_evaluation(self):
        """Test ensemble evaluation."""
        results = self.evaluator.evaluate_ensemble(env=self.env, num_episodes=3, include_diagnostics=True)

        assert "performance" in results
        assert "consensus" in results
        assert "diversity" in results
        assert "stability" in results
        assert "diagnostics" in results

        # Check performance metrics
        perf = results["performance"]
        assert "mean_reward" in perf
        assert "std_reward" in perf
        assert "success_rate" in perf

        # Check consensus metrics
        cons = results["consensus"]
        assert "mean_consensus" in cons
        assert "consensus_stability" in cons

    def test_consensus_calculation(self):
        """Test consensus score calculation."""
        # Test with similar actions (high consensus)
        similar_actions = {
            "agent1": np.array([0.5]),
            "agent2": np.array([0.51]),
        }
        consensus = self.evaluator._calculate_consensus_score(similar_actions)
        assert consensus > 0.8

        # Test with different actions (low consensus)
        different_actions = {
            "agent1": np.array([0.1]),
            "agent2": np.array([0.9]),
        }
        consensus = self.evaluator._calculate_consensus_score(different_actions)
        assert consensus < 0.5

    def test_diversity_metrics(self):
        """Test diversity metrics calculation."""
        agent_actions = {
            "agent1": [np.array([0.1]), np.array([0.2])],
            "agent2": [np.array([0.9]), np.array([0.8])],
        }

        diversity = self.evaluator._calculate_diversity_metrics(agent_actions)

        assert "action_diversity" in diversity
        assert "policy_diversity" in diversity
        assert "overall_diversity" in diversity

        # Should have high diversity with different actions
        assert diversity["overall_diversity"] > 0.3

    def test_stability_metrics(self):
        """Test stability metrics calculation."""
        episode_results = [
            {"reward": 1.0, "avg_consensus": 0.8},
            {"reward": 1.1, "avg_consensus": 0.9},
            {"reward": 0.9, "avg_consensus": 0.7},
        ]

        stability = self.evaluator._calculate_stability_metrics(episode_results)

        assert "reward_stability" in stability
        assert "consensus_stability" in stability
        assert "performance_consistency" in stability
        assert "overall_stability" in stability

    def test_agent_comparison(self):
        """Test agent comparison functionality."""
        comparison = self.evaluator.compare_agents(self.env, num_episodes=2)

        assert "ensemble" in comparison
        assert "agent1" in comparison
        assert "agent2" in comparison

        for results in comparison.values():
            assert "mean_reward" in results
            assert "std_reward" in results
            assert "success_rate" in results

    def test_evaluation_report(self):
        """Test evaluation report generation."""
        # Create mock results
        results = {
            "performance": {
                "mean_reward": 1.5,
                "std_reward": 0.3,
                "min_reward": 1.0,
                "max_reward": 2.0,
                "success_rate": 0.8,
                "mean_episode_length": 10.0,
            },
            "consensus": {
                "mean_consensus": 0.7,
                "std_consensus": 0.2,
                "consensus_stability": 0.8,
                "high_consensus_rate": 0.6,
            },
            "diversity": {
                "action_diversity": 0.4,
                "policy_diversity": 0.3,
                "overall_diversity": 0.35,
            },
            "stability": {
                "reward_stability": 0.8,
                "consensus_stability": 0.7,
                "performance_consistency": 0.9,
                "overall_stability": 0.8,
            },
        }

        report = self.evaluator.generate_evaluation_report(results)

        assert isinstance(report, str)
        assert "ENSEMBLE EVALUATION REPORT" in report
        assert "PERFORMANCE METRICS" in report
        assert "CONSENSUS METRICS" in report
        assert "DIVERSITY METRICS" in report
        assert "STABILITY METRICS" in report


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble components not available")
class TestEnsembleTrainer:
    """Test EnsembleTrainer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnsembleConfig(
            agents={
                "sac": {"enabled": True, "config": {}},
                "td3": {"enabled": True, "config": {}},
            },
            ensemble_method="weighted_voting",
        )

        def mock_env_creator():
            return MockEnvironment()

        self.env_creator = mock_env_creator

    @patch("trading_rl_agent.agents.ensemble_trainer.SACTrainer")
    @patch("trading_rl_agent.agents.ensemble_trainer.TD3Trainer")
    def test_trainer_creation(self, _mock_td3, _mock_sac):
        """Test ensemble trainer creation."""
        trainer = EnsembleTrainer(config=self.config, env_creator=self.env_creator, save_dir="test_outputs")

        assert trainer.config == self.config
        assert trainer.ensemble is None
        assert len(trainer.agents) == 0

    @patch("trading_rl_agent.agents.ensemble_trainer.SACTrainer")
    @patch("trading_rl_agent.agents.ensemble_trainer.TD3Trainer")
    def test_agent_creation(self, mock_td3, mock_sac):
        """Test agent creation in trainer."""
        trainer = EnsembleTrainer(config=self.config, env_creator=self.env_creator, save_dir="test_outputs")

        # Mock the agent creation
        mock_sac_instance = Mock()
        mock_sac_instance.get_policy.return_value = MockPolicy("sac")
        mock_sac.return_value = mock_sac_instance

        mock_td3_instance = Mock()
        mock_td3_instance.get_policy.return_value = MockPolicy("td3")
        mock_td3.return_value = mock_td3_instance

        trainer.create_agents()

        assert len(trainer.agents) == 2
        assert trainer.ensemble is not None
        assert len(trainer.ensemble.policy_map) == 2

    def test_dynamic_agent_management(self):
        """Test dynamic agent addition and removal."""
        trainer = EnsembleTrainer(config=self.config, env_creator=self.env_creator, save_dir="test_outputs")

        # Add agent dynamically
        success = trainer.add_agent_dynamically("ppo", "ppo", {"learning_rate": 1e-4})

        # Should fail without proper setup
        assert not success

        # Test removal
        success = trainer.remove_agent_dynamically("ppo")
        assert not success  # Agent doesn't exist

    def test_ensemble_info(self):
        """Test ensemble information retrieval."""
        trainer = EnsembleTrainer(config=self.config, env_creator=self.env_creator, save_dir="test_outputs")

        info = trainer.get_ensemble_info()

        assert "error" in info  # No ensemble created yet
        assert info["error"] == "Ensemble not initialized"


@pytest.mark.skipif(not ENSEMBLE_AVAILABLE, reason="Ensemble components not available")
class TestEnsembleConfig:
    """Test EnsembleConfig functionality."""

    def test_config_creation(self):
        """Test ensemble configuration creation."""
        config = EnsembleConfig(
            agents={
                "sac": {"enabled": True, "config": {"learning_rate": 1e-4}},
                "td3": {"enabled": False, "config": {}},
            },
            ensemble_method="consensus",
            diversity_penalty=0.2,
        )

        assert config.ensemble_method == "consensus"
        assert config.diversity_penalty == 0.2
        assert len(config.agents) == 2
        assert config.agents["sac"]["enabled"] is True
        assert config.agents["td3"]["enabled"] is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Should raise error for empty agents
        with pytest.raises(ValueError):
            EnsembleConfig(agents={})

    def test_agent_configs_alias(self):
        """Test agent_configs parameter alias."""
        config = EnsembleConfig(
            agent_configs={
                "sac": {"enabled": True, "config": {}},
            }
        )

        assert "sac" in config.agents
        assert config.agents["sac"]["enabled"] is True


# Test basic functionality even without full ensemble components
class TestBasicEnsembleFunctionality:
    """Test basic ensemble functionality that doesn't require full dependencies."""

    def test_mock_policy(self):
        """Test mock policy functionality."""
        policy = MockPolicy("test", action_bias=0.1)
        obs = np.array([0.1, 0.2, 0.3])
        action, _, _ = policy.compute_single_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert policy.get_uncertainty(obs) >= 0

    def test_mock_environment(self):
        """Test mock environment functionality."""
        env = MockEnvironment(max_steps=5)
        obs = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)

        action = np.array([0.5])
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__])
