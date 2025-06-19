"""
Advanced Ensemble Agent with Multi-Agent Consensus

This agent combines multiple RL agents using advanced ensemble methods:
- Weighted voting with dynamic rebalancing
- Confidence-based weighting
- Uncertainty quantification
- Risk-adjusted consensus
- Meta-learning capabilities
"""

from collections import deque
import copy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

# Use absolute imports to avoid relative import issues when running tests
try:
    from configs.hyperparameters import EnsembleConfig, get_agent_config
except ImportError:
    from src.configs.hyperparameters import EnsembleConfig, get_agent_config


class UncertaintyEstimator(nn.Module):
    """Neural network for uncertainty estimation."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64]):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # Output uncertainty in [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty for state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class MetaLearner(nn.Module):
    """Meta-learning network for agent weight adaptation."""

    def __init__(
        self, input_dim: int, num_agents: int, hidden_dims: list[int] = [128, 64]
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_agents))
        layers.append(nn.Softmax(dim=-1))  # Output normalized weights

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict optimal agent weights based on context."""
        return self.network(features)


class EnsembleAgent:
    """
    Advanced Ensemble Agent with multi-agent consensus.

    Features:
    - Dynamic agent weighting based on performance
    - Uncertainty quantification for decision confidence
    - Risk-adjusted consensus mechanisms
    - Meta-learning for adaptive ensemble weights
    - Consensus voting with confidence thresholds
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, Any]] = None,
        state_dim: int = 10,
        action_dim: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize Ensemble Agent.

        Args:
            config: Configuration (dataclass, dict, or file path)
            state_dim: State space dimension
            action_dim: Action space dimension
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Load configuration
        if config is None:
            try:
                from configs.hyperparameters import EnsembleConfig
            except ImportError:
                from src.configs.hyperparameters import EnsembleConfig
            config = EnsembleConfig()

        if is_dataclass(config):
            self.config = config
            try:
                if hasattr(config, "__dataclass_fields__"):
                    # It's a proper dataclass instance
                    self._config_dict = {
                        field: getattr(config, field)
                        for field in config.__dataclass_fields__
                    }
                else:
                    # Fallback
                    self._config_dict = (
                        config.__dict__ if hasattr(config, "__dict__") else {}
                    )
            except Exception:
                # Final fallback
                self._config_dict = (
                    config.__dict__ if hasattr(config, "__dict__") else {}
                )
        else:
            self._config_dict = self._load_config(config)
            self.config = self._config_dict

        # Extract hyperparameters
        self.agents_config = self._config_dict.get(
            "agents",
            {
                "ppo": {"enabled": True, "weight": 0.25},
                "sac": {"enabled": True, "weight": 0.25},
                "td3": {"enabled": True, "weight": 0.25},
                "rainbow_dqn": {"enabled": True, "weight": 0.25},
            },
        )

        self.consensus_method = self._config_dict.get(
            "consensus_method", "weighted_voting"
        )
        self.confidence_threshold = self._config_dict.get("confidence_threshold", 0.7)
        self.dynamic_weighting = self._config_dict.get("dynamic_weighting", True)
        self.weight_update_frequency = self._config_dict.get(
            "weight_update_frequency", 1000
        )
        self.performance_window = self._config_dict.get("performance_window", 100)
        self.min_weight = self._config_dict.get("min_weight", 0.05)
        self.risk_adjustment = self._config_dict.get("risk_adjustment", True)
        self.risk_aversion = self._config_dict.get("risk_aversion", 0.1)
        self.diversification_bonus = self._config_dict.get(
            "diversification_bonus", 0.05
        )
        self.uncertainty_estimation = self._config_dict.get(
            "uncertainty_estimation", True
        )
        self.monte_carlo_samples = self._config_dict.get("monte_carlo_samples", 10)
        self.meta_learning = self._config_dict.get("meta_learning", False)
        self.adaptation_rate = self._config_dict.get("adaptation_rate", 0.01)

        # Initialize agents
        self.agents = {}
        self.agent_weights = {}
        self.agent_performance = {}
        self.agent_confidence = {}

        # Initialize enabled agents
        self._initialize_agents()

        # Performance tracking
        self.performance_history = {
            name: deque(maxlen=self.performance_window) for name in self.agents.keys()
        }
        self.prediction_history = {
            name: deque(maxlen=self.performance_window) for name in self.agents.keys()
        }

        # Uncertainty estimation
        if self.uncertainty_estimation:
            self.uncertainty_estimator = UncertaintyEstimator(
                state_dim + action_dim, [128, 64]
            ).to(self.device)
            self.uncertainty_optimizer = optim.Adam(
                self.uncertainty_estimator.parameters(), lr=1e-3
            )

        # Meta-learning
        if self.meta_learning:
            context_dim = state_dim + len(self.agents) * (
                action_dim + 2
            )  # +2 for confidence and uncertainty
            self.meta_learner = MetaLearner(
                context_dim, len(self.agents), [128, 64]
            ).to(self.device)
            self.meta_optimizer = optim.Adam(
                self.meta_learner.parameters(), lr=self.adaptation_rate
            )

        # Training counters
        self.training_step = 0
        self.last_weight_update = 0
        self.episode_rewards = deque(maxlen=100)

    def _initialize_agents(self):
        """Initialize individual RL agents."""
        from .ppo_agent import PPOAgent
        from .rainbow_dqn_agent import RainbowDQNAgent
        from .sac_agent import SACAgent
        from .td3_agent import TD3Agent

        agent_classes = {
            "ppo": (PPOAgent, False),  # (class, discrete)
            "sac": (SACAgent, False),
            "td3": (TD3Agent, False),
            "rainbow_dqn": (RainbowDQNAgent, True),
        }

        for agent_name, agent_config in self.agents_config.items():
            if agent_config.get("enabled", False):
                if agent_name in agent_classes:
                    agent_class, discrete = agent_classes[agent_name]

                    try:
                        if agent_name == "ppo":
                            agent = agent_class(
                                self.state_dim,
                                self.action_dim,
                                discrete=discrete,
                                device=str(self.device),
                            )
                        elif agent_name == "rainbow_dqn":
                            agent = agent_class(
                                self.state_dim, self.action_dim, device=str(self.device)
                            )
                        else:
                            agent = agent_class(
                                self.state_dim, self.action_dim, device=str(self.device)
                            )

                        self.agents[agent_name] = agent
                        self.agent_weights[agent_name] = agent_config.get(
                            "weight", 1.0 / len(self.agents_config)
                        )
                        self.agent_performance[agent_name] = []
                        self.agent_confidence[agent_name] = 1.0

                        print(f"✅ Initialized {agent_name} agent")

                    except Exception as e:
                        print(f"⚠️ Failed to initialize {agent_name} agent: {e}")
                        # Create a dummy agent placeholder
                        self.agents[agent_name] = None
                        self.agent_weights[agent_name] = 0.0
                        self.agent_performance[agent_name] = []
                        self.agent_confidence[agent_name] = 0.0

        # Normalize weights
        self._normalize_weights()

    def _load_config(self, config: Optional[Union[str, dict, Any]]) -> dict:
        """Load configuration from file, dict, or dataclass."""
        if config is None:
            return {}
        elif isinstance(config, str):
            with open(config) as f:
                return yaml.safe_load(f) or {}
        elif is_dataclass(config):
            try:
                if hasattr(config, "__dataclass_fields__"):
                    return {
                        field: getattr(config, field)
                        for field in config.__dataclass_fields__
                    }
                else:
                    return config.__dict__ if hasattr(config, "__dict__") else {}
            except Exception:
                return config.__dict__ if hasattr(config, "__dict__") else {}
        else:
            return config or {}

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using ensemble consensus."""
        # Get predictions from all agents
        agent_predictions = {}
        agent_confidences = {}
        agent_uncertainties = {}

        for agent_name, agent in self.agents.items():
            if agent is not None and self.agent_weights[agent_name] > 0:
                try:
                    if agent_name == "rainbow_dqn":
                        # Rainbow DQN returns discrete actions
                        action_idx = agent.select_action(state, evaluate)
                        # Convert to continuous action space
                        action = np.zeros(self.action_dim)
                        if action_idx < self.action_dim:
                            action[action_idx] = 1.0
                        agent_predictions[agent_name] = action
                    elif agent_name == "ppo":
                        # PPO returns tuple (action, value, log_prob)
                        action, _, _ = agent.select_action(
                            state, deterministic=evaluate
                        )
                        agent_predictions[agent_name] = action
                    else:
                        # SAC and TD3 return actions directly
                        action = agent.select_action(state, evaluate=(not evaluate))
                        agent_predictions[agent_name] = action

                    # Estimate confidence (simplified)
                    confidence = self._estimate_confidence(
                        agent_name, state, agent_predictions[agent_name]
                    )
                    agent_confidences[agent_name] = confidence

                    # Estimate uncertainty if enabled
                    if self.uncertainty_estimation:
                        uncertainty = self._estimate_uncertainty(
                            state, agent_predictions[agent_name]
                        )
                        agent_uncertainties[agent_name] = uncertainty
                    else:
                        agent_uncertainties[agent_name] = 0.1

                except Exception as e:
                    print(f"⚠️ Error getting prediction from {agent_name}: {e}")
                    continue

        if not agent_predictions:
            # Fallback to random action
            return np.random.uniform(-1, 1, self.action_dim)

        # Apply consensus method
        final_action = self._apply_consensus(
            agent_predictions, agent_confidences, agent_uncertainties
        )

        # Store predictions for performance tracking
        for agent_name, prediction in agent_predictions.items():
            self.prediction_history[agent_name].append(prediction.copy())

        return final_action

    def _estimate_confidence(
        self, agent_name: str, state: np.ndarray, action: np.ndarray
    ) -> float:
        """Estimate agent confidence in its prediction."""
        # Simple confidence based on recent performance
        recent_performance = list(self.performance_history[agent_name])
        if len(recent_performance) > 5:
            confidence = float(np.mean(recent_performance[-5:]))
            confidence = max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]
        else:
            confidence = float(self.agent_confidence[agent_name])

        return confidence

    def _estimate_uncertainty(self, state: np.ndarray, action: np.ndarray) -> float:
        """Estimate uncertainty in state-action pair."""
        if not hasattr(self, "uncertainty_estimator"):
            return 0.1

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            uncertainty = self.uncertainty_estimator(state_tensor, action_tensor)
            return uncertainty.item()

    def _apply_consensus(
        self,
        predictions: dict[str, np.ndarray],
        confidences: dict[str, float],
        uncertainties: dict[str, float],
    ) -> np.ndarray:
        """Apply consensus mechanism to combine predictions."""
        if self.consensus_method == "weighted_voting":
            return self._weighted_voting(predictions, confidences, uncertainties)
        elif self.consensus_method == "majority_voting":
            return self._majority_voting(predictions, confidences)
        elif self.consensus_method == "confidence_weighted":
            return self._confidence_weighted(predictions, confidences, uncertainties)
        else:
            # Default to simple average
            return np.mean(list(predictions.values()), axis=0)

    def _weighted_voting(
        self,
        predictions: dict[str, np.ndarray],
        confidences: dict[str, float],
        uncertainties: dict[str, float],
    ) -> np.ndarray:
        """Combine predictions using weighted voting."""
        combined = np.zeros(self.action_dim)
        total_weight = 0.0

        for agent_name, prediction in predictions.items():
            # Base weight from ensemble configuration
            base_weight = self.agent_weights.get(agent_name, 0.0)

            # Adjust weight by confidence and uncertainty
            confidence = confidences.get(agent_name, 0.5)
            uncertainty = uncertainties.get(agent_name, 0.5)

            # Confidence weighting
            confidence_weight = confidence

            # Uncertainty weighting (lower uncertainty = higher weight)
            uncertainty_weight = 1.0 - uncertainty

            # Combined weight
            weight = base_weight * confidence_weight * uncertainty_weight

            # Risk adjustment
            if self.risk_adjustment:
                risk_penalty = self.risk_aversion * uncertainty
                weight = weight * (1.0 - risk_penalty)

            combined += weight * prediction
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight
        else:
            # Fallback to equal weighting
            combined = np.mean(list(predictions.values()), axis=0)

        return combined

    def _majority_voting(
        self, predictions: dict[str, np.ndarray], confidences: dict[str, float]
    ) -> np.ndarray:
        """Apply majority voting with confidence threshold."""
        # For continuous actions, use clustering-based majority voting
        prediction_list = list(predictions.values())
        confidence_list = [confidences.get(name, 0.5) for name in predictions.keys()]

        # Filter by confidence threshold
        filtered_predictions = []
        for pred, conf in zip(prediction_list, confidence_list):
            if conf >= self.confidence_threshold:
                filtered_predictions.append(pred)

        if not filtered_predictions:
            # No confident predictions, use all
            filtered_predictions = prediction_list

        # Simple average for now (could use more sophisticated clustering)
        return np.mean(filtered_predictions, axis=0)

    def _confidence_weighted(
        self,
        predictions: dict[str, np.ndarray],
        confidences: dict[str, float],
        uncertainties: dict[str, float],
    ) -> np.ndarray:
        """Combine predictions using pure confidence weighting."""
        combined = np.zeros(self.action_dim)
        total_confidence = 0.0

        for agent_name, prediction in predictions.items():
            confidence = confidences.get(agent_name, 0.5)
            uncertainty = uncertainties.get(agent_name, 0.5)

            # Adjust confidence by uncertainty
            adjusted_confidence = confidence * (1.0 - uncertainty)

            combined += adjusted_confidence * prediction
            total_confidence += adjusted_confidence

        if total_confidence > 0:
            combined /= total_confidence
        else:
            combined = np.mean(list(predictions.values()), axis=0)

        return combined

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in all agent replay buffers."""
        # Store in all active agents
        for agent_name, agent in self.agents.items():
            if agent is not None and self.agent_weights[agent_name] > 0:
                try:
                    if agent_name == "rainbow_dqn":
                        # Convert continuous action to discrete
                        action_idx = np.argmax(action) if self.action_dim > 1 else 0
                        agent.store_experience(
                            state, action_idx, reward, next_state, done
                        )
                    elif agent_name == "ppo":
                        # PPO stores differently - need value and log_prob
                        # For now, skip or use dummy values
                        pass
                    else:
                        agent.store_experience(state, action, reward, next_state, done)
                except Exception as e:
                    print(f"⚠️ Error storing experience in {agent_name}: {e}")

        # Track episode rewards
        if done:
            self.episode_rewards.append(reward)

    def update(self) -> dict[str, float]:
        """Update all agents and ensemble weights."""
        self.training_step += 1

        metrics = {}
        total_loss = 0.0
        successful_updates = 0

        # Update individual agents
        for agent_name, agent in self.agents.items():
            if agent is not None and self.agent_weights[agent_name] > 0:
                try:
                    agent_metrics = agent.update()
                    if agent_metrics:
                        # Extract loss metric (different names for different agents)
                        loss_keys = ["loss", "policy_loss", "critic_loss", "actor_loss"]
                        agent_loss = 0.0
                        for key in loss_keys:
                            if key in agent_metrics:
                                agent_loss = agent_metrics[key]
                                break

                        total_loss += agent_loss
                        successful_updates += 1

                        # Store agent-specific metrics
                        for key, value in agent_metrics.items():
                            metrics[f"{agent_name}_{key}"] = value

                        # Update performance tracking
                        performance = 1.0 / (
                            1.0 + agent_loss
                        )  # Convert loss to performance
                        self.performance_history[agent_name].append(performance)

                except Exception as e:
                    print(f"⚠️ Error updating {agent_name}: {e}")

        # Update ensemble weights if enabled
        if (
            self.dynamic_weighting
            and self.training_step % self.weight_update_frequency == 0
        ):
            self._update_ensemble_weights()

        # Update uncertainty estimator
        if self.uncertainty_estimation and len(self.episode_rewards) > 10:
            self._update_uncertainty_estimator()

        # Meta-learning update
        if self.meta_learning and len(self.episode_rewards) > 20:
            self._update_meta_learner()

        # Ensemble metrics
        metrics.update(
            {
                "ensemble_loss": total_loss / max(1, successful_updates),
                "total_agents": len(self.agents),
                "active_agents": successful_updates,
                "training_step": self.training_step,
            }
        )

        # Add agent weights
        for agent_name, weight in self.agent_weights.items():
            metrics[f"{agent_name}_weight"] = weight

        return metrics

    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance."""
        # Calculate performance for each agent
        agent_performances = {}

        for agent_name in self.agents.keys():
            if len(self.performance_history[agent_name]) > 5:
                # Use recent performance average
                recent_perf = list(self.performance_history[agent_name])[-10:]
                agent_performances[agent_name] = np.mean(recent_perf)
            else:
                agent_performances[agent_name] = 0.5  # Default performance

        # Softmax normalization for weights
        performances = np.array(list(agent_performances.values()))
        if len(performances) > 0:
            # Add small epsilon to avoid zero weights
            performances += 1e-8
            weights = np.exp(performances) / np.sum(np.exp(performances))

            # Ensure minimum weight
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)  # Renormalize

            # Update weights
            for i, agent_name in enumerate(agent_performances.keys()):
                self.agent_weights[agent_name] = weights[i]

    def _update_uncertainty_estimator(self):
        """Update uncertainty estimation network."""
        if not hasattr(self, "uncertainty_estimator"):
            return

        # Simple training on recent episode variance
        if len(self.episode_rewards) < 10:
            return

        # Use variance of recent rewards as uncertainty target
        recent_rewards = list(self.episode_rewards)[-10:]
        uncertainty_target = np.var(recent_rewards)
        uncertainty_target = min(1.0, uncertainty_target)  # Normalize

        # Dummy training step (in practice, would use proper state-action pairs)
        dummy_state = torch.randn(1, self.state_dim, device=self.device)
        dummy_action = torch.randn(1, self.action_dim, device=self.device)

        predicted_uncertainty = self.uncertainty_estimator(dummy_state, dummy_action)
        target_uncertainty = torch.tensor([[uncertainty_target]], device=self.device)

        loss = F.mse_loss(predicted_uncertainty, target_uncertainty)

        self.uncertainty_optimizer.zero_grad()
        loss.backward()
        self.uncertainty_optimizer.step()

    def _update_meta_learner(self):
        """Update meta-learning network for weight adaptation."""
        if not hasattr(self, "meta_learner"):
            return

        # Create context features (simplified)
        recent_rewards = list(self.episode_rewards)[-5:]
        if len(recent_rewards) < 5:
            return

        # Context: recent performance, agent diversities, etc.
        context_features = []

        # Recent reward statistics
        context_features.extend(
            [
                np.mean(recent_rewards),
                np.std(recent_rewards),
                np.min(recent_rewards),
                np.max(recent_rewards),
            ]
        )

        # Agent performance statistics
        for agent_name in self.agents.keys():
            if len(self.performance_history[agent_name]) > 0:
                context_features.append(
                    np.mean(list(self.performance_history[agent_name])[-5:])
                )
            else:
                context_features.append(0.5)

        # Pad or truncate to expected input size
        expected_size = self.state_dim + len(self.agents) * (self.action_dim + 2)
        while len(context_features) < expected_size:
            context_features.append(0.0)
        context_features = context_features[:expected_size]

        context_tensor = (
            torch.FloatTensor(context_features).unsqueeze(0).to(self.device)
        )

        # Predict optimal weights
        predicted_weights = self.meta_learner(context_tensor)

        # Target weights based on recent performance
        target_weights = []
        for agent_name in self.agents.keys():
            target_weights.append(self.agent_weights[agent_name])
        target_weights = torch.FloatTensor(target_weights).unsqueeze(0).to(self.device)

        # Meta-learning loss
        meta_loss = F.mse_loss(predicted_weights, target_weights)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Optionally update weights with meta-learner predictions
        if np.random.random() < 0.1:  # 10% chance to use meta-learner weights
            new_weights = predicted_weights.detach().cpu().numpy()[0]
            for i, agent_name in enumerate(self.agents.keys()):
                self.agent_weights[agent_name] = new_weights[i]
            self._normalize_weights()

    def _normalize_weights(self):
        """Normalize agent weights to sum to 1."""
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_name in self.agent_weights:
                self.agent_weights[agent_name] /= total_weight

    def save(self, filepath: str):
        """Save ensemble agent state."""
        # Save individual agents
        agent_states = {}
        for agent_name, agent in self.agents.items():
            if agent is not None:
                agent_path = filepath.replace(".pth", f"_{agent_name}.pth")
                agent.save(agent_path)
                agent_states[agent_name] = agent_path

        # Save ensemble state
        ensemble_state = {
            "agent_weights": self.agent_weights,
            "agent_performance": self.agent_performance,
            "agent_confidence": self.agent_confidence,
            "training_step": self.training_step,
            "config": self.config,
            "agent_states": agent_states,
        }

        # Save uncertainty estimator
        if hasattr(self, "uncertainty_estimator"):
            ensemble_state["uncertainty_estimator"] = (
                self.uncertainty_estimator.state_dict()
            )

        # Save meta-learner
        if hasattr(self, "meta_learner"):
            ensemble_state["meta_learner"] = self.meta_learner.state_dict()

        torch.save(ensemble_state, filepath)

    def load(self, filepath: str):
        """Load ensemble agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load ensemble state
        self.agent_weights = checkpoint.get("agent_weights", {})
        self.agent_performance = checkpoint.get("agent_performance", {})
        self.agent_confidence = checkpoint.get("agent_confidence", {})
        self.training_step = checkpoint.get("training_step", 0)

        # Load individual agents
        agent_states = checkpoint.get("agent_states", {})
        for agent_name, agent_path in agent_states.items():
            if agent_name in self.agents and self.agents[agent_name] is not None:
                try:
                    self.agents[agent_name].load(agent_path)
                except Exception as e:
                    print(f"⚠️ Failed to load {agent_name} agent: {e}")

        # Load uncertainty estimator
        if (
            hasattr(self, "uncertainty_estimator")
            and "uncertainty_estimator" in checkpoint
        ):
            self.uncertainty_estimator.load_state_dict(
                checkpoint["uncertainty_estimator"]
            )

        # Load meta-learner
        if hasattr(self, "meta_learner") and "meta_learner" in checkpoint:
            self.meta_learner.load_state_dict(checkpoint["meta_learner"])


if __name__ == "__main__":
    # Test Enhanced Ensemble Agent
    print("=== Advanced Ensemble Agent Test ===")

    # Create agent
    state_dim = 10
    action_dim = 3
    agent = EnsembleAgent(None, state_dim=state_dim, action_dim=action_dim)

    print(f"✅ Created Ensemble agent with {len(agent.agents)} sub-agents")
    print(
        f"   Active agents: {[name for name, a in agent.agents.items() if a is not None]}"
    )

    # Test action selection
    dummy_state = np.random.randn(state_dim)
    action = agent.select_action(dummy_state)
    print(f"✅ Action selection test: action={action}")

    # Test experience storage
    print("\n📈 Testing experience collection...")
    for i in range(10):
        state = np.random.randn(state_dim)
        action = np.random.uniform(-1, 1, action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i == 9  # Last episode

        agent.store_experience(state, action, reward, next_state, done)

    print("✅ Stored experiences in active agents")

    # Test update
    print("\n🔄 Testing ensemble update...")
    metrics = agent.update()

    print(f"✅ Update metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n🎯 Advanced Ensemble agent test complete!")
