from __future__ import annotations

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymnasium import spaces


class CallablePolicy(Policy):
    """Wrap a callable into an RLlib Policy."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        func: Callable,
    ):
        super().__init__(obs_space, action_space, {})
        if not callable(func):
            raise ValueError("func must be a callable")
        self.func = func

    def compute_actions(self, obs_batch: list[np.ndarray], **kwargs: Any) -> tuple[np.ndarray, list, dict[str, Any]]:
        actions = [self.func(obs) for obs in obs_batch]
        # Ensure consistent 2D array shape: (batch_size, action_dim)
        actions_array = np.array(actions)
        if actions_array.ndim == 1:
            # Wrap 1D output into a column vector
            actions_array = actions_array.reshape(-1, 1)
        elif actions_array.ndim > 2:
            raise ValueError(f"Actions must be 1D or 2D, got {actions_array.ndim}D")
        return actions_array, [], {}


def weighted_policy_mapping(weights: dict[str, float]) -> Callable[[str, Any | None, Any | None, dict[str, Any]], str]:
    """Create a policy mapping function using normalized weights."""
    total = sum(weights.values()) or 1.0
    norm_weights = {k: v / total for k, v in weights.items()}

    if not norm_weights:
        raise ValueError("Weights dictionary cannot be empty")

    # Pre-compute the choices and weights for efficiency
    choices, wts = zip(*norm_weights.items())

    def mapping_fn(agent_id: str, episode: Any | None = None, worker: Any | None = None, **kwargs: Any) -> str:
        return str(random.choices(choices, weights=wts, k=1)[0])

    return mapping_fn  # type: ignore[return-value]


class WeightedEnsembleAgent:
    """Simple ensemble agent using RLlib's policy mapping utilities."""

    def __init__(self, policies: dict[str, Policy], weights: dict[str, float]):
        # Validate that weights.keys() matches policies.keys()
        if set(weights.keys()) != set(policies.keys()):
            raise ValueError("Mismatch between policies and weights keys.")

        # Normalize weights to ensure they sum to 1
        total = sum(weights.values()) or 1.0
        normalized_weights = {k: v / total for k, v in weights.items()}

        self.policy_map: PolicyMap = PolicyMap()
        for name, policy in policies.items():
            self.policy_map[name] = policy
        self.mapping_fn = weighted_policy_mapping(normalized_weights)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        policy_id = self.mapping_fn("agent0", None, None, {})
        action, _, _ = self.policy_map[policy_id].compute_single_action(obs)
        return action


class EnsembleAgent(WeightedEnsembleAgent):
    """Advanced multi-agent ensemble with voting mechanisms, diversity measures, and dynamic weighting."""

    def __init__(
        self,
        policies: dict[str, Policy],
        weights: dict[str, float] | None = None,
        ensemble_method: str = "weighted_voting",
        diversity_penalty: float = 0.1,
        performance_window: int = 100,
        min_weight: float = 0.05,
        risk_adjustment: bool = True,
        consensus_threshold: float = 0.6,
    ):
        # Initialize with equal weights if not provided
        if weights is None:
            weights = {name: 1.0 / len(policies) for name in policies}

        super().__init__(policies, weights)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.ensemble_method = ensemble_method
        self.diversity_penalty = diversity_penalty
        self.performance_window = performance_window
        self.min_weight = min_weight
        self.risk_adjustment = risk_adjustment
        self.consensus_threshold = consensus_threshold

        # Performance tracking
        self.agent_performances: dict[str, deque] = {name: deque(maxlen=performance_window) for name in policies}

        # Action history for diversity calculation
        self.action_history: deque = deque(maxlen=performance_window)

        # Ensemble diagnostics
        self.diagnostics = {
            "diversity_score": 0.0,
            "consensus_rate": 0.0,
            "weight_stability": 0.0,
            "performance_variance": 0.0,
        }

        # Previous weights for stability calculation
        self._previous_weights: dict[str, float] = {}

        self.logger.info(f"Advanced ensemble initialized with {len(policies)} agents using {ensemble_method}")

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using advanced ensemble methods."""
        if self.ensemble_method == "weighted_voting":
            return self._weighted_voting(obs)
        if self.ensemble_method == "consensus":
            return self._consensus_voting(obs)
        if self.ensemble_method == "diversity_aware":
            return self._diversity_aware_voting(obs)
        if self.ensemble_method == "risk_adjusted":
            return self._risk_adjusted_voting(obs)
        return super().select_action(obs)

    def _weighted_voting(self, obs: np.ndarray) -> np.ndarray:
        """Weighted voting based on agent performance."""
        actions = {}
        total_weight = 0.0

        for name, policy in self.policy_map.items():
            action, _, _ = policy.compute_single_action(obs)
            actions[name] = action
            total_weight += self._get_current_weight(name)

        # Weighted average of actions
        weighted_action = np.zeros_like(next(iter(actions.values())))
        for name, action in actions.items():
            weight = self._get_current_weight(name) / total_weight
            weighted_action += weight * action

        return weighted_action

    def _consensus_voting(self, obs: np.ndarray) -> np.ndarray:
        """Consensus voting with threshold-based agreement."""
        actions = []
        for policy in self.policy_map.values():
            action, _, _ = policy.compute_single_action(obs)
            actions.append(action)

        # Check for consensus
        if len(actions) > 1:
            # For continuous actions, check if they're within a threshold
            action_array = np.array(actions)
            mean_action = np.mean(action_array, axis=0)
            std_action = np.std(action_array, axis=0)

            # If standard deviation is low, use consensus
            if np.all(std_action < self.consensus_threshold):
                return mean_action
            # Use weighted voting as fallback
            return self._weighted_voting(obs)

        return actions[0] if actions else np.array([0.0])

    def _diversity_aware_voting(self, obs: np.ndarray) -> np.ndarray:
        """Voting that considers agent diversity."""
        actions = {}
        for name, policy in self.policy_map.items():
            action, _, _ = policy.compute_single_action(obs)
            actions[name] = action

        # Calculate diversity penalty
        action_array = np.array(list(actions.values()))
        diversity_score = self._calculate_diversity(action_array)

        # Adjust weights based on diversity
        adjusted_weights = {}
        for name in actions:
            base_weight = self._get_current_weight(name)
            # Encourage diversity by reducing weight for similar agents
            adjusted_weights[name] = base_weight * (1.0 + self.diversity_penalty * diversity_score)

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        # Weighted voting with adjusted weights
        weighted_action = np.zeros_like(next(iter(actions.values())))
        for name, action in actions.items():
            weight = adjusted_weights[name]
            weighted_action += weight * action

        return weighted_action

    def _risk_adjusted_voting(self, obs: np.ndarray) -> np.ndarray:
        """Risk-adjusted voting considering agent uncertainty."""
        actions = {}
        uncertainties = {}

        for name, policy in self.policy_map.items():
            action, _, info = policy.compute_single_action(obs)
            actions[name] = action

            # Estimate uncertainty from policy info or action variance
            if hasattr(policy, "get_uncertainty"):
                uncertainties[name] = policy.get_uncertainty(obs)
            else:
                # Simple uncertainty estimate based on action magnitude
                uncertainties[name] = np.linalg.norm(action)

        # Adjust weights based on uncertainty (lower uncertainty = higher weight)
        total_uncertainty = sum(uncertainties.values())
        if total_uncertainty > 0:
            risk_adjusted_weights = {}
            for name in actions:
                base_weight = self._get_current_weight(name)
                uncertainty = uncertainties[name]
                # Invert uncertainty (lower uncertainty = higher weight)
                risk_adjusted_weights[name] = base_weight * (1.0 - uncertainty / total_uncertainty)

            # Normalize weights
            total_weight = sum(risk_adjusted_weights.values())
            if total_weight > 0:
                risk_adjusted_weights = {k: v / total_weight for k, v in risk_adjusted_weights.items()}

            # Weighted voting with risk-adjusted weights
            weighted_action = np.zeros_like(next(iter(actions.values())))
            for name, action in actions.items():
                weight = risk_adjusted_weights[name]
                weighted_action += weight * action

            return weighted_action

        return self._weighted_voting(obs)

    def update_weights(self, performance_metrics: dict[str, float]) -> None:
        """Update agent weights based on recent performance."""
        for name, metric in performance_metrics.items():
            if name in self.agent_performances:
                self.agent_performances[name].append(metric)

        # Calculate new weights based on average performance
        new_weights = {}
        total_performance = 0.0

        for name in self.policy_map:
            if self.agent_performances.get(name):
                avg_performance = np.mean(list(self.agent_performances[name]))
                new_weights[name] = max(self.min_weight, avg_performance)
                total_performance += new_weights[name]
            else:
                new_weights[name] = self.min_weight
                total_performance += self.min_weight

        # Normalize weights
        if total_performance > 0:
            self.weights = {k: v / total_performance for k, v in new_weights.items()}

        # Update mapping function
        self.mapping_fn = weighted_policy_mapping(self.weights)

        self.logger.debug(f"Updated weights: {self.weights}")

    def _get_current_weight(self, agent_name: str) -> float:
        """Get current weight for an agent."""
        return float(self.weights.get(agent_name, self.min_weight))

    def _calculate_diversity(self, actions: np.ndarray) -> float:
        """Calculate diversity score based on action variance."""
        if len(actions) < 2:
            return 0.0

        # Calculate pairwise distances between actions
        distances = []
        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                distance = np.linalg.norm(actions[i] - actions[j])
                distances.append(distance)

        if distances:
            return float(np.mean(distances))
        return 0.0

    def get_ensemble_diagnostics(self) -> dict[str, float]:
        """Get ensemble diagnostics and metrics."""
        # Calculate diversity score
        if self.action_history:
            recent_actions = list(self.action_history)[-min(10, len(self.action_history)) :]
            if recent_actions:
                action_array = np.array(recent_actions)
                self.diagnostics["diversity_score"] = self._calculate_diversity(action_array)

        # Calculate consensus rate
        if self.action_history:
            consensus_count = 0
            total_decisions = len(self.action_history) - 1
            if total_decisions > 0:
                for i in range(total_decisions):
                    if self._is_consensus(list(self.action_history)[i : i + 2]):
                        consensus_count += 1
                self.diagnostics["consensus_rate"] = consensus_count / total_decisions

        # Calculate weight stability
        if hasattr(self, "_previous_weights"):
            weight_changes = []
            for name in self.weights:
                if name in self._previous_weights:
                    change = abs(self.weights[name] - self._previous_weights[name])
                    weight_changes.append(change)
            if weight_changes:
                self.diagnostics["weight_stability"] = 1.0 - np.mean(weight_changes)

        # Calculate performance variance
        all_performances: list[float] = []
        for performances in self.agent_performances.values():
            all_performances.extend(performances)
        if all_performances:
            self.diagnostics["performance_variance"] = np.var(all_performances)

        # Store current weights for next stability calculation
        self._previous_weights = self.weights.copy()

        return self.diagnostics.copy()

    def _is_consensus(self, actions: list) -> bool:
        """Check if actions represent consensus."""
        if len(actions) < 2:
            return True

        action_array = np.array(actions)
        std_action = np.std(action_array, axis=0)
        return bool(np.all(std_action < self.consensus_threshold))

    def add_agent(self, name: str, policy: Policy, initial_weight: float | None = None) -> None:
        """Add a new agent to the ensemble."""
        if name in self.policy_map:
            self.logger.warning(f"Agent {name} already exists, overwriting")

        self.policy_map[name] = policy
        if initial_weight is None:
            initial_weight = 1.0 / (len(self.policy_map) + 1)

        self.weights[name] = initial_weight
        self.agent_performances[name] = deque(maxlen=self.performance_window)

        # Renormalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        self.mapping_fn = weighted_policy_mapping(self.weights)
        self.logger.info(f"Added agent {name} with weight {initial_weight}")

    def remove_agent(self, name: str) -> None:
        """Remove an agent from the ensemble."""
        if name not in self.policy_map:
            self.logger.warning(f"Agent {name} not found in ensemble")
            return

        del self.policy_map[name]
        del self.weights[name]
        del self.agent_performances[name]

        # Renormalize weights
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
            self.mapping_fn = weighted_policy_mapping(self.weights)

        self.logger.info(f"Removed agent {name}")

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about all agents in the ensemble."""
        info: dict[str, Any] = {
            "num_agents": len(self.policy_map),
            "weights": self.weights.copy(),
            "ensemble_method": self.ensemble_method,
            "agent_performances": {},
            "diagnostics": self.get_ensemble_diagnostics(),
        }

        for name in self.policy_map:
            if self.agent_performances.get(name):
                info["agent_performances"][name] = {
                    "recent_performance": list(self.agent_performances[name])[-10:],
                    "avg_performance": np.mean(list(self.agent_performances[name])),
                    "std_performance": np.std(list(self.agent_performances[name])),
                }
            else:
                info["agent_performances"][name] = {
                    "recent_performance": [],
                    "avg_performance": 0.0,
                    "std_performance": 0.0,
                }

        return info
