from __future__ import annotations

import random
from typing import Callable, Dict

from gymnasium import spaces
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap


class CallablePolicy(Policy):
    """Wrap a callable into an RLlib Policy."""

    def __init__(self, obs_space: spaces.Space, action_space: spaces.Space, func: Callable):
        super().__init__(obs_space, action_space, {})
        if not callable(func):
            raise ValueError("func must be a callable")
        self.func = func

    def compute_actions(self, obs_batch, **kwargs):
        actions = [self.func(obs) for obs in obs_batch]
        # Ensure consistent 2D array shape: (batch_size, action_dim)
        actions_array = np.array(actions)
        if actions_array.ndim == 1:
            # Wrap 1D output into a column vector
            actions_array = actions_array.reshape(-1, 1)
        elif actions_array.ndim > 2:
            raise ValueError(f"Actions must be 1D or 2D, got {actions_array.ndim}D")
        return actions_array, [], {}


def weighted_policy_mapping(weights: Dict[str, float]):
    """Create a policy mapping function using normalized weights."""
    total = sum(weights.values()) or 1.0
    norm_weights = {k: v / total for k, v in weights.items()}

    if not norm_weights:
        raise ValueError("Weights dictionary cannot be empty")

    # Pre-compute the choices and weights for efficiency
    choices, wts = zip(*norm_weights.items())

    def mapping_fn(agent_id: str, episode=None, worker=None, **kwargs) -> str:
        return random.choices(choices, weights=wts, k=1)[0]

    return mapping_fn


class WeightedEnsembleAgent:
    """Simple ensemble agent using RLlib's policy mapping utilities."""

    def __init__(self, policies: Dict[str, Policy], weights: Dict[str, float]):
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
        policy_id = self.mapping_fn("agent0")
        action, _, _ = self.policy_map[policy_id].compute_single_action(obs)
        return action
