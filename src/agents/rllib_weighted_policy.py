"""Utilities for managing ensembles of RLlib policies with weights."""

from __future__ import annotations

import random
from typing import Callable, Dict

from gymnasium import spaces
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap


class WeightedPolicyManager:
    """Lightweight wrapper around :class:`PolicyMap` supporting weighted policies."""

    def __init__(self, policies: dict[str, Policy], weights: dict[str, float]):
        self.policy_map: PolicyMap = PolicyMap()
        for name, policy in policies.items():
            self.policy_map[name] = policy

        missing_weights = set(policies.keys()) - set(weights.keys())
        if missing_weights:
            raise ValueError(f"Missing weights for policies: {missing_weights}")
        extra_weights = set(weights.keys()) - set(policies.keys())
        if extra_weights:
            raise ValueError(
                f"Extra weights provided for non-existent policies: {extra_weights}"
            )

        self.weights = dict(weights)
        self._normalize()

    def _normalize(self) -> None:
        total = sum(self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] /= total

    def update_weights(self, new_weights: dict[str, float]) -> None:
        self.weights.update(new_weights)
        self._normalize()

    def policy_mapping_fn(
        self, agent_id: str, episode=None, worker=None, **kwargs
    ) -> str:
        """Select a policy according to the configured weights."""
        # Use agent_id for seeding to ensure reproducibility per agent
        random.seed(hash(agent_id) % (2**32))
        r = random.random()
        cum = 0.0
        for name, w in self.weights.items():
            cum += w
            if r <= cum:
                return name
        return list(self.weights)[-1]


class CallablePolicy(Policy):
    """Wrap a callable into an RLlib policy."""

    def __init__(
        self, obs_space: spaces.Space, action_space: spaces.Space, func: Callable
    ):
        super().__init__(obs_space, action_space, {})
        if not callable(func):
            raise ValueError("func must be a callable")
        self.func = func

    def compute_actions(self, obs_batch, **kwargs):
        """Compute actions using the wrapped callable."""
        try:
            actions = [self.func(obs) for obs in obs_batch]
            actions_array = np.asarray(actions)
            # Ensure actions are in the correct shape
            if actions_array.ndim == 1 and len(obs_batch) > 1:
                actions_array = actions_array.reshape(len(obs_batch), -1)

            return actions_array, [], {}
        except Exception as e:
            raise RuntimeError(f"Error in compute_actions: {e}") from e
