"""Utilities for managing ensembles of RLlib policies with weights."""
from __future__ import annotations

import random
from typing import Dict

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from gymnasium import spaces
import numpy as np


class WeightedPolicyManager:
    """Lightweight wrapper around :class:`PolicyMap` supporting weighted policies."""

    def __init__(self, policies: Dict[str, Policy], weights: Dict[str, float]):
        self.policy_map: PolicyMap = PolicyMap()
        for name, policy in policies.items():
            self.policy_map[name] = policy
        self.weights = dict(weights)
        self._normalize()

    def _normalize(self) -> None:
        total = sum(self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] /= total

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        self.weights.update(new_weights)
        self._normalize()

    def policy_mapping_fn(self, agent_id: str, *_) -> str:
        """Select a policy according to the configured weights."""
        r = random.random()
        cum = 0.0
        for name, w in self.weights.items():
            cum += w
            if r <= cum:
                return name
        return list(self.weights)[-1]


class CallablePolicy(Policy):
    """Wrap a callable into an RLlib policy."""

    def __init__(self, obs_space: spaces.Space, action_space: spaces.Space, func):
        super().__init__(obs_space, action_space, {})
        self.func = func

    def compute_actions(self, obs_batch, **kwargs):
        actions = [self.func(obs) for obs in obs_batch]
        return np.asarray(actions), [], {}
