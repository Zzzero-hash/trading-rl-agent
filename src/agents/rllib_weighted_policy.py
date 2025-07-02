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
        """
        Initialize the WeightedPolicyManager with a set of policies and their associated weights.
        
        Parameters:
            policies (Dict[str, Policy]): Mapping of policy names to RLlib Policy instances.
            weights (Dict[str, float]): Mapping of policy names to their corresponding weights.
        """
        self.policy_map: PolicyMap = PolicyMap()
        for name, policy in policies.items():
            self.policy_map[name] = policy
        self.weights = dict(weights)
        self._normalize()

    def _normalize(self) -> None:
        """
        Normalize the policy weights so that their sum equals 1.0.
        
        If the total weight is zero, normalization uses 1.0 as the denominator to avoid division by zero.
        """
        total = sum(self.weights.values()) or 1.0
        for k in self.weights:
            self.weights[k] /= total

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update the policy weights with new values and normalize them so their sum equals 1.0.
        
        Parameters:
            new_weights (Dict[str, float]): Dictionary of policy names and their new weights.
        """
        self.weights.update(new_weights)
        self._normalize()

    def policy_mapping_fn(self, agent_id: str, *_) -> str:
        """
        Selects and returns a policy name based on the current normalized weights using weighted random sampling.
        
        Parameters:
            agent_id (str): The identifier of the agent for which to select a policy.
        
        Returns:
            str: The name of the selected policy.
        """
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
        """
        Initialize a policy that wraps a callable function for action computation.
        
        Parameters:
            obs_space (spaces.Space): The observation space for the policy.
            action_space (spaces.Space): The action space for the policy.
            func: A callable that takes an observation and returns an action.
        """
        super().__init__(obs_space, action_space, {})
        self.func = func

    def compute_actions(self, obs_batch, **kwargs):
        """
        Compute actions for a batch of observations using the wrapped callable.
        
        Parameters:
            obs_batch: A batch of observations to process.
        
        Returns:
            actions (np.ndarray): Array of actions computed by applying the callable to each observation.
            state_out (list): Empty list, as no recurrent state is used.
            extra_info (dict): Empty dictionary, as no extra information is provided.
        """
        actions = [self.func(obs) for obs in obs_batch]
        return np.asarray(actions), [], {}
