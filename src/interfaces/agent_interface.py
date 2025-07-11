from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import Env


class IAgent(ABC):
    """Abstract interface for RL agents."""

    @abstractmethod
    def train(self, env: Env, config: dict[str, Any]) -> None:
        """Train the agent on the provided environment using config settings."""
        raise NotImplementedError

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Return an action for the given state."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent's parameters to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's parameters from disk."""
        raise NotImplementedError
