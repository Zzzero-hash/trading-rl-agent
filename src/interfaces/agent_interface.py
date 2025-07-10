from abc import ABC, abstractmethod


class IAgent(ABC):
    """Abstract interface for RL agents."""

    @abstractmethod
    def train(self, env, config: dict):
        """Train the agent on the provided environment using config settings."""

    @abstractmethod
    def act(self, state):
        """Return an action for the given state."""

    @abstractmethod
    def save(self, path: str):
        """Persist the agent's parameters to disk."""

    @abstractmethod
    def load(self, path: str):
        """Load the agent's parameters from disk."""
