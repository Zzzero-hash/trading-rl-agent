"""
Base class for supervised learning models.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseSupervisedModel(ABC):
    """Base class for all supervised learning models."""

    def __init__(self, model_name: str = "base_model"):
        """Initialize the base model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.is_trained = False
        self.feature_names: list[str] | None = None
        self.model_params: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> "BaseSupervisedModel":
        """Fit the model to the training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        return None

    def get_model_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.model_params.copy()

    def set_model_params(self, params: dict[str, Any]) -> None:
        """Set model parameters.

        Args:
            params: Dictionary of model parameters
        """
        self.model_params.update(params)

    def save_model(self, filepath: str) -> None:
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("Model saving not implemented")

    def load_model(self, filepath: str) -> None:
        """Load the model from disk.

        Args:
            filepath: Path to load the model from
        """
        raise NotImplementedError("Model loading not implemented")
