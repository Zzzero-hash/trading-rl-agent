"""
Trading signal classification models.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from .base_model import BaseSupervisedModel


class SignalClassifier(BaseSupervisedModel):
    """Trading signal classification model."""

    def __init__(self, model_type: str = "random_forest", **kwargs: Any) -> None:
        """Initialize the signal classifier.

        Args:
            model_type: Type of model to use ('random_forest', 'logistic')
            **kwargs: Additional model parameters
        """
        super().__init__(model_name=f"signal_classifier_{model_type}")
        self.model_type = model_type
        self.classes_ = None
        self.set_model_params(kwargs)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        if self.model_type == "random_forest":
            params = self.model_params.get("random_forest", {})
            self.model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=params.get("random_state", 42),
                **params,
            )
        elif self.model_type == "logistic":
            params = self.model_params.get("logistic", {})
            self.model = LogisticRegression(random_state=params.get("random_state", 42), **params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> "SignalClassifier":
        """Fit the model to the training data.

        Args:
            X: Training features
            y: Training targets (trading signals: buy, sell, hold)

        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = None

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Fit the model
        self.model.fit(X_array, y_array)
        self.classes_ = self.model.classes_
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make signal predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Signal predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_names is None:
            return None
        if hasattr(self.model, "feature_importances_"):
            importance_scores = self.model.feature_importances_
            return dict(zip(self.feature_names, importance_scores))
        if hasattr(self.model, "coef_"):
            coef_scores = np.abs(self.model.coef_0)  # Take first class for multiclass
            return dict(zip(self.feature_names, coef_scores))
        raise AttributeError("Model does not have feature importances or coefficients.")

    def evaluate(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> dict[str, float]:
        """Evaluate the model performance.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else y

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }

    def get_classification_report(self, X: np.ndarray, y_true: np.ndarray) -> str:
        """Get detailed classification report.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Classification report string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        # Ensure the returned value is a string
        return str(classification_report(y_true, y_pred))

    def get_class_distribution(self, y: np.ndarray | pd.Series) -> dict[str, int]:
        """Get class distribution in the data.

        Args:
            y: Target values

        Returns:
            Dictionary with class counts
        """
        y_array = y.values if isinstance(y, pd.Series) else y
        unique, counts = np.unique(y_array, return_counts=True)
        return dict(zip(unique, counts))
