"""
Price prediction models for financial time series.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base_model import BaseSupervisedModel


class PricePredictor(BaseSupervisedModel):
    """Price prediction model using various algorithms."""

    def __init__(self, model_type: str = "random_forest", **kwargs: Any) -> None:
        """Initialize the price predictor.

        Args:
            model_type: Type of model to use ('random_forest', 'linear')
            **kwargs: Additional model parameters
        """
        super().__init__(model_name=f"price_predictor_{model_type}")
        self.model_type = model_type
        self.set_model_params(kwargs)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        if self.model_type == "random_forest":
            params = self.model_params.get("random_forest", {})
            self.model = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=params.get("random_state", 42),
                **params,
            )
        elif self.model_type == "linear":
            params = self.model_params.get("linear", {})
            self.model = LinearRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> "PricePredictor":
        """Fit the model to the training data.

        Args:
            X: Training features
            y: Training targets (price changes or returns)

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
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make price predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Price predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

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
            coef_scores = np.abs(self.model.coef_)
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
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
