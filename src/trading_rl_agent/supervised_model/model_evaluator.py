"""
Model evaluation utilities for supervised learning models.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from .base_model import BaseSupervisedModel


class ModelEvaluator:
    """Evaluator for supervised learning models."""

    def __init__(self) -> None:
        """Initialize the model evaluator."""

    def cross_validate(
        self,
        model: BaseSupervisedModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        scoring: str = "default",
    ) -> dict[str, list[float]]:
        """Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary with cross-validation results
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Determine scoring metric based on model type
        if scoring == "default":
            if hasattr(model, "classes_"):  # Classification model
                scoring = "accuracy"
            else:  # Regression model
                scoring = "neg_mean_squared_error"

        # Perform cross-validation
        if not hasattr(model, "model") or model.model is None:
            raise RuntimeError("Model is not initialized.")
        scores = cross_val_score(model.model, X_array, y_array, cv=cv, scoring=scoring)

        # Convert negative MSE to positive RMSE for regression
        if scoring == "neg_mean_squared_error":
            scores = np.sqrt(-scores)

        return {
            "scores": scores.tolist(),
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }

    def train_test_evaluate(
        self,
        model: BaseSupervisedModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Evaluate model using train-test split.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            test_size: Proportion of data for testing
            random_state: Random seed

        Returns:
            Dictionary with evaluation results
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array,
            y_array,
            test_size=test_size,
            random_state=random_state,
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate on test set
        if hasattr(model, "classes_"):  # Classification model
            y_pred = model.predict(X_test)
            metrics = {"accuracy": accuracy_score(y_test, y_pred), "test_size": len(X_test), "train_size": len(X_train)}
        else:  # Regression model
            y_pred = model.predict(X_test)
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_size": len(X_test),
                "train_size": len(X_train),
            }

        return metrics

    def compare_models(
        self,
        models: list[BaseSupervisedModel],
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple models using cross-validation.

        Args:
            models: List of models to compare
            X: Features
            y: Targets
            cv: Number of cross-validation folds

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for model in models:
            cv_results = self.cross_validate(model, X, y, cv=cv)
            results[model.model_name] = cv_results

        return results

    def get_best_model(
        self,
        models: list[BaseSupervisedModel],
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv: int = 5,
        metric: str = "mean",
    ) -> BaseSupervisedModel:
        """Get the best performing model.

        Args:
            models: List of models to compare
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            metric: Metric to use for comparison ('mean', 'min', 'max')

        Returns:
            Best performing model
        """
        comparison = self.compare_models(models, X, y, cv)

        best_model_name = None
        best_score = float("-inf") if metric in ["mean", "max"] else float("inf")

        for model_name, results in comparison.items():
            score = results[metric]

            if metric in ["mean", "max"]:
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            elif score < best_score:
                best_score = score
                best_model_name = model_name

        # Find the corresponding model
        for model in models:
            if model.model_name == best_model_name:
                return model

        raise ValueError("Could not find best model")

    def generate_report(
        self,
        model: BaseSupervisedModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> str:
        """Generate a comprehensive evaluation report.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets

        Returns:
            Evaluation report string
        """
        report_lines = []
        report_lines.append("Model Evaluation Report")
        report_lines.append("=====================")
        report_lines.append(f"Model: {model.model_name}")
        report_lines.append(f"Model Type: {type(model).__name__}")
        report_lines.append("")

        # Cross-validation results
        cv_results = self.cross_validate(model, X, y)
        report_lines.append("Cross-Validation Results:")
        report_lines.append(f"  Mean: {cv_results['mean']:.4f}")
        report_lines.append(f"  Std:  {cv_results['std']:.4f}")
        report_lines.append(f"  Min:  {cv_results['min']:.4f}")
        report_lines.append(f"  Max:  {cv_results['max']:.4f}")
        report_lines.append("")

        # Train-test results
        tt_results = self.train_test_evaluate(model, X, y)
        report_lines.append("Train-Test Split Results:")
        for metric, value in tt_results.items():
            if isinstance(value, float):
                report_lines.append(f"  {metric}: {value:.4f}")
            else:
                report_lines.append(f"  {metric}: {value}")
        report_lines.append("")

        # Feature importance if available
        feature_importance = model.get_feature_importance()
        if feature_importance:
            report_lines.append("Top 10 Feature Importance:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                report_lines.append(f"  {feature}: {importance:.4f}")

        return "\n".join(report_lines)
