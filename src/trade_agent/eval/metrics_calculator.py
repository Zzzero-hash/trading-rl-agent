"""
Metrics Calculator for Trading Model Evaluation

This module provides comprehensive metrics calculation for evaluating
trading model performance, including prediction accuracy, risk metrics,
and trading-specific metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from trade_agent.utils.metrics import calculate_comprehensive_metrics


class MetricsCalculator:
    """
    Comprehensive metrics calculator for trading model evaluation.

    Provides metrics for:
    - Prediction accuracy (regression and classification)
    - Risk-adjusted returns
    - Trading-specific metrics
    - Statistical significance testing
    """

    def __init__(self) -> None:
        """Initialize the metrics calculator."""

    def calculate_prediction_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "regression",
    ) -> dict[str, float]:
        """
        Calculate comprehensive prediction metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: Type of task ("regression" or "classification")

        Returns:
            Dictionary of metrics
        """
        if task_type == "regression":
            return self._calculate_regression_metrics(y_true, y_pred)
        if task_type == "classification":
            return self._calculate_classification_metrics(y_true, y_pred)
        raise ValueError(f"Unsupported task type: {task_type}")

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate regression metrics."""

        # Basic regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        mape = self._calculate_mape(y_true, y_pred)
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

        # Directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape),
            "correlation": float(correlation),
            "directional_accuracy": float(directional_accuracy),
        }

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate classification metrics."""

        # Convert to binary classification if needed
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        y_true = y_true.astype(int)

        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    def calculate_trading_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """
        Calculate trading-specific performance metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary of trading metrics
        """

        # Use existing comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            returns,
            benchmark_returns,
            risk_free_rate,
        )

        # Add additional trading-specific metrics
        additional_metrics = self._calculate_additional_trading_metrics(returns)
        metrics.update(additional_metrics)

        return dict(metrics)  # Ensure it's a dict[str, float]

    def _calculate_additional_trading_metrics(self, returns: np.ndarray) -> dict[str, float]:
        """Calculate additional trading-specific metrics."""

        if len(returns) == 0:
            return {}

        # Convert to pandas Series for easier calculations
        returns_series = pd.Series(returns)

        # Maximum consecutive losses
        consecutive_losses = self._calculate_max_consecutive_losses(returns_series)

        # Maximum consecutive wins
        consecutive_wins = self._calculate_max_consecutive_wins(returns_series)

        # Average win and loss
        wins = returns_series[returns_series > 0]
        losses = returns_series[returns_series < 0]

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        # Largest win and loss
        largest_win = float(returns_series.max()) if len(returns_series) > 0 else 0.0
        largest_loss = float(returns_series.min()) if len(returns_series) > 0 else 0.0

        # Skewness and kurtosis
        skewness = float(returns_series.skew()) if len(returns_series) > 2 else 0.0
        kurtosis = float(returns_series.kurtosis()) if len(returns_series) > 2 else 0.0

        return {
            "max_consecutive_losses": consecutive_losses,
            "max_consecutive_wins": consecutive_wins,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        if len(y_true) == 0:
            return 0.0

        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)

    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (correct sign predictions)."""
        if len(y_true) < 2:
            return 0.0

        # Calculate direction changes
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        # Calculate accuracy
        accuracy = np.mean(true_direction == pred_direction)
        return float(accuracy)

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses."""
        if len(returns) == 0:
            return 0

        # Create a series of 1s for losses, 0s for wins
        losses = (returns < 0).astype(int)

        # Find consecutive losses
        max_consecutive = 0
        current_consecutive = 0

        for loss in losses:
            if loss == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_max_consecutive_wins(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive wins."""
        if len(returns) == 0:
            return 0

        # Create a series of 1s for wins, 0s for losses
        wins = (returns > 0).astype(int)

        # Find consecutive wins
        max_consecutive = 0
        current_consecutive = 0

        for win in wins:
            if win == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def calculate_bootstrap_confidence_intervals(
        self,
        metric_values: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap_samples: int = 1000,
    ) -> tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for a metric.

        Args:
            metric_values: Array of metric values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap_samples: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(metric_values) == 0:
            return 0.0, 0.0

        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap_samples):
            sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
            bootstrap_means.append(np.mean(sample))

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)

        return float(lower_bound), float(upper_bound)

    def calculate_statistical_significance(
        self,
        metric_values_1: np.ndarray,
        metric_values_2: np.ndarray,
        test_type: str = "t_test",
    ) -> dict[str, float]:
        """
        Calculate statistical significance between two sets of metric values.

        Args:
            metric_values_1: First set of metric values
            metric_values_2: Second set of metric values
            test_type: Type of statistical test ("t_test", "wilcoxon", "mann_whitney")

        Returns:
            Dictionary with test results
        """
        from scipy import stats

        if len(metric_values_1) == 0 or len(metric_values_2) == 0:
            return {"p_value": 1.0, "statistic": 0.0, "significant": False}

        if test_type == "t_test":
            statistic, p_value = stats.ttest_ind(metric_values_1, metric_values_2)
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(metric_values_1, metric_values_2)
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(metric_values_1, metric_values_2, alternative="two-sided")
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        return {
            "p_value": float(p_value),
            "statistic": float(statistic),
            "significant": p_value < 0.05,
        }
