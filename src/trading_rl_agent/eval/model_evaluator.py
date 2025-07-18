"""
Model Evaluator for Trading Model Assessment

This module provides comprehensive model evaluation capabilities including
model comparison, performance analysis, and statistical validation.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .metrics_calculator import MetricsCalculator
from .statistical_tests import StatisticalTests

console = Console()
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for trading models.

    Provides capabilities for:
    - Model performance assessment
    - Model comparison and ranking
    - Statistical validation
    - Performance attribution analysis
    """

    def __init__(self) -> None:
        """Initialize the model evaluator."""
        self.metrics_calculator = MetricsCalculator()
        self.statistical_tests = StatisticalTests()

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        task_type: str = "regression",
        benchmark_returns: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single model comprehensively.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for reporting
            task_type: Type of task ("regression" or "classification")
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with comprehensive evaluation results
        """
        console.print(f"[bold blue]Evaluating {model_name}...[/bold blue]")

        # Generate predictions
        predictions = self._generate_predictions(model, X_test)

        # Calculate prediction metrics
        prediction_metrics = self.metrics_calculator.calculate_prediction_metrics(y_test, predictions, task_type)

        # Calculate trading metrics if applicable
        trading_metrics = {}
        if task_type == "regression":
            returns = self._calculate_returns(predictions, y_test)
            trading_metrics = self.metrics_calculator.calculate_trading_metrics(returns, benchmark_returns)

        # Perform statistical tests on residuals
        residuals = y_test - predictions
        residual_tests = self.statistical_tests.test_model_residuals(residuals, predictions)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(prediction_metrics)

        results = {
            "model_name": model_name,
            "task_type": task_type,
            "prediction_metrics": prediction_metrics,
            "trading_metrics": trading_metrics,
            "residual_tests": residual_tests,
            "confidence_intervals": confidence_intervals,
            "predictions": predictions,
            "actuals": y_test,
            "residuals": residuals,
        }

        console.print(f"[bold green]✅ {model_name} evaluation complete[/bold green]")
        return results

    def compare_models(
        self,
        models: dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "regression",
        benchmark_returns: np.ndarray | None = None,
        comparison_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare multiple models comprehensively.

        Args:
            models: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test targets
            task_type: Type of task ("regression" or "classification")
            benchmark_returns: Optional benchmark returns for comparison
            comparison_metrics: List of metrics to use for comparison

        Returns:
            Dictionary with comparison results
        """
        console.print("[bold blue]Comparing models...[/bold blue]")

        # Evaluate each model
        model_results = {}
        for model_name, model in models.items():
            results = self.evaluate_model(model, X_test, y_test, model_name, task_type, benchmark_returns)
            model_results[model_name] = results

        # Perform model comparison
        comparison_results = self._perform_model_comparison(model_results, comparison_metrics)

        # Generate ranking
        ranking = self._generate_model_ranking(model_results, comparison_metrics)

        # Statistical significance testing
        significance_tests = self._perform_significance_tests(model_results)

        results = {
            "model_results": model_results,
            "comparison_results": comparison_results,
            "ranking": ranking,
            "significance_tests": significance_tests,
        }

        console.print("[bold green]✅ Model comparison complete[/bold green]")
        return results

    def _generate_predictions(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions from model."""

        if hasattr(model, "predict"):
            # Scikit-learn style model
            predictions = model.predict(X_test)
        elif hasattr(model, "forward"):
            # PyTorch model
            import torch

            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                predictions = model(X_tensor).cpu().numpy().flatten()
        else:
            # Try to call the model directly
            try:
                predictions = model(X_test)
                if hasattr(predictions, "numpy"):
                    predictions = predictions.numpy()
                predictions = np.array(predictions).flatten()
            except Exception as e:
                logger.exception(f"Error generating predictions: {e}")
                raise ValueError(f"Cannot generate predictions from model: {e}") from e

        return predictions

    def _calculate_returns(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Calculate trading returns based on predictions."""

        # Simple strategy: long when prediction > 0, short when prediction < 0
        positions = np.sign(predictions)
        return positions * actuals

    def _calculate_confidence_intervals(
        self,
        metrics: dict[str, float],
        confidence_level: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Calculate confidence intervals for metrics."""

        # This is a simplified version - in practice, you'd want to use
        # bootstrap methods or other statistical techniques
        confidence_intervals = {}

        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Simple approximation - in practice, use proper statistical methods
                std_error = abs(value) * 0.1  # 10% of value as approximation
                lower_bound = value - 1.96 * std_error
                upper_bound = value + 1.96 * std_error
                confidence_intervals[metric] = (float(lower_bound), float(upper_bound))

        return confidence_intervals

    def _perform_model_comparison(
        self,
        model_results: dict[str, dict[str, Any]],
        comparison_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive model comparison."""

        if not model_results:
            return {}

        # Default comparison metrics
        if comparison_metrics is None:
            comparison_metrics = ["mae", "rmse", "r2_score", "sharpe_ratio", "max_drawdown"]

        comparison_results = {}

        for metric in comparison_metrics:
            metric_values = {}
            for model_name, results in model_results.items():
                # Check in prediction metrics first
                if metric in results.get("prediction_metrics", {}):
                    metric_values[model_name] = results["prediction_metrics"][metric]
                # Then check in trading metrics
                elif metric in results.get("trading_metrics", {}):
                    metric_values[model_name] = results["trading_metrics"][metric]

            if metric_values:
                comparison_results[metric] = {
                    "values": metric_values,
                    "best_model": (
                        min(metric_values, key=lambda k: metric_values[k])
                        if metric in ["mae", "rmse", "max_drawdown"]
                        else max(metric_values, key=lambda k: metric_values[k])
                    ),
                    "worst_model": (
                        max(metric_values, key=lambda k: metric_values[k])
                        if metric in ["mae", "rmse", "max_drawdown"]
                        else min(metric_values, key=lambda k: metric_values[k])
                    ),
                    "mean": float(np.mean(list(metric_values.values()))),
                    "std": float(np.std(list(metric_values.values()))),
                }

        return comparison_results

    def _generate_model_ranking(
        self,
        model_results: dict[str, dict[str, Any]],
        comparison_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate model ranking based on multiple metrics."""

        if not model_results:
            return {}

        # Default ranking metrics
        if comparison_metrics is None:
            comparison_metrics = ["mae", "rmse", "r2_score", "sharpe_ratio", "max_drawdown"]

        # Calculate scores for each model
        model_scores = {}
        for model_name in model_results:
            model_scores[model_name] = 0

        for metric in comparison_metrics:
            metric_values = {}
            for model_name, results in model_results.items():
                # Check in prediction metrics first
                if metric in results.get("prediction_metrics", {}):
                    metric_values[model_name] = results["prediction_metrics"][metric]
                # Then check in trading metrics
                elif metric in results.get("trading_metrics", {}):
                    metric_values[model_name] = results["trading_metrics"][metric]

            if metric_values:
                # Sort models by metric value
                if metric in ["mae", "rmse", "max_drawdown"]:
                    # Lower is better
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    # Higher is better
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)

                # Assign scores (higher rank = higher score)
                for rank, (model_name, _) in enumerate(sorted_models):
                    model_scores[model_name] += len(sorted_models) - rank

        # Generate final ranking
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "overall_scores": dict(sorted_models),
            "ranked_models": [model_name for model_name, _ in sorted_models],
            "best_model": sorted_models[0][0] if sorted_models else None,
            "worst_model": sorted_models[-1][0] if sorted_models else None,
        }

    def _perform_significance_tests(
        self,
        model_results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Perform statistical significance tests between models."""

        if len(model_results) < 2:
            return {}

        significance_tests = {}
        model_names = list(model_results.keys())

        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i + 1 :], i + 1):
                pair_name = f"{model1}_vs_{model2}"

                # Get residuals for both models
                residuals1 = model_results[model1]["residuals"]
                residuals2 = model_results[model2]["residuals"]

                # Perform statistical tests
                tests = {}

                # T-test on residuals
                tests["residual_t_test"] = self.metrics_calculator.calculate_statistical_significance(
                    residuals1,
                    residuals2,
                    "t_test",
                )

                # Wilcoxon test on residuals
                tests["residual_wilcoxon"] = self.metrics_calculator.calculate_statistical_significance(
                    residuals1,
                    residuals2,
                    "wilcoxon",
                )

                # Compare key metrics
                key_metrics = ["mae", "rmse", "r2_score"]
                for metric in key_metrics:
                    metric1 = model_results[model1]["prediction_metrics"].get(metric, 0.0)
                    metric2 = model_results[model2]["prediction_metrics"].get(metric, 0.0)

                    # Create arrays for testing (simplified approach)
                    # In practice, you'd want to use proper statistical methods
                    tests[f"{metric}_comparison"] = {
                        "model1_value": metric1,
                        "model2_value": metric2,
                        "difference": metric1 - metric2,
                        "model1_better": metric1 < metric2 if metric in ["mae", "rmse"] else metric1 > metric2,
                    }

                significance_tests[pair_name] = tests

        return significance_tests

    def print_evaluation_summary(self, evaluation_results: dict[str, Any]) -> None:
        """Print a summary of evaluation results."""

        if "model_name" in evaluation_results:
            # Single model evaluation
            self._print_single_model_summary(evaluation_results)
        else:
            # Multiple model comparison
            self._print_comparison_summary(evaluation_results)

    def _print_single_model_summary(self, results: dict[str, Any]) -> None:
        """Print summary for a single model evaluation."""

        model_name = results["model_name"]

        # Create summary table
        table = Table(title=f"Evaluation Summary - {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("CI Lower", style="yellow")
        table.add_column("CI Upper", style="yellow")

        # Add prediction metrics
        prediction_metrics = results.get("prediction_metrics", {})
        for metric, value in prediction_metrics.items():
            ci_lower, ci_upper = results["confidence_intervals"].get(metric, (0.0, 0.0))
            table.add_row(
                metric.replace("_", " ").title(),
                f"{value:.4f}",
                f"{ci_lower:.4f}",
                f"{ci_upper:.4f}",
            )

        # Add trading metrics
        trading_metrics = results.get("trading_metrics", {})
        for metric, value in trading_metrics.items():
            ci_lower, ci_upper = results["confidence_intervals"].get(metric, (0.0, 0.0))
            table.add_row(
                metric.replace("_", " ").title(),
                f"{value:.4f}",
                f"{ci_lower:.4f}",
                f"{ci_upper:.4f}",
            )

        console.print(table)

        # Print residual test results
        residual_tests = results.get("residual_tests", {})
        if residual_tests:
            console.print("\n[bold]Residual Tests:[/bold]")
            for test_name, test_result in residual_tests.items():
                status = "✅" if test_result.get("is_normal", True) else "❌"
                console.print(f"  {test_name}: {status} p={test_result.get('p_value', 0.0):.4f}")

    def _print_comparison_summary(self, results: dict[str, Any]) -> None:
        """Print summary for model comparison."""

        # Print ranking
        ranking = results.get("ranking", {})
        if ranking:
            console.print("\n[bold]Model Ranking:[/bold]")
            for i, model_name in enumerate(ranking.get("ranked_models", []), 1):
                score = ranking["overall_scores"].get(model_name, 0)
                console.print(f"  {i}. {model_name} (Score: {score})")

        # Print comparison results
        comparison_results = results.get("comparison_results", {})
        if comparison_results:
            console.print("\n[bold]Metric Comparison:[/bold]")
            for metric, comparison in comparison_results.items():
                best_model = comparison.get("best_model", "N/A")
                worst_model = comparison.get("worst_model", "N/A")
                mean_val = comparison.get("mean", 0.0)
                std_val = comparison.get("std", 0.0)

                console.print(f"  {metric}:")
                console.print(f"    Best: {best_model} (Mean: {mean_val:.4f} ± {std_val:.4f})")
                console.print(f"    Worst: {worst_model}")

        # Print significance tests
        significance_tests = results.get("significance_tests", {})
        if significance_tests:
            console.print("\n[bold]Statistical Significance Tests:[/bold]")
            for pair_name, tests in significance_tests.items():
                console.print(f"  {pair_name}:")
                for test_name, test_result in tests.items():
                    if "p_value" in test_result:
                        significant = "✅" if test_result["significant"] else "❌"
                        console.print(f"    {test_name}: {significant} p={test_result['p_value']:.4f}")

    def save_evaluation_results(
        self,
        results: dict[str, Any],
        output_path: str | Path,
        file_format: str = "json",
    ) -> None:
        """Save evaluation results to file."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        elif file_format == "csv":
            # Save key metrics to CSV
            if "model_results" in results:
                # Multiple model comparison
                df_data = []
                for model_name, model_results in results["model_results"].items():
                    row = {"model_name": model_name}
                    row.update(model_results.get("prediction_metrics", {}))
                    row.update(model_results.get("trading_metrics", {}))
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            else:
                # Single model evaluation
                row = {"model_name": results.get("model_name", "unknown")}
                row.update(results.get("prediction_metrics", {}))
                row.update(results.get("trading_metrics", {}))
                df = pd.DataFrame([row])
                df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        console.print(f"[green]Results saved to {output_path}[/green]")
