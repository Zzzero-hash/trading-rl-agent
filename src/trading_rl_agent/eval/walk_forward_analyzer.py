"""
Walk-Forward Analysis for Trading Model Evaluation

This module implements walk-forward analysis to evaluate model performance
across multiple time windows, providing robust out-of-sample validation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy import stats

from trading_rl_agent.core.unified_config import ModelConfig
from trading_rl_agent.training.optimized_trainer import OptimizedTrainingManager
from trading_rl_agent.utils.metrics import calculate_comprehensive_metrics

from .metrics_calculator import MetricsCalculator
from .statistical_tests import StatisticalTests

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    # Window configuration
    train_window_size: int = 252  # ~1 year of trading days
    validation_window_size: int = 63  # ~3 months
    test_window_size: int = 63  # ~3 months
    step_size: int = 21  # ~1 month step

    # Model configuration
    model_type: str = "cnn_lstm"  # "cnn_lstm", "rl", "hybrid"
    retrain_models: bool = True
    save_models: bool = False
    model_save_dir: str | None = None

    # Evaluation configuration
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    include_benchmark: bool = True
    benchmark_column: str = "benchmark_returns"

    # Output configuration
    output_dir: str = "walk_forward_results"
    generate_plots: bool = True
    save_results: bool = True


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # Model performance
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]

    # Predictions and actual values
    test_predictions: np.ndarray
    test_actuals: np.ndarray
    test_returns: np.ndarray

    # Model information
    model_path: str | None = None
    training_time: float = 0.0
    inference_time: float = 0.0


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for robust model evaluation.

    Implements walk-forward analysis to evaluate model performance across
    multiple time windows, providing robust out-of-sample validation.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        model_config: ModelConfig | None = None,
    ):
        """Initialize the walk-forward analyzer."""
        self.config = config
        self.model_config = model_config or ModelConfig()

        # Add training_config attribute with default values
        self.training_config = type(
            "TrainingConfig",
            (),
            {"batch_size": 32, "learning_rate": 0.001, "epochs": 100, "early_stopping_patience": 10},
        )()

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.statistical_tests = StatisticalTests()

        # Results storage
        self.window_results: list[WindowResult] = []
        self.overall_metrics: dict[str, Any] = {}
        self.stability_analysis: dict[str, Any] = {}

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"WalkForwardAnalyzer initialized with {config.model_type} model type")

    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str = "returns",
        feature_columns: list[str] | None = None,
        benchmark_returns: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Perform complete walk-forward analysis.

        Args:
            data: DataFrame with features and target
            target_column: Column name for target variable
            feature_columns: List of feature column names
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary containing comprehensive analysis results
        """
        console.print("[bold blue]🚀 Starting Walk-Forward Analysis[/bold blue]")

        # Prepare data
        feature_columns = feature_columns or [col for col in data.columns if col != target_column]

        # Create time windows
        windows = self._create_time_windows(data)
        console.print(f"Created {len(windows)} time windows for analysis")

        # Process each window
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing windows...", total=len(windows))

            for window_id, (train_idx, val_idx, test_idx) in enumerate(windows):
                progress.update(task, description=f"Processing window {window_id + 1}/{len(windows)}")

                # Extract window data
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                test_data = data.iloc[test_idx]

                # Train and evaluate model
                window_result = self._process_window(
                    window_id=window_id,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    benchmark_returns=benchmark_returns,
                )

                self.window_results.append(window_result)
                progress.advance(task)

        # Calculate overall metrics and stability analysis
        self._calculate_overall_metrics()
        self._perform_stability_analysis()

        # Generate reports and visualizations
        if self.config.generate_plots:
            self._generate_visualizations()

        if self.config.save_results:
            self._save_results()

        console.print("[bold green]✅ Walk-Forward Analysis Complete[/bold green]")

        return {
            "window_results": self.window_results,
            "overall_metrics": self.overall_metrics,
            "stability_analysis": self.stability_analysis,
        }

    def _create_time_windows(self, data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create time windows for walk-forward analysis."""

        total_samples = len(data)
        train_size = self.config.train_window_size
        val_size = self.config.validation_window_size
        test_size = self.config.test_window_size
        step_size = self.config.step_size

        windows = []
        start_idx = 0

        while start_idx + train_size + val_size + test_size <= total_samples:
            train_end = start_idx + train_size
            val_end = train_end + val_size
            test_end = val_end + test_size

            train_idx = np.arange(start_idx, train_end)
            val_idx = np.arange(train_end, val_end)
            test_idx = np.arange(val_end, test_end)

            windows.append((train_idx, val_idx, test_idx))
            start_idx += step_size

        return windows

    def _process_window(
        self,
        window_id: int,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        benchmark_returns: np.ndarray | None,
    ) -> WindowResult:
        """Process a single walk-forward window."""

        import time

        # Prepare data
        X_train = train_data[feature_columns].values
        y_train = train_data[target_column].values
        X_val = val_data[feature_columns].values
        y_val = val_data[target_column].values
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values

        # Train model
        start_time = time.time()
        model, train_metrics, val_metrics = self._train_model(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # Evaluate on test set
        start_time = time.time()
        test_predictions, test_metrics = self._evaluate_model(model, X_test, y_test)
        inference_time = time.time() - start_time

        # Calculate returns
        test_returns = self._calculate_returns(test_predictions, y_test)

        # Add benchmark comparison if available
        if benchmark_returns is not None and len(benchmark_returns) >= len(test_returns):
            benchmark_window = benchmark_returns[-len(test_returns) :]
            test_metrics.update(calculate_comprehensive_metrics(test_returns, benchmark_window))

        # Save model if requested
        model_path = None
        if self.config.save_models and self.config.model_save_dir:
            model_path = self._save_model(model, window_id)

        return WindowResult(
            window_id=window_id,
            train_start=train_data.index[0],
            train_end=train_data.index[-1],
            validation_start=val_data.index[0],
            validation_end=val_data.index[-1],
            test_start=test_data.index[0],
            test_end=test_data.index[-1],
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            test_predictions=test_predictions,
            test_actuals=y_test,
            test_returns=test_returns,
            model_path=model_path,
            training_time=training_time,
            inference_time=inference_time,
        )

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[Any, dict[str, float], dict[str, float]]:
        """Train model for current window."""

        if self.config.model_type == "cnn_lstm":
            return self._train_cnn_lstm(X_train, y_train, X_val, y_val)
        if self.config.model_type == "rl":
            return self._train_rl_agent(X_train, y_train, X_val, y_val)
        raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def _train_cnn_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[Any, dict[str, float], dict[str, float]]:
        """Train CNN+LSTM model."""

        # Create model
        from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

        model = CNNLSTMModel(self.model_config)

        # Create trainer
        trainer = OptimizedTrainingManager(
            model=model,
            device="auto",
            enable_amp=True,
        )

        # Prepare data loaders
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        # Train model
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
        )
        criterion = torch.nn.MSELoss()

        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=self.training_config.epochs,
            early_stopping_patience=self.training_config.early_stopping_patience,
        )

        return model, training_results["train_metrics"], training_results["val_metrics"]

    def _train_rl_agent(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[Any, dict[str, float], dict[str, float]]:
        """Train RL agent."""

        # Placeholder for RL training
        # This would integrate with the existing RL training infrastructure
        logger.warning("RL agent training not yet implemented in walk-forward analysis")

        # Return dummy results for now
        dummy_model = None
        train_metrics = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
        val_metrics = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}

        return dummy_model, train_metrics, val_metrics

    def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Evaluate model on test data."""

        if model is None:
            # Return dummy predictions for RL placeholder
            predictions = np.random.normal(0, 0.1, len(y_test))
        else:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                predictions = model(X_tensor).cpu().numpy().flatten()

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_prediction_metrics(y_test, predictions)

        return predictions, metrics

    def _calculate_returns(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """Calculate trading returns based on predictions."""

        # Simple strategy: long when prediction > 0, short when prediction < 0
        positions = np.sign(predictions)
        return positions * actuals

    def _save_model(self, model: Any, window_id: int) -> str:
        """Save trained model."""

        if self.config.model_save_dir is None:
            raise ValueError("model_save_dir is not configured")
        model_dir = Path(self.config.model_save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"model_window_{window_id}.pth"

        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), model_path)

        return str(model_path)

    def _calculate_overall_metrics(self) -> None:
        """Calculate overall performance metrics across all windows."""

        if not self.window_results:
            return

        # Extract test metrics from all windows
        test_metrics_list = [result.test_metrics for result in self.window_results]
        test_returns_list = [result.test_returns for result in self.window_results]

        # Calculate aggregate metrics
        self.overall_metrics = {
            "num_windows": len(self.window_results),
            "avg_metrics": self._calculate_average_metrics(test_metrics_list),
            "std_metrics": self._calculate_std_metrics(test_metrics_list),
            "confidence_intervals": self._calculate_confidence_intervals(test_metrics_list),
            "performance_degradation": self._calculate_performance_degradation(),
            "stability_metrics": self._calculate_stability_metrics(test_returns_list),
        }

    def _calculate_average_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Calculate average metrics across windows."""

        avg_metrics = {}
        metric_names = metrics_list[0].keys()

        for metric in metric_names:
            values = [metrics.get(metric, 0.0) for metrics in metrics_list]
            avg_metrics[metric] = float(np.mean(values))

        return avg_metrics

    def _calculate_std_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Calculate standard deviation of metrics across windows."""

        std_metrics = {}
        metric_names = metrics_list[0].keys()

        for metric in metric_names:
            values = [metrics.get(metric, 0.0) for metrics in metrics_list]
            std_metrics[metric] = float(np.std(values))

        return std_metrics

    def _calculate_confidence_intervals(
        self,
        metrics_list: list[dict[str, float]],
        confidence_level: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Calculate confidence intervals for metrics."""

        confidence_intervals = {}
        metric_names = metrics_list[0].keys()

        for metric in metric_names:
            values = [metrics.get(metric, 0.0) for metrics in metrics_list]
            values = np.array(values)

            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)

            confidence_intervals[metric] = (float(lower_bound), float(upper_bound))

        return confidence_intervals

    def _calculate_performance_degradation(self) -> dict[str, float]:
        """Calculate performance degradation over time."""

        if len(self.window_results) < 2:
            return {}

        # Calculate correlation between window ID and performance metrics
        window_ids = [result.window_id for result in self.window_results]
        key_metrics = ["sharpe_ratio", "total_return", "max_drawdown"]

        degradation_metrics = {}

        for metric in key_metrics:
            values = [result.test_metrics.get(metric, 0.0) for result in self.window_results]

            if len(values) > 1:
                correlation, p_value = stats.pearsonr(window_ids, values)
                degradation_metrics[f"{metric}_degradation_correlation"] = correlation
                degradation_metrics[f"{metric}_degradation_p_value"] = p_value

        return degradation_metrics

    def _calculate_stability_metrics(self, returns_list: list[np.ndarray]) -> dict[str, float]:
        """Calculate stability metrics across windows."""

        if not returns_list:
            return {}

        # Calculate coefficient of variation for key metrics
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []

        for returns in returns_list:
            if len(returns) > 0:
                metrics = calculate_comprehensive_metrics(returns)
                sharpe_ratios.append(metrics.get("sharpe_ratio", 0.0))
                total_returns.append(metrics.get("total_return", 0.0))
                max_drawdowns.append(metrics.get("max_drawdown", 0.0))

        stability_metrics = {}

        if sharpe_ratios:
            stability_metrics["sharpe_cv"] = (
                np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else 0
            )
        if total_returns:
            stability_metrics["returns_cv"] = (
                np.std(total_returns) / np.mean(total_returns) if np.mean(total_returns) != 0 else 0
            )
        if max_drawdowns:
            stability_metrics["drawdown_cv"] = (
                np.std(max_drawdowns) / np.mean(max_drawdowns) if np.mean(max_drawdowns) != 0 else 0
            )

        return stability_metrics

    def _perform_stability_analysis(self) -> None:
        """Perform comprehensive stability analysis."""

        if not self.window_results:
            return

        # Extract all test returns
        all_returns = []
        for result in self.window_results:
            all_returns.extend(result.test_returns)

        all_returns = np.array(all_returns)

        # Perform statistical tests
        self.stability_analysis = {
            "normality_test": self.statistical_tests.test_normality(all_returns),
            "stationarity_test": self.statistical_tests.test_stationarity(all_returns),
            "autocorrelation_test": self.statistical_tests.test_autocorrelation(all_returns),
            "volatility_clustering": self.statistical_tests.test_volatility_clustering(all_returns),
        }

    def _generate_visualizations(self) -> None:
        """Generate comprehensive visualizations."""

        if not self.window_results:
            return

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Walk-Forward Analysis Results", fontsize=16, fontweight="bold")

        # 1. Performance over time
        self._plot_performance_over_time(axes[0, 0])

        # 2. Metric distributions
        self._plot_metric_distributions(axes[0, 1])

        # 3. Confidence intervals
        self._plot_confidence_intervals(axes[0, 2])

        # 4. Performance degradation
        self._plot_performance_degradation(axes[1, 0])

        # 5. Stability analysis
        self._plot_stability_analysis(axes[1, 1])

        # 6. Returns distribution
        self._plot_returns_distribution(axes[1, 2])

        plt.tight_layout()
        plt.savefig(self.output_dir / "walk_forward_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_performance_over_time(self, ax: plt.Axes) -> None:
        """Plot performance metrics over time."""

        window_ids = [result.window_id for result in self.window_results]
        sharpe_ratios = [result.test_metrics.get("sharpe_ratio", 0.0) for result in self.window_results]
        total_returns = [result.test_metrics.get("total_return", 0.0) for result in self.window_results]

        ax.plot(window_ids, sharpe_ratios, "o-", label="Sharpe Ratio", alpha=0.7)
        ax_twin = ax.twinx()
        ax_twin.plot(window_ids, total_returns, "s-", color="orange", label="Total Return", alpha=0.7)

        ax.set_xlabel("Window ID")
        ax.set_ylabel("Sharpe Ratio", color="blue")
        ax_twin.set_ylabel("Total Return", color="orange")
        ax.set_title("Performance Over Time")
        ax.grid(True, alpha=0.3)

    def _plot_metric_distributions(self, ax: plt.Axes) -> None:
        """Plot distributions of key metrics."""

        sharpe_ratios = [result.test_metrics.get("sharpe_ratio", 0.0) for result in self.window_results]
        max_drawdowns = [result.test_metrics.get("max_drawdown", 0.0) for result in self.window_results]

        ax.hist(sharpe_ratios, bins=10, alpha=0.7, label="Sharpe Ratio", color="blue")
        ax.hist(max_drawdowns, bins=10, alpha=0.7, label="Max Drawdown", color="red")

        ax.set_xlabel("Metric Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Metric Distributions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confidence_intervals(self, ax: plt.Axes) -> None:
        """Plot confidence intervals for key metrics."""

        if not self.overall_metrics.get("confidence_intervals"):
            return

        metrics = ["sharpe_ratio", "total_return", "max_drawdown"]
        means = []
        lower_bounds = []
        upper_bounds = []

        for metric in metrics:
            if metric in self.overall_metrics["confidence_intervals"]:
                lower, upper = self.overall_metrics["confidence_intervals"][metric]
                mean = self.overall_metrics["avg_metrics"].get(metric, 0.0)

                means.append(mean)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

        x_pos = np.arange(len(metrics))
        ax.errorbar(
            x_pos,
            means,
            yerr=[np.array(means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(means)],
            fmt="o",
            capsize=5,
            capthick=2,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylabel("Metric Value")
        ax.set_title("Confidence Intervals")
        ax.grid(True, alpha=0.3)

    def _plot_performance_degradation(self, ax: plt.Axes) -> None:
        """Plot performance degradation analysis."""

        if not self.overall_metrics.get("performance_degradation"):
            return

        degradation = self.overall_metrics["performance_degradation"]
        metrics = []
        correlations = []

        for key, value in degradation.items():
            if "correlation" in key:
                metric_name = key.replace("_degradation_correlation", "")
                metrics.append(metric_name)
                correlations.append(value)

        colors = ["red" if cor < 0 else "green" for cor in correlations]
        bars = ax.bar(metrics, correlations, color=colors, alpha=0.7)

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Correlation with Window ID")
        ax.set_title("Performance Degradation")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{corr:.3f}",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

    def _plot_stability_analysis(self, ax: plt.Axes) -> None:
        """Plot stability analysis results."""

        if not self.stability_analysis:
            return

        stability_metrics = self.overall_metrics.get("stability_metrics", {})

        if not stability_metrics:
            return

        metrics = list(stability_metrics.keys())
        values = list(stability_metrics.values())

        bars = ax.bar(metrics, values, alpha=0.7, color="purple")
        ax.set_xlabel("Stability Metrics")
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title("Model Stability")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.3f}", ha="center", va="bottom")

    def _plot_returns_distribution(self, ax: plt.Axes) -> None:
        """Plot returns distribution."""

        all_returns = []
        for result in self.window_results:
            all_returns.extend(result.test_returns)

        if not all_returns:
            return

        all_returns = np.array(all_returns)

        ax.hist(all_returns, bins=50, alpha=0.7, density=True, color="green")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Zero Return")

        # Add normal distribution overlay
        mu, sigma = np.mean(all_returns), np.std(all_returns)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal Distribution")

        ax.set_xlabel("Returns")
        ax.set_ylabel("Density")
        ax.set_title("Returns Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _save_results(self) -> None:
        """Save analysis results to files."""

        # Save window results
        results_df = pd.DataFrame(
            [
                {
                    "window_id": result.window_id,
                    "train_start": result.train_start,
                    "train_end": result.train_end,
                    "test_start": result.test_start,
                    "test_end": result.test_end,
                    **result.test_metrics,
                    "training_time": result.training_time,
                    "inference_time": result.inference_time,
                }
                for result in self.window_results
            ]
        )

        results_df.to_csv(self.output_dir / "window_results.csv", index=False)

        # Save overall metrics
        overall_metrics_df = pd.DataFrame([self.overall_metrics["avg_metrics"]])
        overall_metrics_df.to_csv(self.output_dir / "overall_metrics.csv", index=False)

        # Save stability analysis
        stability_df = pd.DataFrame([self.stability_analysis])
        stability_df.to_csv(self.output_dir / "stability_analysis.csv", index=False)

        # Generate summary report
        self._generate_summary_report()

    def _generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""

        report_path = self.output_dir / "walk_forward_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("Analysis Configuration:\n")
            f.write(f"- Model Type: {self.config.model_type}\n")
            f.write(f"- Number of Windows: {len(self.window_results)}\n")
            f.write(f"- Train Window Size: {self.config.train_window_size}\n")
            f.write(f"- Validation Window Size: {self.config.validation_window_size}\n")
            f.write(f"- Test Window Size: {self.config.test_window_size}\n")
            f.write(f"- Step Size: {self.config.step_size}\n\n")

            f.write("Overall Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            f.writelines(f"{metric}: {value:.4f}\n" for metric, value in self.overall_metrics["avg_metrics"].items())
            f.write("\n")

            f.write("Performance Stability:\n")
            f.write("-" * 40 + "\n")
            f.writelines(
                f"{metric}: {value:.4f}\n"
                for metric, value in self.overall_metrics.get("stability_metrics", {}).items()
            )
            f.write("\n")

            f.write("Performance Degradation Analysis:\n")
            f.write("-" * 40 + "\n")
            f.writelines(
                f"{metric}: {value:.4f}\n"
                for metric, value in self.overall_metrics.get("performance_degradation", {}).items()
            )
            f.write("\n")

            f.write("Statistical Tests:\n")
            f.write("-" * 40 + "\n")
            f.writelines(f"{test}: {result}\n" for test, result in self.stability_analysis.items())
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

    def print_summary(self) -> None:
        """Print summary of walk-forward analysis results."""

        if not self.overall_metrics:
            console.print("[red]No analysis results available. Run analyze() first.[/red]")
            return

        # Create summary table
        table = Table(title="Walk-Forward Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Std Dev", style="yellow")

        avg_metrics = self.overall_metrics["avg_metrics"]
        std_metrics = self.overall_metrics["std_metrics"]

        for metric in ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]:
            if metric in avg_metrics:
                avg_val = avg_metrics[metric]
                std_val = std_metrics.get(metric, 0.0)
                table.add_row(metric.replace("_", " ").title(), f"{avg_val:.4f}", f"{std_val:.4f}")

        console.print(table)

        # Print stability metrics
        stability_metrics = self.overall_metrics.get("stability_metrics", {})
        if stability_metrics:
            console.print("\n[bold]Stability Metrics:[/bold]")
            for metric, value in stability_metrics.items():
                console.print(f"  {metric}: {value:.4f}")

        # Print degradation analysis
        degradation = self.overall_metrics.get("performance_degradation", {})
        if degradation:
            console.print("\n[bold]Performance Degradation:[/bold]")
            for metric, value in degradation.items():
                if "correlation" in metric:
                    color = "red" if value < 0 else "green"
                    console.print(f"  {metric}: [{color}]{value:.4f}[/{color}]")
