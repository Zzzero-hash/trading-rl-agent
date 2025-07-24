"""
Benchmarking Framework for Policy Optimization Methods.

This module provides comprehensive benchmarking capabilities for comparing
different policy optimization algorithms with standardized metrics and visualizations.
"""


import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .advanced_trainer import AdvancedTrainer
from .configs import (
    AdvancedPPOConfig,
    NaturalPolicyGradientConfig,
    TRPOConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""

    # Environment parameters
    state_dim: int = 50
    action_dim: int = 3
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000

    # Training parameters
    batch_size: int = 256
    eval_frequency: int = 100
    save_frequency: int = 500

    # Benchmark parameters
    num_runs: int = 5
    confidence_level: float = 0.95

    # Output parameters
    save_dir: str = "benchmark_results"
    save_plots: bool = True
    save_data: bool = True

    # Algorithm configurations
    algorithms: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    algorithm_name: str
    run_id: int
    episode_rewards: list[float]
    episode_lengths: list[float]
    training_metrics: list[dict[str, float]]
    evaluation_metrics: dict[str, float]
    training_time: float
    memory_usage: float
    convergence_episode: int | None = None

    def compute_statistics(self) -> dict[str, float]:
        """Compute summary statistics for this run."""
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "final_avg_reward": float(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)),
            "convergence_episode": self.convergence_episode or len(rewards),
        }


class BenchmarkFramework:
    """Comprehensive benchmarking framework for policy optimization methods."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: dict[str, list[BenchmarkResult]] = {}
        self.summary_stats: dict[str, dict[str, float]] = {}

        # Create output directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize default algorithm configurations
        self._setup_default_configs()

    def _setup_default_configs(self) -> None:
        """Setup default algorithm configurations if not provided."""
        if not self.config.algorithms:
            self.config.algorithms = {
                "advanced_ppo": AdvancedPPOConfig(),
                "trpo": TRPOConfig(),
                "natural_policy_gradient": NaturalPolicyGradientConfig(),
            }

    def run_benchmark(
        self,
        env_creator: Callable[[], Any],
        algorithms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive benchmark of policy optimization algorithms."""
        self.logger.info("Starting comprehensive benchmark")

        if algorithms is None:
            algorithms = list(self.config.algorithms.keys())

        start_time = time.time()

        for algorithm_name in algorithms:
            if algorithm_name not in self.config.algorithms:
                self.logger.warning(f"Algorithm {algorithm_name} not found in config, skipping")
                continue

            self.logger.info(f"Benchmarking {algorithm_name}")
            algorithm_results = []

            for run_id in range(self.config.num_runs):
                self.logger.info(f"  Run {run_id + 1}/{self.config.num_runs}")

                # Create environment
                env = env_creator()

                # Create trainer
                trainer = AdvancedTrainer(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    device="cpu",  # Use CPU for consistent benchmarking
                )

                # Run training
                run_result = self._run_single_benchmark(trainer, env, algorithm_name, run_id)
                algorithm_results.append(run_result)

            self.results[algorithm_name] = algorithm_results

        total_time = time.time() - start_time
        self.logger.info(f"Benchmark completed in {total_time:.2f} seconds")

        # Compute summary statistics
        self._compute_summary_statistics()

        # Generate reports and visualizations
        if self.config.save_plots:
            self._generate_plots()

        if self.config.save_data:
            self._save_results()

        return self._generate_benchmark_report()

    def _run_single_benchmark(
        self,
        trainer: AdvancedTrainer,
        env: Any,
        algorithm_name: str,
        run_id: int,
    ) -> BenchmarkResult:
        """Run a single benchmark for one algorithm."""
        start_time = time.time()

        # Get algorithm configuration
        config = self.config.algorithms[algorithm_name]

        # Train the agent
        training_results = trainer.train(
            algorithm_name,
            config,
            env,
            num_episodes=self.config.num_episodes,
            eval_frequency=self.config.eval_frequency,
            save_frequency=self.config.save_frequency,
        )

        training_time = time.time() - start_time

        # Extract metrics
        episode_rewards = [ep["episode_reward"] for ep in trainer.training_history]
        episode_lengths = [ep["episode_length"] for ep in trainer.training_history]
        training_metrics = training_results.get("training_metrics", [])
        evaluation_metrics = training_results.get("final_evaluation", {})

        # Estimate memory usage (simplified)
        memory_usage = self._estimate_memory_usage(trainer)

        # Detect convergence
        convergence_episode = self._detect_convergence(episode_rewards)

        return BenchmarkResult(
            algorithm_name=algorithm_name,
            run_id=run_id,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            training_time=training_time,
            memory_usage=memory_usage,
            convergence_episode=convergence_episode,
        )

    def _estimate_memory_usage(self, trainer: AdvancedTrainer) -> float:
        """Estimate memory usage of the trainer."""
        # Count parameters
        policy_params = sum(p.numel() for p in trainer.policy_net.parameters())
        value_params = sum(p.numel() for p in trainer.value_net.parameters())
        total_params = policy_params + value_params

        # Estimate memory in MB (assuming float32)
        return float(total_params * 4 / (1024 * 1024))

    def _detect_convergence(self, rewards: list[float], window_size: int = 100) -> int | None:
        """Detect convergence point in training."""
        if len(rewards) < window_size:
            return None

        # Use rolling average to detect convergence
        rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()

        # Find point where rolling average stabilizes
        threshold = 0.01  # 1% change threshold
        for i in range(window_size, len(rolling_avg)):
            if i + window_size >= len(rolling_avg):
                break

            current_avg = rolling_avg.iloc[i]
            future_avg = rolling_avg.iloc[i + window_size]

            if abs(future_avg - current_avg) / (abs(current_avg) + 1e-8) < threshold:
                return i

        return None

    def _compute_summary_statistics(self) -> None:
        """Compute summary statistics across all runs."""
        for algorithm_name, results in self.results.items():
            # Compute statistics for each run
            run_stats = [result.compute_statistics() for result in results]

            # Aggregate across runs
            summary: dict[str, float] = {}
            for metric in run_stats[0]:
                values = [stats[metric] for stats in run_stats]
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))

            # Add training time and memory usage
            training_times = [result.training_time for result in results]
            memory_usages = [result.memory_usage for result in results]

            summary["training_time_mean"] = float(np.mean(training_times))
            summary["training_time_std"] = float(np.std(training_times))
            summary["memory_usage_mean"] = float(np.mean(memory_usages))
            summary["memory_usage_std"] = float(np.std(memory_usages))

            self.summary_stats[algorithm_name] = summary

    def _generate_plots(self) -> None:
        """Generate comprehensive visualization plots."""
        self.logger.info("Generating benchmark plots")

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Policy Optimization Algorithm Benchmark", fontsize=16)

        # Plot 1: Learning curves
        self._plot_learning_curves(axes[0, 0])

        # Plot 2: Performance comparison
        self._plot_performance_comparison(axes[0, 1])

        # Plot 3: Training time comparison
        self._plot_training_time_comparison(axes[1, 0])

        # Plot 4: Convergence analysis
        self._plot_convergence_analysis(axes[1, 1])

        plt.tight_layout()

        # Save plot
        plot_path = self.save_dir / "benchmark_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Plots saved to {plot_path}")

    def _plot_learning_curves(self, ax: plt.Axes) -> None:
        """Plot learning curves for all algorithms."""
        ax.set_title("Learning Curves")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")

        for algorithm_name, results in self.results.items():
            # Plot mean and confidence interval
            all_rewards = [result.episode_rewards for result in results]
            max_length = max(len(rewards) for rewards in all_rewards)

            # Pad shorter sequences
            padded_rewards = []
            for rewards in all_rewards:
                if len(rewards) < max_length:
                    padded_rewards.append(rewards + [rewards[-1]] * (max_length - len(rewards)))
                else:
                    padded_rewards.append(rewards)

            rewards_array = np.array(padded_rewards)
            mean_rewards = np.mean(rewards_array, axis=0)
            std_rewards = np.std(rewards_array, axis=0)

            episodes = np.arange(len(mean_rewards))
            ax.plot(episodes, mean_rewards, label=algorithm_name, linewidth=2)
            ax.fill_between(
                episodes,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.3,
            )

        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_comparison(self, ax: plt.Axes) -> None:
        """Plot final performance comparison."""
        ax.set_title("Final Performance Comparison")
        ax.set_ylabel("Mean Reward")

        algorithms = list(self.summary_stats.keys())
        final_rewards = [self.summary_stats[alg]["final_avg_reward_mean"] for alg in algorithms]
        final_stds = [self.summary_stats[alg]["final_avg_reward_std"] for alg in algorithms]

        x_pos = np.arange(len(algorithms))
        bars = ax.bar(x_pos, final_rewards, yerr=final_stds, capsize=5, alpha=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, reward in zip(bars, final_rewards, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{reward:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_training_time_comparison(self, ax: plt.Axes) -> None:
        """Plot training time comparison."""
        ax.set_title("Training Time Comparison")
        ax.set_ylabel("Training Time (seconds)")

        algorithms = list(self.summary_stats.keys())
        training_times = [self.summary_stats[alg]["training_time_mean"] for alg in algorithms]
        time_stds = [self.summary_stats[alg]["training_time_std"] for alg in algorithms]

        x_pos = np.arange(len(algorithms))
        bars = ax.bar(x_pos, training_times, yerr=time_stds, capsize=5, alpha=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, time_val in zip(bars, training_times, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
            )

    def _plot_convergence_analysis(self, ax: plt.Axes) -> None:
        """Plot convergence analysis."""
        ax.set_title("Convergence Analysis")
        ax.set_ylabel("Convergence Episode")

        algorithms = list(self.summary_stats.keys())
        convergence_episodes = []

        for alg in algorithms:
            conv_episodes = [result.convergence_episode for result in self.results[alg]]
            conv_episodes = [ep for ep in conv_episodes if ep is not None]
            if conv_episodes:
                convergence_episodes.append(float(np.mean(conv_episodes)))
            else:
                convergence_episodes.append(float(self.config.num_episodes))

        x_pos = np.arange(len(algorithms))
        bars = ax.bar(x_pos, convergence_episodes, alpha=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, conv_ep in zip(bars, convergence_episodes, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{conv_ep:.0f}",
                ha="center",
                va="bottom",
            )

    def _save_results(self) -> None:
        """Save benchmark results to files."""
        # Save summary statistics
        summary_path = self.save_dir / "summary_statistics.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary_stats, f, indent=2, default=str)

        # Save detailed results
        detailed_results: dict[str, Any] = {}
        for algorithm_name, results in self.results.items():
            detailed_results[algorithm_name] = []
            for result in results:
                detailed_results[algorithm_name].append(
                    {
                        "run_id": result.run_id,
                        "episode_rewards": result.episode_rewards,
                        "episode_lengths": result.episode_lengths,
                        "training_time": result.training_time,
                        "memory_usage": result.memory_usage,
                        "convergence_episode": result.convergence_episode,
                        "evaluation_metrics": result.evaluation_metrics,
                    },
                )

        detailed_path = self.save_dir / "detailed_results.json"
        with open(detailed_path, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        # Save as CSV for easy analysis
        self._save_csv_results()

        self.logger.info(f"Results saved to {self.save_dir}")

    def _save_csv_results(self) -> None:
        """Save results in CSV format for easy analysis."""
        # Create DataFrame for summary statistics
        summary_data = []
        for algorithm_name, stats in self.summary_stats.items():
            row = {"algorithm": algorithm_name}
            row.update({k: str(v) for k, v in stats.items()})
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.save_dir / "summary_statistics.csv", index=False)

        # Create DataFrame for learning curves
        learning_curve_data = []
        for algorithm_name, results in self.results.items():
            for run_id, result in enumerate(results):
                for episode, reward in enumerate(result.episode_rewards):
                    learning_curve_data.append(
                        {
                            "algorithm": algorithm_name,
                            "run_id": run_id,
                            "episode": episode,
                            "reward": reward,
                        },
                    )

        learning_curve_df = pd.DataFrame(learning_curve_data)
        learning_curve_df.to_csv(self.save_dir / "learning_curves.csv", index=False)

    def _generate_benchmark_report(self) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "benchmark_config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "num_episodes": self.config.num_episodes,
                "num_runs": self.config.num_runs,
            },
            "summary_statistics": self.summary_stats,
            "algorithm_rankings": self._compute_algorithm_rankings(),
            "recommendations": self._generate_recommendations(),
        }

        # Save report
        report_path = self.save_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _compute_algorithm_rankings(self) -> dict[str, list[str]]:
        """Compute rankings for different metrics."""
        algorithms = list(self.summary_stats.keys())

        rankings = {}

        # Rank by final performance
        final_performance = [(alg, self.summary_stats[alg]["final_avg_reward_mean"]) for alg in algorithms]
        rankings["final_performance"] = [alg for alg, _ in sorted(final_performance, key=lambda x: x[1], reverse=True)]

        # Rank by training speed
        training_speed = [(alg, self.summary_stats[alg]["training_time_mean"]) for alg in algorithms]
        rankings["training_speed"] = [alg for alg, _ in sorted(training_speed, key=lambda x: x[1])]

        # Rank by convergence speed
        convergence_speed = [
            (
                alg,
                self.summary_stats[alg].get("convergence_episode_mean", self.config.num_episodes),
            )
            for alg in algorithms
        ]
        rankings["convergence_speed"] = [alg for alg, _ in sorted(convergence_speed, key=lambda x: x[1])]

        # Rank by stability (lowest std)
        stability = [(alg, self.summary_stats[alg]["final_avg_reward_std"]) for alg in algorithms]
        rankings["stability"] = [alg for alg, _ in sorted(stability, key=lambda x: x[1])]

        return rankings

    def _generate_recommendations(self) -> dict[str, str]:
        """Generate recommendations based on benchmark results."""
        rankings = self._compute_algorithm_rankings()

        recommendations = {
            "best_overall": rankings["final_performance"][0],
            "fastest_training": rankings["training_speed"][0],
            "most_stable": rankings["stability"][0],
            "fastest_convergence": rankings["convergence_speed"][0],
        }

        # Add specific recommendations
        best_performer = rankings["final_performance"][0]
        best_performance = self.summary_stats[best_performer]["final_avg_reward_mean"]

        if best_performance > 0.8:
            recommendations["use_case"] = "production"
        elif best_performance > 0.5:
            recommendations["use_case"] = "development"
        else:
            recommendations["use_case"] = "research"

        return recommendations

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("POLICY OPTIMIZATION BENCHMARK SUMMARY")
        print("=" * 60)

        print("\nConfiguration:")
        print(f"  State dimension: {self.config.state_dim}")
        print(f"  Action dimension: {self.config.action_dim}")
        print(f"  Episodes per run: {self.config.num_episodes}")
        print(f"  Number of runs: {self.config.num_runs}")

        print("\nAlgorithm Performance (Final Average Reward):")
        for algorithm_name, stats in self.summary_stats.items():
            reward = stats["final_avg_reward_mean"]
            std = stats["final_avg_reward_std"]
            time = stats["training_time_mean"]
            print(f"  {algorithm_name:25s}: {reward:.4f} ± {std:.4f} ({time:.1f}s)")

        rankings = self._compute_algorithm_rankings()
        print("\nRankings:")
        print(f"  Best Performance: {rankings['final_performance'][0]}")
        print(f"  Fastest Training: {rankings['training_speed'][0]}")
        print(f"  Most Stable: {rankings['stability'][0]}")
        print(f"  Fastest Convergence: {rankings['convergence_speed'][0]}")

        recommendations = self._generate_recommendations()
        print("\nRecommendations:")
        print(f"  Best Overall: {recommendations['best_overall']}")
        print(f"  Use Case: {recommendations['use_case']}")

        print(f"\nResults saved to: {self.save_dir}")
        print("=" * 60)


def create_simple_env(state_dim: int = 50, action_dim: int = 3) -> Any:
    """Create a simple environment for benchmarking."""

    class SimpleEnv:
        def __init__(self, state_dim: int, action_dim: int):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.step_count = 0
            self.max_steps = 1000

        def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
            self.step_count = 0
            return np.random.randn(self.state_dim), {}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
            self.step_count += 1

            # Simple reward function
            reward = np.random.normal(0, 1)
            if action == 0:  # Hold
                reward *= 0.1
            elif action == 1:  # Buy
                reward *= 1.0
            else:  # Sell
                reward *= 0.5

            next_state = np.random.randn(self.state_dim)
            done = self.step_count >= self.max_steps

            return next_state, reward, done, {}

        def render(self) -> None:
            pass

    return SimpleEnv(state_dim, action_dim)


def run_quick_benchmark(
    algorithms: list[str] | None = None,
    num_episodes: int = 100,
    num_runs: int = 3,
) -> dict[str, Any]:
    """Run a quick benchmark for testing purposes."""
    config = BenchmarkConfig(
        num_episodes=num_episodes,
        num_runs=num_runs,
        save_plots=True,
        save_data=True,
    )

    framework = BenchmarkFramework(config)

    def env_creator() -> Any:
        return create_simple_env(config.state_dim, config.action_dim)

    results = framework.run_benchmark(env_creator, algorithms)
    framework.print_summary()

    return results
