"""
CLI for Advanced Policy Optimization.

This module provides command-line interfaces for running advanced policy optimization
algorithms, benchmarking, and comparing different methods.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from .advanced_policy_optimization import (
    AdvancedPPOConfig,
    NaturalPolicyGradientConfig,
    TRPOConfig,
)
from .advanced_trainer import AdvancedTrainer, MultiObjectiveTrainer
from .benchmark_framework import BenchmarkConfig, BenchmarkFramework, run_quick_benchmark
from .configs import MultiObjectiveConfig

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_algorithm_config(
    algorithm: str,
    **kwargs: Any,
) -> AdvancedPPOConfig | TRPOConfig | NaturalPolicyGradientConfig:
    """Create algorithm configuration based on type."""
    if algorithm == "advanced_ppo":
        algorithm_config: AdvancedPPOConfig | TRPOConfig | NaturalPolicyGradientConfig = AdvancedPPOConfig()
    elif algorithm == "trpo":
        algorithm_config = TRPOConfig()
    elif algorithm == "natural_policy_gradient":
        algorithm_config = NaturalPolicyGradientConfig()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(algorithm_config, key):
            setattr(algorithm_config, key, value)

    return algorithm_config


def train_agent(args: argparse.Namespace) -> None:
    """Train an agent using advanced policy optimization."""
    logger.info(f"Training agent with {args.algorithm}")

    # Create configuration
    config = create_algorithm_config(
        args.algorithm,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    # Create trainer
    trainer = AdvancedTrainer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=args.device,
        save_dir=args.save_dir,
    )

    # Create simple environment for demonstration
    def env_creator() -> Any:
        from .benchmark_framework import create_simple_env

        return create_simple_env(args.state_dim, args.action_dim)

    env = env_creator()

    # Train the agent
    results = trainer.train(
        args.algorithm,
        config,
        env,
        num_episodes=args.num_episodes,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
    )

    logger.info("Training completed!")
    logger.info(f"Final average reward: {results['final_avg_reward']:.4f}")

    # Save results
    import json

    results_path = Path(args.save_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark comparison of different algorithms."""
    logger.info("Running algorithm benchmark")

    # Create benchmark configuration
    config = BenchmarkConfig(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_episodes=args.num_episodes,
        num_runs=args.num_runs,
        save_dir=args.save_dir,
        save_plots=args.save_plots,
        save_data=args.save_data,
    )

    # Setup algorithm configurations
    config.algorithms = {}

    if "advanced_ppo" in args.algorithms:
        config.algorithms["advanced_ppo"] = AdvancedPPOConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
        )

    if "trpo" in args.algorithms:
        config.algorithms["trpo"] = TRPOConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
        )

    if "natural_policy_gradient" in args.algorithms:
        config.algorithms["natural_policy_gradient"] = NaturalPolicyGradientConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
        )

    # Create benchmark framework
    framework = BenchmarkFramework(config)

    # Create environment
    def env_creator() -> Any:
        from .benchmark_framework import create_simple_env

        return create_simple_env(config.state_dim, config.action_dim)

    # Run benchmark
    results = framework.run_benchmark(env_creator, list(config.algorithms.keys()))

    # Print summary
    framework.print_summary()

    logger.info("Benchmark completed!")


def multi_objective_training(args: argparse.Namespace) -> None:
    """Train agent with multi-objective optimization."""
    logger.info("Training with multi-objective optimization")

    # Create multi-objective configuration
    multi_obj_config = MultiObjectiveConfig(
        return_weight=args.return_weight,
        risk_weight=args.risk_weight,
        sharpe_weight=args.sharpe_weight,
        max_drawdown_weight=args.max_drawdown_weight,
    )

    # Create algorithm configuration
    algorithm_config = create_algorithm_config(
        args.algorithm,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    # Create multi-objective trainer
    trainer = MultiObjectiveTrainer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        multi_obj_config=multi_obj_config,
        device=args.device,
        save_dir=args.save_dir,
    )

    # Create environment
    def env_creator() -> Any:
        from .benchmark_framework import create_simple_env

        return create_simple_env(args.state_dim, args.action_dim)

    env = env_creator()

    # Train with multi-objective optimization
    results = trainer.train(
        args.algorithm,
        algorithm_config,
        env,
        num_episodes=args.num_episodes,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
    )

    logger.info("Multi-objective training completed!")
    logger.info(f"Final average reward: {results['final_avg_reward']:.4f}")

    # Save results
    import json

    results_path = Path(args.save_dir) / "multi_objective_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_path}")


def quick_benchmark(args: argparse.Namespace) -> None:
    """Run a quick benchmark for testing."""
    logger.info("Running quick benchmark")

    algorithms = args.algorithms if args.algorithms else ["advanced_ppo", "trpo"]

    results = run_quick_benchmark(
        algorithms=algorithms,
        num_episodes=args.num_episodes,
        num_runs=args.num_runs,
    )

    logger.info("Quick benchmark completed!")
    logger.info(f"Results: {results}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Policy Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with Advanced PPO
  python -m trading_rl_agent.agents.cli_advanced_optimization train --algorithm advanced_ppo --num-episodes 1000

  # Run benchmark comparison
  python -m trading_rl_agent.agents.cli_advanced_optimization benchmark --algorithms advanced_ppo trpo --num-episodes 500

  # Multi-objective training
  python -m trading_rl_agent.agents.cli_advanced_optimization multi-objective --algorithm advanced_ppo --return-weight 0.8 --risk-weight 0.2

  # Quick benchmark
  python -m trading_rl_agent.agents.cli_advanced_optimization quick-benchmark --num-episodes 100
        """,
    )

    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--algorithm",
        "-a",
        required=True,
        choices=["advanced_ppo", "trpo", "natural_policy_gradient"],
        help="Algorithm to use",
    )
    train_parser.add_argument("--state-dim", type=int, default=50, help="State dimension")
    train_parser.add_argument("--action-dim", type=int, default=3, help="Action dimension")
    train_parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    train_parser.add_argument("--eval-frequency", type=int, default=100, help="Evaluation frequency")
    train_parser.add_argument("--save-frequency", type=int, default=500, help="Save frequency")
    train_parser.add_argument("--save-dir", default="outputs", help="Save directory")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run algorithm benchmark")
    benchmark_parser.add_argument(
        "--algorithms",
        "-a",
        nargs="+",
        choices=["advanced_ppo", "trpo", "natural_policy_gradient"],
        default=["advanced_ppo", "trpo"],
        help="Algorithms to benchmark",
    )
    benchmark_parser.add_argument("--state-dim", type=int, default=50, help="State dimension")
    benchmark_parser.add_argument("--action-dim", type=int, default=3, help="Action dimension")
    benchmark_parser.add_argument("--num-episodes", type=int, default=500, help="Episodes per run")
    benchmark_parser.add_argument("--num-runs", type=int, default=3, help="Number of runs")
    benchmark_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    benchmark_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    benchmark_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    benchmark_parser.add_argument("--save-dir", default="benchmark_results", help="Save directory")
    benchmark_parser.add_argument("--save-plots", action="store_true", help="Save plots")
    benchmark_parser.add_argument("--save-data", action="store_true", help="Save data")

    # Multi-objective training command
    multi_obj_parser = subparsers.add_parser("multi-objective", help="Multi-objective training")
    multi_obj_parser.add_argument(
        "--algorithm",
        "-a",
        required=True,
        choices=["advanced_ppo", "trpo", "natural_policy_gradient"],
        help="Algorithm to use",
    )
    multi_obj_parser.add_argument("--state-dim", type=int, default=50, help="State dimension")
    multi_obj_parser.add_argument("--action-dim", type=int, default=3, help="Action dimension")
    multi_obj_parser.add_argument("--num-episodes", type=int, default=1000, help="Number of episodes")
    multi_obj_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    multi_obj_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    multi_obj_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    multi_obj_parser.add_argument("--return-weight", type=float, default=0.9, help="Return weight")
    multi_obj_parser.add_argument("--risk-weight", type=float, default=0.1, help="Risk weight")
    multi_obj_parser.add_argument("--sharpe-weight", type=float, default=0.0, help="Sharpe ratio weight")
    multi_obj_parser.add_argument("--max-drawdown-weight", type=float, default=0.0, help="Max drawdown weight")
    multi_obj_parser.add_argument("--eval-frequency", type=int, default=100, help="Evaluation frequency")
    multi_obj_parser.add_argument("--save-frequency", type=int, default=500, help="Save frequency")
    multi_obj_parser.add_argument("--save-dir", default="outputs", help="Save directory")

    # Quick benchmark command
    quick_parser = subparsers.add_parser("quick-benchmark", help="Quick benchmark for testing")
    quick_parser.add_argument(
        "--algorithms",
        "-a",
        nargs="+",
        choices=["advanced_ppo", "trpo", "natural_policy_gradient"],
        help="Algorithms to benchmark",
    )
    quick_parser.add_argument("--num-episodes", type=int, default=100, help="Episodes per run")
    quick_parser.add_argument("--num-runs", type=int, default=2, help="Number of runs")

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    # Execute command
    if args.command == "train":
        train_agent(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "multi-objective":
        multi_objective_training(args)
    elif args.command == "quick-benchmark":
        quick_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
