#!/usr/bin/env python3
"""
Multi-Agent Ensemble Trading Example

This example demonstrates how to use the new ensemble system for trading:
1. Creating a multi-agent ensemble with SAC, TD3, and PPO agents
2. Training the ensemble with dynamic weight updates
3. Evaluating ensemble performance and diversity
4. Comparing individual agents vs ensemble performance
5. Generating comprehensive diagnostics and reports

Usage:
    python examples/ensemble_trading_example.py
"""

import logging
import sys
from pathlib import Path
from typing import Callable

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl_agent.agents import (
    EnsembleConfig,
    EnsembleEvaluator,
    EnsembleTrainer,
)
from trading_rl_agent.envs.trading_env import TradingEnv


def create_trading_env() -> Callable[[], TradingEnv]:
    """Create a trading environment for the example."""
    # Simple trading environment configuration
    env_config = {
        "data_path": "data/sample_data.csv",  # You'll need to provide your own data
        "initial_balance": 10000,
        "transaction_fee": 0.001,
        "max_position": 1.0,
        "window_size": 50,
        "features": ["close", "volume", "returns", "volatility"],
    }

    return lambda: TradingEnv(**env_config)


def create_ensemble_config() -> EnsembleConfig:
    """Create ensemble configuration with multiple agents."""
    return EnsembleConfig(
        agents={
            "sac": {
                "enabled": True,
                "config": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "batch_size": 256,
                    "hidden_dims": [256, 256],
                    "automatic_entropy_tuning": True,
                    "target_entropy": -1.0,
                },
            },
            "td3": {
                "enabled": True,
                "config": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "batch_size": 256,
                    "hidden_dims": [256, 256],
                    "policy_delay": 2,
                    "target_noise": 0.2,
                    "noise_clip": 0.5,
                    "exploration_noise": 0.1,
                },
            },
            "ppo": {
                "enabled": True,
                "config": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_ratio": 0.2,
                    "batch_size": 256,
                    "minibatch_size": 64,
                    "n_epochs": 10,
                    "hidden_dims": [256, 256],
                    "activation": "tanh",
                    "vf_coef": 0.5,
                    "ent_coef": 0.01,
                    "target_kl": 0.01,
                },
            },
        },
        ensemble_method="weighted_voting",
        diversity_penalty=0.1,
        performance_window=100,
        min_weight=0.05,
        risk_adjustment=True,
    )


def main() -> None:
    """Main example function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Multi-Agent Ensemble Trading Example")

    try:
        # Create environment
        logger.info("Creating trading environment...")
        env_creator = create_trading_env()

        # Create ensemble configuration
        logger.info("Creating ensemble configuration...")
        config = create_ensemble_config()

        # Initialize ensemble trainer
        logger.info("Initializing ensemble trainer...")
        trainer = EnsembleTrainer(
            config=config, env_creator=env_creator, save_dir="outputs/ensemble_example", device="auto"
        )

        # Create agents
        logger.info("Creating ensemble agents...")
        trainer.create_agents()

        # Train ensemble
        logger.info("Starting ensemble training...")
        training_results = trainer.train_ensemble(
            total_iterations=500,  # Reduced for example
            eval_frequency=25,
            save_frequency=50,
            early_stopping_patience=25,
        )

        logger.info("Training completed!")
        logger.info(f"Best ensemble reward: {training_results['best_reward']:.3f}")
        logger.info(f"Total iterations: {training_results['total_iterations']}")

        # Load the best ensemble
        logger.info("Loading best ensemble...")
        trainer.load_ensemble("best")

        # Create evaluator
        logger.info("Creating ensemble evaluator...")
        evaluator = EnsembleEvaluator(trainer.ensemble)

        # Evaluate ensemble
        logger.info("Evaluating ensemble...")
        env = env_creator()
        evaluation_results = evaluator.evaluate_ensemble(
            env=env,
            num_episodes=50,  # Reduced for example
            include_diagnostics=True,
            save_results=True,
            results_path="outputs/ensemble_example/evaluation_results.json",
        )

        # Generate evaluation report
        logger.info("Generating evaluation report...")
        report = evaluator.generate_evaluation_report(evaluation_results)
        print("\n" + "=" * 60)
        print("ENSEMBLE EVALUATION REPORT")
        print("=" * 60)
        print(report)

        # Compare agents
        logger.info("Comparing individual agents vs ensemble...")
        comparison_results = evaluator.compare_agents(env, num_episodes=25)

        print("\n" + "=" * 60)
        print("AGENT COMPARISON RESULTS")
        print("=" * 60)
        for agent_name, results in comparison_results.items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            print(f"  Success Rate: {results['success_rate']:.1%}")
            print(f"  Mean Episode Length: {results['mean_length']:.1f}")

        # Get ensemble info
        logger.info("Getting ensemble information...")
        ensemble_info = trainer.get_ensemble_info()

        print("\n" + "=" * 60)
        print("ENSEMBLE INFORMATION")
        print("=" * 60)
        print(f"Number of agents: {ensemble_info['num_agents']}")
        print(f"Ensemble method: {ensemble_info['ensemble_method']}")
        print(f"Current weights: {ensemble_info['weights']}")

        # Demonstrate dynamic agent management
        logger.info("Demonstrating dynamic agent management...")

        # Add a new agent dynamically
        logger.info("Adding a new SAC agent dynamically...")
        new_sac_config = {
            "learning_rate": 1e-4,  # Different learning rate
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 128,
            "hidden_dims": [128, 128],  # Different architecture
        }

        success = trainer.add_agent_dynamically("sac_v2", "sac", new_sac_config)
        if success:
            logger.info("Successfully added new agent!")

            # Re-evaluate with new agent
            logger.info("Re-evaluating ensemble with new agent...")
            new_evaluation = evaluator.evaluate_ensemble(env, num_episodes=25)
            print(f"\nNew ensemble performance: {new_evaluation['performance']['mean_reward']:.3f}")

            # Remove the agent
            logger.info("Removing the dynamically added agent...")
            trainer.remove_agent_dynamically("sac_v2")
            logger.info("Agent removed successfully!")

        # Get evaluation summary
        logger.info("Getting evaluation summary...")
        summary = evaluator.get_evaluation_summary()

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Number of evaluations: {summary['num_evaluations']}")
        print(f"Performance trend: {summary['performance_trend']['reward_trend']}")
        print(f"Consensus trend: {summary['consensus_trend']['consensus_trend']}")
        print(f"Diversity trend: {summary['diversity_trend']['diversity_trend']}")
        print(f"Stability trend: {summary['stability_trend']['stability_trend']}")

        logger.info("Example completed successfully!")

    except Exception as e:
        logger.exception(f"Error in ensemble example: {e}")
        raise


def demonstrate_ensemble_methods() -> None:
    """Demonstrate different ensemble voting methods."""
    logger = logging.getLogger(__name__)
    logger.info("Demonstrating different ensemble voting methods...")

    # Create a simple environment for demonstration
    env_creator = create_trading_env()
    config = create_ensemble_config()

    # Test different ensemble methods
    ensemble_methods = ["weighted_voting", "consensus", "diversity_aware", "risk_adjusted"]

    for method in ensemble_methods:
        logger.info(f"Testing ensemble method: {method}")

        # Update config
        config.ensemble_method = method

        # Create trainer and agents
        trainer = EnsembleTrainer(config, env_creator, save_dir=f"outputs/ensemble_{method}")
        trainer.create_agents()

        # Quick evaluation
        evaluator = EnsembleEvaluator(trainer.ensemble)
        env = env_creator()
        results = evaluator.evaluate_ensemble(env, num_episodes=10)

        print(
            f"{method}: Mean Reward = {results['performance']['mean_reward']:.3f}, "
            f"Diversity = {results['diversity']['overall_diversity']:.3f}"
        )


if __name__ == "__main__":
    main()

    # Uncomment to demonstrate different ensemble methods
    # demonstrate_ensemble_methods()
