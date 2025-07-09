"""Hyperparameter optimization for RL models.

This module provides utilities for hyperparameter tuning of RL models
using Ray Tune. It includes specialized sampling distributions and
configuration spaces for common RL algorithms.

Note: TD3 has been removed from Ray RLlib 2.38.0+. Use SAC for continuous control tasks.

Example usage:

>>> from trading_rl_agent.optimization.rl_optimization import optimize_sac_hyperparams
>>> results = optimize_sac_hyperparams(
...     env_config=env_config,
...     num_samples=20,
...     max_iterations_per_trial=100,
...     gpu_per_trial=0.5
... )
>>> print(results.best_config)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from ray import tune

# TD3 has been removed from Ray RLlib 2.38.0+, use SAC instead
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from trading_rl_agent.envs.finrl_trading_env import TradingEnv
from trading_rl_agent.models.concat_model import ConcatModel
from trading_rl_agent.utils.cluster import get_available_devices, init_ray

torch, _ = try_import_torch()
logger = logging.getLogger(__name__)


def create_env(config):
    """Create trading environment factory function."""
    return TradingEnv(config)


def _get_default_sac_search_space() -> dict[str, Any]:
    """Get default SAC hyperparameter search space for continuous control."""
    return {
        "twin_q": True,
        "q_model_config": {
            "fcnet_hiddens": tune.choice([[256, 256], [512, 512]]),
            "fcnet_activation": tune.choice(["relu", "tanh"]),
        },
        "policy_model_config": {
            "fcnet_hiddens": tune.choice([[256, 256], [512, 512]]),
            "fcnet_activation": tune.choice(["relu", "tanh"]),
        },
        # Learning rates
        "actor_lr": tune.loguniform(1e-5, 1e-3),
        "critic_lr": tune.loguniform(1e-5, 1e-3),
        "alpha_lr": tune.loguniform(1e-5, 1e-3),
        "tau": tune.loguniform(1e-4, 1e-2),
        # Entropy parameters
        "target_entropy": "auto",
        "initial_alpha": tune.loguniform(0.1, 1.0),
        # Training parameters
        "train_batch_size": tune.choice([64, 128, 256, 512]),
        "gamma": tune.uniform(0.95, 0.999),
        # Replay buffer
        "replay_buffer_config": {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": tune.choice([100000, 1000000]),
            "prioritized_replay_alpha": tune.uniform(0.4, 0.8),
            "prioritized_replay_beta": tune.uniform(0.4, 0.6),
        },
    }


def register_models_and_envs():
    """Register custom models and environments with Ray."""
    from ray.rllib.models import ModelCatalog

    # Register the custom model
    ModelCatalog.register_custom_model("concat_model", ConcatModel)

    # Register the environment
    register_env("TradingEnv", lambda cfg: create_env(cfg))


def optimize_sac_hyperparams(
    env_config: dict[str, Any],
    num_samples: int = 20,
    max_iterations_per_trial: int = 100,
    output_dir: str = "./rl_optimization",
    custom_search_space: dict[str, Any] | None = None,
    cpu_per_trial: float = 1.0,
    gpu_per_trial: float = 0.0,
    use_best_model: bool = True,
) -> tune.ExperimentAnalysis:
    """Optimize SAC hyperparameters using Ray Tune.

    Parameters
    ----------
    env_config : dict
        Environment configuration
    num_samples : int, default 20
        Number of trials to run
    max_iterations_per_trial : int, default 100
        Maximum training iterations per trial
    output_dir : str, default "./rl_optimization"
        Directory to save results
    custom_search_space : dict, optional
        Custom hyperparameter search space (overrides default)
    cpu_per_trial : float, default 1.0
        CPUs to allocate per trial
    gpu_per_trial : float, default 0.0
        GPUs to allocate per trial
    use_best_model : bool, default True
        Whether to return the best performing model

    Returns
    -------
    ExperimentAnalysis
        Ray Tune experiment analysis object
    """
    # Initialize Ray if not already done
    if not ray.is_initialized():
        init_ray()

    # Register models and environments
    register_models_and_envs()

    # Prepare search space
    search_space = _get_default_sac_search_space()
    if custom_search_space:
        search_space.update(custom_search_space)

    # Add environment configuration
    search_space["env"] = "TradingEnv"
    search_space["env_config"] = env_config
    search_space["framework"] = "torch"

    # Configure search algorithm
    search_alg = OptunaSearch(
        metric="episode_reward_mean",
        mode="max",
    )

    # Configure scheduler
    scheduler = ASHAScheduler(
        max_t=max_iterations_per_trial,
        grace_period=min(10, max(1, max_iterations_per_trial // 10)),
        reduction_factor=3,
        brackets=1,
    )

    # Setup output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Run optimization
    analysis = tune.run(
        "SAC",
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        stop={"training_iteration": max_iterations_per_trial},
        checkpoint_at_end=use_best_model,
        checkpoint_freq=max(1, max_iterations_per_trial // 5),
        keep_checkpoints_num=2,
        storage_path=output_dir,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        verbose=2,
        metric="episode_reward_mean",  # Add metric for Ray 2.0+
        mode="max",  # Add mode for Ray 2.0+
    )

    # Log best config
    best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
    logger.info(f"Best SAC config: {best_config}")

    return analysis
