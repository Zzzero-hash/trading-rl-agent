"""Utilities for hyperparameter tuning with Ray Tune."""

from pathlib import Path
from typing import Any, cast

import ray
import yaml
from ray import tune

from trade_agent.envs.finrl_trading_env import register_env
from trade_agent.utils.cluster import validate_cluster_health
from trade_agent.utils.ray_utils import robust_ray_init


def _convert_value(value: Any) -> Any:
    """Convert YAML search spec into Ray Tune objects."""
    if isinstance(value, dict):
        if "grid_search" in value:
            return tune.grid_search(value["grid_search"])
        if "choice" in value:
            return tune.choice(value["choice"])
        if "uniform" in value and isinstance(value["uniform"], list | tuple):
            low, high = value["uniform"]
            return tune.uniform(low, high)
        if "randint" in value and isinstance(value["randint"], list | tuple):
            low, high = value["randint"]
            return tune.randint(low, high)
    return value


def _load_search_space(path: str) -> dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f) or {}

    def _recurse_convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _recurse_convert(_convert_value(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recurse_convert(v) for v in obj]
        return obj

    return cast("dict[str, Any]", _recurse_convert(cfg))


def run_tune(config_paths: str | list[str]) -> None:
    """Run Ray Tune with a search space defined in YAML files.

    Parameters
    ----------
    config_paths : list[str] or str
        One or more YAML files containing parameter search spaces.
    """
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    search_space = {}
    for path in config_paths:
        loaded_space = _load_search_space(path)
        if not isinstance(loaded_space, dict):
            raise ValueError(
                f"Expected a dict from _load_search_space, got {type(loaded_space).__name__}",
            )
        search_space.update(loaded_space)

    algorithm = search_space.pop("algorithm", "PPO")
    # env_cfg = search_space.pop("env_config", {})  # Retrieved but not used currently

    search_space.setdefault("env", "TraderEnv")

    # Initialize Ray with robust error handling
    if not ray.is_initialized():
        success, info = robust_ray_init(show_cluster_info=False)
        if not success:
            raise RuntimeError(f"Failed to initialize Ray for tuning: {info.get('error', 'Unknown error')}")

        # Validate cluster is suitable for hyperparameter tuning
        health = validate_cluster_health()
        if not health["healthy"]:
            print(f"⚠️  Starting tuning on degraded cluster: {health['reason']}")

    register_env()

    # analysis = tune.run(  # Results not used currently
    tune.run(
        algorithm,
        config=search_space,
        storage_path="tune",
        metric="episode_reward_mean",  # Add default metric for RL
        mode="max",  # Add default mode for RL
    )
    print("Tuning completed. Results are in 'tune' directory")
    ray.shutdown()
