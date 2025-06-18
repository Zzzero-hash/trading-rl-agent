"""Utilities for hyperparameter tuning with Ray Tune."""

import yaml
import ray
from ray import tune

from src.envs.trading_env import register_env


def _convert_value(value):
    """Convert YAML search spec into Ray Tune objects."""
    if isinstance(value, dict):
        if "grid_search" in value:
            return tune.grid_search(value["grid_search"])
        if "choice" in value:
            return tune.choice(value["choice"])
        if "uniform" in value and isinstance(value["uniform"], (list, tuple)):
            low, high = value["uniform"]
            return tune.uniform(low, high)
        if "randint" in value and isinstance(value["randint"], (list, tuple)):
            low, high = value["randint"]
            return tune.randint(low, high)
    return value


def _load_search_space(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    def _recurse_convert(obj):
        if isinstance(obj, dict):
            return {k: _recurse_convert(_convert_value(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recurse_convert(v) for v in obj]
        return obj

    return _recurse_convert(cfg)


def run_tune(config_paths):
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
            raise ValueError(f"Expected a dict from _load_search_space, got {type(loaded_space).__name__}")
        search_space.update(loaded_space)

    algorithm = search_space.pop("algorithm", "PPO")
    env_cfg = search_space.pop("env_config", {})
    
    search_space.setdefault("env", "TraderEnv")

    if not ray.is_initialized():
        ray.init()
    register_env()

    analysis = tune.run(
        algorithm, 
        config=search_space, 
        storage_path="tune",
        metric="episode_reward_mean",  # Add default metric for RL
        mode="max"                     # Add default mode for RL
    )
    print("Tuning completed. Results are in 'tune' directory")
    ray.shutdown()