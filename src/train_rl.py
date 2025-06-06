"""Example RLlib training script using the TradingEnv with model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.utils.cluster import init_ray, get_available_devices

from src.envs.trading_env import TradingEnv
from src.models.concat_model import ConcatModel


def create_env(cfg):
    return TradingEnv(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file with market data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to supervised model checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument(
        "--cluster-config",
        type=str,
        help="Path to ray cluster yaml config (optional)",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run Ray in local mode for debugging",
    )
    args = parser.parse_args()

    init_ray(config_path=args.cluster_config, local_mode=args.local_mode)
    resources = get_available_devices()
    if args.num_workers == 0:
        # use all CPUs minus one for the driver
        args.num_workers = max(1, int(resources["CPU"] - 1))
    if args.num_gpus == 0 and resources.get("GPU", 0) > 0:
        args.num_gpus = int(resources["GPU"])

    env_config = {
        "dataset_paths": [args.data],
        "window_size": 50,
        "model_path": args.model_path,
    }

    register_env("TradingEnvRL", lambda cfg: create_env(cfg))
    ModelCatalog.register_custom_model("concat_model", ConcatModel)

    config = (
        PPOConfig()
        .environment("TradingEnvRL", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_workers)
        .resources(num_gpus=args.num_gpus)
        .training(model={"custom_model": "concat_model"})
    )

    algo = config.build()
    for _ in range(5):
        result = algo.train()
        print("iteration", result["training_iteration"], "reward", result["episode_reward_mean"]) 

    checkpoint_dir = Path("./rl_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    algo.save(str(checkpoint_dir))


if __name__ == "__main__":
    main()
