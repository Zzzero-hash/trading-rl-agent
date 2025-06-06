"""Example RLlib training script using the TradingEnv with model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

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
    args = parser.parse_args()

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
