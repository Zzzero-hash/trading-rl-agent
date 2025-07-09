"""Example RLlib training script using the TradingEnv with model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from trading_rl_agent.core.config import ConfigManager
from trading_rl_agent.envs.finrl_trading_env import TradingEnv, register_env
from trading_rl_agent.models.concat_model import ConcatModel
from trading_rl_agent.utils.cluster import get_available_devices, init_ray


def create_env(cfg):
    return TradingEnv(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to system config YAML",
    )
    parser.add_argument("--data", type=str, help="CSV file with market data")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to supervised model checkpoint",
    )
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

    if args.config:
        cfg_mgr = ConfigManager(args.config)
        cfg = cfg_mgr.load_config()
        data_path = args.data or cfg.data.data_path
        model_path = args.model_path or cfg.model.model_save_path
        if args.num_workers == 0:
            args.num_workers = cfg.infrastructure.num_workers
        if args.num_gpus == 0:
            args.num_gpus = (
                resources.get("GPU", 0) if cfg.infrastructure.gpu_enabled else 0
            )
        window_size = cfg.data.feature_window
    else:
        if args.num_workers == 0:
            # use all CPUs minus one for the driver
            args.num_workers = max(1, int(resources["CPU"] - 1))
        if args.num_gpus == 0 and resources.get("GPU", 0) > 0:
            args.num_gpus = int(resources["GPU"])
        if not args.data:
            raise ValueError(
                "Argument '--data' is required when '--config' is not provided."
            )
        if not args.model_path:
            raise ValueError(
                "Argument '--model-path' is required when '--config' is not provided."
            )
        data_path = args.data
        model_path = args.model_path
        window_size = 50

    env_config = {
        "dataset_paths": [data_path],
        "window_size": window_size,
        "model_path": model_path,
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
        print(
            "iteration",
            result["training_iteration"],
            "reward",
            result["episode_reward_mean"],
        )

    checkpoint_dir = Path("./rl_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    algo.save(str(checkpoint_dir))


if __name__ == "__main__":
    main()
