# Basic Trainer implementation
import glob
import os

import ray
from ray import tune

from trading_rl_agent.envs.finrl_trading_env import register_env

try:
    from ray.rllib.algorithms.ppo import PPOTrainer
except ImportError:  # Ray >=2.3 renames Trainer classes
    from ray.rllib.algorithms.ppo import PPO as PPOTrainer

try:
    from ray.rllib.algorithms.dqn import DQNTrainer
except ImportError:
    from ray.rllib.algorithms.dqn import DQN as DQNTrainer


class Trainer:
    def __init__(self, env_cfg, model_cfg, trainer_cfg, seed=42, save_dir="outputs"):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.seed = seed
        self.save_dir = save_dir

        self.ray_address = (
            self.trainer_cfg.get("ray_address")
            if isinstance(self.trainer_cfg, dict)
            else None
        )
        self.algorithm = (
            self.trainer_cfg.get("algorithm", "ppo").lower()
            if isinstance(self.trainer_cfg, dict)
            else "ppo"
        )
        self.num_iterations = (
            self.trainer_cfg.get(
                "num_iterations", self.trainer_cfg.get("total_episodes", 10)
            )
            if isinstance(self.trainer_cfg, dict)
            else 10
        )
        self.ray_config = (
            self.trainer_cfg.get("ray_config", {})
            if isinstance(self.trainer_cfg, dict)
            else {}
        )
        self.ray_config.setdefault("env", "TraderEnv")
        self.ray_config.setdefault("env_config", self.env_cfg)
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init()
        register_env()

        os.makedirs(self.save_dir, exist_ok=True)
        print(
            f"Initialized Trainer with seed={self.seed}, save_dir='{self.save_dir}', ray_address={self.ray_address}"
        )

    def train(self):
        print(
            f"[Trainer] Starting training with configs:\n  env={self.env_cfg}\n  model={self.model_cfg}\n  trainer={self.trainer_cfg}"
        )

        algo_cls = PPOTrainer if self.algorithm == "ppo" else DQNTrainer
        # Use RunConfig from ray.air.config for Ray >=2.0
        from ray.air.config import RunConfig

        run_config = RunConfig(
            stop={"training_iteration": self.num_iterations},
            storage_path=f"file://{self.save_dir}",
            verbose=1,
        )
        tuner = tune.Tuner(algo_cls, param_space=self.ray_config, run_config=run_config)
        tuner.fit()

        ray.shutdown()

    def evaluate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
