# Basic Trainer implementation
import os
import glob
import ray

from envs.trader_env import register_env
from ray.rllib.algorithms.ppo import PPOTrainer
from ray.rllib.algorithms.dqn import DQNTrainer

class Trainer:
    def __init__(self, env_cfg, model_cfg, trainer_cfg, seed=42, save_dir='outputs'):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.seed = seed
        self.save_dir = save_dir

        self.ray_address = self.trainer_cfg.get('ray_address') if isinstance(self.trainer_cfg, dict) else None
        self.algorithm = self.trainer_cfg.get('algorithm', 'ppo').lower() if isinstance(self.trainer_cfg, dict) else 'ppo'
        self.num_iterations = self.trainer_cfg.get('num_iterations', self.trainer_cfg.get('total_episodes', 10)) if isinstance(self.trainer_cfg, dict) else 10
        self.ray_config = self.trainer_cfg.get('ray_config', {}) if isinstance(self.trainer_cfg, dict) else {}
        self.ray_config.setdefault('env', 'TraderEnv')
        self.ray_config.setdefault('env_config', self.env_cfg)
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

        trainer_cls = PPOTrainer if self.algorithm == "ppo" else DQNTrainer
        trainer = trainer_cls(config=self.ray_config)

        for i in range(self.num_iterations):
            result = trainer.train()
            print(f"Iteration {i+1}/{self.num_iterations}: reward_mean={result.get('episode_reward_mean')}")
            checkpoint = trainer.save(self.save_dir)
            print(f"Saved checkpoint to {checkpoint}")

        ray.shutdown()

    def evaluate(self):
        print(f"[Trainer] Starting evaluation using models in '{self.save_dir}'")
        trainer_cls = PPOTrainer if self.algorithm == "ppo" else DQNTrainer
        trainer = trainer_cls(config=self.ray_config)

        checkpoints = sorted(
            glob.glob(os.path.join(self.save_dir, "checkpoint_*")),
            key=os.path.getmtime,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.save_dir}")
        latest = checkpoints[-1]
        trainer.restore(latest)
        results = trainer.evaluate()
        print(results)
        ray.shutdown()
