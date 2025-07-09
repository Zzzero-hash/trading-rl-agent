# Basic Trainer implementation
import glob
import os

import ray
from ray import tune

from trading_rl_agent.envs.finrl_trading_env import TradingEnv, register_env
from risk import RiskfolioConfig, RiskfolioRiskManager
import gymnasium as gym
import numpy as np


class RiskAwareEnv(gym.Wrapper):
    """Environment wrapper applying risk management."""

    def __init__(self, env: gym.Env, risk_manager: RiskfolioRiskManager, terminate: bool = True) -> None:
        super().__init__(env)
        self.risk_manager = risk_manager
        self.terminate = terminate
        self._returns: list[float] = []

    def reset(self, **kwargs):
        self._returns = []
        return self.env.reset(**kwargs)

    def step(self, action):
        # Handle array-like actions
        act_arr = np.asarray(action)
        scalar_action = float(act_arr.mean())
        valid = self.risk_manager.validate_action(scalar_action, self._returns)
        if not valid:
            scalar_action = self.risk_manager.risk_adjusted_action(scalar_action, self._returns)
            act_arr = np.zeros_like(act_arr) + scalar_action
            if self.terminate:
                obs, reward, done, truncated, info = self.env.step(act_arr)
                self._returns.append(float(reward))
                info = info or {}
                info["risk_violation"] = True
                return obs, reward, True, truncated, info
        obs, reward, done, truncated, info = self.env.step(act_arr)
        self._returns.append(float(reward))
        metrics = self.risk_manager.calculate_risk(self._returns)
        info = info or {}
        info["risk_metrics"] = metrics
        if self.terminate and metrics["var"] > self.risk_manager.config.var_limit:
            info["risk_violation"] = True
            done = True
        return obs, reward, done, truncated, info

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

        risk_cfg = self.trainer_cfg.get("risk_management", {}) if isinstance(self.trainer_cfg, dict) else {}
        self.risk_enabled = bool(risk_cfg.get("enabled"))
        if self.risk_enabled:
            rc = RiskfolioConfig(
                max_position=risk_cfg.get("max_position", 1.0),
                min_position=risk_cfg.get("min_position", -1.0),
                var_limit=risk_cfg.get("var_limit", 0.02),
                var_alpha=risk_cfg.get("var_alpha", 0.05),
            )
            self.risk_manager = RiskfolioRiskManager(rc)
            self.terminate_on_violation = risk_cfg.get("terminate_on_violation", True)
        else:
            self.risk_manager = None
            self.terminate_on_violation = False
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init()
        register_env()
        self._register_risk_env()

        os.makedirs(self.save_dir, exist_ok=True)
        print(
            f"Initialized Trainer with seed={self.seed}, save_dir='{self.save_dir}', ray_address={self.ray_address}"
        )

    def _register_risk_env(self) -> None:
        """Register trading environment with optional risk management."""

        from ray.tune.registry import register_env as ray_register_env

        def creator(cfg):
            env = TradingEnv(cfg)
            if self.risk_enabled and self.risk_manager is not None:
                env = RiskAwareEnv(env, self.risk_manager, self.terminate_on_violation)
            return env

        ray_register_env("TraderEnv", creator)
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
