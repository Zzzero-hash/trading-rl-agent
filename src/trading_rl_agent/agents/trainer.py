# Basic Trainer implementation
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
from ray import tune

from trading_rl_agent.core.exceptions import ConfigurationError
from trading_rl_agent.data.synthetic import fetch_synthetic_data
from trading_rl_agent.envs.finrl_trading_env import TradingEnv, register_env
from trading_rl_agent.risk.riskfolio import RiskfolioConfig, RiskfolioRiskManager

if TYPE_CHECKING:
    from trading_rl_agent.core.config import SystemConfig


class RiskAwareEnv(gym.Wrapper):
    """Environment wrapper applying risk management."""

    def __init__(
        self,
        env: gym.Env,
        risk_manager: RiskfolioRiskManager,
        terminate: bool = True,
    ) -> None:
        super().__init__(env)
        self.risk_manager = risk_manager
        self.terminate = terminate
        self._returns: np.ndarray = np.array([])

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._returns = np.array([])
        return cast(tuple[Any, dict[str, Any]], super().reset(**kwargs))

    def step(self, action: np.ndarray, **kwargs: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Handle array-like actions
        act_arr = np.asarray(action)
        scalar_action = float(act_arr.mean())
        valid = self.risk_manager.validate_action(scalar_action, self._returns)
        if not valid:
            scalar_action = self.risk_manager.risk_adjusted_action(
                scalar_action,
                self._returns,
            )
            act_arr = np.zeros_like(act_arr) + scalar_action
            if self.terminate:
                obs, reward, done, truncated, info = self.env.step(act_arr)
                self._returns = np.append(self._returns, float(reward))
                info = info or {}
                info["risk_violation"] = True
                return obs, reward, True, truncated, info
        obs, reward, done, truncated, info = self.env.step(act_arr)
        self._returns = np.append(self._returns, float(reward))
        metrics = self.risk_manager.calculate_risk(self._returns)
        info = info or {}
        info["risk_metrics"] = metrics
        if self.terminate and metrics["var"] > self.risk_manager.config.var_limit:
            info["risk_violation"] = True
            done = True
        return obs, reward, done, truncated, info


try:
    from ray.rllib.algorithms.ppo import PPO as PPOTrainer
except ImportError:  # Ray >=2.3 renames Trainer classes
    from ray.rllib.algorithms.ppo import PPO as PPOTrainer

try:
    from ray.rllib.algorithms.sac import SAC as SACTrainer
except ImportError:
    from ray.rllib.algorithms.sac import SAC as SACTrainer


class Trainer:
    def __init__(
        self,
        system_cfg: "SystemConfig",
        seed: int = 42,
        save_dir: str = "outputs",
    ):
        self.cfg = system_cfg
        self.seed = seed
        self.save_dir = str(Path(save_dir).resolve())

        self.ray_address = self.cfg.infrastructure.ray_address
        self.algorithm = self.cfg.agent.agent_type.lower()
        if self.algorithm not in {"ppo", "sac", "td3"}:
            raise ConfigurationError(
                f"Invalid algorithm specified: {self.cfg.agent.agent_type}. "
                "Supported algorithms are: ppo, sac, td3.",
            )
        self.num_iterations = self.cfg.agent.total_timesteps  # Simplified for now

        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(ignore_reinit_error=True)

        # Translate our ModelConfig to the format rllib expects
        model_config = {
            "conv_filters": self.cfg.model.cnn_filters,
            "conv_activation": "relu",
            "post_fcnet_hiddens": [self.cfg.model.lstm_units],
            "post_fcnet_activation": "relu",
            "vf_share_layers": True,
        }

        self.ray_config = {
            "env": "TraderEnv",
            "env_config": self.cfg.data.__dict__,
            "model": model_config,
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            # Disable the new API stack to avoid conflicts with legacy model configs
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False,
        }

        self.risk_enabled = self.cfg.risk.max_position_size < 1.0  # Example condition
        if self.risk_enabled:
            rc = RiskfolioConfig(
                max_position=self.cfg.risk.max_position_size,
                min_position=-self.cfg.risk.max_position_size,  # Assuming symmetrical
                var_limit=self.cfg.risk.max_drawdown,
            )
            self.risk_manager = RiskfolioRiskManager(rc)
            self.terminate_on_violation = True  # Or get from config
        else:
            self.risk_manager = None
            self.terminate_on_violation = False

        register_env()
        self._register_risk_env()

        os.makedirs(self.save_dir, exist_ok=True)
        print(
            f"Initialized Trainer with seed={self.seed}, save_dir='{self.save_dir}', ray_address={self.ray_address}",
        )

    def _register_risk_env(self) -> None:
        """Register trading environment with optional risk management."""

        from ray.tune.registry import register_env as ray_register_env

        def creator(cfg: dict[str, Any]) -> gym.Env:
            env = TradingEnv(cfg)
            if self.risk_enabled and self.risk_manager is not None:
                env = RiskAwareEnv(env, self.risk_manager, self.terminate_on_violation)
            return env

        ray_register_env("TraderEnv", creator)

    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        # For now, using synthetic data generation from train_sample.py
        # This should be replaced with the proper data pipeline later
        print("Loading and preparing data...")
        df = fetch_synthetic_data(n_samples=1200)  # Larger dataset for realistic training

        # The FeaturePipeline is not used for now, as generate_features does the job.
        # pipeline = FeaturePipeline()
        # df_processed = pipeline.transform(df)

        from trading_rl_agent.data.features import generate_features

        df_processed = generate_features(df)

        df_processed = df_processed.select_dtypes(include=["number"]).dropna()

        # Save the processed data to a file
        processed_data_path = Path(self.save_dir) / "processed_data.csv"
        df_processed.to_csv(processed_data_path, index=False)
        print(f"Processed data saved to {processed_data_path}")

        # Update env_config with processed data info
        self.ray_config["env_config"]["dataset_paths"] = [str(processed_data_path)]
        self.ray_config["env_config"]["df"] = None  # Ensure df is not passed directly
        self.ray_config["env_config"]["initial_capital"] = 100000
        self.ray_config["env_config"]["max_leverage"] = self.cfg.risk.max_leverage

        print("Data preparation complete.")
        return df_processed

    def train(self) -> None:
        print(
            f"[Trainer] Starting training with algorithm: {self.algorithm}",
        )
        self._load_and_prepare_data()

        if self.algorithm == "ppo":
            algo_cls = PPOTrainer
        else:
            algo_cls = SACTrainer

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

    def evaluate(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError
