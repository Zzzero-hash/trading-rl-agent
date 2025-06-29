"""Custom trading environment using Gym interface compatible with RLlib."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from src.data.features import generate_features
from src.supervised_model import load_model, predict_features


class TradingEnv(gym.Env):
    """A simple trading environment with configurable parameters."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_cfg: dict | None = None, **kwargs):
        """Initialize the environment.

        Parameters can be provided either as a configuration dictionary
        ``env_cfg`` or directly as keyword arguments. Keyword arguments will
        override values in ``env_cfg`` when both are supplied. This preserves
        backward compatibility with existing code that passes a single
        configuration dictionary while allowing a more pythonic style of
        initialization.
        """

        if env_cfg is not None and not isinstance(env_cfg, dict):
            raise TypeError("env_cfg must be a dict if provided")

        cfg = {**(env_cfg or {}), **kwargs}
        self.config = cfg  # Store config for potential cleanup access
        self.data_paths = cfg.get("dataset_paths", [])
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.window_size = int(cfg.get("window_size", 50))
        self.initial_balance = float(cfg.get("initial_balance", 1_0000))
        self.transaction_cost = float(cfg.get("transaction_cost", 0.001))
        self.include_features = bool(cfg.get("include_features", False))
        self.model_path = cfg.get("model_path")
        self.model = None
        if self.model_path:
            self.model = load_model(self.model_path)
            self.model_output_size = self.model.config.output_size
        else:
            self.model_output_size = 0

        # New: allow continuous action space for TD3
        self.continuous_actions = bool(cfg.get("continuous_actions", False))

        self.data = self._load_data()
        if len(self.data) <= self.window_size:
            raise ValueError("Not enough data for the specified window_size")

        # Dynamically determine feature count for observation/action spaces
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        obs_shape = (self.window_size, len(self.numeric_cols))

        if self.continuous_actions:
            # For TD3: continuous action in range [-1, 1], shape (1,)
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Discrete(3)  # hold/buy/sell

        base_box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        if self.model:
            pred_box = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.model_output_size,),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(
                {"market_features": base_box, "model_pred": pred_box}
            )
        else:
            self.observation_space = base_box

        # Initialize step-dependent attributes before using them
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.position = 0

        # Log and assert observation shape for debugging
        obs = self._get_observation()
        if isinstance(obs, dict):
            obs_val = obs.get("market_features", obs)
        else:
            obs_val = obs
        obs_arr = np.asarray(obs_val)
        print(f"[TradingEnv] Initial observation shape: {obs_arr.shape}")
        if self.continuous_actions:
            assert (
                obs_arr.ndim == 1
            ), f"Continuous action env must return flat obs, got shape {obs_arr.shape}"
            # TD3 expects obs shape to match state_dim; help debug mismatches
            expected_state_dim = cfg.get("state_dim")
            if (
                expected_state_dim is not None
                and obs_arr.shape[0] != expected_state_dim
            ):
                raise ValueError(
                    f"[TradingEnv] For TD3, observation shape {obs_arr.shape} does not match expected state_dim={expected_state_dim}. "
                    f"Set window_size * num_features = {expected_state_dim} in your config."
                )
        self.reset()

    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        frames = []
        for path in self.data_paths:
            df = pd.read_csv(path)
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        if self.include_features:
            data = generate_features(data)
        # Explicitly cast price/volume columns to float32
        price_cols = ["open", "high", "low", "close", "volume"]
        for col in price_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)
        # Optionally cast other known numeric feature columns if present
        numeric_feature_cols = [
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "rsi",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "bb_position",
            "volume_sma",
            "volume_ratio",
            "price_change",
            "price_change_5",
            "volatility",
            "news_sentiment",
            "social_sentiment",
            "composite_sentiment",
            "sentiment_volume",
            "label",
        ]
        for col in numeric_feature_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)
        return data

    def _get_observation(self):
        # Only use numeric columns for the observation
        obs = self.data.iloc[self.current_step - self.window_size : self.current_step][
            self.numeric_cols
        ].values.astype(np.float32)
        if self.model:
            pred = predict_features(self.model, obs).numpy().astype(np.float32)
            # Ensure prediction has the expected shape for the observation space
            if pred.ndim == 0:  # scalar
                pred = pred.reshape(1)
            elif pred.ndim > 1:  # multi-dimensional
                pred = pred.flatten()
            # Ensure it matches the expected model output size
            if len(pred) != self.model_output_size:
                # Pad or truncate to match expected size
                if len(pred) < self.model_output_size:
                    pred = np.pad(pred, (0, self.model_output_size - len(pred)))
                else:
                    pred = pred[: self.model_output_size]
            obs_dict = {"market_features": obs, "model_pred": pred}
            if self.continuous_actions:
                # For TD3, flatten the market_features for agent compatibility
                obs_dict["market_features"] = obs.flatten()
            return obs_dict
        if self.continuous_actions:
            return obs.flatten()
        return obs

    # Gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.position = 0
        return self._get_observation(), {}

    def step(self, action):
        # Accept both discrete and continuous actions
        if self.continuous_actions:
            # For continuous, clip and discretize for internal logic if needed
            action = np.clip(action, -1.0, 1.0)
            # Example: map [-1, 1] to sell/hold/buy for reward logic
            if action <= -0.33:
                action_idx = 2  # sell
            elif action >= 0.33:
                action_idx = 1  # buy
            else:
                action_idx = 0  # hold
        else:
            assert self.action_space.contains(action), "Invalid action"
            action_idx = action

        # Check bounds before accessing data
        if self.current_step - 1 >= len(self.data):
            # Environment is already done, return terminal state
            if isinstance(self.observation_space, gym.spaces.Dict):
                obs = {
                    "market_features": np.zeros(
                        (self.window_size, len(self.numeric_cols)), dtype=np.float32
                    ),
                    "model_pred": np.zeros(self.model_output_size, dtype=np.float32),
                }
            else:
                obs = np.zeros_like(self.observation_space.sample())
            return obs, 0.0, True, False, {"balance": self.balance}

        prev_price = self.data.loc[self.current_step - 1, "close"]
        try:
            prev_price = float(np.asarray(prev_price).astype(np.float32))
        except Exception:
            prev_price = np.nan

        new_position = {0: self.position, 1: 1, 2: -1}[action_idx]
        cost = 0.0
        if new_position != self.position:
            cost = self.transaction_cost * abs(new_position - self.position)
        self.position = new_position

        self.current_step += 1

        # Check for episode termination BEFORE accessing data
        done = self.current_step >= len(self.data)

        if done:
            # Episode is done, use last available price or set a neutral reward
            current_price = prev_price  # No price change if episode is done
            reward = -cost  # Only transaction cost applies
        else:
            # Normal step, get current price
            current_price = self.data.loc[self.current_step - 1, "close"]
            try:
                current_price = float(np.asarray(current_price).astype(np.float32))
            except Exception:
                current_price = np.nan
            price_diff = current_price - prev_price
            reward = float(self.position * price_diff - cost)

        self.balance += reward

        if done:
            if isinstance(self.observation_space, gym.spaces.Dict):
                obs = {
                    "market_features": np.zeros(
                        (self.window_size, len(self.numeric_cols)), dtype=np.float32
                    ),
                    "model_pred": np.zeros(self.model_output_size, dtype=np.float32),
                }
            else:
                obs = np.zeros_like(self.observation_space.sample())
        else:
            obs = self._get_observation()
        info = {"balance": self.balance}
        return obs, reward, done, False, info

    def render(self):
        print(
            f"Step: {self.current_step}, Price: {self.data.loc[self.current_step - 1, 'close']}, "
            f"Position: {self.position}, Balance: {self.balance}"
        )


# Registration helpers ---------------------------------------------------------


def env_creator(env_cfg):
    return TradingEnv(env_cfg)


def register_env(name: str = "TradingEnv"):
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env(name, lambda cfg: TradingEnv(cfg))
