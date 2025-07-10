from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO


class DummyEnv(gym.Env):
    """Minimal environment used to initialize SB3 agents."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class PPOAgent:
    """Wrapper around :class:`stable_baselines3.PPO`."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict | None = None,
        device: str = "cpu",
    ):
        config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        env = DummyEnv(state_dim, action_dim)
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device=device,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.model.predict(state, deterministic=evaluate)[0]

    def train(self, total_timesteps: int = 1) -> dict:
        self.model.learn(total_timesteps=total_timesteps)
        return {}

    update = train

    def save(self, filepath: str) -> None:
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        # Load the model onto the agent’s configured device
        self.model = PPO.load(filepath, device=self.device)

        # Validate loaded model matches agent configuration
        expected_obs_shape = (self.state_dim,)
        expected_action_shape = (self.action_dim,)
        if self.model.observation_space.shape != expected_obs_shape:
            raise ValueError(
                f"Loaded model observation space {self.model.observation_space.shape} "
                f"doesn't match agent state_dim {expected_obs_shape}",
            )
        if self.model.action_space.shape != expected_action_shape:
            raise ValueError(
                f"Loaded model action space {self.model.action_space.shape} "
                f"doesn't match agent action_dim {expected_action_shape}",
            )
