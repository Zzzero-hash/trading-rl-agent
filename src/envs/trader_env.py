import gymnasium as gym
import numpy as np
import pandas as pd


class TraderEnv(gym.Env):
    """Simple trading environment for demonstration purposes."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, data_paths, initial_balance=10000, window_size=50, transaction_cost=0.001
    ):
        super().__init__()
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_cost = transaction_cost

        self.data = self._load_data()
        if len(self.data) <= self.window_size:
            raise ValueError("Not enough data for the specified window_size")

        self.action_space = gym.spaces.Discrete(3)  # 0 hold, 1 buy, 2 sell
        obs_shape = (self.window_size, self.data.shape[1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        dfs = []
        for path in self.data_paths:
            df = pd.read_csv(path)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        # ensure float32 for observations
        return data.astype(np.float32)

    def _get_observation(self):
        obs = self.data.iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        return obs.astype(np.float32)

    # Gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.position = 0  # -1 short, 0 flat, 1 long
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        prev_price = self.data.loc[self.current_step - 1, "close"]

        # Update position with transaction cost when changing
        new_position = {0: self.position, 1: 1, 2: -1}[action]
        cost = 0.0
        if new_position != self.position:
            cost = self.transaction_cost * abs(new_position - self.position)
        self.position = new_position

        self.current_step += 1
        done = self.current_step >= len(self.data)
        current_price = self.data.loc[self.current_step - 1, "close"]
        price_diff = current_price - prev_price
        reward = float(self.position * price_diff - cost)
        self.balance += reward
        obs = (
            self._get_observation()
            if not done
            else np.zeros_like(self.observation_space.sample())
        )
        info = {"balance": self.balance}
        return obs, reward, done, False, info

    def render(self):
        print(
            f"Step: {self.current_step}, Price: {self.data.loc[self.current_step - 1, 'close']}, "
            f"Position: {self.position}, Balance: {self.balance}"
        )


# Registration -------------------------------------------------------------


def env_creator(env_cfg):
    data_paths = env_cfg.get("dataset_paths", [])
    return TraderEnv(
        data_paths,
        initial_balance=env_cfg.get("initial_balance", 10000),
        window_size=env_cfg.get("window_size", 50),
        transaction_cost=env_cfg.get("transaction_cost", 0.001),
    )


def register_env():
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env("TraderEnv", env_creator)
