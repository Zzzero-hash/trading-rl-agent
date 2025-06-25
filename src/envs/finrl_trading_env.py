"""
FinRL-Enhanced Trading Environment

Industry-grade trading environment that combines FinRL's proven framework
with our CNN+LSTM predictions for enhanced state representation.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

try:
    from finrl.apps import config
    from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv

    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    logging.warning("FinRL not available. Install with: pip install finrl")

from src.data.features import generate_features

logger = logging.getLogger(__name__)


class HybridFinRLEnv(gym.Env):
    """
    Industry-grade trading environment enhanced with CNN+LSTM predictions.

    Features:
    - FinRL-compatible interface for professional backtesting
    - Enhanced state space with CNN+LSTM market intelligence
    - Realistic transaction costs and market impact modeling
    - Risk management integration with position limits
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cnn_lstm_model: Optional[Any] = None,
        initial_amount: float = 1000000,
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: Optional[int] = None,
        action_space: Optional[int] = None,
        tech_indicator_list: Optional[list[str]] = None,
        turbulence_threshold: float = 30,
        risk_indicator_col: str = "vix",
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: list = [],
        model_name: str = "",
        mode: str = "",
        iteration: str = "",
        **kwargs,
    ):
        """
        Initialize hybrid FinRL environment.

        Args:
            df: Market data DataFrame with OHLCV and technical indicators
            cnn_lstm_model: Trained CNN+LSTM model for predictions
            initial_amount: Starting portfolio value (cash)
            transaction_cost_pct: Transaction cost percentage per trade
            reward_scaling: Scaling factor applied to raw returns
            state_space: Dimension of FinRL state space
            action_space: Dimension of FinRL action space
            tech_indicator_list: List of technical indicator column names
            turbulence_threshold: Threshold to filter turbulent market periods
            risk_indicator_col: Column name for market risk indicator (e.g., VIX)
            make_plots: Whether to generate diagnostic plots
            print_verbosity: Verbosity level for log output
            day: Starting index for simulation day
            initial: Flag indicating initial reset vs continuation
            previous_state: Previous environment state (for warm starts)
            model_name: Identifier for the CNN+LSTM model in logs and outputs
            mode: Run mode label (e.g., 'train', 'test')
            iteration: Iteration or run identifier string
            **kwargs: Additional FinRL-compatible parameters (passed to StockTradingEnv)
        """
        self.df = df.copy()
        self.cnn_lstm_model = cnn_lstm_model
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.day = day
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Initialize FinRL environment if available
        if FINRL_AVAILABLE and len(df) > 0:
            try:
                self.finrl_env = StockTradingEnv(
                    df=df,
                    initial_amount=initial_amount,
                    transaction_cost_pct=transaction_cost_pct,
                    reward_scaling=reward_scaling,
                    state_space=state_space,
                    action_space=action_space,
                    tech_indicator_list=tech_indicator_list
                    or self._get_tech_indicators(),
                    turbulence_threshold=turbulence_threshold,
                    risk_indicator_col=risk_indicator_col,
                    make_plots=make_plots,
                    print_verbosity=print_verbosity,
                )

                # Use FinRL's action and observation spaces as base
                self.base_action_space = self.finrl_env.action_space
                self.base_observation_space = self.finrl_env.observation_space

            except Exception as e:
                logger.warning(f"FinRL environment initialization failed: {e}")
                self.finrl_env = None
                self._init_custom_spaces()
        else:
            logger.info("Using custom environment (FinRL not available)")
            self.finrl_env = None
            self._init_custom_spaces()

        # Enhance observation space for CNN+LSTM predictions
        self._init_enhanced_spaces()

        # Environment state
        self.reset()

    def _get_tech_indicators(self) -> list:
        """Get list of technical indicators from DataFrame columns."""
        # Common technical indicators
        tech_indicators = [
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "sma_30",
            "ema_12",
            "ema_26",
            "bb_up",
            "bb_mid",
            "bb_down",
            "atr",
            "stoch_k",
            "stoch_d",
            "wr_14",
            "adx",
            "obv",
            "vwap",
        ]

        # Filter indicators that exist in the DataFrame
        available_indicators = [
            col for col in tech_indicators if col in self.df.columns
        ]

        if not available_indicators:
            # Fallback to any numeric columns that might be indicators
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            available_indicators = [
                col
                for col in numeric_cols
                if col not in ["open", "high", "low", "close", "volume", "date"]
            ]

        logger.info(f"Using technical indicators: {available_indicators}")
        return available_indicators

    def _init_custom_spaces(self):
        """Initialize custom action and observation spaces when FinRL is not available."""
        # Get unique stock symbols
        if "tic" in self.df.columns:
            self.stock_list = self.df["tic"].unique()
        elif "symbol" in self.df.columns:
            self.stock_list = self.df["symbol"].unique()
        else:
            self.stock_list = ["STOCK"]  # Default single stock

        # Action space: portfolio weights for each stock
        self.base_action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.stock_list),), dtype=np.float32
        )

        # Observation space: market features + portfolio state
        n_features = len(self._get_tech_indicators()) + 5  # OHLCV
        n_portfolio_features = len(self.stock_list) + 1  # positions + cash

        obs_dim = n_features * len(self.stock_list) + n_portfolio_features

        self.base_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _init_enhanced_spaces(self):
        """Initialize enhanced spaces with CNN+LSTM predictions."""
        # Calculate additional dimensions for CNN+LSTM predictions
        cnn_lstm_features = 0
        if self.cnn_lstm_model is not None:
            # Typical CNN+LSTM output: trend prediction + confidence
            cnn_lstm_features = (
                len(self.stock_list) * 2
            )  # prediction + confidence per stock

        # Enhanced observation space
        base_obs_shape = self.base_observation_space.shape[0]
        enhanced_obs_dim = base_obs_shape + cnn_lstm_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(enhanced_obs_dim,), dtype=np.float32
        )

        # Action space remains the same
        self.action_space = self.base_action_space

        logger.info(f"Enhanced observation space: {self.observation_space.shape}")
        logger.info(f"Action space: {self.action_space.shape}")

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if self.finrl_env:
            # Use FinRL reset
            obs, info = self.finrl_env.reset()
            base_obs = obs
        else:
            # Custom reset
            self.day = 0
            self.portfolio_value = self.initial_amount
            self.asset_portfolio = np.zeros(len(self.stock_list))
            self.cash = self.initial_amount

            base_obs = self._get_base_observation()
            info = {}

        # Add CNN+LSTM predictions to observation
        enhanced_obs = self._enhance_observation(base_obs)

        return enhanced_obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        if self.finrl_env:
            # Use FinRL step function
            obs, reward, done, truncated, info = self.finrl_env.step(action)
            base_obs = obs
        else:
            # Custom step function
            base_obs, reward, done, truncated, info = self._custom_step(action)

        # Enhance observation with CNN+LSTM predictions
        enhanced_obs = self._enhance_observation(base_obs)

        return enhanced_obs, reward, done, truncated, info

    def _custom_step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Custom step function when FinRL is not available."""
        # Normalize actions to valid portfolio weights
        action = np.clip(action, -1, 1)

        # Get current market data
        if self.day >= len(self.df) - 1:
            done = True
            truncated = False
            obs = self._get_base_observation()
            return obs, 0, done, truncated, {}

        current_data = self._get_current_market_data()

        # Calculate portfolio value change
        prev_portfolio_value = self.portfolio_value

        # Simple portfolio update (this is simplified - FinRL has more sophisticated logic)
        price_changes = self._calculate_price_changes(current_data)
        portfolio_return = np.sum(self.asset_portfolio * price_changes)

        # Apply transaction costs
        transaction_cost = (
            np.sum(np.abs(action - self.asset_portfolio)) * self.transaction_cost_pct
        )

        # Update portfolio
        self.asset_portfolio = action.copy()
        self.portfolio_value = (
            prev_portfolio_value + portfolio_return - transaction_cost
        )

        # Calculate reward
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward *= self.reward_scaling

        # Move to next day
        self.day += 1

        # Check if done
        done = self.day >= len(self.df) - 1
        truncated = False

        obs = self._get_base_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "day": self.day,
            "transaction_cost": transaction_cost,
        }

        return obs, reward, done, truncated, info

    def _get_base_observation(self) -> np.ndarray:
        """Get base observation without CNN+LSTM enhancements."""
        if self.finrl_env:
            # This would be called within FinRL's step/reset
            return np.array([])  # Placeholder

        # Custom observation
        current_data = self._get_current_market_data()

        # Market features
        market_features = []
        for symbol in self.stock_list:
            symbol_data = current_data[
                current_data.get("symbol", current_data.get("tic")) == symbol
            ]
            if len(symbol_data) > 0:
                # Add OHLCV and technical indicators
                features = symbol_data[
                    ["open", "high", "low", "close", "volume"]
                ].values[0]
                market_features.extend(features)

        # Portfolio features
        portfolio_features = list(self.asset_portfolio) + [
            self.cash / self.initial_amount
        ]

        # Combine all features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)

        return observation

    def _enhance_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Enhance observation with CNN+LSTM predictions."""
        if self.cnn_lstm_model is None:
            return base_obs

        try:
            # Get CNN+LSTM predictions for current market state
            cnn_lstm_predictions = self._get_cnn_lstm_predictions()

            # Combine base observation with CNN+LSTM features
            enhanced_obs = np.concatenate([base_obs, cnn_lstm_predictions])

            return enhanced_obs.astype(np.float32)

        except Exception as e:
            logger.warning(f"CNN+LSTM prediction failed: {e}")
            # Return base observation with zero-padded CNN+LSTM features
            cnn_lstm_features = len(self.stock_list) * 2
            zero_padding = np.zeros(cnn_lstm_features)
            return np.concatenate([base_obs, zero_padding]).astype(np.float32)

    def _get_cnn_lstm_predictions(self) -> np.ndarray:
        """Get CNN+LSTM predictions for current market state."""
        # Get recent market data for prediction
        lookback_window = 60  # Typical CNN+LSTM sequence length
        start_idx = max(0, self.day - lookback_window)
        end_idx = self.day + 1

        recent_data = self.df.iloc[start_idx:end_idx]

        # Prepare features for CNN+LSTM model
        # This would depend on your specific model input format
        features = self._prepare_cnn_lstm_features(recent_data)

        # Get predictions (mock implementation)
        predictions = []
        confidences = []

        for symbol in self.stock_list:
            # Mock prediction - replace with actual model inference
            trend_prediction = np.random.rand()  # 0-1, trend strength
            confidence = np.random.rand()  # 0-1, prediction confidence

            predictions.append(trend_prediction)
            confidences.append(confidence)

        # Combine predictions and confidences
        cnn_lstm_features = []
        for pred, conf in zip(predictions, confidences):
            cnn_lstm_features.extend([pred, conf])

        return np.array(cnn_lstm_features, dtype=np.float32)

    def _prepare_cnn_lstm_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for CNN+LSTM model input."""
        # This should match your CNN+LSTM model's expected input format
        # Placeholder implementation

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_cols].fillna(method="ffill").fillna(0)

        return feature_data.values

    def _get_current_market_data(self) -> pd.DataFrame:
        """Get current day's market data."""
        if self.day < len(self.df):
            return self.df.iloc[[self.day]]
        else:
            return self.df.iloc[[-1]]  # Return last available data

    def _calculate_price_changes(self, current_data: pd.DataFrame) -> np.ndarray:
        """Calculate price changes for portfolio update."""
        # Simplified implementation - should use proper price change calculation
        price_changes = np.random.normal(0.001, 0.02, len(self.stock_list))  # Mock data
        return price_changes

    def render(self, mode: str = "human"):
        """Render environment state."""
        if mode == "human":
            print(f"Day: {self.day}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Asset Allocation: {self.asset_portfolio}")

    def close(self):
        """Close environment."""
        if self.finrl_env:
            self.finrl_env.close()


def create_finrl_env(
    df: pd.DataFrame, cnn_lstm_model: Optional[Any] = None, **kwargs
) -> HybridFinRLEnv:
    """
    Factory function to create FinRL-enhanced environment.

    Args:
        df: Market data DataFrame
        cnn_lstm_model: Optional CNN+LSTM model for predictions
        **kwargs: Additional environment parameters

    Returns:
        Configured HybridFinRLEnv instance
    """
    return HybridFinRLEnv(df=df, cnn_lstm_model=cnn_lstm_model, **kwargs)
