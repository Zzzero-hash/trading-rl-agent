"""
Reward Function Utilities

This module provides various reward functions for trading RL environments.
Includes risk-adjusted rewards, portfolio-based rewards, and custom reward shaping.
"""

from typing import Any, Callable, Dict, Optional, Union
import warnings

import numpy as np
import pandas as pd


def simple_profit_reward(prev_balance: float, current_balance: float) -> float:
    """
    Simple profit-based reward function.

    Args:
        prev_balance: Previous portfolio balance
        current_balance: Current portfolio balance

    Returns:
        Reward based on profit/loss
    """
    return (current_balance - prev_balance) / prev_balance if prev_balance > 0 else 0.0


def risk_adjusted_reward(
    returns: Union[float, np.ndarray],
    volatility: float = None,
    risk_penalty: float = 0.1,
) -> float:
    """
    Risk-adjusted reward that penalizes volatility.

    Args:
        returns: Portfolio returns
        volatility: Portfolio volatility (computed if None)
        risk_penalty: Risk penalty coefficient

    Returns:
        Risk-adjusted reward
    """
    if isinstance(returns, (list, np.ndarray)):
        return_value = np.mean(returns)
        if volatility is None:
            volatility = np.std(returns)
    else:
        return_value = returns
        if volatility is None:
            volatility = 0.0

    # Sharpe-like reward with risk penalty
    risk_adjusted = return_value - (risk_penalty * volatility)
    return float(risk_adjusted)


def drawdown_penalty_reward(
    current_return: float,
    current_drawdown: float,
    max_drawdown_threshold: float = 0.1,
    penalty_multiplier: float = 2.0,
) -> float:
    """
    Reward function with drawdown penalty.

    Args:
        current_return: Current period return
        current_drawdown: Current drawdown percentage
        max_drawdown_threshold: Maximum acceptable drawdown
        penalty_multiplier: Penalty multiplier for excessive drawdown

    Returns:
        Reward with drawdown penalty
    """
    base_reward = current_return

    if current_drawdown > max_drawdown_threshold:
        excess_drawdown = current_drawdown - max_drawdown_threshold
        penalty = excess_drawdown * penalty_multiplier
        base_reward -= penalty

    return float(base_reward)


def sharpe_based_reward(
    returns: np.ndarray, risk_free_rate: float = 0.0, window_size: int = 252
) -> float:
    """
    Sharpe ratio based reward function.

    Args:
        returns: Historical returns
        risk_free_rate: Risk-free rate
        window_size: Window for Sharpe calculation

    Returns:
        Sharpe ratio reward
    """
    if len(returns) < 2:
        return 0.0

    # Use last window_size returns
    recent_returns = returns[-window_size:] if len(returns) > window_size else returns

    excess_returns = recent_returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return float(sharpe)


def portfolio_diversification_reward(
    weights: np.ndarray,
    correlation_matrix: Optional[np.ndarray] = None,
    diversification_bonus: float = 0.1,
) -> float:
    """
    Reward for portfolio diversification.

    Args:
        weights: Portfolio weights
        correlation_matrix: Asset correlation matrix
        diversification_bonus: Bonus for diversified portfolios

    Returns:
        Diversification reward
    """
    # Concentration penalty (Herfindahl index)
    concentration = np.sum(weights**2)
    diversification_score = 1.0 - concentration

    # Additional correlation penalty if provided
    if correlation_matrix is not None:
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
        correlation_penalty = portfolio_variance * 0.1
        diversification_score -= correlation_penalty

    return float(diversification_score * diversification_bonus)


def transaction_cost_penalty(
    action: Union[int, np.ndarray],
    prev_position: Union[int, np.ndarray],
    cost_rate: float = 0.001,
) -> float:
    """
    Penalty for transaction costs.

    Args:
        action: Current action/position
        prev_position: Previous position
        cost_rate: Transaction cost rate

    Returns:
        Transaction cost penalty (negative reward)
    """
    if isinstance(action, (int, float)) and isinstance(prev_position, (int, float)):
        trade_size = abs(action - prev_position)
    else:
        trade_size = np.sum(np.abs(np.array(action) - np.array(prev_position)))

    cost = trade_size * cost_rate
    return -float(cost)


def momentum_reward(
    price_changes: np.ndarray, action: Union[int, float], momentum_window: int = 5
) -> float:
    """
    Reward for following momentum trends.

    Args:
        price_changes: Recent price changes
        action: Current action (1=long, 0=hold, -1=short)
        momentum_window: Window for momentum calculation

    Returns:
        Momentum alignment reward
    """
    if len(price_changes) < momentum_window:
        return 0.0

    recent_changes = price_changes[-momentum_window:]
    momentum = np.mean(recent_changes)

    # Reward alignment with momentum
    if momentum > 0 and action > 0:  # Upward momentum + long position
        return float(abs(momentum) * 0.1)
    elif momentum < 0 and action < 0:  # Downward momentum + short position
        return float(abs(momentum) * 0.1)
    elif momentum * action < 0:  # Counter-momentum penalty
        return float(-abs(momentum) * 0.05)

    return 0.0


def custom_trading_reward(
    portfolio_return: float,
    benchmark_return: float = 0.0,
    volatility: float = 0.0,
    drawdown: float = 0.0,
    transaction_costs: float = 0.0,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """
    Comprehensive custom trading reward function.

    Args:
        portfolio_return: Portfolio return for the period
        benchmark_return: Benchmark return for comparison
        volatility: Portfolio volatility
        drawdown: Current drawdown
        transaction_costs: Transaction costs incurred
        weights: Portfolio weights for diversification
        **kwargs: Additional parameters

    Returns:
        Comprehensive trading reward
    """
    # Base return vs benchmark
    excess_return = portfolio_return - benchmark_return

    # Risk adjustments
    risk_penalty = kwargs.get("risk_penalty", 0.1) * volatility
    drawdown_penalty = kwargs.get("drawdown_penalty", 1.0) * max(0, drawdown - 0.05)

    # Transaction cost penalty
    cost_penalty = transaction_costs

    # Diversification bonus
    diversification_bonus = 0.0
    if weights is not None:
        diversification_bonus = portfolio_diversification_reward(
            weights, diversification_bonus=kwargs.get("diversification_bonus", 0.02)
        )

    # Combine all components
    total_reward = (
        excess_return
        - risk_penalty
        - drawdown_penalty
        - cost_penalty
        + diversification_bonus
    )

    return float(total_reward)


class RewardFunction:
    """
    Configurable reward function class for trading environments.
    """

    def __init__(
        self,
        reward_type: str = "profit",
        risk_penalty: float = 0.1,
        transaction_cost: float = 0.001,
        drawdown_threshold: float = 0.1,
        **kwargs,
    ):
        """
        Initialize reward function.

        Args:
            reward_type: Type of reward ('profit', 'sharpe', 'risk_adjusted', 'custom')
            risk_penalty: Risk penalty coefficient
            transaction_cost: Transaction cost rate
            drawdown_threshold: Maximum acceptable drawdown
            **kwargs: Additional parameters
        """
        self.reward_type = reward_type
        self.risk_penalty = risk_penalty
        self.transaction_cost = transaction_cost
        self.drawdown_threshold = drawdown_threshold
        self.kwargs = kwargs

        # Initialize reward function
        self.reward_fn = self._get_reward_function()

    def _get_reward_function(self) -> Callable:
        """Get the appropriate reward function."""
        if self.reward_type == "profit":
            return simple_profit_reward
        elif self.reward_type == "sharpe":
            return lambda prev_bal, curr_bal, **kwargs: sharpe_based_reward(
                kwargs.get("returns", np.array([curr_bal / prev_bal - 1]))
            )
        elif self.reward_type == "risk_adjusted":
            return lambda prev_bal, curr_bal, **kwargs: risk_adjusted_reward(
                curr_bal / prev_bal - 1, kwargs.get("volatility"), self.risk_penalty
            )
        elif self.reward_type == "custom":
            return self._custom_reward
        else:
            warnings.warn(f"Unknown reward type {self.reward_type}, using profit")
            return simple_profit_reward

    def _custom_reward(
        self, prev_balance: float, current_balance: float, **kwargs
    ) -> float:
        """Custom reward implementation."""
        portfolio_return = (current_balance - prev_balance) / prev_balance

        return custom_trading_reward(
            portfolio_return=portfolio_return,
            benchmark_return=kwargs.get("benchmark_return", 0.0),
            volatility=kwargs.get("volatility", 0.0),
            drawdown=kwargs.get("drawdown", 0.0),
            transaction_costs=kwargs.get("transaction_costs", 0.0),
            weights=kwargs.get("weights"),
            risk_penalty=self.risk_penalty,
            drawdown_penalty=2.0,
            diversification_bonus=0.02,
        )

    def __call__(self, prev_balance: float, current_balance: float, **kwargs) -> float:
        """Calculate reward."""
        return self.reward_fn(prev_balance, current_balance, **kwargs)


# Preset reward functions
def get_conservative_reward_fn() -> RewardFunction:
    """Get conservative reward function with high risk penalty."""
    return RewardFunction(
        reward_type="risk_adjusted",
        risk_penalty=0.2,
        transaction_cost=0.002,
        drawdown_threshold=0.05,
    )


def get_aggressive_reward_fn() -> RewardFunction:
    """Get aggressive reward function with low risk penalty."""
    return RewardFunction(
        reward_type="profit",
        risk_penalty=0.05,
        transaction_cost=0.001,
        drawdown_threshold=0.15,
    )


def get_sharpe_optimized_reward_fn() -> RewardFunction:
    """Get Sharpe ratio optimized reward function."""
    return RewardFunction(
        reward_type="sharpe",
        risk_penalty=0.1,
        transaction_cost=0.001,
        drawdown_threshold=0.1,
    )


def compute_reward(reward_type: str, *args, **kwargs) -> float:
    """Dispatch to the appropriate reward function.

    Parameters
    ----------
    reward_type : str
        Identifier of the reward function to call. Supported types include
        ``"simple_profit"``, ``"risk_adjusted"``, ``"drawdown_penalty"``,
        ``"sharpe"``, ``"diversification"``, ``"transaction_cost"``,
        ``"momentum"`` and ``"custom"``.
    *args, **kwargs
        Arguments forwarded to the underlying reward function.

    Returns
    -------
    float
        Computed reward value.
    """

    dispatch_map: Dict[str, Callable[..., float]] = {
        "simple_profit": simple_profit_reward,
        "profit": simple_profit_reward,
        "risk_adjusted": risk_adjusted_reward,
        "drawdown_penalty": drawdown_penalty_reward,
        "sharpe": sharpe_based_reward,
        "diversification": portfolio_diversification_reward,
        "transaction_cost": transaction_cost_penalty,
        "momentum": momentum_reward,
        "custom": custom_trading_reward,
    }

    if reward_type not in dispatch_map:
        raise ValueError(f"Unknown reward type: {reward_type}")

    return dispatch_map[reward_type](*args, **kwargs)

