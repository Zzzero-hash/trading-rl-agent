"""
Agent Scenario Evaluator for Comprehensive Agent Assessment

This module provides a comprehensive evaluation framework using synthetic data
scenarios to test agent performance across different market regimes, evaluate
robustness to market shocks, and measure adaptation to changing conditions.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.trade_agent.data.synthetic import generate_gbm_prices

from .metrics_calculator import MetricsCalculator
from .model_evaluator import ModelEvaluator

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class MarketScenario:
    """Definition of a specific market scenario for agent evaluation."""

    name: str
    description: str
    duration_days: int
    market_regime: str  # trend_following, mean_reversion, volatility_breakout, crisis, calm

    # Market parameters
    base_volatility: float
    drift: float
    regime_changes: list[dict[str, Any]] = field(default_factory=list)

    # Success criteria
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.15
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.2

    # Scenario-specific metrics
    custom_metrics: dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scenario parameters."""
        if self.duration_days <= 0:
            raise ValueError("Duration must be positive")
        if self.base_volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            raise ValueError("Max drawdown must be between 0 and 1")


@dataclass
class ScenarioResult:
    """Results from a single scenario evaluation."""

    scenario: MarketScenario
    agent_name: str

    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    volatility: float

    # Scenario-specific metrics
    custom_metrics: dict[str, float] = field(default_factory=dict)

    # Success indicators
    passed_criteria: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    # Detailed data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns_series: pd.Series = field(default_factory=pd.Series)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def evaluate_success(self) -> bool:
        """Evaluate if the agent passed the scenario criteria."""
        failures = []

        if self.sharpe_ratio < self.scenario.min_sharpe_ratio:
            failures.append(f"Sharpe ratio {self.sharpe_ratio:.3f} < {self.scenario.min_sharpe_ratio}")

        if self.max_drawdown > self.scenario.max_drawdown:
            failures.append(f"Max drawdown {self.max_drawdown:.3f} > {self.scenario.max_drawdown}")

        if self.win_rate < self.scenario.min_win_rate:
            failures.append(f"Win rate {self.win_rate:.3f} < {self.scenario.min_win_rate}")

        if self.profit_factor < self.scenario.min_profit_factor:
            failures.append(f"Profit factor {self.profit_factor:.3f} < {self.scenario.min_profit_factor}")

        self.failure_reasons = failures
        self.passed_criteria = len(failures) == 0
        return self.passed_criteria


class MarketScenarioGenerator:
    """Generates synthetic market data for different scenarios."""

    def __init__(self, seed: int | None = None):
        """Initialize the scenario generator."""
        if seed is not None:
            np.random.seed(seed)
        self.metrics_calculator = MetricsCalculator()

    def generate_trend_following_scenario(
        self,
        duration_days: int = 252,
        trend_strength: float = 0.001,
        volatility: float = 0.015,
        trend_changes: int = 2,
    ) -> pd.DataFrame:
        """Generate trend-following market scenario."""

        # Create trend segments
        segment_length = duration_days // (trend_changes + 1)
        prices = []
        current_price = 100.0

        for segment in range(trend_changes + 1):
            # Random trend direction for each segment
            trend_direction = np.random.choice([-1, 1])
            segment_trend = trend_direction * trend_strength

            # Generate prices for this segment
            segment_prices = self._generate_trend_segment(
                length=segment_length,
                start_price=current_price,
                trend=segment_trend,
                volatility=volatility,
            )
            prices.extend(segment_prices)
            current_price = segment_prices[-1]

        # Ensure exact length
        prices = prices[:duration_days]

        return self._create_ohlcv_data(prices, volatility)

    def generate_mean_reversion_scenario(
        self,
        duration_days: int = 252,
        mean_price: float = 100.0,
        reversion_strength: float = 0.1,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Generate mean-reverting market scenario."""

        prices = [mean_price]

        for i in range(1, duration_days):
            current_price = prices[-1]

            # Mean reversion component
            reversion_component = reversion_strength * (mean_price - current_price)

            # Random walk component
            random_component = volatility * np.random.normal(0, 1)

            # Update price
            new_price = current_price + reversion_component + random_component
            prices.append(max(0.1, new_price))  # Ensure positive prices

        return self._create_ohlcv_data(prices, volatility)

    def generate_volatility_breakout_scenario(
        self,
        duration_days: int = 252,
        base_volatility: float = 0.01,
        breakout_volatility: float = 0.05,
        breakout_probability: float = 0.1,
    ) -> pd.DataFrame:
        """Generate volatility breakout scenario."""

        prices = [100.0]
        volatilities = [base_volatility]

        for i in range(1, duration_days):
            current_price = prices[-1]
            current_vol = volatilities[-1]

            # Determine if breakout occurs
            if np.random.random() < breakout_probability:
                current_vol = breakout_volatility
            else:
                # Gradual return to base volatility
                current_vol = current_vol * 0.95 + base_volatility * 0.05

            # Generate price movement
            price_change = current_vol * np.random.normal(0, 1)
            new_price = current_price * (1 + price_change)

            prices.append(max(0.1, new_price))
            volatilities.append(current_vol)

        return self._create_ohlcv_data(prices, volatilities)

    def generate_crisis_scenario(
        self,
        duration_days: int = 252,
        crisis_start: int = 126,  # Middle of the period
        crisis_duration: int = 30,
        _crisis_severity: float = 0.3,
    ) -> pd.DataFrame:
        """Generate market crisis scenario."""

        prices = [100.0]

        for i in range(1, duration_days):
            current_price = prices[-1]

            # Determine if we're in crisis period
            in_crisis = crisis_start <= i <= crisis_start + crisis_duration

            if in_crisis:
                # High volatility and negative drift during crisis
                volatility = 0.05
                drift = -0.01
            else:
                # Normal market conditions
                volatility = 0.015
                drift = 0.0002

            # Generate price movement
            price_change = drift + volatility * np.random.normal(0, 1)
            new_price = current_price * (1 + price_change)

            prices.append(max(0.1, new_price))

        return self._create_ohlcv_data(prices, 0.02)

    def generate_regime_change_scenario(
        self,
        duration_days: int = 252,
        regime_changes: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Generate scenario with multiple regime changes."""

        if regime_changes is None:
            regime_changes = [
                {"start_day": 50, "regime": "trend_up", "duration": 50},
                {"start_day": 100, "regime": "volatile", "duration": 30},
                {"start_day": 130, "regime": "trend_down", "duration": 50},
                {"start_day": 180, "regime": "calm", "duration": 72},
            ]

        prices = [100.0]
        current_regime = "normal"

        for i in range(1, duration_days):
            current_price = prices[-1]

            # Determine current regime
            for change in regime_changes:
                if change["start_day"] <= i <= change["start_day"] + change["duration"]:
                    current_regime = change["regime"]
                    break

            # Generate price based on regime
            if current_regime == "trend_up":
                volatility, drift = 0.015, 0.001
            elif current_regime == "trend_down":
                volatility, drift = 0.015, -0.001
            elif current_regime == "volatile":
                volatility, drift = 0.04, 0.0001
            elif current_regime == "calm":
                volatility, drift = 0.008, 0.0002
            else:  # normal
                volatility, drift = 0.015, 0.0002

            # Generate price movement
            price_change = drift + volatility * np.random.normal(0, 1)
            new_price = current_price * (1 + price_change)

            prices.append(max(0.1, new_price))

        return self._create_ohlcv_data(prices, 0.02)

    def _generate_trend_segment(
        self,
        length: int,
        start_price: float,
        trend: float,
        volatility: float,
    ) -> list[float]:
        """Generate a price segment with consistent trend."""

        prices = [start_price]

        for i in range(1, length):
            current_price = prices[-1]

            # Trend component + random noise
            price_change = trend + volatility * np.random.normal(0, 1)
            new_price = current_price * (1 + price_change)

            prices.append(max(0.1, new_price))

        return prices

    def _create_ohlcv_data(
        self,
        prices: list[float],
        volatility: float | list[float],
    ) -> pd.DataFrame:
        """Create OHLCV data from price series."""

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=len(prices)),
            periods=len(prices),
            freq="D",
        )

        # Generate OHLC from close prices
        opens = []
        highs = []
        lows = []
        volumes = []

        for i, close_price in enumerate(prices):
            # Use volatility for intraday range
            if isinstance(volatility, list):
                vol = volatility[i] if i < len(volatility) else volatility[-1]
            else:
                vol = volatility

            # Generate realistic OHLC
            intraday_range = close_price * vol * np.random.uniform(0.5, 1.5)

            if i == 0:
                open_price = close_price * (1 + np.random.normal(0, vol * 0.5))
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, vol * 0.3))

            high = max(open_price, close_price) + intraday_range * 0.3
            low = min(open_price, close_price) - intraday_range * 0.3

            # Ensure realistic OHLC relationships
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Generate volume (correlated with price movement)
            price_change = abs(close_price - open_price) / open_price
            volume = int(np.random.normal(1000000, 200000) * (1 + 10 * price_change))

            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            volumes.append(volume)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
        )


class AgentScenarioEvaluator:
    """
    Comprehensive agent evaluation framework using synthetic market scenarios.

    This evaluator tests agent performance across different market regimes,
    evaluates robustness to market shocks, and measures adaptation to
    changing market conditions.
    """

    def __init__(
        self,
        scenarios: list[MarketScenario] | None = None,
        seed: int | None = None,
    ):
        """Initialize the scenario evaluator."""
        self.scenario_generator = MarketScenarioGenerator(seed=seed)
        self.metrics_calculator = MetricsCalculator()
        self.model_evaluator = ModelEvaluator()

        # Default scenarios if none provided
        if scenarios is None:
            self.scenarios = self._create_default_scenarios()
        else:
            self.scenarios = scenarios

        self.results: list[ScenarioResult] = []

    def _create_default_scenarios(self) -> list[MarketScenario]:
        """Create default market scenarios for evaluation."""

        return [
            MarketScenario(
                name="Trend Following",
                description="Market with clear directional trends",
                duration_days=252,
                market_regime="trend_following",
                base_volatility=0.015,
                drift=0.0002,
                min_sharpe_ratio=0.8,
                max_drawdown=0.12,
                min_win_rate=0.45,
                min_profit_factor=1.3,
            ),
            MarketScenario(
                name="Mean Reversion",
                description="Market that reverts to mean price levels",
                duration_days=252,
                market_regime="mean_reversion",
                base_volatility=0.02,
                drift=0.0,
                min_sharpe_ratio=0.6,
                max_drawdown=0.10,
                min_win_rate=0.50,
                min_profit_factor=1.4,
            ),
            MarketScenario(
                name="Volatility Breakout",
                description="Market with sudden volatility spikes",
                duration_days=252,
                market_regime="volatility_breakout",
                base_volatility=0.01,
                drift=0.0001,
                min_sharpe_ratio=0.4,
                max_drawdown=0.20,
                min_win_rate=0.35,
                min_profit_factor=1.1,
            ),
            MarketScenario(
                name="Market Crisis",
                description="Simulated market crisis with high volatility",
                duration_days=252,
                market_regime="crisis",
                base_volatility=0.05,
                drift=-0.005,
                min_sharpe_ratio=0.2,
                max_drawdown=0.25,
                min_win_rate=0.30,
                min_profit_factor=1.0,
            ),
            MarketScenario(
                name="Regime Changes",
                description="Market with multiple regime transitions",
                duration_days=252,
                market_regime="regime_changes",
                base_volatility=0.02,
                drift=0.0001,
                min_sharpe_ratio=0.5,
                max_drawdown=0.15,
                min_win_rate=0.40,
                min_profit_factor=1.2,
            ),
        ]

    def evaluate_agent(
        self,
        agent: Any,
        agent_name: str = "agent",
        custom_scenarios: list[MarketScenario] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate agent across all scenarios.

        Args:
            agent: The agent to evaluate
            agent_name: Name of the agent for reporting
            custom_scenarios: Optional custom scenarios to use instead of defaults

        Returns:
            Dictionary with comprehensive evaluation results
        """
        console.print(f"[bold blue]Evaluating {agent_name} across market scenarios...[/bold blue]")

        scenarios_to_evaluate = custom_scenarios if custom_scenarios else self.scenarios
        results = []

        for scenario in scenarios_to_evaluate:
            console.print(f"  Testing scenario: {scenario.name}")

            # Generate scenario data
            scenario_data = self._generate_scenario_data(scenario)

            # Run agent on scenario
            scenario_result = self._evaluate_scenario(
                agent=agent,
                agent_name=agent_name,
                scenario=scenario,
                data=scenario_data,
            )

            results.append(scenario_result)

        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(results)

        # Generate comprehensive report
        evaluation_report = {
            "agent_name": agent_name,
            "evaluation_date": datetime.now().isoformat(),
            "scenario_results": [result.__dict__ for result in results],
            "aggregate_metrics": aggregate_results,
            "overall_score": self._calculate_overall_score(results),
            "robustness_score": self._calculate_robustness_score(results),
            "adaptation_score": self._calculate_adaptation_score(results),
        }

        console.print(f"[bold green]✅ {agent_name} evaluation complete[/bold green]")
        return evaluation_report

    def _generate_scenario_data(self, scenario: MarketScenario) -> pd.DataFrame:
        """Generate synthetic data for a specific scenario."""

        if scenario.market_regime == "trend_following":
            return self.scenario_generator.generate_trend_following_scenario(
                duration_days=scenario.duration_days,
                volatility=scenario.base_volatility,
                trend_strength=scenario.drift,
            )
        if scenario.market_regime == "mean_reversion":
            return self.scenario_generator.generate_mean_reversion_scenario(
                duration_days=scenario.duration_days,
                volatility=scenario.base_volatility,
            )
        if scenario.market_regime == "volatility_breakout":
            return self.scenario_generator.generate_volatility_breakout_scenario(
                duration_days=scenario.duration_days,
                base_volatility=scenario.base_volatility,
            )
        if scenario.market_regime == "crisis":
            return self.scenario_generator.generate_crisis_scenario(
                duration_days=scenario.duration_days,
            )
        if scenario.market_regime == "regime_changes":
            return self.scenario_generator.generate_regime_change_scenario(
                duration_days=scenario.duration_days,
                regime_changes=scenario.regime_changes,
            )
        # Fallback to GBM
        return generate_gbm_prices(
            n_days=scenario.duration_days,
            mu=scenario.drift,
            sigma=scenario.base_volatility,
        )

    def _evaluate_scenario(
        self,
        agent: Any,
        agent_name: str,
        scenario: MarketScenario,
        data: pd.DataFrame,
    ) -> ScenarioResult:
        """Evaluate agent on a specific scenario."""

        # Prepare data for agent
        features = self._prepare_features(data)

        # Generate agent predictions/actions
        predictions = self._generate_agent_predictions(agent, features)

        # Calculate returns based on predictions
        returns = self._calculate_strategy_returns(data, predictions)

        # Calculate performance metrics
        metrics = self.metrics_calculator.calculate_trading_metrics(returns)

        # Create scenario result
        result = ScenarioResult(
            scenario=scenario,
            agent_name=agent_name,
            total_return=metrics.get("total_return", 0.0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics.get("sortino_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            win_rate=metrics.get("win_rate", 0.0),
            profit_factor=metrics.get("profit_factor", 0.0),
            volatility=metrics.get("volatility", 0.0),
            equity_curve=pd.Series(returns).cumsum(),
            returns_series=pd.Series(returns),
        )

        # Evaluate success criteria
        result.evaluate_success()

        # Calculate custom metrics if defined
        if scenario.custom_metrics:
            for metric_name, metric_func in scenario.custom_metrics.items():
                result.custom_metrics[metric_name] = metric_func(returns, data)

        return result

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for agent input."""
        # Simple feature engineering - can be enhanced
        features = []

        # Price-based features
        features.append(data["close"].values)
        features.append(data["volume"].values)

        # Technical indicators
        close_prices = data["close"].values
        returns = np.diff(close_prices, prepend=close_prices[0])

        # Moving averages
        for window in [5, 10, 20]:
            ma = pd.Series(close_prices).rolling(window=window).mean().bfill().values
            features.append(ma)

        # Volatility
        volatility = pd.Series(returns).rolling(window=20).std().bfill().values
        features.append(volatility)

        # RSI-like indicator
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = pd.Series(gains).rolling(window=14).mean().bfill().values
        avg_loss = pd.Series(losses).rolling(window=14).mean().bfill().values
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)

        # Stack features
        feature_matrix = np.column_stack(features)

        # Normalize features
        return (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)

    def _generate_agent_predictions(self, agent: Any, features: np.ndarray) -> np.ndarray:
        """Generate predictions from agent."""

        try:
            if hasattr(agent, "predict"):
                # Scikit-learn style model
                predictions = agent.predict(features)
            elif hasattr(agent, "forward"):
                # PyTorch model
                import torch

                agent.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features)
                    predictions = agent(features_tensor).cpu().numpy().flatten()
            elif callable(agent):
                # Function-based agent - pass the entire feature matrix
                predictions = agent(features)
                if not isinstance(predictions, np.ndarray):
                    predictions = np.array(predictions)
            else:
                # Try to call directly
                predictions = agent(features)
                if hasattr(predictions, "numpy"):
                    predictions = predictions.numpy()
                predictions = np.array(predictions).flatten()

            return predictions

        except Exception as e:
            logger.warning(f"Error generating predictions: {e}")
            # Return random predictions as fallback
            return np.random.normal(0, 1, len(features))

    def _calculate_strategy_returns(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Calculate strategy returns based on predictions."""

        # Simple strategy: long when prediction > 0, short when prediction < 0
        positions = np.sign(predictions)

        # Calculate price returns
        prices = data["close"].values
        price_returns = np.diff(prices, prepend=prices[0]) / prices

        # Strategy returns
        strategy_returns = positions * price_returns

        # Apply transaction costs (simplified)
        position_changes = np.diff(positions, prepend=0)
        transaction_costs = np.abs(position_changes) * 0.001  # 0.1% transaction cost

        return strategy_returns - transaction_costs

    def _calculate_aggregate_metrics(self, results: list[ScenarioResult]) -> dict[str, float]:
        """Calculate aggregate metrics across all scenarios."""

        if not results:
            return {}

        # Collect metrics
        sharpe_ratios = [r.sharpe_ratio for r in results]
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        profit_factors = [r.profit_factor for r in results]

        # Calculate aggregate statistics
        return {
            "avg_sharpe_ratio": np.mean(sharpe_ratios),
            "std_sharpe_ratio": np.std(sharpe_ratios),
            "avg_total_return": np.mean(total_returns),
            "std_total_return": np.std(total_returns),
            "avg_max_drawdown": np.mean(max_drawdowns),
            "worst_drawdown": np.max(max_drawdowns),
            "avg_win_rate": np.mean(win_rates),
            "avg_profit_factor": np.mean(profit_factors),
            "scenarios_passed": sum(1 for r in results if r.passed_criteria),
            "total_scenarios": len(results),
            "pass_rate": sum(1 for r in results if r.passed_criteria) / len(results),
        }

    def _calculate_overall_score(self, results: list[ScenarioResult]) -> float:
        """Calculate overall performance score."""

        if not results:
            return 0.0

        # Weighted average of key metrics
        weights = {
            "sharpe_ratio": 0.3,
            "total_return": 0.25,
            "max_drawdown": 0.2,
            "win_rate": 0.15,
            "profit_factor": 0.1,
        }

        scores = []
        for result in results:
            # Normalize metrics to 0-1 scale
            sharpe_score = min(1.0, max(0.0, result.sharpe_ratio / 2.0))
            return_score = min(1.0, max(0.0, (result.total_return + 0.5) / 1.0))
            drawdown_score = min(1.0, max(0.0, (0.3 - result.max_drawdown) / 0.3))
            win_rate_score = result.win_rate
            profit_factor_score = min(1.0, result.profit_factor / 2.0)

            # Calculate weighted score
            score = (
                weights["sharpe_ratio"] * sharpe_score
                + weights["total_return"] * return_score
                + weights["max_drawdown"] * drawdown_score
                + weights["win_rate"] * win_rate_score
                + weights["profit_factor"] * profit_factor_score
            )
            scores.append(score)

        return float(np.mean(scores))

    def _calculate_robustness_score(self, results: list[ScenarioResult]) -> float:
        """Calculate robustness score based on consistency across scenarios."""

        if not results:
            return 0.0

        # Calculate coefficient of variation for key metrics
        sharpe_ratios = [r.sharpe_ratio for r in results]
        total_returns = [r.total_return for r in results]

        # Robustness is inversely related to variability
        sharpe_cv = np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8)
        return_cv = np.std(total_returns) / (np.mean(total_returns) + 1e-8)

        # Convert to 0-1 score (lower CV = higher robustness)
        return float(1.0 / (1.0 + sharpe_cv + return_cv))

    def _calculate_adaptation_score(self, results: list[ScenarioResult]) -> float:
        """Calculate adaptation score based on performance in challenging scenarios."""

        if not results:
            return 0.0

        # Identify challenging scenarios (high volatility, crisis, etc.)
        challenging_scenarios = [r for r in results if r.scenario.market_regime in ["crisis", "volatility_breakout"]]

        if not challenging_scenarios:
            return 0.5  # Neutral score if no challenging scenarios

        # Calculate average performance in challenging scenarios
        challenging_scores = []
        for result in challenging_scenarios:
            # Normalize performance relative to scenario difficulty
            if result.scenario.market_regime == "crisis":
                # Crisis scenarios are very difficult
                score = min(1.0, max(0.0, (result.sharpe_ratio + 1.0) / 2.0))
            else:
                # Volatility breakout scenarios are moderately difficult
                score = min(1.0, max(0.0, result.sharpe_ratio / 1.5))

            challenging_scores.append(score)

        return float(np.mean(challenging_scores))

    def generate_evaluation_report(
        self,
        evaluation_results: dict[str, Any],
        output_path: Path | None = None,
    ) -> str:
        """Generate comprehensive evaluation report."""

        agent_name = evaluation_results["agent_name"]
        scenario_results = evaluation_results["scenario_results"]
        aggregate_metrics = evaluation_results["aggregate_metrics"]

        # Create report
        report = f"""
# Agent Scenario Evaluation Report

## Agent: {agent_name}
## Evaluation Date: {evaluation_results["evaluation_date"]}

## Executive Summary

- **Overall Score**: {evaluation_results["overall_score"]:.3f}
- **Robustness Score**: {evaluation_results["robustness_score"]:.3f}
- **Adaptation Score**: {evaluation_results["adaptation_score"]:.3f}
- **Scenarios Passed**: {aggregate_metrics["scenarios_passed"]}/{aggregate_metrics["total_scenarios"]}
- **Pass Rate**: {aggregate_metrics["pass_rate"]:.1%}

## Aggregate Performance Metrics

- **Average Sharpe Ratio**: {aggregate_metrics["avg_sharpe_ratio"]:.3f} ± {aggregate_metrics["std_sharpe_ratio"]:.3f}
- **Average Total Return**: {aggregate_metrics["avg_total_return"]:.1%} ± {aggregate_metrics["std_total_return"]:.1%}
- **Average Max Drawdown**: {aggregate_metrics["avg_max_drawdown"]:.1%}
- **Worst Drawdown**: {aggregate_metrics["worst_drawdown"]:.1%}
- **Average Win Rate**: {aggregate_metrics["avg_win_rate"]:.1%}
- **Average Profit Factor**: {aggregate_metrics["avg_profit_factor"]:.2f}

## Scenario-by-Scenario Results

"""

        for result in scenario_results:
            status = "✅ PASSED" if result["passed_criteria"] else "❌ FAILED"
            report += f"""
### {result["scenario"].name} - {status}

**Description**: {result["scenario"].description}
**Market Regime**: {result["scenario"].market_regime}

**Performance Metrics**:
- Total Return: {result["total_return"]:.1%}
- Sharpe Ratio: {result["sharpe_ratio"]:.3f}
- Max Drawdown: {result["max_drawdown"]:.1%}
- Win Rate: {result["win_rate"]:.1%}
- Profit Factor: {result["profit_factor"]:.2f}

**Success Criteria**:
- Min Sharpe Ratio: {result["scenario"].min_sharpe_ratio} (Actual: {result["sharpe_ratio"]:.3f})
- Max Drawdown: {result["scenario"].max_drawdown:.1%} (Actual: {result["max_drawdown"]:.1%})
- Min Win Rate: {result["scenario"].min_win_rate:.1%} (Actual: {result["win_rate"]:.1%})
- Min Profit Factor: {result["scenario"].min_profit_factor:.2f} (Actual: {result["profit_factor"]:.2f})

"""

            if result["failure_reasons"]:
                report += "**Failure Reasons**:\n"
                for reason in result["failure_reasons"]:
                    report += f"- {reason}\n"

        # Save report if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

        return report

    def create_visualization(
        self,
        evaluation_results: dict[str, Any],
        output_path: Path | None = None,
    ) -> None:
        """Create comprehensive visualization of evaluation results."""

        scenario_results = evaluation_results["scenario_results"]
        aggregate_metrics = evaluation_results["aggregate_metrics"]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Agent Scenario Evaluation: {evaluation_results['agent_name']}",
            fontsize=16,
        )

        # 1. Sharpe Ratio by Scenario
        scenario_names = [r["scenario"].name for r in scenario_results]
        sharpe_ratios = [r["sharpe_ratio"] for r in scenario_results]
        colors = ["green" if r["passed_criteria"] else "red" for r in scenario_results]

        axes[0, 0].bar(scenario_names, sharpe_ratios, color=colors, alpha=0.7)
        axes[0, 0].set_title("Sharpe Ratio by Scenario")
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Total Returns by Scenario
        total_returns = [r["total_return"] for r in scenario_results]
        axes[0, 1].bar(scenario_names, total_returns, color=colors, alpha=0.7)
        axes[0, 1].set_title("Total Returns by Scenario")
        axes[0, 1].set_ylabel("Total Return")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Max Drawdown by Scenario
        max_drawdowns = [r["max_drawdown"] for r in scenario_results]
        axes[0, 2].bar(scenario_names, max_drawdowns, color=colors, alpha=0.7)
        axes[0, 2].set_title("Max Drawdown by Scenario")
        axes[0, 2].set_ylabel("Max Drawdown")
        axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. Performance Radar Chart
        metrics = [
            "Sharpe Ratio",
            "Total Return",
            "Win Rate",
            "Profit Factor",
            "Robustness",
        ]
        values = [
            aggregate_metrics["avg_sharpe_ratio"] / 2.0,  # Normalize to 0-1
            min(1.0, (aggregate_metrics["avg_total_return"] + 0.5) / 1.0),
            aggregate_metrics["avg_win_rate"],
            min(1.0, aggregate_metrics["avg_profit_factor"] / 2.0),
            evaluation_results["robustness_score"],
        ]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Close the plot
        angles += angles[:1]

        axes[1, 0].plot(angles, values, "o-", linewidth=2)
        axes[1, 0].fill(angles, values, alpha=0.25)
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].set_title("Performance Radar Chart")
        axes[1, 0].set_ylim(0, 1)

        # 5. Pass/Fail Summary
        passed = aggregate_metrics["scenarios_passed"]
        failed = aggregate_metrics["total_scenarios"] - passed
        axes[1, 1].pie(
            [passed, failed],
            labels=["Passed", "Failed"],
            colors=["green", "red"],
            autopct="%1.1f%%",
        )
        axes[1, 1].set_title("Scenario Pass/Fail Summary")

        # 6. Score Comparison
        scores = [
            evaluation_results["overall_score"],
            evaluation_results["robustness_score"],
            evaluation_results["adaptation_score"],
        ]
        score_labels = ["Overall", "Robustness", "Adaptation"]
        axes[1, 2].bar(score_labels, scores, color=["blue", "orange", "green"], alpha=0.7)
        axes[1, 2].set_title("Performance Scores")
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].set_ylim(0, 1)

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.show()

    def print_evaluation_summary(self, evaluation_results: dict[str, Any]) -> None:
        """Print a summary of evaluation results to console."""

        agent_name = evaluation_results["agent_name"]
        aggregate_metrics = evaluation_results["aggregate_metrics"]

        # Create summary table
        table = Table(title=f"Agent Scenario Evaluation Summary: {agent_name}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        # Overall scores
        table.add_row(
            "Overall Score",
            f"{evaluation_results['overall_score']:.3f}",
            "✅" if evaluation_results["overall_score"] > 0.5 else "❌",
        )
        table.add_row(
            "Robustness Score",
            f"{evaluation_results['robustness_score']:.3f}",
            "✅" if evaluation_results["robustness_score"] > 0.5 else "❌",
        )
        table.add_row(
            "Adaptation Score",
            f"{evaluation_results['adaptation_score']:.3f}",
            "✅" if evaluation_results["adaptation_score"] > 0.5 else "❌",
        )

        # Performance metrics
        table.add_row("Avg Sharpe Ratio", f"{aggregate_metrics['avg_sharpe_ratio']:.3f}", "")
        table.add_row("Avg Total Return", f"{aggregate_metrics['avg_total_return']:.1%}", "")
        table.add_row("Avg Max Drawdown", f"{aggregate_metrics['avg_max_drawdown']:.1%}", "")
        table.add_row(
            "Pass Rate",
            f"{aggregate_metrics['pass_rate']:.1%}",
            "✅" if aggregate_metrics["pass_rate"] > 0.6 else "❌",
        )

        console.print(table)

        # Print scenario results
        console.print("\n[bold]Scenario Results:[/bold]")
        for result in evaluation_results["scenario_results"]:
            status = "✅ PASSED" if result["passed_criteria"] else "❌ FAILED"
            console.print(f"  {result['scenario'].name}: {status}")
            if result["failure_reasons"]:
                for reason in result["failure_reasons"]:
                    console.print(f"    - {reason}")
