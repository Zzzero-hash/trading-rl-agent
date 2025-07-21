#!/usr/bin/env python3
"""
Simplified Ensemble Trading Example
===================================

This example demonstrates ensemble trading concepts without the complex
Ray RLlib integration that may have compatibility issues.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEnsembleAgent:
    """Simplified ensemble agent for demonstration purposes."""

    def __init__(self, name: str, strategy_type: str) -> None:
        self.name = name
        self.strategy_type = strategy_type
        self.performance_history: list[float] = []

    def predict(self, data: pd.DataFrame) -> float:
        """Make a simple prediction based on strategy type."""
        if self.strategy_type == "momentum":
            # Simple momentum strategy
            returns = data["close"].pct_change().dropna()
            if len(returns) >= 5:
                momentum = returns.tail(5).mean()
                return float(np.clip(momentum * 10, -1, 1))  # Scale and clip
            return 0.0

        if self.strategy_type == "mean_reversion":
            # Simple mean reversion strategy
            if len(data) >= 20:
                current_price = data["close"].iloc[-1]
                avg_price = data["close"].tail(20).mean()
                deviation = (current_price - avg_price) / avg_price
                return float(np.clip(-deviation * 5, -1, 1))  # Revert to mean
            return 0.0

        if self.strategy_type == "volatility":
            # Volatility-based strategy
            if len(data) >= 10:
                volatility = data["close"].pct_change().tail(10).std()
                return float(np.clip(volatility * 20, -1, 1))
            return 0.0

        return 0.0


class SimpleEnsemble:
    """Simplified ensemble for combining multiple strategies."""

    def __init__(self, agents: list[SimpleEnsembleAgent], weights: list[float] | None = None) -> None:
        self.agents = agents
        self.weights = weights if weights else [1.0 / len(agents)] * len(agents)
        self.performance_history: list[float] = []

    def predict(self, data: pd.DataFrame) -> float:
        """Combine predictions from all agents."""
        predictions = []
        for agent in self.agents:
            try:
                pred = agent.predict(data)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Agent {agent.name} failed: {e}")
                predictions.append(0.0)

        # Weighted average
        weighted_pred = sum(p * w for p, w in zip(predictions, self.weights, strict=False))
        return float(np.clip(weighted_pred, -1, 1))

    def update_weights(self, performances: list[float]) -> None:
        """Update weights based on recent performance."""
        if len(performances) == len(self.agents):
            # Simple performance-based weighting
            total_perf = sum(max(p, 0) for p in performances) + 1e-6
            self.weights = [max(p, 0) / total_perf for p in performances]


def generate_sample_data(n_days: int = 252) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    np.random.seed(42)

    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100

    # Add some trend and volatility clustering
    trend = np.linspace(0, 0.1, n_days)
    prices = prices * (1 + trend)

    # Add volatility clustering
    volatility = np.abs(returns) * 0.5 + 0.01
    prices = prices * np.exp(np.random.normal(0, volatility))

    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        # Simple OHLC generation
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price

        data.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": np.random.randint(1000000, 10000000),
            }
        )

    return pd.DataFrame(data)


def backtest_ensemble(data: pd.DataFrame, ensemble: SimpleEnsemble, initial_capital: float = 100000) -> dict[str, Any]:
    """Simple backtesting of the ensemble strategy."""

    capital = initial_capital
    position = 0
    trades = []
    equity_curve = []

    for i in range(20, len(data)):  # Start after enough data for strategies
        current_data = data.iloc[: i + 1]
        signal = ensemble.predict(current_data)

        current_price = data.iloc[i]["close"]

        # Simple position sizing
        if signal > 0.1:  # Buy signal
            if position <= 0:
                position = capital * 0.95 / current_price  # Use 95% of capital
                capital -= position * current_price
                trades.append(
                    {
                        "date": data.iloc[i]["date"],
                        "action": "buy",
                        "price": current_price,
                        "position": position,
                    }
                )
        elif signal < -0.1 and position > 0:  # Sell signal
            capital += position * current_price
            position = 0
            trades.append(
                {
                    "date": data.iloc[i]["date"],
                    "action": "sell",
                    "price": current_price,
                    "position": 0,
                }
            )

        # Calculate current equity
        current_equity = capital + (position * current_price)
        equity_curve.append({"date": data.iloc[i]["date"], "equity": current_equity, "signal": signal})

    # Close final position
    if position > 0:
        final_price = data.iloc[-1]["close"]
        capital += position * final_price

    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    returns = equity_df["equity"].pct_change().dropna()

    total_return = (capital - initial_capital) / initial_capital
    sharpe_ratio = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
    max_drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min()

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_capital": capital,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def plot_results(data: pd.DataFrame, results: dict[str, Any], ensemble: SimpleEnsemble) -> None:
    """Plot the backtesting results."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Price and signals
    equity_df = pd.DataFrame(results["equity_curve"])
    axes[0, 0].plot(data["date"], data["close"], label="Price", alpha=0.7)
    axes[0, 0].plot(equity_df["date"], equity_df["equity"], label="Portfolio Value", linewidth=2)
    axes[0, 0].set_title("Price and Portfolio Performance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Trading signals
    axes[0, 1].plot(equity_df["date"], equity_df["signal"], label="Ensemble Signal", alpha=0.7)
    axes[0, 1].axhline(y=0.1, color="g", linestyle="--", alpha=0.5, label="Buy Threshold")
    axes[0, 1].axhline(y=-0.1, color="r", linestyle="--", alpha=0.5, label="Sell Threshold")
    axes[0, 1].set_title("Trading Signals")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Individual agent predictions
    agent_predictions = {}
    for agent in ensemble.agents:
        predictions = []
        for i in range(20, len(data)):
            current_data = data.iloc[: i + 1]
            pred = agent.predict(current_data)
            predictions.append(pred)
        agent_predictions[agent.name] = predictions

    for agent_name, predictions in agent_predictions.items():
        axes[1, 0].plot(data["date"].iloc[20:], predictions, label=agent_name, alpha=0.7)

    axes[1, 0].set_title("Individual Agent Predictions")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Performance metrics
    metrics = ["Total Return", "Sharpe Ratio", "Max Drawdown"]
    values = [
        results["total_return"] * 100,
        results["sharpe_ratio"],
        results["max_drawdown"] * 100,
    ]

    bars = axes[1, 1].bar(metrics, values, color=["green", "blue", "red"], alpha=0.7)
    axes[1, 1].set_title("Performance Metrics")
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values, strict=False):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("ensemble_trading_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    """Main function to run the ensemble trading example."""
    logger.info("Starting Simplified Ensemble Trading Example")

    # Generate sample data
    logger.info("Generating sample market data...")
    data = generate_sample_data(252)  # One year of data

    # Create ensemble agents
    logger.info("Creating ensemble agents...")
    agents = [
        SimpleEnsembleAgent("Momentum", "momentum"),
        SimpleEnsembleAgent("MeanReversion", "mean_reversion"),
        SimpleEnsembleAgent("Volatility", "volatility"),
    ]

    # Create ensemble
    ensemble = SimpleEnsemble(agents)

    # Run backtest
    logger.info("Running ensemble backtest...")
    results = backtest_ensemble(data, ensemble)

    # Print results
    print("\n" + "=" * 50)
    print("ENSEMBLE TRADING RESULTS")
    print("=" * 50)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Number of Trades: {len(results['trades'])}")
    print("=" * 50)

    # Plot results
    logger.info("Generating visualization...")
    plot_results(data, results, ensemble)

    # Save results
    output_dir = Path("outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    equity_df = pd.DataFrame(results["equity_curve"])
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

    # Save trades
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(output_dir / "trades.csv", index=False)

    # Save summary
    summary = {
        "total_return": results["total_return"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown": results["max_drawdown"],
        "final_capital": results["final_capital"],
        "num_trades": len(results["trades"]),
        "ensemble_weights": ensemble.weights,
        "agent_names": [agent.name for agent in ensemble.agents],
    }

    import json

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info("Ensemble trading example completed successfully!")


if __name__ == "__main__":
    main()
