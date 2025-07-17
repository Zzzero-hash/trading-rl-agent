"""
Example: Agent Scenario Evaluation with Synthetic Data

This example demonstrates how to use the AgentScenarioEvaluator to test
agent performance across different market regimes and scenarios.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_rl_agent.eval import AgentScenarioEvaluator


def create_simple_moving_average_agent(window: int = 20) -> Callable[[np.ndarray], np.ndarray]:
    """Create a simple moving average crossover agent."""

    def agent(features: np.ndarray) -> np.ndarray:
        """Simple moving average strategy."""
        # Extract close prices (first feature)
        close_prices = features[:, 0]

        # Calculate moving averages
        ma_short = pd.Series(close_prices).rolling(window=window // 2).mean().fillna(method="bfill").values
        ma_long = pd.Series(close_prices).rolling(window=window).mean().fillna(method="bfill").values

        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signals = np.zeros_like(close_prices)
        signals[ma_short > ma_long] = 1  # Buy signal
        signals[ma_short < ma_long] = -1  # Sell signal

        return signals

    return agent


def create_momentum_agent(lookback: int = 10) -> Callable[[np.ndarray], np.ndarray]:
    """Create a momentum-based agent."""

    def agent(features: np.ndarray) -> np.ndarray:
        """Momentum strategy based on price changes."""
        # Extract close prices
        close_prices = features[:, 0]

        # Calculate momentum (rate of change)
        momentum = pd.Series(close_prices).pct_change(lookback).fillna(0).values

        # Generate signals based on momentum
        signals = np.zeros_like(close_prices)
        signals[momentum > 0.02] = 1  # Strong positive momentum
        signals[momentum < -0.02] = -1  # Strong negative momentum

        return signals

    return agent


def create_mean_reversion_agent(lookback: int = 20) -> Callable[[np.ndarray], np.ndarray]:
    """Create a mean reversion agent."""

    def agent(features: np.ndarray) -> np.ndarray:
        """Mean reversion strategy."""
        # Extract close prices
        close_prices = features[:, 0]

        # Calculate rolling mean and standard deviation
        rolling_mean = pd.Series(close_prices).rolling(lookback).mean().fillna(method="bfill").values
        rolling_std = pd.Series(close_prices).rolling(lookback).std().fillna(method="bfill").values

        # Calculate z-score
        z_score = (close_prices - rolling_mean) / (rolling_std + 1e-8)

        # Generate signals: buy when oversold, sell when overbought
        signals = np.zeros_like(close_prices)
        signals[z_score < -1.5] = 1  # Oversold - buy
        signals[z_score > 1.5] = -1  # Overbought - sell

        return signals

    return agent


def create_volatility_breakout_agent(vol_window: int = 20) -> Callable[[np.ndarray], np.ndarray]:
    """Create a volatility breakout agent."""

    def agent(features: np.ndarray) -> np.ndarray:
        """Volatility breakout strategy."""
        # Extract close prices and volatility
        close_prices = features[:, 0]
        volatility = features[:, 4]  # Assuming volatility is the 5th feature

        # Calculate rolling volatility
        rolling_vol = pd.Series(volatility).rolling(vol_window).mean().fillna(method="bfill").values

        # Calculate price changes
        price_changes = pd.Series(close_prices).pct_change().fillna(0).values

        # Generate signals based on volatility breakouts
        signals = np.zeros_like(close_prices)

        # Buy on high volatility with positive price movement
        high_vol_mask = volatility > rolling_vol * 1.5
        positive_move_mask = price_changes > 0.01
        signals[high_vol_mask & positive_move_mask] = 1

        # Sell on high volatility with negative price movement
        negative_move_mask = price_changes < -0.01
        signals[high_vol_mask & negative_move_mask] = -1

        return signals

    return agent


def create_custom_scenarios() -> list[dict[str, Any]]:
    """Create custom market scenarios for evaluation."""

    return [
        {
            "name": "bull_market",
            "duration_days": 60,
            "regime_changes": [
                {
                    "day": 0,
                    "volatility_multiplier": 0.8,
                    "trend_strength": 0.3,
                    "correlation_shift": 0.1,
                }
            ],
        },
        {
            "name": "bear_market",
            "duration_days": 60,
            "regime_changes": [
                {
                    "day": 0,
                    "volatility_multiplier": 1.5,
                    "trend_strength": -0.4,
                    "correlation_shift": -0.2,
                }
            ],
        },
        {
            "name": "high_volatility",
            "duration_days": 60,
            "regime_changes": [
                {
                    "day": 0,
                    "volatility_multiplier": 2.0,
                    "trend_strength": 0.0,
                    "correlation_shift": 0.0,
                }
            ],
        },
    ]


def main() -> None:
    """Main example function."""

    print("🚀 Agent Scenario Evaluation Example")
    print("=" * 50)

    # Create agents
    agents = {
        "Moving Average": create_simple_moving_average_agent(window=20),
        "Momentum": create_momentum_agent(lookback=10),
        "Mean Reversion": create_mean_reversion_agent(lookback=20),
        "Volatility Breakout": create_volatility_breakout_agent(vol_window=20),
    }

    # Initialize scenario evaluator
    evaluator = AgentScenarioEvaluator(seed=42)

    # Create output directory
    output_dir = Path("outputs/scenario_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each agent
    all_results = {}

    for agent_name, agent in agents.items():
        print(f"\n📊 Evaluating {agent_name} agent...")

        # Run evaluation
        results = evaluator.evaluate_agent(
            agent=agent,
            agent_name=agent_name,
        )

        all_results[agent_name] = results

        # Print summary
        evaluator.print_evaluation_summary(results)

        # Generate detailed report
        report = evaluator.generate_evaluation_report(
            results, output_path=output_dir / f"{agent_name.lower().replace(' ', '_')}_report.md"
        )

        # Create visualization
        evaluator.create_visualization(
            results, output_path=output_dir / f"{agent_name.lower().replace(' ', '_')}_evaluation.png"
        )

    # Compare agents
    print("\n" + "=" * 50)
    print("🏆 AGENT COMPARISON")
    print("=" * 50)

    comparison_table = pd.DataFrame(
        {
            agent_name: {
                "Overall Score": results["overall_score"],
                "Robustness Score": results["robustness_score"],
                "Adaptation Score": results["adaptation_score"],
                "Pass Rate": results["aggregate_metrics"]["pass_rate"],
                "Avg Sharpe": results["aggregate_metrics"]["avg_sharpe_ratio"],
                "Avg Return": results["aggregate_metrics"]["avg_total_return"],
                "Worst Drawdown": results["aggregate_metrics"]["worst_drawdown"],
            }
            for agent_name, results in all_results.items()
        }
    ).T

    print(comparison_table.round(3))

    # Find best performing agent
    best_agent = comparison_table["Overall Score"].idxmax()
    print(f"\n🥇 Best Overall Agent: {best_agent}")
    print(f"   Overall Score: {comparison_table.loc[best_agent, 'Overall Score']:.3f}")

    # Test with custom scenarios
    print("\n" + "=" * 50)
    print("🎯 CUSTOM SCENARIO TESTING")
    print("=" * 50)

    custom_scenarios = create_custom_scenarios()
    best_agent_func = agents[best_agent]

    custom_results = evaluator.evaluate_agent(
        agent=best_agent_func,
        agent_name=f"{best_agent} (Custom Scenarios)",
        custom_scenarios=custom_scenarios,
    )

    evaluator.print_evaluation_summary(custom_results)

    # Generate comprehensive report
    print("\n📝 Generating comprehensive evaluation report...")

    comprehensive_report = f"""
# Comprehensive Agent Scenario Evaluation Report

## Summary
- **Best Agent**: {best_agent}
- **Evaluation Date**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Agents Tested**: {len(agents)}
- **Scenarios Per Agent**: {len(evaluator.scenarios)}

## Agent Rankings

{comparison_table.to_markdown()}

## Detailed Results

Each agent has been evaluated across multiple market scenarios including:
- Trend Following Markets
- Mean Reversion Markets
- Volatility Breakout Markets
- Market Crisis Scenarios
- Regime Change Scenarios

Detailed reports and visualizations have been saved to: {output_dir}

## Key Insights

1. **Best Overall Performance**: {best_agent} achieved the highest overall score
2. **Most Robust**: {comparison_table["Robustness Score"].idxmax()} showed the most consistent performance
3. **Best Adaptation**: {comparison_table["Adaptation Score"].idxmax()} adapted best to challenging scenarios

## Recommendations

- Use {best_agent} for general market conditions
- Consider scenario-specific agent selection for specialized strategies
- Monitor performance during regime changes and market crises
- Regular re-evaluation recommended as market conditions evolve
"""

    with open(output_dir / "comprehensive_report.md", "w") as f:
        f.write(comprehensive_report)

    print(f"✅ Evaluation complete! Results saved to: {output_dir}")
    print("\n📁 Generated Files:")
    for file_path in output_dir.glob("*"):
        print(f"   - {file_path.name}")


if __name__ == "__main__":
    main()
