#!/usr/bin/env python3
"""
Example: Using the Unified BacktestEvaluator

This example demonstrates how to use the new BacktestEvaluator to:
1. Run backtests with realistic transaction cost modeling
2. Compare multiple strategies
3. Generate detailed performance reports
4. Analyze transaction costs and their impact on returns
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.trade_agent.core.unified_config import BacktestConfig
from src.trade_agent.eval.backtest_evaluator import BacktestEvaluator
from src.trade_agent.portfolio.transaction_costs import (
    BrokerType,
    TransactionCostModel,
)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sample_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load historical data for backtesting."""
    print(f"Loading data for {symbols} from {start_date} to {end_date}...")

    data_frames = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data["symbol"] = symbol
        data_frames.append(data)

    # Combine all data
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Rename columns to lowercase
    combined_data.columns = [col.lower() for col in combined_data.columns]

    print(f"Loaded {len(combined_data)} data points")
    return combined_data


def create_momentum_strategy(data: pd.DataFrame) -> pd.Series:
    """Create a simple momentum strategy."""
    returns = data["close"].pct_change()
    signals = pd.Series(0, index=data.index)

    # Buy when returns are above 20-day moving average
    signals[returns > returns.rolling(20).mean()] = 1
    # Sell when returns are below 20-day moving average
    signals[returns < returns.rolling(20).mean()] = -1

    return signals


def create_mean_reversion_strategy(data: pd.DataFrame) -> pd.Series:
    """Create a simple mean reversion strategy."""
    sma_20 = data["close"].rolling(20).mean()
    data["close"].rolling(50).mean()

    signals = pd.Series(0, index=data.index)

    # Buy when price is significantly below 20-day SMA
    signals[data["close"] < sma_20 * 0.95] = 1
    # Sell when price is significantly above 20-day SMA
    signals[data["close"] > sma_20 * 1.05] = -1

    return signals


def create_volatility_strategy(data: pd.DataFrame) -> pd.Series:
    """Create a volatility-based strategy."""
    # Calculate rolling volatility
    returns = data["close"].pct_change()
    volatility = returns.rolling(20).std()

    signals = pd.Series(0, index=data.index)

    # Buy when volatility is low (mean reversion opportunity)
    low_vol_threshold = volatility.quantile(0.25)
    signals[volatility < low_vol_threshold] = 1

    # Sell when volatility is high (risk management)
    high_vol_threshold = volatility.quantile(0.75)
    signals[volatility > high_vol_threshold] = -1

    return signals


def main() -> None:
    """Main example function."""
    print("=" * 60)
    print("Unified BacktestEvaluator Example")
    print("=" * 60)

    # Configuration
    symbols = ["AAPL", "GOOGL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000.0

    # Load data
    data = load_sample_data(symbols, start_date, end_date)

    # Create strategies
    strategies = {
        "momentum": create_momentum_strategy(data),
        "mean_reversion": create_mean_reversion_strategy(data),
        "volatility": create_volatility_strategy(data),
    }

    # Configure backtest
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1% commission
        slippage_rate=0.0001,  # 0.01% slippage
        max_position_size=0.1,  # 10% max position size
        max_leverage=1.0,
        stop_loss_pct=0.02,  # 2% stop loss
        take_profit_pct=0.05,  # 5% take profit
    )

    # Create different transaction cost models for comparison
    cost_models = {
        "low_cost": TransactionCostModel.create_broker_model(BrokerType.PREMIUM),
        "standard_cost": TransactionCostModel.create_broker_model(BrokerType.RETAIL),
        "high_cost": TransactionCostModel.create_broker_model(BrokerType.DISCOUNT),
    }

    print("\n" + "=" * 60)
    print("Strategy Comparison with Different Cost Models")
    print("=" * 60)

    # Compare strategies with different cost models
    for cost_model_name, cost_model in cost_models.items():
        print(f"\n--- {cost_model_name.upper()} COST MODEL ---")

        # Initialize evaluator
        evaluator = BacktestEvaluator(backtest_config, cost_model)

        # Compare strategies
        results = evaluator.compare_strategies(data, strategies)

        # Display results
        print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'WinRate':<8} {'Trades':<8} {'Costs':<12}")
        print("-" * 75)

        for strategy_name, result in results.items():
            print(
                f"{strategy_name:<15} "
                f"{result.total_return:>8.2%} "
                f"{result.sharpe_ratio:>7.2f} "
                f"{result.max_drawdown:>7.2%} "
                f"{result.win_rate:>7.2%} "
                f"{result.num_trades:>7d} "
                f"${result.total_transaction_costs:>10.2f}"
            )

    print("\n" + "=" * 60)
    print("Detailed Analysis: Standard Cost Model")
    print("=" * 60)

    # Detailed analysis with standard cost model
    evaluator = BacktestEvaluator(backtest_config, cost_models["standard_cost"])

    # Run backtest for momentum strategy
    momentum_results = evaluator.run_backtest(data, strategies["momentum"], strategy_name="momentum")

    # Generate detailed report
    report = evaluator.generate_performance_report(momentum_results)
    print(report)

    # Save detailed report
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "momentum_strategy_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nDetailed report saved to: {report_path!s}")

    # Transaction cost analysis
    print("\n" + "=" * 60)
    print("Transaction Cost Analysis")
    print("=" * 60)

    print(f"Total Commission: ${momentum_results.total_commission:.2f}")
    print(f"Total Slippage: ${momentum_results.total_slippage:.2f}")
    print(f"Total Market Impact: ${momentum_results.total_market_impact:.2f}")
    print(f"Total Spread Cost: ${momentum_results.total_spread_cost:.2f}")
    print(f"Total Transaction Costs: ${momentum_results.total_transaction_costs:.2f}")
    print(f"Cost Drag on Returns: {momentum_results.cost_drag:.2%}")

    # Trade analysis
    if momentum_results.trades:
        print(f"\nNumber of Trades: {momentum_results.num_trades}")
        print(f"Average Trade Value: ${np.mean([t.trade_value for t in momentum_results.trades]):.2f}")
        print(f"Average Cost per Trade: ${momentum_results.total_transaction_costs / momentum_results.num_trades:.2f}")

        # Show first few trades
        print("\nFirst 5 Trades:")
        print(f"{'Date':<12} {'Side':<6} {'Quantity':<10} {'Price':<10} {'Value':<12} {'Cost':<10}")
        print("-" * 70)

        for trade in momentum_results.trades[:5]:
            print(
                f"{trade.timestamp.strftime('%Y-%m-%d'):<12} "
                f"{trade.side:<6} "
                f"{trade.quantity:>9.0f} "
                f"${trade.price:>8.2f} "
                f"${trade.trade_value:>10.2f} "
                f"${trade.costs['total_cost']:>8.2f}"
            )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
