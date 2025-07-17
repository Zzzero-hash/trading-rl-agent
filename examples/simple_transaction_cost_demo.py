#!/usr/bin/env python3
"""
Simple Transaction Cost Modeling Demonstration

This script demonstrates the basic functionality of the transaction cost modeling system
without requiring external dependencies like numpy or pandas.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def simple_math_sqrt(x):
    """Simple square root implementation for demonstration."""
    if x <= 0:
        return 0
    guess = x / 2
    for _ in range(10):
        guess = (guess + x / guess) / 2
    return guess


def simple_random():
    """Simple random number generator for demonstration."""
    import time

    return (time.time() * 1000) % 1.0


# Mock numpy functions for demonstration
class MockNumpy:
    @staticmethod
    def random():
        return simple_random()

    @staticmethod
    def uniform(low, high):
        return low + (high - low) * simple_random()

    @staticmethod
    def normal(mean, std):
        # Simple normal approximation
        return mean + std * (simple_random() - 0.5) * 2

    @staticmethod
    def sqrt(x):
        return simple_math_sqrt(x)

    @staticmethod
    def choice(choices, size=None, p=None):  # noqa: ARG004
        if size is None:
            return choices[int(simple_random() * len(choices))]
        return [choices[int(simple_random() * len(choices))] for _ in range(size)]


# Mock pandas for demonstration
class MockPandas:
    @staticmethod
    def DataFrame(data):
        return MockDataFrame(data)

    @staticmethod
    def Series(data, index=None):
        return MockSeries(data, index)


class MockDataFrame:
    def __init__(self, data):
        self.data = data
        self.columns = list(data[0].keys()) if data else []

    def pivot(self, index, columns, values):
        return MockPivotTable(self.data, index, columns, values)

    def to_string(self, index=False, float_format="%.2f"):
        if not self.data:
            return "Empty DataFrame"

        # Simple table formatting
        lines = []
        if index:
            lines.append("Index | " + " | ".join(self.columns))
            lines.append("-" * (len(lines[0]) + 10))

        for i, row in enumerate(self.data):
            if index:
                line = f"{i:5d} | "
            else:
                line = ""
            line += " | ".join(f"{row.get(col, ''):.2f}" for col in self.columns)
            lines.append(line)

        return "\n".join(lines)


class MockSeries:
    def __init__(self, data, index=None):
        self.data = data
        self.index = index or list(range(len(data)))

    def rolling(self, window):
        return MockRolling(self.data, window)

    def mean(self):
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)

    def std(self):
        if not self.data:
            return 0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return simple_math_sqrt(variance)


class MockRolling:
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def mean(self):
        if len(self.data) < self.window:
            return [0] * len(self.data)

        result = []
        for i in range(len(self.data)):
            if i < self.window - 1:
                result.append(0)
            else:
                window_data = self.data[i - self.window + 1 : i + 1]
                result.append(sum(window_data) / len(window_data))
        return result


class MockPivotTable:
    def __init__(self, data, index, columns, values):
        self.data = data
        self.index = index
        self.columns = columns
        self.values = values

    def to_string(self, float_format="%.2f"):
        if not self.data:
            return "Empty PivotTable"

        # Simple pivot table formatting
        lines = []
        lines.append(f"{self.index} | " + " | ".join(self.columns))
        lines.append("-" * (len(lines[0]) + 10))

        # Group by index
        grouped = {}
        for row in self.data:
            idx_val = row.get(self.index, "")
            col_val = row.get(self.columns, "")
            val = row.get(self.values, 0)

            if idx_val not in grouped:
                grouped[idx_val] = {}
            grouped[idx_val][col_val] = val

        for idx_val in sorted(grouped.keys()):
            line = f"{idx_val} | "
            line += " | ".join(f"{grouped[idx_val].get(col, 0):.2f}" for col in self.columns)
            lines.append(line)

        return "\n".join(lines)


# Mock scipy for demonstration
class MockScipy:
    @staticmethod
    def stats():
        return MockStats()


class MockStats:
    pass


# Replace imports with mocks
sys.modules["numpy"] = MockNumpy()
sys.modules["pandas"] = MockPandas()
sys.modules["scipy"] = MockScipy()

# Now import our transaction cost classes
from trading_rl_agent.portfolio.transaction_costs import (
    BrokerType,
    FlatRateCommission,
    MarketCondition,
    MarketData,
    OrderType,
    PerShareCommission,
    TieredCommission,
    TransactionCostModel,
)


def create_sample_market_data(
    price=100.0, volatility=0.02, avg_daily_volume=1000000, market_cap=1000000000, sector="Technology"
):
    """Create sample market data for testing."""
    spread = 0.0002  # 2 bps spread
    bid = price * (1 - spread / 2)
    ask = price * (1 + spread / 2)

    return MarketData(
        timestamp=datetime.now(),
        bid=bid,
        ask=ask,
        mid_price=price,
        volume=avg_daily_volume * 0.1,  # 10% of daily volume
        volatility=volatility,
        avg_daily_volume=avg_daily_volume,
        market_cap=market_cap,
        sector=sector,
    )


def demonstrate_broker_types():
    """Demonstrate different broker types and their cost structures."""
    print("=" * 60)
    print("BROKER TYPE COMPARISON")
    print("=" * 60)

    # Create sample market data
    market_data = create_sample_market_data(price=100.0)
    trade_value = 50000  # $50k trade
    quantity = 500  # 500 shares

    broker_types = [
        BrokerType.RETAIL,
        BrokerType.INSTITUTIONAL,
        BrokerType.DISCOUNT,
        BrokerType.PREMIUM,
        BrokerType.CRYPTO,
    ]

    results = []
    for broker_type in broker_types:
        # Create broker-specific cost model
        cost_model = TransactionCostModel.create_broker_model(broker_type)

        # Calculate costs
        cost_breakdown = cost_model.calculate_total_cost(
            trade_value=trade_value,
            quantity=quantity,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        results.append(
            {
                "broker_type": broker_type.value,
                "total_cost": cost_breakdown["total_cost"],
                "cost_pct": cost_breakdown["cost_pct"] * 100,
                "commission": cost_breakdown["commission"],
                "slippage": cost_breakdown["slippage"],
                "market_impact": cost_breakdown["market_impact"],
                "spread_cost": cost_breakdown["spread_cost"],
            }
        )

    # Display results
    df = MockPandas.DataFrame(results)
    print(df.to_string(index=False, float_format="%.2f"))
    print()


def demonstrate_market_conditions():
    """Demonstrate how market conditions affect transaction costs."""
    print("=" * 60)
    print("MARKET CONDITION IMPACT")
    print("=" * 60)

    # Create sample market data
    market_data = create_sample_market_data(price=100.0)
    trade_value = 100000  # $100k trade
    quantity = 1000  # 1000 shares

    # Use premium broker model for better market condition handling
    cost_model = TransactionCostModel.create_broker_model(BrokerType.PREMIUM)

    market_conditions = [
        MarketCondition.NORMAL,
        MarketCondition.VOLATILE,
        MarketCondition.LIQUID,
        MarketCondition.ILLIQUID,
        MarketCondition.CRISIS,
    ]

    results = []
    for condition in market_conditions:
        cost_breakdown = cost_model.calculate_total_cost(
            trade_value=trade_value,
            quantity=quantity,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=condition,
        )

        results.append(
            {
                "market_condition": condition.value,
                "total_cost": cost_breakdown["total_cost"],
                "cost_pct": cost_breakdown["cost_pct"] * 100,
                "condition_multiplier": cost_breakdown["condition_multiplier"],
            }
        )

    # Display results
    df = MockPandas.DataFrame(results)
    print(df.to_string(index=False, float_format="%.4f"))
    print()


def demonstrate_commission_structures():
    """Demonstrate different commission structures."""
    print("=" * 60)
    print("COMMISSION STRUCTURE COMPARISON")
    print("=" * 60)

    # Test different trade values
    trade_values = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    quantity = 1000  # Fixed quantity for comparison

    commission_structures = [
        ("Flat Rate (0.1%)", FlatRateCommission(rate=0.001)),
        ("Tiered", TieredCommission()),
        ("Per Share ($0.005)", PerShareCommission(rate_per_share=0.005)),
    ]

    results = []
    for trade_value in trade_values:
        for structure_name, structure in commission_structures:
            commission = structure.calculate_commission(trade_value, quantity)
            commission_pct = (commission / trade_value) * 100

            results.append(
                {
                    "trade_value": f"${trade_value:,}",
                    "structure": structure_name,
                    "commission": commission,
                    "commission_pct": commission_pct,
                }
            )

    # Display results
    df = MockPandas.DataFrame(results)
    pivot_df = df.pivot(index="trade_value", columns="structure", values="commission_pct")
    print(pivot_df.to_string(float_format="%.4f"))
    print()


def demonstrate_cost_optimization():
    """Demonstrate cost optimization recommendations."""
    print("=" * 60)
    print("COST OPTIMIZATION ANALYSIS")
    print("=" * 60)

    # Create a cost model and simulate some trades
    cost_model = TransactionCostModel.create_broker_model(BrokerType.RETAIL)

    # Simulate various trades to build up history
    market_data = create_sample_market_data(price=100.0)

    # Simulate different trade scenarios
    trade_scenarios = [
        (1000, 100.0, OrderType.MARKET, MarketCondition.NORMAL),  # Small trade
        (5000, 100.0, OrderType.MARKET, MarketCondition.NORMAL),  # Medium trade
        (10000, 100.0, OrderType.MARKET, MarketCondition.VOLATILE),  # Large trade, volatile
        (20000, 100.0, OrderType.LIMIT, MarketCondition.ILLIQUID),  # Very large trade, illiquid
        (5000, 100.0, OrderType.MARKET, MarketCondition.CRISIS),  # Medium trade, crisis
    ]

    print("Simulating trades for cost analysis...")
    for quantity, price, order_type, market_condition in trade_scenarios:
        # Update market data
        market_data.mid_price = price
        market_data.bid = price * 0.999
        market_data.ask = price * 1.001

        # Simulate execution
        execution_result = cost_model.simulate_execution(
            requested_quantity=quantity,
            market_data=market_data,
            order_type=order_type,
            market_condition=market_condition,
        )

        print(f"  Trade: {quantity} shares at ${price:.2f} ({order_type.value}, {market_condition.value})")
        print(f"    Executed: {execution_result.executed_quantity} shares at ${execution_result.executed_price:.2f}")
        print(
            f"    Cost: ${execution_result.total_cost:.2f} ({(execution_result.total_cost / (quantity * price) * 100):.3f}%)"
        )
        print(f"    Delay: {execution_result.delay_seconds:.2f}s")
        print()

    # Generate optimization recommendations
    recommendations = cost_model.generate_optimization_recommendations()

    print("Cost Optimization Recommendations:")
    print("-" * 40)

    if not recommendations:
        print("No specific recommendations at this time.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.recommendation_type.replace('_', ' ').title()}")
            print(f"   Description: {rec.description}")
            print(f"   Expected Savings: ${rec.expected_savings:,.2f}")
            print(f"   Confidence: {rec.confidence:.1%}")
            print(f"   Priority: {rec.priority.upper()}")
            print(f"   Implementation: {rec.implementation_difficulty}")
            print()

    # Get cost summary
    summary = cost_model.get_cost_summary()
    print("Cost Summary:")
    print("-" * 40)
    print(f"Total Trades: {summary['num_trades']}")
    print(f"Total Transaction Costs: ${summary['total_transaction_costs']:,.2f}")
    print(f"Average Cost per Trade: ${summary['avg_cost_per_trade']:,.2f}")
    print(f"Average Delay per Trade: {summary['avg_delay_per_trade']:.2f} seconds")


def main():
    """Run all demonstrations."""
    print("TRANSACTION COST MODELING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()

    try:
        # Run demonstrations
        demonstrate_broker_types()
        demonstrate_market_conditions()
        demonstrate_commission_structures()
        demonstrate_cost_optimization()

        print("=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("The transaction cost modeling system provides:")
        print("✅ Realistic bid-ask spread modeling")
        print("✅ Configurable commission structures for different brokers")
        print("✅ Market impact modeling based on order size")
        print("✅ Slippage modeling for different market conditions")
        print("✅ Execution delay simulation")
        print("✅ Partial fill simulation")
        print("✅ Cost optimization recommendations")
        print("✅ Integration with portfolio management and backtesting")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
