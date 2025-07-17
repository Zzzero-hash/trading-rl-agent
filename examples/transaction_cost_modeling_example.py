#!/usr/bin/env python3
"""
Comprehensive Transaction Cost Modeling Example

This example demonstrates the new transaction cost modeling system with:
- Different broker types and commission structures
- Market impact modeling based on order size
- Slippage modeling for different market conditions
- Execution delay simulation
- Partial fill simulation
- Cost optimization recommendations
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl_agent.portfolio.transaction_costs import (
    AdaptiveImpactModel,
    BrokerType,
    ConstantDelayModel,
    ConstantSlippageModel,
    FlatRateCommission,
    LinearImpactModel,
    MarketCondition,
    MarketConditionDelayModel,
    MarketData,
    OrderType,
    PerShareCommission,
    SizeBasedDelayModel,
    SpreadBasedSlippageModel,
    SquareRootImpactModel,
    TieredCommission,
    TransactionCostAnalyzer,
    TransactionCostModel,
    VolumeBasedSlippageModel,
)


def create_sample_market_data(
    price: float = 100.0,
    volatility: float = 0.02,
    avg_daily_volume: float = 1000000,
    market_cap: float = 1000000000,
    sector: str = "Technology",
) -> MarketData:
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
    df = pd.DataFrame(results)
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
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.4f"))
    print()


def demonstrate_order_types():
    """Demonstrate how different order types affect execution costs."""
    print("=" * 60)
    print("ORDER TYPE COMPARISON")
    print("=" * 60)

    # Create sample market data
    market_data = create_sample_market_data(price=100.0)
    trade_value = 75000  # $75k trade
    quantity = 750  # 750 shares

    # Use institutional broker model
    cost_model = TransactionCostModel.create_broker_model(BrokerType.INSTITUTIONAL)

    order_types = [
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP,
        OrderType.TWAP,
        OrderType.VWAP,
    ]

    results = []
    for order_type in order_types:
        # Simulate execution
        execution_result = cost_model.simulate_execution(
            requested_quantity=quantity,
            market_data=market_data,
            order_type=order_type,
            market_condition=MarketCondition.NORMAL,
        )

        results.append(
            {
                "order_type": order_type.value,
                "executed_quantity": execution_result.executed_quantity,
                "executed_price": execution_result.executed_price,
                "total_cost": execution_result.total_cost,
                "delay_seconds": execution_result.delay_seconds,
                "success": execution_result.success,
                "fill_rate": execution_result.executed_quantity / quantity if quantity > 0 else 0,
            }
        )

    # Display results
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.2f"))
    print()


def demonstrate_market_impact_models():
    """Demonstrate different market impact models."""
    print("=" * 60)
    print("MARKET IMPACT MODEL COMPARISON")
    print("=" * 60)

    # Create sample market data
    market_data = create_sample_market_data(price=100.0, avg_daily_volume=500000)

    # Test different order sizes
    order_sizes = [10000, 50000, 100000, 250000, 500000]  # $10k to $500k

    impact_models = [
        ("Linear", LinearImpactModel()),
        ("Square Root", SquareRootImpactModel()),
        ("Adaptive", AdaptiveImpactModel()),
    ]

    results = []
    for order_size in order_sizes:
        for model_name, model in impact_models:
            impact = model.calculate_impact(order_size, market_data)
            impact_pct = (impact / order_size) * 100

            results.append(
                {
                    "order_size": f"${order_size:,}",
                    "model": model_name,
                    "impact": impact,
                    "impact_pct": impact_pct,
                }
            )

    # Display results
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="order_size", columns="model", values="impact_pct")
    print(pivot_df.to_string(float_format="%.4f"))
    print()


def demonstrate_slippage_models():
    """Demonstrate different slippage models."""
    print("=" * 60)
    print("SLIPPAGE MODEL COMPARISON")
    print("=" * 60)

    # Create sample market data with different spreads
    market_data_normal = create_sample_market_data(price=100.0, volatility=0.02)
    market_data_wide = create_sample_market_data(price=100.0, volatility=0.05)
    market_data_wide.bid = 99.5
    market_data_wide.ask = 100.5

    trade_value = 100000  # $100k trade

    slippage_models = [
        ("Constant", ConstantSlippageModel()),
        ("Volume Based", VolumeBasedSlippageModel()),
        ("Spread Based", SpreadBasedSlippageModel()),
    ]

    results = []
    for model_name, model in slippage_models:
        # Test with normal spread
        slippage_normal = model.calculate_slippage(trade_value, market_data_normal, OrderType.MARKET)

        # Test with wide spread
        slippage_wide = model.calculate_slippage(trade_value, market_data_wide, OrderType.MARKET)

        results.append(
            {
                "model": model_name,
                "normal_spread_slippage": slippage_normal,
                "wide_spread_slippage": slippage_wide,
                "normal_spread_pct": (slippage_normal / trade_value) * 100,
                "wide_spread_pct": (slippage_wide / trade_value) * 100,
            }
        )

    # Display results
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.4f"))
    print()


def demonstrate_execution_delays():
    """Demonstrate execution delay models."""
    print("=" * 60)
    print("EXECUTION DELAY COMPARISON")
    print("=" * 60)

    # Create sample market data
    market_data = create_sample_market_data(price=100.0, avg_daily_volume=1000000)

    # Test different order sizes
    order_sizes = [10000, 50000, 100000, 500000, 1000000]  # $10k to $1M

    delay_models = [
        ("Constant", ConstantDelayModel()),
        ("Size Based", SizeBasedDelayModel()),
        ("Market Condition", MarketConditionDelayModel()),
    ]

    results = []
    for order_size in order_sizes:
        for model_name, model in delay_models:
            delay = model.calculate_delay(order_size, market_data, OrderType.MARKET)

            results.append(
                {
                    "order_size": f"${order_size:,}",
                    "model": model_name,
                    "delay_seconds": delay,
                }
            )

    # Display results
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="order_size", columns="model", values="delay_seconds")
    print(pivot_df.to_string(float_format="%.2f"))
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

    # Generate detailed cost report
    analyzer = TransactionCostAnalyzer(cost_model)
    cost_report = analyzer.generate_cost_report()

    print("Detailed Cost Analysis Report:")
    print("-" * 40)
    print(cost_report)


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
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="trade_value", columns="structure", values="commission_pct")
    print(pivot_df.to_string(float_format="%.4f"))
    print()


def main():
    """Run all demonstrations."""
    print("TRANSACTION COST MODELING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()

    try:
        # Run all demonstrations
        demonstrate_broker_types()
        demonstrate_market_conditions()
        demonstrate_order_types()
        demonstrate_market_impact_models()
        demonstrate_slippage_models()
        demonstrate_execution_delays()
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
