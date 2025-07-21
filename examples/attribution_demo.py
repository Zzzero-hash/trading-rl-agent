#!/usr/bin/env python3
"""
Performance Attribution Demo

Comprehensive demonstration of performance attribution analysis:
- Factor analysis and decomposition
- Brinson attribution for sector allocation
- Risk attribution and decomposition
- Visualization and reporting
- Automated workflow integration

Usage:
    python attribution_demo.py [--output-dir OUTPUT_DIR]
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from trade_agent.portfolio.attribution import (
        AttributionConfig,
        PerformanceAttributor,
    )
    from trade_agent.portfolio.attribution_integration import (
        AutomatedAttributionWorkflow,
    )
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

if TYPE_CHECKING:
    from collections.abc import Callable


def create_sample_data() -> dict[str, Any]:
    """Create realistic sample data for attribution analysis."""
    print("Creating sample data...")

    # Date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Asset universe
    symbols = [
        "AAPL",
        "GOOGL",
        "MSFT",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "NFLX",
        "CRM",
        "ADBE",
    ]
    sectors = [
        "Technology",
        "Technology",
        "Technology",
        "Consumer",
        "Technology",
        "Technology",
        "Technology",
        "Consumer",
        "Technology",
        "Technology",
    ]

    # Create sector mapping
    sector_data = pd.DataFrame({"sector": sectors}, index=symbols)

    # Generate realistic asset returns with sector effects
    np.random.seed(42)  # For reproducible results

    # Base returns for each sector
    sector_returns = {
        "Technology": 0.0008,  # Higher growth for tech
        "Consumer": 0.0005,  # Moderate growth for consumer
    }

    # Generate asset returns with sector effects and idiosyncratic components
    asset_returns = pd.DataFrame(index=symbols, columns=dates)

    for symbol in symbols:
        sector = sector_data.loc[symbol, "sector"]
        base_return = sector_returns[sector]

        # Add sector effect, market effect, and idiosyncratic noise
        for date in dates:
            # Market effect (correlated across all assets)
            market_effect = np.random.normal(0.0002, 0.01)

            # Sector effect (correlated within sector)
            sector_effect = np.random.normal(base_return, 0.005)

            # Idiosyncratic effect (unique to each asset)
            idiosyncratic = np.random.normal(0, 0.008)

            # Combine effects
            total_return = market_effect + sector_effect + idiosyncratic
            asset_returns.loc[symbol, date] = total_return

    # Create portfolio weights (active management)
    portfolio_weights = pd.DataFrame(index=symbols, columns=dates)

    # Start with equal weights
    initial_weights = np.ones(len(symbols)) / len(symbols)

    for i, date in enumerate(dates):
        if i == 0:
            # Initial weights
            portfolio_weights.loc[:, date] = initial_weights
        else:
            # Simulate active management with some drift
            prev_weights = portfolio_weights.iloc[:, i - 1]

            # Add some active bets (overweight tech, underweight consumer)
            active_bets = np.zeros(len(symbols))
            for j, symbol in enumerate(symbols):
                sector = sector_data.loc[symbol, "sector"]
                if sector == "Technology":
                    active_bets[j] = 0.02  # Overweight tech
                else:
                    active_bets[j] = -0.02  # Underweight consumer

            # Apply active bets with some rebalancing
            new_weights = prev_weights + active_bets * 0.1

            # Normalize weights
            new_weights = np.maximum(new_weights, 0)  # No short positions
            new_weights = new_weights / new_weights.sum()

            portfolio_weights.loc[:, date] = new_weights

    # Create benchmark weights (market cap weighted)
    benchmark_weights = pd.DataFrame(index=symbols, columns=dates)

    # Simulate market cap weights with some variation
    market_caps = np.array([2.5, 1.8, 2.2, 1.5, 0.8, 0.9, 1.2, 0.3, 0.2, 0.3])  # Trillions
    market_cap_weights = market_caps / market_caps.sum()

    for date in dates:
        # Add some variation to market cap weights
        variation = np.random.normal(0, 0.01, len(symbols))
        weights = market_cap_weights + variation
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        benchmark_weights.loc[:, date] = weights

    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=dates, dtype=float)
    for date in dates:
        weights = portfolio_weights.loc[:, date]
        returns = asset_returns.loc[:, date]
        portfolio_returns[date] = (weights * returns).sum()

    # Calculate benchmark returns
    benchmark_returns = pd.Series(index=dates, dtype=float)
    for date in dates:
        weights = benchmark_weights.loc[:, date]
        returns = asset_returns.loc[:, date]
        benchmark_returns[date] = (weights * returns).sum()

    return {
        "asset_returns": asset_returns,
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "sector_data": sector_data,
        "dates": dates,
        "symbols": symbols,
    }


def demo_factor_analysis(attributor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate factor analysis capabilities."""
    print("\n" + "=" * 60)
    print("FACTOR ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Run factor analysis
    results: dict[str, Any] = attributor.analyze_performance(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        asset_returns=data["asset_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_data=data["sector_data"],
    )

    # Display factor attribution results
    factor_results = results.get("factor_attribution", {})

    if factor_results:
        print("\nFactor Attribution Summary:")
        print("-" * 40)
        for factor, contribution in factor_results.items():
            print(f"{factor:20s}: {contribution:8.6f}")

        # Calculate systematic vs idiosyncratic volatility
        systematic_vol = np.sqrt(sum(contrib**2 for contrib in factor_results.values()))
        idiosyncratic_vol = results.get("idiosyncratic_volatility", 0.0)

        print("\nReturn Decomposition:")
        print("-" * 40)
        print(f"Systematic Volatility:   {systematic_vol:.4f}")
        print(f"Idiosyncratic Volatility: {idiosyncratic_vol:.4f}")
        print(f"Total Volatility:        {systematic_vol + idiosyncratic_vol:.4f}")

    return results


def demo_brinson_attribution(attributor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate Brinson attribution analysis."""
    print("\n" + "=" * 60)
    print("BRINSON ATTRIBUTION DEMONSTRATION")
    print("=" * 60)

    # Run attribution analysis
    results: dict[str, Any] = attributor.analyze_performance(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        asset_returns=data["asset_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_data=data["sector_data"],
    )

    # Display Brinson attribution results
    brinson_results = results.get("brinson_attribution", {})

    if brinson_results:
        print("\nSector Attribution Summary:")
        print("-" * 40)

        # Aggregate results over time
        allocation_effects = []
        selection_effects = []
        interaction_effects = []

        for attribution in brinson_results.values():
            allocation_effects.append(attribution["allocation"])
            selection_effects.append(attribution["selection"])
            interaction_effects.append(attribution["interaction"])

        print(f"Average Allocation Effect: {np.mean(allocation_effects):.6f}")
        print(f"Average Selection Effect:  {np.mean(selection_effects):.6f}")
        print(f"Average Interaction Effect: {np.mean(interaction_effects):.6f}")
        total_attribution = np.mean(allocation_effects) + np.mean(selection_effects) + np.mean(interaction_effects)
        print(f"Total Attribution:         {total_attribution:.6f}")

    return results


def demo_risk_analysis(attributor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate risk-adjusted analysis."""
    print("\n" + "=" * 60)
    print("RISK-ADJUSTED ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Run risk analysis
    results: dict[str, Any] = attributor.analyze_performance(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        asset_returns=data["asset_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_data=data["sector_data"],
    )

    # Display risk metrics
    risk_results = results.get("risk_analysis", {})

    print("\nRisk Metrics:")
    print("-" * 40)
    print(f"Portfolio Sharpe Ratio:     {risk_results.get('portfolio_sharpe', 0):.4f}")
    print(f"Benchmark Sharpe Ratio:     {risk_results.get('benchmark_sharpe', 0):.4f}")
    print(f"Information Ratio:          {risk_results.get('information_ratio', 0):.4f}")
    print(f"Portfolio Volatility:       {risk_results.get('portfolio_volatility', 0):.4f}")
    print(f"Benchmark Volatility:       {risk_results.get('benchmark_volatility', 0):.4f}")
    print(f"Tracking Error:             {risk_results.get('tracking_error', 0):.4f}")
    print(f"Maximum Drawdown:           {risk_results.get('max_drawdown', 0):.4f}")
    print(f"Value at Risk (95%):        {risk_results.get('var_95', 0):.4f}")

    return results


def demo_visualization(attributor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)

    # Run analysis
    results: dict[str, Any] = attributor.analyze_performance(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        asset_returns=data["asset_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_data=data["sector_data"],
    )

    # Create output directory
    output_dir = Path("attribution_output")
    output_dir.mkdir(exist_ok=True)

    # Create additional plots
    create_additional_plots(data, results, output_dir)

    print(f"\nVisualizations saved to: {output_dir}")
    print("Generated files:")
    for file_path in output_dir.glob("*.html"):
        print(f"  - {file_path}")

    return results


def create_additional_plots(data: dict[str, Any], results: dict[str, Any], output_dir: Path) -> None:
    """Create additional visualization plots."""
    # Create cumulative returns plot
    portfolio_cumulative = (1 + data["portfolio_returns"]).cumprod()
    benchmark_cumulative = (1 + data["benchmark_returns"]).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Cumulative returns
    axes[0, 0].plot(
        portfolio_cumulative.index,
        portfolio_cumulative.values,
        label="Portfolio",
        linewidth=2,
    )
    axes[0, 0].plot(
        benchmark_cumulative.index,
        benchmark_cumulative.values,
        label="Benchmark",
        linewidth=2,
    )
    axes[0, 0].set_title("Cumulative Returns")
    axes[0, 0].set_ylabel("Cumulative Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Rolling volatility
    rolling_vol = data["portfolio_returns"].rolling(30).std() * np.sqrt(252)
    axes[0, 1].plot(rolling_vol.index, rolling_vol.values, label="Portfolio Volatility", linewidth=2)
    axes[0, 1].set_title("30-Day Rolling Volatility")
    axes[0, 1].set_ylabel("Annualized Volatility")
    axes[0, 1].grid(True, alpha=0.3)

    # Factor contributions
    if "factor_attribution" in results:
        factors = list(results["factor_attribution"].keys())
        contributions = list(results["factor_attribution"].values())
        axes[1, 0].bar(factors, contributions)
        axes[1, 0].set_title("Factor Contributions")
        axes[1, 0].set_ylabel("Contribution")
        axes[1, 0].tick_params(axis="x", rotation=45)

    # Sector weights over time
    sector_weights = data["portfolio_weights"].groupby(data["sector_data"]["sector"]).sum()
    sector_weights.T.plot(ax=axes[1, 1], kind="area", stacked=True)
    axes[1, 1].set_title("Sector Weights Over Time")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "attribution_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create interactive plot with plotly if available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create interactive cumulative returns plot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Cumulative Returns",
                "Rolling Volatility",
                "Factor Contributions",
                "Sector Weights",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                name="Portfolio",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name="Benchmark",
            ),
            row=1,
            col=1,
        )

        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Portfolio Volatility"),
            row=1,
            col=2,
        )

        # Factor contributions
        if "factor_attribution" in results:
            fig.add_trace(
                go.Bar(x=factors, y=contributions, name="Factor Contributions"),
                row=2,
                col=1,
            )

        # Sector weights
        for sector in sector_weights.index:
            fig.add_trace(
                go.Scatter(
                    x=sector_weights.columns,
                    y=sector_weights.loc[sector],
                    name=sector,
                    stackgroup="sectors",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(height=800, title_text="Attribution Analysis Dashboard")
        fig.write_html(output_dir / "attribution_dashboard.html")

    except ImportError:
        print("Plotly not available, skipping interactive plots")


def demo_automation(data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate automation capabilities."""
    print("\n" + "=" * 60)
    print("AUTOMATION DEMONSTRATION")
    print("=" * 60)

    # Create automated workflow

    # Mock portfolio manager for demonstration
    class MockPortfolioManager:
        def __init__(self) -> None:
            self.config = type("Config", (), {"benchmark_symbol": "SPY"})()
            self.performance_history = pd.DataFrame(
                {
                    "timestamp": data["dates"],
                    "portfolio_value": (1 + data["portfolio_returns"]).cumprod() * 1000000,
                    "benchmark_value": (1 + data["benchmark_returns"]).cumprod() * 1000000,
                }
            )

    portfolio_manager = MockPortfolioManager()
    workflow = AutomatedAttributionWorkflow(portfolio_manager)

    # Simulate scheduled analysis
    print("Running scheduled analysis...")
    results = workflow.run_scheduled_analysis()
    print("✓ Scheduled analysis completed")

    # Simulate portfolio rebalance trigger
    print("\nSimulating portfolio rebalance...")
    rebalance_results = workflow.on_portfolio_rebalance()
    print("✓ Rebalance analysis completed")

    # Simulate performance milestone
    print("\nSimulating performance milestone...")
    milestone_results = workflow.on_performance_milestone("quarterly")
    print("✓ Milestone analysis completed")

    return {
        "scheduled": results,
        "rebalance": rebalance_results,
        "milestone": milestone_results,
    }


def demo_reporting(attributor: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate automated reporting capabilities."""
    print("\n" + "=" * 60)
    print("REPORTING DEMONSTRATION")
    print("=" * 60)

    # Run analysis
    results: dict[str, Any] = attributor.analyze_performance(
        portfolio_returns=data["portfolio_returns"],
        benchmark_returns=data["benchmark_returns"],
        asset_returns=data["asset_returns"],
        portfolio_weights=data["portfolio_weights"],
        benchmark_weights=data["benchmark_weights"],
        sector_data=data["sector_data"],
    )

    # Create output directory
    output_dir = Path("attribution_reports")
    output_dir.mkdir(exist_ok=True)

    # Generate comprehensive report
    report_path = output_dir / "comprehensive_report.html"
    report_content = attributor.generate_report(str(report_path))

    print(f"\nComprehensive report generated: {report_path}")
    print(f"Report length: {len(report_content)} characters")

    # Export data to Excel
    data_path = output_dir / "attribution_data.xlsx"
    with pd.ExcelWriter(data_path) as writer:
        data["asset_returns"].to_excel(writer, sheet_name="Asset Returns")
        data["portfolio_weights"].to_excel(writer, sheet_name="Portfolio Weights")
        data["benchmark_weights"].to_excel(writer, sheet_name="Benchmark Weights")
        data["portfolio_returns"].to_excel(writer, sheet_name="Portfolio Returns")
        data["benchmark_returns"].to_excel(writer, sheet_name="Benchmark Returns")
        data["sector_data"].to_excel(writer, sheet_name="Sector Data")

    print(f"Data exported to: {data_path}")

    return results


def main() -> None:
    """Main demonstration function."""
    print("Portfolio Attribution Analysis Demonstration")
    print("=" * 60)

    # Create sample data
    data = create_sample_data()

    # Initialize attributor

    config = AttributionConfig()
    attributor = PerformanceAttributor(config)

    # Run demonstrations
    demos: list[Callable] = [
        demo_factor_analysis,
        demo_brinson_attribution,
        demo_risk_analysis,
        demo_visualization,
        demo_automation,
        demo_reporting,
    ]

    for demo in demos:
        try:
            if demo == demo_automation:
                demo(data)
            else:
                demo(attributor, data)
        except Exception as e:
            print(f"Demo failed: {e}")
            continue

    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("Check the output directories for generated files:")
    print("  - attribution_output/")
    print("  - attribution_reports/")


if __name__ == "__main__":
    main()
