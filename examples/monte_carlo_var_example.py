"""
Comprehensive example demonstrating advanced Monte Carlo VaR capabilities.

This example showcases:
1. Historical simulation with bootstrapping
2. Parametric VaR with multiple distribution assumptions
3. Monte Carlo simulation with correlated asset movements
4. Stress testing scenarios for extreme market conditions
5. Parallel processing for large simulations
6. VaR backtesting and validation
7. Real-time VaR monitoring setup
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trade_agent.risk.monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig
from trade_agent.risk.parallel_var import ParallelVaRCalculator, ParallelVaRConfig

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def generate_sample_data(n_days: int = 1000, n_assets: int = 10) -> pd.DataFrame:
    """
    Generate sample asset returns data for demonstration.

    Args:
        n_days: Number of trading days
        n_assets: Number of assets

    Returns:
        DataFrame with asset returns
    """
    np.random.seed(42)  # For reproducible results

    # Generate correlated returns using a factor model
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Market factor
    market_factor = np.random.normal(0, 0.02, n_days)

    # Asset-specific factors and betas
    asset_returns = {}
    for i in range(n_assets):
        asset_name = f"ASSET_{i + 1:02d}"

        # Asset-specific beta to market
        beta = np.random.uniform(0.5, 1.5)

        # Idiosyncratic risk
        idiosyncratic = np.random.normal(0, 0.015, n_days)

        # Generate returns with some correlation structure
        returns = beta * market_factor + idiosyncratic

        # Add some fat tails (t-distribution characteristics)
        if np.random.random() > 0.7:  # 30% chance of fat tails
            returns = np.random.standard_t(df=3, size=n_days) * 0.02

        asset_returns[asset_name] = returns

    returns_df = pd.DataFrame(asset_returns, index=dates)

    # Add some extreme events (market crashes)
    crash_dates = [100, 300, 500, 700, 900]
    for crash_date in crash_dates:
        if crash_date < len(returns_df):
            returns_df.iloc[crash_date] *= -3  # 3x negative shock

    return returns_df


def demonstrate_basic_var_calculations() -> None:
    """Demonstrate basic VaR calculation methods."""
    print("=" * 60)
    print("BASIC VaR CALCULATIONS")
    print("=" * 60)

    # Generate sample data
    returns_data = generate_sample_data(n_days=500, n_assets=5)
    print(f"Generated {len(returns_data)} days of data for {len(returns_data.columns)} assets")

    # Define portfolio weights
    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    # Configure VaR calculator
    var_config = MonteCarloVaRConfig(
        n_simulations=5000,
        confidence_level=0.05,  # 95% VaR
        time_horizon=1,
        lookback_period=252,
        distribution_type="normal",
    )

    # Initialize VaR calculator
    var_calculator = MonteCarloVaR(var_config)
    var_calculator.update_data(returns_data)

    # 1. Historical Simulation VaR
    print("\n1. Historical Simulation VaR:")
    hist_var = var_calculator.historical_simulation_var(weights, use_bootstrap=True)
    print(f"   VaR: {hist_var.var_value:.4f}")
    print(f"   CVaR: {hist_var.cvar_value:.4f}")
    print(f"   Method: {hist_var.method}")
    print(f"   Calculation time: {hist_var.calculation_time:.3f}s")

    # 2. Parametric VaR with different distributions
    print("\n2. Parametric VaR (Normal Distribution):")
    param_var_normal = var_calculator.parametric_var(weights, distribution="normal")
    print(f"   VaR: {param_var_normal.var_value:.4f}")
    print(f"   CVaR: {param_var_normal.cvar_value:.4f}")
    print(f"   Portfolio mean: {param_var_normal.additional_metrics['portfolio_mean']:.6f}")
    print(f"   Portfolio std: {param_var_normal.additional_metrics['portfolio_std']:.6f}")

    print("\n3. Parametric VaR (t-Distribution):")
    param_var_t = var_calculator.parametric_var(weights, distribution="t")
    print(f"   VaR: {param_var_t.var_value:.4f}")
    print(f"   CVaR: {param_var_t.cvar_value:.4f}")

    # 3. Monte Carlo VaR
    print("\n4. Monte Carlo VaR:")
    mc_var = var_calculator.monte_carlo_var(weights, use_correlation=True)
    print(f"   VaR: {mc_var.var_value:.4f}")
    print(f"   CVaR: {mc_var.cvar_value:.4f}")
    print(f"   Simulations: {mc_var.simulation_count}")
    print(f"   Correlation used: {mc_var.additional_metrics['correlation_used']}")


def demonstrate_stress_testing() -> None:
    """Demonstrate stress testing capabilities."""
    print("\n" + "=" * 60)
    print("STRESS TESTING")
    print("=" * 60)

    # Generate data
    returns_data = generate_sample_data(n_days=500, n_assets=5)

    # Define weights
    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    # Configure VaR calculator with custom stress scenarios
    custom_stress_scenarios = {
        "mild_correction": {
            "volatility_multiplier": 1.5,
            "correlation_increase": 0.1,
            "mean_shift": -0.005,
        },
        "severe_crash": {
            "volatility_multiplier": 4.0,
            "correlation_increase": 0.4,
            "mean_shift": -0.03,
        },
        "flash_crash": {
            "volatility_multiplier": 6.0,
            "correlation_increase": 0.6,
            "mean_shift": -0.08,
        },
    }

    var_config = MonteCarloVaRConfig(
        n_simulations=3000,
        confidence_level=0.05,
        stress_scenarios=custom_stress_scenarios,
    )

    var_calculator = MonteCarloVaR(var_config)
    var_calculator.update_data(returns_data)

    # Perform stress testing
    print("Performing stress testing...")
    stress_results = var_calculator.stress_test_var(weights)

    print("\nStress Test Results:")
    print("-" * 40)

    # Calculate baseline VaR for comparison
    baseline_var = var_calculator.monte_carlo_var(weights)
    print(f"Baseline VaR: {baseline_var.var_value:.4f}")

    for scenario, result in stress_results.items():
        var_increase = (result.var_value - baseline_var.var_value) / baseline_var.var_value * 100
        print(f"{scenario.replace('_', ' ').title()}:")
        print(f"  VaR: {result.var_value:.4f} ({var_increase:+.1f}%)")
        print(f"  CVaR: {result.cvar_value:.4f}")


def demonstrate_parallel_processing() -> None:
    """Demonstrate parallel processing capabilities."""
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING")
    print("=" * 60)

    # Generate larger dataset for parallel processing demo
    returns_data = generate_sample_data(n_days=1000, n_assets=10)

    weights = {f"ASSET_{i + 1:02d}": 1.0 / 10 for i in range(10)}

    # Configure parallel processing
    parallel_config = ParallelVaRConfig(
        use_multiprocessing=True,
        n_processes=4,
        n_threads=2,
        chunk_size=2000,
        memory_limit_gb=2.0,
    )

    var_config = MonteCarloVaRConfig(
        n_simulations=20000,  # Large simulation for parallel processing
        confidence_level=0.05,
        time_horizon=1,
    )

    print("Running parallel Monte Carlo VaR calculation...")

    with ParallelVaRCalculator(parallel_config) as parallel_calc:
        # Parallel Monte Carlo VaR
        start_time = datetime.now()
        parallel_result = parallel_calc.parallel_monte_carlo_var(var_config, returns_data, weights)
        end_time = datetime.now()

        print(f"Parallel calculation completed in {end_time - start_time}")
        print(f"VaR: {parallel_result.var_value:.4f}")
        print(f"CVaR: {parallel_result.cvar_value:.4f}")
        print(f"Total simulations: {parallel_result.simulation_count}")
        print(f"Number of chunks: {parallel_result.additional_metrics['n_chunks']}")

        # Compare with sequential calculation
        print("\nComparing with sequential calculation...")
        sequential_config = MonteCarloVaRConfig(n_simulations=20000, confidence_level=0.05, use_parallel=False)

        sequential_calc = MonteCarloVaR(sequential_config)
        sequential_calc.update_data(returns_data)

        start_time = datetime.now()
        sequential_result = sequential_calc.monte_carlo_var(weights)
        end_time = datetime.now()

        print(f"Sequential calculation completed in {end_time - start_time}")
        print(f"VaR: {sequential_result.var_value:.4f}")
        print(f"CVaR: {sequential_result.cvar_value:.4f}")

        # Performance comparison
        parallel_time = parallel_result.calculation_time
        sequential_time = sequential_result.calculation_time
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        print("\nPerformance Comparison:")
        print(f"Parallel time: {parallel_time:.3f}s")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")


def demonstrate_backtesting() -> None:
    """Demonstrate VaR backtesting capabilities."""
    print("\n" + "=" * 60)
    print("VaR BACKTESTING")
    print("=" * 60)

    # Generate data
    returns_data = generate_sample_data(n_days=800, n_assets=5)

    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    var_config = MonteCarloVaRConfig(n_simulations=2000, confidence_level=0.05, backtest_window=252)

    var_calculator = MonteCarloVaR(var_config)
    var_calculator.update_data(returns_data)

    # Perform backtesting for different methods
    methods = ["monte_carlo", "historical_simulation", "parametric"]

    print("Performing VaR backtesting...")

    for method in methods:
        print(f"\nBacktesting {method.replace('_', ' ').title()}:")
        backtest_results = var_calculator.backtest_var(weights, method)

        print(f"  Total predictions: {backtest_results['total_predictions']}")
        print(f"  Breach count: {backtest_results['breach_count']}")
        print(f"  Breach rate: {backtest_results['breach_rate']:.4f}")
        print(f"  Expected breach rate: {backtest_results['expected_breach_rate']:.4f}")
        print(f"  Kupiec test p-value: {backtest_results['kupiec_pvalue']:.4f}")
        print(f"  Christoffersen test p-value: {backtest_results['christoffersen_pvalue']:.4f}")

        # Interpret results
        if backtest_results["kupiec_pvalue"] > 0.05:
            print("  ✓ Kupiec test: VaR model is accurate")
        else:
            print("  ✗ Kupiec test: VaR model may be inaccurate")

        if backtest_results["christoffersen_pvalue"] > 0.05:
            print("  ✓ Christoffersen test: VaR breaches are independent")
        else:
            print("  ✗ Christoffersen test: VaR breaches may be clustered")


def demonstrate_method_comparison() -> None:
    """Demonstrate comparison of different VaR methods."""
    print("\n" + "=" * 60)
    print("VaR METHOD COMPARISON")
    print("=" * 60)

    # Generate data
    returns_data = generate_sample_data(n_days=500, n_assets=5)

    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    var_config = MonteCarloVaRConfig(n_simulations=5000, confidence_level=0.05)

    var_calculator = MonteCarloVaR(var_config)
    var_calculator.update_data(returns_data)

    # Calculate VaR using different methods
    methods = {
        "Historical Simulation": lambda: var_calculator.historical_simulation_var(weights),
        "Parametric (Normal)": lambda: var_calculator.parametric_var(weights, "normal"),
        "Parametric (t-Dist)": lambda: var_calculator.parametric_var(weights, "t"),
        "Monte Carlo": lambda: var_calculator.monte_carlo_var(weights),
    }

    results = {}

    print("Calculating VaR using different methods...")

    for method_name, method_func in methods.items():
        try:
            result = method_func()
            results[method_name] = result
            print(f"{method_name}: VaR = {result.var_value:.4f}, CVaR = {result.cvar_value:.4f}")
        except Exception as e:
            print(f"{method_name}: Error - {e}")

    # Create comparison table
    print("\nMethod Comparison Summary:")
    print("-" * 80)
    print(f"{'Method':<20} {'VaR':<10} {'CVaR':<10} {'Time (s)':<10} {'Simulations':<12}")
    print("-" * 80)

    for method_name, result in results.items():
        print(
            f"{method_name:<20} {result.var_value:<10.4f} {result.cvar_value:<10.4f} "
            f"{result.calculation_time:<10.3f} {result.simulation_count:<12}"
        )


def demonstrate_real_time_monitoring() -> None:
    """Demonstrate real-time VaR monitoring setup."""
    print("\n" + "=" * 60)
    print("REAL-TIME VaR MONITORING")
    print("=" * 60)

    # Configure parallel processing for real-time monitoring
    parallel_config = ParallelVaRConfig(use_multiprocessing=True, n_processes=2, n_threads=4, chunk_size=1000)

    var_config = MonteCarloVaRConfig(n_simulations=3000, confidence_level=0.05, time_horizon=1)

    # Sample portfolio weights
    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    with ParallelVaRCalculator(parallel_config) as parallel_calc:
        # Set up real-time monitoring
        monitoring_config = parallel_calc.real_time_var_monitoring(
            var_config,
            pd.DataFrame(),  # Empty for demo
            weights,
            update_frequency=60,  # Update every minute
        )

        print("Real-time VaR monitoring configuration:")
        for key, value in monitoring_config.items():
            if key != "weights":  # Skip printing weights
                print(f"  {key}: {value}")

        print("\nReal-time monitoring features:")
        print("  ✓ Parallel VaR calculations")
        print("  ✓ Configurable update frequency")
        print("  ✓ Memory-efficient chunking")
        print("  ✓ Automatic resource management")
        print("  ✓ Error handling and recovery")


def create_visualizations() -> None:
    """Create visualizations for VaR analysis."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Generate data
    returns_data = generate_sample_data(n_days=500, n_assets=5)

    weights = {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1,
    }

    var_config = MonteCarloVaRConfig(n_simulations=5000, confidence_level=0.05)

    var_calculator = MonteCarloVaR(var_config)
    var_calculator.update_data(returns_data)

    # Calculate portfolio returns
    portfolio_returns = var_calculator._calculate_portfolio_returns(weights)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Monte Carlo VaR Analysis", fontsize=16)

    # 1. Portfolio returns distribution
    axes[0, 0].hist(portfolio_returns, bins=50, alpha=0.7, density=True, color="skyblue")
    axes[0, 0].axvline(x=-0.02, color="red", linestyle="--", label="2% Loss Threshold")
    axes[0, 0].set_title("Portfolio Returns Distribution")
    axes[0, 0].set_xlabel("Returns")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Asset correlation heatmap
    if var_calculator._correlation_matrix is not None:
        sns.heatmap(
            var_calculator._correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Asset Correlation Matrix")

    # 3. VaR comparison by method
    methods = ["Historical", "Parametric", "Monte Carlo"]
    var_values = []

    try:
        hist_var = var_calculator.historical_simulation_var(weights)
        var_values.append(hist_var.var_value)
    except Exception:
        # Optionally log or print(e)
        var_values.append(0)

    try:
        param_var = var_calculator.parametric_var(weights)
        var_values.append(param_var.var_value)
    except Exception:
        # Optionally log or print(e)
        var_values.append(0)

    try:
        mc_var = var_calculator.monte_carlo_var(weights)
        var_values.append(mc_var.var_value)
    except Exception:
        # Optionally log or print(e)
        var_values.append(0)

    axes[1, 0].bar(methods, var_values, color=["lightblue", "lightgreen", "lightcoral"])
    axes[1, 0].set_title("VaR Comparison by Method")
    axes[1, 0].set_ylabel("VaR Value")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Cumulative returns with VaR bands
    cumulative_returns = (1 + portfolio_returns).cumprod()
    axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values, label="Portfolio Value")

    # Add VaR bands (simplified)
    if var_values[2] > 0:  # Monte Carlo VaR
        var_band = 1 - var_values[2]
        axes[1, 1].axhline(
            y=var_band,
            color="red",
            linestyle="--",
            label=f"VaR Band ({var_values[2]:.2%})",
        )

    axes[1, 1].set_title("Portfolio Cumulative Returns")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Portfolio Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_file = "monte_carlo_var_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as: {output_file}")

    plt.show()


def main() -> None:
    """Run all Monte Carlo VaR demonstrations."""
    print("MONTE CARLO VaR COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates advanced Monte Carlo VaR capabilities including:")
    print("• Historical simulation with bootstrapping")
    print("• Parametric VaR with multiple distribution assumptions")
    print("• Monte Carlo simulation with correlated asset movements")
    print("• Stress testing scenarios for extreme market conditions")
    print("• Parallel processing for large simulations")
    print("• VaR backtesting and validation methods")
    print("• Real-time VaR monitoring setup")
    print("=" * 80)

    try:
        # Run all demonstrations
        demonstrate_basic_var_calculations()
        demonstrate_stress_testing()
        demonstrate_parallel_processing()
        demonstrate_backtesting()
        demonstrate_method_comparison()
        demonstrate_real_time_monitoring()

        # Create visualizations
        create_visualizations()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey features demonstrated:")
        print("✓ Multiple VaR calculation methods")
        print("✓ Advanced statistical distributions")
        print("✓ Stress testing capabilities")
        print("✓ Parallel processing optimization")
        print("✓ Comprehensive backtesting")
        print("✓ Real-time monitoring setup")
        print("✓ Visualization and analysis tools")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
