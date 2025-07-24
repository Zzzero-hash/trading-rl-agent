"""
Performance tests for risk calculation components.

Tests include:
- Monte Carlo VaR calculation speed
- Parallel risk calculations
- Risk alert system performance
- Memory usage in risk calculations
- Stress testing for risk models
- Risk calculation accuracy under load
"""

import time

import numpy as np
import pandas as pd
import pytest

from trade_agent.risk.alert_system import RiskAlertSystem
from trade_agent.risk.monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig
from trade_agent.risk.parallel_var import ParallelVaRCalculator


class TestRiskCalculationPerformance:
    """Performance tests for risk calculation components."""

    @pytest.fixture
    def portfolio_data(self, benchmark_data):
        """Prepare portfolio data for risk calculations."""
        # Convert benchmark data to portfolio format
        test_data = benchmark_data.copy()

        # Calculate returns for each symbol
        portfolio_returns = {}
        for symbol in test_data["symbol"].unique()[:20]:  # Use 20 symbols
            symbol_data = test_data[test_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")
            symbol_data["returns"] = symbol_data["close"].pct_change()
            symbol_data = symbol_data.dropna()

            if len(symbol_data) > 100:  # Only use symbols with sufficient data
                portfolio_returns[symbol] = symbol_data["returns"].values

        # Create returns DataFrame
        returns_df = pd.DataFrame(portfolio_returns)
        return returns_df.dropna()

    @pytest.fixture
    def portfolio_weights(self, portfolio_data):
        """Generate portfolio weights."""
        symbols = portfolio_data.columns
        weights = np.random.dirichlet(np.ones(len(symbols)))  # Random weights that sum to 1
        return dict(zip(symbols, weights, strict=False))

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_monte_carlo_var_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test Monte Carlo VaR calculation performance."""
        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=10000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=4,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Benchmark VaR calculation
        def calculate_var():
            return var_calculator.monte_carlo_var(portfolio_weights)

        # Measure performance
        start_time = time.time()
        result = calculate_var()
        end_time = time.time()

        performance_monitor.record_measurement("var_calculation_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert result.var_value > 0
        assert end_time - start_time < 30  # Should complete within 30 seconds
        assert metrics["peak_memory_mb"] < 1024  # Should use less than 1GB

        # Log performance metrics
        print("Monte Carlo VaR performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  VaR value: {result.var_value:.6f}")
        print(f"  Simulations: {config.n_simulations}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_parallel_var_calculation_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test parallel VaR calculation performance."""
        # Initialize parallel VaR calculator
        var_calculator = ParallelVaRCalculator(n_workers=4, n_simulations=5000, confidence_level=0.05)

        performance_monitor.start_monitoring()

        # Benchmark parallel VaR calculation
        def calculate_parallel_var():
            return var_calculator.calculate_var(returns_data=portfolio_data, weights=portfolio_weights)

        # Measure performance
        start_time = time.time()
        result = calculate_parallel_var()
        end_time = time.time()

        performance_monitor.record_measurement("parallel_var_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert result["var_value"] > 0
        assert end_time - start_time < 20  # Should complete within 20 seconds
        assert metrics["peak_memory_mb"] < 1024  # Should use less than 1GB

        # Log performance metrics
        print("Parallel VaR calculation performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  VaR value: {result['var_value']:.6f}")
        print(f"  Workers used: {var_calculator.n_workers}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_historical_simulation_var_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test historical simulation VaR performance."""
        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=1000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=False,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Benchmark historical simulation
        def calculate_historical_var():
            return var_calculator.historical_simulation_var(portfolio_weights, use_bootstrap=True)

        # Measure performance
        start_time = time.time()
        result = calculate_historical_var()
        end_time = time.time()

        performance_monitor.record_measurement("historical_var_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert result.var_value > 0
        assert end_time - start_time < 10  # Should complete within 10 seconds
        assert metrics["peak_memory_mb"] < 512  # Should use less than 512MB

        # Log performance metrics
        print("Historical simulation VaR performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  VaR value: {result.var_value:.6f}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_parametric_var_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test parametric VaR performance."""
        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=1000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Benchmark parametric VaR with different distributions
        distributions = ["normal", "t", "laplace", "logistic"]
        results = {}

        def calculate_parametric_var():
            for dist in distributions:
                results[dist] = var_calculator.parametric_var(portfolio_weights, distribution=dist)
            return results

        # Measure performance
        start_time = time.time()
        result = calculate_parametric_var()
        end_time = time.time()

        performance_monitor.record_measurement("parametric_var_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == len(distributions)
        assert end_time - start_time < 5  # Should complete within 5 seconds
        assert metrics["peak_memory_mb"] < 256  # Should use less than 256MB

        # Log performance metrics
        print("Parametric VaR performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Distributions tested: {len(distributions)}")
        for dist, var_result in result.items():
            print(f"    {dist}: {var_result.var_value:.6f}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_stress_testing_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test stress testing performance."""
        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=5000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Benchmark stress testing
        def run_stress_tests():
            scenarios = ["market_crash", "flash_crash", "liquidity_crisis"]
            results = {}
            for scenario in scenarios:
                results[scenario] = var_calculator.stress_test_var(portfolio_weights, scenario=scenario)
            return results

        # Measure performance
        start_time = time.time()
        result = run_stress_tests()
        end_time = time.time()

        performance_monitor.record_measurement("stress_testing_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == 3  # Three scenarios
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 1536  # Should use less than 1.5GB

        # Log performance metrics
        print("Stress testing performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Scenarios tested: {len(result)}")
        for scenario, var_result in result.items():
            print(f"    {scenario}: {var_result['market_crash'].var_value:.6f}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_risk_alert_system_performance(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test risk alert system performance."""
        # Initialize risk alert system
        alert_system = RiskAlertSystem(
            var_threshold=0.02,  # 2% VaR threshold
            cvar_threshold=0.03,  # 3% CVaR threshold
            check_interval=1,  # Check every second
            alert_cooldown=60,  # 1 minute cooldown
        )

        # Calculate current VaR
        config = MonteCarloVaRConfig(n_simulations=1000, confidence_level=0.05)
        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)
        var_result = var_calculator.monte_carlo_var(portfolio_weights)

        performance_monitor.start_monitoring()

        # Benchmark alert system
        def run_alert_checks():
            alerts = []
            for _ in range(10):  # Simulate 10 risk checks
                alert = alert_system.check_risk_metrics(
                    var_value=var_result.var_value,
                    cvar_value=var_result.cvar_value,
                    portfolio_value=1000000,
                )
                if alert:
                    alerts.append(alert)
            return alerts

        # Measure performance
        start_time = time.time()
        result = run_alert_checks()
        end_time = time.time()

        performance_monitor.record_measurement("alert_system_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert end_time - start_time < 1  # Should complete within 1 second
        assert metrics["peak_memory_mb"] < 128  # Should use less than 128MB

        # Log performance metrics
        print("Risk alert system performance:")
        print(f"  Time: {end_time - start_time:.3f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print("  Risk checks: 10")
        print(f"  Alerts generated: {len(result)}")

    @pytest.mark.performance
    @pytest.mark.memory
    def test_memory_usage_in_risk_calculations(self, portfolio_data, portfolio_weights, memory_profiler):
        """Test memory usage in risk calculations."""
        # Initialize VaR calculator with large simulation count
        config = MonteCarloVaRConfig(
            n_simulations=50000,  # Large number of simulations
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=4,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        # Profile memory usage for large VaR calculation
        def calculate_large_var():
            return var_calculator.monte_carlo_var(portfolio_weights)

        memory_metrics = memory_profiler(calculate_large_var)

        # Assertions
        assert memory_metrics["max_memory_mb"] < 2048  # Should use less than 2GB
        assert memory_metrics["avg_memory_mb"] < 1024  # Average should be less than 1GB

        # Log memory metrics
        print("Memory usage in risk calculations:")
        print(f"  Max memory: {memory_metrics['max_memory_mb']:.2f} MB")
        print(f"  Avg memory: {memory_metrics['avg_memory_mb']:.2f} MB")
        print(f"  Min memory: {memory_metrics['min_memory_mb']:.2f} MB")
        print(f"  Simulations: {config.n_simulations}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_concurrent_risk_calculations(self, portfolio_data, performance_monitor):
        """Test performance with concurrent risk calculations."""

        # Prepare multiple portfolios
        portfolios = []
        symbols = portfolio_data.columns

        for i in range(5):  # Create 5 different portfolios
            # Random weights for each portfolio
            weights = np.random.dirichlet(np.ones(len(symbols)))
            portfolio_weights = dict(zip(symbols, weights, strict=False))
            portfolios.append(portfolio_weights)

        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=2000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=False,  # We'll handle concurrency manually
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Benchmark concurrent calculations
        def calculate_portfolio_var(weights):
            return var_calculator.monte_carlo_var(weights)

        def concurrent_risk_calculation():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(calculate_portfolio_var, weights) for weights in portfolios]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

        # Measure performance
        start_time = time.time()
        result = concurrent_risk_calculation()
        end_time = time.time()

        performance_monitor.record_measurement("concurrent_risk_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == len(portfolios)
        assert end_time - start_time < 30  # Should complete within 30 seconds
        assert metrics["peak_memory_mb"] < 1536  # Should use less than 1.5GB

        # Log performance metrics
        print("Concurrent risk calculations:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Portfolios calculated: {len(result)}")
        for i, var_result in enumerate(result):
            print(f"    Portfolio {i + 1}: {var_result.var_value:.6f}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_risk_calculation_accuracy_under_load(self, portfolio_data, portfolio_weights, performance_monitor):
        """Test risk calculation accuracy under load."""
        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=1000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(portfolio_data)

        performance_monitor.start_monitoring()

        # Run multiple VaR calculations to test consistency
        var_results = []

        def run_multiple_calculations():
            for _ in range(10):  # Run 10 calculations
                result = var_calculator.monte_carlo_var(portfolio_weights)
                var_results.append(result.var_value)
            return var_results

        # Measure performance
        start_time = time.time()
        result = run_multiple_calculations()
        end_time = time.time()

        performance_monitor.record_measurement("accuracy_test_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate accuracy metrics
        var_array = np.array(result)
        mean_var = np.mean(var_array)
        std_var = np.std(var_array)
        coefficient_of_variation = std_var / mean_var if mean_var > 0 else 0

        # Assertions
        assert len(result) == 10
        assert end_time - start_time < 20  # Should complete within 20 seconds
        assert coefficient_of_variation < 0.1  # Should have low variation (< 10%)
        assert metrics["peak_memory_mb"] < 512  # Should use less than 512MB

        # Log performance metrics
        print("Risk calculation accuracy under load:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Calculations: {len(result)}")
        print(f"  Mean VaR: {mean_var:.6f}")
        print(f"  Std VaR: {std_var:.6f}")
        print(f"  Coefficient of variation: {coefficient_of_variation:.4f}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_large_portfolio_risk_calculation(self, large_portfolio_data, performance_monitor):
        """Test risk calculation performance with large portfolios."""
        # Prepare large portfolio data
        test_data = large_portfolio_data.copy()

        # Calculate returns for large portfolio
        portfolio_returns = {}
        for symbol in test_data["symbol"].unique()[:100]:  # Use 100 symbols
            symbol_data = test_data[test_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")
            symbol_data["returns"] = symbol_data["close"].pct_change()
            symbol_data = symbol_data.dropna()

            if len(symbol_data) > 500:  # Only use symbols with sufficient data
                portfolio_returns[symbol] = symbol_data["returns"].values

        # Create returns DataFrame
        returns_df = pd.DataFrame(portfolio_returns)
        returns_df = returns_df.dropna()

        # Generate portfolio weights
        symbols = returns_df.columns
        weights = np.random.dirichlet(np.ones(len(symbols)))
        portfolio_weights = dict(zip(symbols, weights, strict=False))

        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=2000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=4,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_df)

        performance_monitor.start_monitoring()

        # Benchmark large portfolio VaR calculation
        def calculate_large_portfolio_var():
            return var_calculator.monte_carlo_var(portfolio_weights)

        # Measure performance
        start_time = time.time()
        result = calculate_large_portfolio_var()
        end_time = time.time()

        performance_monitor.record_measurement("large_portfolio_var_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert result.var_value > 0
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("Large portfolio risk calculation:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  VaR value: {result.var_value:.6f}")
        print(f"  Portfolio size: {len(symbols)} assets")
        print(f"  Data points: {len(returns_df)}")
