"""
Stress testing framework for Trading RL Agent.

Tests include:
- High-frequency data processing stress
- Large portfolio management stress
- Concurrent operations stress
- System resource limits stress
- Memory pressure stress
- CPU-intensive operations stress
- Network latency stress
- Rapid state changes stress
"""

import gc
import time

import numpy as np
import pandas as pd
import psutil
import pytest

from trade_agent.data.features import FeatureEngineer
from trade_agent.data.parallel_data_fetcher import ParallelDataManager
from trade_agent.risk.monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig


class TestStressTesting:
    """Stress testing for the trading system."""

    @pytest.mark.stress
    @pytest.mark.performance
    def test_high_frequency_data_processing_stress(self, high_frequency_data, performance_monitor):
        """Test system under high-frequency data processing stress."""
        # Prepare high-frequency data
        test_data = high_frequency_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=8)

        performance_monitor.start_monitoring()

        # Simulate high-frequency processing
        def process_high_frequency_stress():
            results = []
            # Process data in rapid succession
            for i in range(0, len(test_data), 1000):  # Process in chunks
                chunk = test_data.iloc[i : i + 1000]
                if len(chunk) > 0:
                    # Calculate features
                    features = feature_engineer.calculate_all_features(chunk)
                    results.append(features)
            return results

        # Measure performance under stress
        start_time = time.time()
        result = process_high_frequency_stress()
        end_time = time.time()

        performance_monitor.record_measurement("high_frequency_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate throughput
        records_processed = sum(len(r) for r in result)
        throughput = records_processed / (end_time - start_time)

        # Assertions
        assert len(result) > 0
        assert throughput > 1000  # Should process at least 1000 records per second
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB
        assert metrics["average_cpu_percent"] < 90  # Should not max out CPU

        # Log stress test results
        print("High-frequency data processing stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Records processed: {records_processed}")
        print(f"  Throughput: {throughput:.2f} records/second")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_large_portfolio_management_stress(self, large_portfolio_data, performance_monitor):
        """Test system under large portfolio management stress."""
        # Prepare large portfolio data
        test_data = large_portfolio_data.copy()

        # Calculate returns for large portfolio
        portfolio_returns = {}
        symbols = test_data["symbol"].unique()[:500]  # Use 500 symbols for stress test

        for symbol in symbols:
            symbol_data = test_data[test_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")
            symbol_data["returns"] = symbol_data["close"].pct_change()
            symbol_data = symbol_data.dropna()

            if len(symbol_data) > 100:
                portfolio_returns[symbol] = symbol_data["returns"].values

        # Create returns DataFrame
        returns_df = pd.DataFrame(portfolio_returns)
        returns_df = returns_df.dropna()

        # Generate multiple portfolio weights
        portfolios = []
        for i in range(10):  # Create 10 different portfolios
            weights = np.random.dirichlet(np.ones(len(returns_df.columns)))
            portfolio_weights = dict(zip(returns_df.columns, weights, strict=False))
            portfolios.append(portfolio_weights)

        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=1000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=4,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_df)

        performance_monitor.start_monitoring()

        # Stress test large portfolio management
        def manage_large_portfolios():
            results = []
            for i, weights in enumerate(portfolios):
                # Calculate VaR for each portfolio
                var_result = var_calculator.monte_carlo_var(weights)
                results.append(
                    {
                        "portfolio_id": i,
                        "var_value": var_result.var_value,
                        "cvar_value": var_result.cvar_value,
                        "symbols": len(weights),
                    }
                )
            return results

        # Measure performance under stress
        start_time = time.time()
        result = manage_large_portfolios()
        end_time = time.time()

        performance_monitor.record_measurement("large_portfolio_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == len(portfolios)
        assert end_time - start_time < 120  # Should complete within 2 minutes
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB
        assert all(r["var_value"] > 0 for r in result)

        # Log stress test results
        print("Large portfolio management stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Portfolios managed: {len(result)}")
        print(f"  Total symbols: {len(returns_df.columns)}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_concurrent_operations_stress(self, benchmark_data, performance_monitor):
        """Test system under concurrent operations stress."""
        # Prepare test data
        test_data = benchmark_data.copy()
        symbols = test_data["symbol"].unique()[:50]

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=8)

        performance_monitor.start_monitoring()

        # Define concurrent operations
        def feature_engineering_task(data):
            return feature_engineer.calculate_all_features(data)

        def data_fetching_task(symbol_list):
            return data_manager.fetch_multiple_symbols(
                symbols=symbol_list,
                start_date="2020-01-01",
                end_date="2023-01-01",
                interval="1d",
                show_progress=False,
            )

        def risk_calculation_task(data):
            # Simulate risk calculation
            returns = data["close"].pct_change().dropna()
            return {
                "volatility": returns.std(),
                "var_95": np.percentile(returns, 5),
                "max_drawdown": (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            }

        # Stress test with concurrent operations
        def run_concurrent_stress():
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit multiple concurrent tasks
                feature_futures = []
                fetch_futures = []
                risk_futures = []

                # Feature engineering tasks
                for i in range(5):
                    chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineering_task, chunk)
                        feature_futures.append(future)

                # Data fetching tasks
                for i in range(3):
                    symbol_chunk = symbols[i * 10 : (i + 1) * 10]
                    future = executor.submit(data_fetching_task, symbol_chunk)
                    fetch_futures.append(future)

                # Risk calculation tasks
                for i in range(5):
                    chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                    if len(chunk) > 0:
                        future = executor.submit(risk_calculation_task, chunk)
                        risk_futures.append(future)

                # Collect results
                feature_results = [f.result() for f in feature_futures]
                fetch_results = [f.result() for f in fetch_futures]
                risk_results = [f.result() for f in risk_futures]

                return {
                    "feature_results": feature_results,
                    "fetch_results": fetch_results,
                    "risk_results": risk_results,
                }

        # Measure performance under stress
        start_time = time.time()
        result = run_concurrent_stress()
        end_time = time.time()

        performance_monitor.record_measurement("concurrent_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result["feature_results"]) > 0
        assert len(result["fetch_results"]) > 0
        assert len(result["risk_results"]) > 0
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB

        # Log stress test results
        print("Concurrent operations stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Feature tasks: {len(result['feature_results'])}")
        print(f"  Fetch tasks: {len(result['fetch_results'])}")
        print(f"  Risk tasks: {len(result['risk_results'])}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_memory_pressure_stress(self, large_portfolio_data, memory_profiler):
        """Test system under memory pressure stress."""
        # Prepare large dataset
        test_data = large_portfolio_data.copy()

        # Create memory-intensive operations
        def memory_intensive_operations():
            # Create large arrays
            large_arrays = []
            for i in range(10):
                # Create large numpy arrays
                array = np.random.random((10000, 1000))  # 10M elements per array
                large_arrays.append(array)

            # Process data with memory pressure
            feature_engineer = FeatureEngineer()
            results = []

            for i in range(0, len(test_data), 5000):
                chunk = test_data.iloc[i : i + 5000]
                if len(chunk) > 0:
                    features = feature_engineer.calculate_all_features(chunk)
                    results.append(features)

            # Clean up large arrays
            del large_arrays
            gc.collect()

            return results

        # Profile memory usage under stress
        memory_metrics = memory_profiler(memory_intensive_operations)

        # Assertions
        assert memory_metrics["max_memory_mb"] < 8192  # Should use less than 8GB
        assert memory_metrics["avg_memory_mb"] < 4096  # Average should be less than 4GB

        # Log stress test results
        print("Memory pressure stress test:")
        print(f"  Max memory: {memory_metrics['max_memory_mb']:.2f} MB")
        print(f"  Avg memory: {memory_metrics['avg_memory_mb']:.2f} MB")
        print(f"  Min memory: {memory_metrics['min_memory_mb']:.2f} MB")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_cpu_intensive_operations_stress(self, benchmark_data, performance_monitor):
        """Test system under CPU-intensive operations stress."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()

        performance_monitor.start_monitoring()

        # CPU-intensive operations
        def cpu_intensive_operations():
            results = []

            # Multiple feature engineering passes
            for iteration in range(5):
                # Calculate all features multiple times
                features = feature_engineer.calculate_all_features(test_data)

                # Additional CPU-intensive calculations
                if len(features) > 0:
                    # Calculate rolling statistics
                    for col in features.select_dtypes(include=[np.number]).columns[:10]:
                        rolling_mean = features[col].rolling(20).mean()
                        rolling_std = features[col].rolling(20).std()
                        rolling_corr = features[col].rolling(50).corr(features.iloc[:, 0])

                        results.append(
                            {
                                "iteration": iteration,
                                "column": col,
                                "mean": rolling_mean.mean(),
                                "std": rolling_std.mean(),
                                "corr": rolling_corr.mean(),
                            }
                        )

            return results

        # Measure performance under stress
        start_time = time.time()
        result = cpu_intensive_operations()
        end_time = time.time()

        performance_monitor.record_measurement("cpu_intensive_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 180  # Should complete within 3 minutes
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log stress test results
        print("CPU-intensive operations stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Operations completed: {len(result)}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")
        print(f"  Peak CPU: {metrics['peak_cpu_percent']:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_rapid_state_changes_stress(self, benchmark_data, performance_monitor):
        """Test system under rapid state changes stress."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()

        performance_monitor.start_monitoring()

        # Simulate rapid state changes
        def rapid_state_changes():
            results = []
            current_state = test_data.copy()

            # Rapid state modifications
            for i in range(100):  # 100 rapid state changes
                # Modify state
                if i % 10 == 0:
                    # Add new data
                    new_data = test_data.iloc[:100].copy()
                    new_data["timestamp"] = pd.Timestamp.now() + pd.Timedelta(seconds=i)
                    current_state = pd.concat([current_state, new_data])
                elif i % 5 == 0:
                    # Remove old data
                    current_state = current_state.iloc[100:]
                else:
                    # Update existing data
                    current_state["close"] = current_state["close"] * (1 + np.random.normal(0, 0.001))

                # Process current state
                if len(current_state) > 0:
                    features = feature_engineer.calculate_all_features(current_state)
                    results.append(
                        {
                            "iteration": i,
                            "state_size": len(current_state),
                            "features_calculated": (len(features.columns) if len(features) > 0 else 0),
                        }
                    )

            return results

        # Measure performance under stress
        start_time = time.time()
        result = rapid_state_changes()
        end_time = time.time()

        performance_monitor.record_measurement("rapid_state_changes_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == 100
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log stress test results
        print("Rapid state changes stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  State changes: {len(result)}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_system_resource_limits_stress(self, benchmark_data, performance_monitor):
        """Test system behavior near resource limits."""
        # Get system information
        process = psutil.Process()
        system_memory = psutil.virtual_memory()

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()

        performance_monitor.start_monitoring()

        # Test near resource limits
        def resource_limit_stress():
            results = []

            # Monitor system resources
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_limit = system_memory.total / 1024 / 1024 * 0.8  # 80% of system memory

            # Process data while monitoring resources
            for i in range(0, len(test_data), 1000):
                chunk = test_data.iloc[i : i + 1000]
                if len(chunk) > 0:
                    # Check memory usage
                    current_memory = process.memory_info().rss / 1024 / 1024

                    if current_memory < memory_limit:
                        # Process normally
                        features = feature_engineer.calculate_all_features(chunk)
                        results.append(
                            {
                                "chunk_id": i // 1000,
                                "memory_usage": current_memory,
                                "features_calculated": (len(features.columns) if len(features) > 0 else 0),
                            }
                        )
                    else:
                        # Force garbage collection
                        gc.collect()
                        results.append(
                            {
                                "chunk_id": i // 1000,
                                "memory_usage": current_memory,
                                "features_calculated": 0,
                                "gc_triggered": True,
                            }
                        )

            return results

        # Measure performance under stress
        start_time = time.time()
        result = resource_limit_stress()
        end_time = time.time()

        performance_monitor.record_measurement("resource_limit_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 120  # Should complete within 2 minutes
        assert (
            metrics["peak_memory_mb"] < system_memory.total / 1024 / 1024 * 0.9
        )  # Should not exceed 90% of system memory

        # Log stress test results
        print("System resource limits stress test:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Chunks processed: {len(result)}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  System memory: {system_memory.total / 1024 / 1024:.2f} MB")
        print(f"  Memory usage %: {metrics['peak_memory_mb'] / (system_memory.total / 1024 / 1024) * 100:.1f}%")

    @pytest.mark.stress
    @pytest.mark.performance
    def test_end_to_end_stress_scenario(self, benchmark_data, performance_monitor):
        """Test complete end-to-end stress scenario."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize all components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=4)

        # Prepare portfolio data for risk calculations
        portfolio_returns = {}
        for symbol in test_data["symbol"].unique()[:20]:
            symbol_data = test_data[test_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")
            symbol_data["returns"] = symbol_data["close"].pct_change()
            symbol_data = symbol_data.dropna()

            if len(symbol_data) > 100:
                portfolio_returns[symbol] = symbol_data["returns"].values

        returns_df = pd.DataFrame(portfolio_returns)
        returns_df = returns_df.dropna()

        # Initialize VaR calculator
        config = MonteCarloVaRConfig(
            n_simulations=1000,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=2,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_df)

        performance_monitor.start_monitoring()

        # End-to-end stress scenario
        def end_to_end_stress():
            results = {
                "data_processing": [],
                "feature_engineering": [],
                "risk_calculations": [],
                "concurrent_operations": [],
            }

            # Phase 1: Data processing stress
            for i in range(5):
                chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                if len(chunk) > 0:
                    processed_data = chunk.copy()
                    results["data_processing"].append({"chunk_id": i, "records": len(processed_data)})

            # Phase 2: Feature engineering stress
            for i in range(3):
                chunk = test_data.iloc[i * 2000 : (i + 1) * 2000]
                if len(chunk) > 0:
                    features = feature_engineer.calculate_all_features(chunk)
                    results["feature_engineering"].append(
                        {
                            "chunk_id": i,
                            "features": (len(features.columns) if len(features) > 0 else 0),
                        }
                    )

            # Phase 3: Risk calculations stress
            for i in range(3):
                weights = np.random.dirichlet(np.ones(len(returns_df.columns)))
                portfolio_weights = dict(zip(returns_df.columns, weights, strict=False))
                var_result = var_calculator.monte_carlo_var(portfolio_weights)
                results["risk_calculations"].append({"portfolio_id": i, "var_value": var_result.var_value})

            # Phase 4: Concurrent operations stress
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                # Submit multiple concurrent tasks
                for i in range(4):
                    chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineer.calculate_all_features, chunk)
                        futures.append(future)

                # Collect results
                concurrent_results = [f.result() for f in futures]
                results["concurrent_operations"] = [len(r) for r in concurrent_results]

            return results

        # Measure performance under stress
        start_time = time.time()
        result = end_to_end_stress()
        end_time = time.time()

        performance_monitor.record_measurement("end_to_end_stress_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result["data_processing"]) > 0
        assert len(result["feature_engineering"]) > 0
        assert len(result["risk_calculations"]) > 0
        assert len(result["concurrent_operations"]) > 0
        assert end_time - start_time < 180  # Should complete within 3 minutes
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB

        # Log stress test results
        print("End-to-end stress scenario:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Data processing chunks: {len(result['data_processing'])}")
        print(f"  Feature engineering chunks: {len(result['feature_engineering'])}")
        print(f"  Risk calculations: {len(result['risk_calculations'])}")
        print(f"  Concurrent operations: {len(result['concurrent_operations'])}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")
