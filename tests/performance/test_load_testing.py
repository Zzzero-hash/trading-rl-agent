"""
Load testing framework for Trading RL Agent.

Tests include:
- Light load testing
- Medium load testing
- Heavy load testing
- Stress load testing
- Scalability testing
- Concurrent user simulation
- System throughput testing
"""

import concurrent.futures
import time

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.features import FeatureEngineer
from trading_rl_agent.data.parallel_data_fetcher import ParallelDataManager
from trading_rl_agent.risk.monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig


class TestLoadTesting:
    """Load testing for the trading system."""

    @pytest.mark.load
    @pytest.mark.performance
    def test_light_load_scenario(self, benchmark_data, performance_monitor, load_test_scenarios):
        """Test system under light load conditions."""
        scenario = load_test_scenarios["light_load"]

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=2)

        performance_monitor.start_monitoring()

        # Simulate light load
        def simulate_light_load():
            results = []

            # Simulate concurrent users
            with concurrent.futures.ThreadPoolExecutor(max_workers=scenario["concurrent_users"]) as executor:
                futures = []

                # Submit requests
                for i in range(scenario["concurrent_users"]):
                    chunk = test_data.iloc[i * 500 : (i + 1) * 500]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineer.calculate_all_features, chunk)
                        futures.append(future)

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(max(0, len(result)))

            return results

        # Measure performance under light load
        start_time = time.time()
        result = simulate_light_load()
        end_time = time.time()

        performance_monitor.record_measurement("light_load_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate load metrics
        total_time = end_time - start_time
        requests_per_second = len(result) / total_time

        # Assertions
        assert len(result) == scenario["concurrent_users"]
        assert total_time < scenario["duration_seconds"]
        assert requests_per_second >= scenario["requests_per_second"]
        assert metrics["peak_memory_mb"] < 1024  # Should use less than 1GB
        assert metrics["average_cpu_percent"] < 50  # Should not exceed 50% CPU

        # Log load test results
        print("Light load test results:")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Concurrent users: {scenario['concurrent_users']}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.load
    @pytest.mark.performance
    def test_medium_load_scenario(self, benchmark_data, performance_monitor, load_test_scenarios):
        """Test system under medium load conditions."""
        scenario = load_test_scenarios["medium_load"]

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=4)

        performance_monitor.start_monitoring()

        # Simulate medium load
        def simulate_medium_load():
            results = {
                "feature_engineering": [],
                "data_fetching": [],
                "risk_calculations": [],
            }

            # Simulate concurrent users with different operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=scenario["concurrent_users"]) as executor:
                futures = []

                # Feature engineering requests
                for i in range(scenario["concurrent_users"] // 2):
                    chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineer.calculate_all_features, chunk)
                        futures.append(("feature", future))

                # Data fetching requests
                symbols = test_data["symbol"].unique()[:20]
                for i in range(scenario["concurrent_users"] // 4):
                    symbol_chunk = symbols[i * 5 : (i + 1) * 5]
                    future = executor.submit(
                        data_manager.fetch_multiple_symbols,
                        symbols=symbol_chunk,
                        start_date="2020-01-01",
                        end_date="2023-01-01",
                        interval="1d",
                        show_progress=False,
                    )
                    futures.append(("fetch", future))

                # Risk calculation requests
                for i in range(scenario["concurrent_users"] // 4):
                    chunk = test_data.iloc[i * 1000 : (i + 1) * 1000]
                    if len(chunk) > 0:
                        future = executor.submit(self._calculate_risk_metrics, chunk)
                        futures.append(("risk", future))

                # Collect results
                for op_type, future in futures:
                    try:
                        result = future.result(timeout=30)
                        results[op_type].append(result)
                    except Exception as e:
                        results[op_type].append({"error": str(e)})

            return results

        # Measure performance under medium load
        start_time = time.time()
        result = simulate_medium_load()
        end_time = time.time()

        performance_monitor.record_measurement("medium_load_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate load metrics
        total_time = end_time - start_time
        total_requests = sum(len(v) for v in result.values())
        requests_per_second = total_requests / total_time

        # Assertions
        assert total_requests >= scenario["concurrent_users"]
        assert total_time < scenario["duration_seconds"]
        assert requests_per_second >= scenario["requests_per_second"]
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB
        assert metrics["average_cpu_percent"] < 70  # Should not exceed 70% CPU

        # Log load test results
        print("Medium load test results:")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Concurrent users: {scenario['concurrent_users']}")
        print(f"  Total requests: {total_requests}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.load
    @pytest.mark.performance
    def test_heavy_load_scenario(self, benchmark_data, performance_monitor, load_test_scenarios):
        """Test system under heavy load conditions."""
        scenario = load_test_scenarios["heavy_load"]

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=8)

        # Prepare portfolio data for risk calculations
        portfolio_returns = {}
        for symbol in test_data["symbol"].unique()[:30]:
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
            n_workers=4,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_df)

        performance_monitor.start_monitoring()

        # Simulate heavy load
        def simulate_heavy_load():
            results = {
                "feature_engineering": [],
                "data_fetching": [],
                "risk_calculations": [],
                "portfolio_management": [],
            }

            # Simulate concurrent users with intensive operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=scenario["concurrent_users"]) as executor:
                futures = []

                # Feature engineering requests
                for i in range(scenario["concurrent_users"] // 3):
                    chunk = test_data.iloc[i * 2000 : (i + 1) * 2000]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineer.calculate_all_features, chunk)
                        futures.append(("feature", future))

                # Data fetching requests
                symbols = test_data["symbol"].unique()[:40]
                for i in range(scenario["concurrent_users"] // 4):
                    symbol_chunk = symbols[i * 8 : (i + 1) * 8]
                    future = executor.submit(
                        data_manager.fetch_multiple_symbols,
                        symbols=symbol_chunk,
                        start_date="2020-01-01",
                        end_date="2023-01-01",
                        interval="1d",
                        show_progress=False,
                    )
                    futures.append(("fetch", future))

                # Risk calculation requests
                for i in range(scenario["concurrent_users"] // 4):
                    chunk = test_data.iloc[i * 1500 : (i + 1) * 1500]
                    if len(chunk) > 0:
                        future = executor.submit(self._calculate_risk_metrics, chunk)
                        futures.append(("risk", future))

                # Portfolio management requests
                for i in range(scenario["concurrent_users"] // 6):
                    weights = np.random.dirichlet(np.ones(len(returns_df.columns)))
                    portfolio_weights = dict(zip(returns_df.columns, weights, strict=False))
                    future = executor.submit(var_calculator.monte_carlo_var, portfolio_weights)
                    futures.append(("portfolio", future))

                # Collect results
                for op_type, future in futures:
                    try:
                        result = future.result(timeout=60)
                        results[op_type].append(result)
                    except Exception as e:
                        results[op_type].append({"error": str(e)})

            return results

        # Measure performance under heavy load
        start_time = time.time()
        result = simulate_heavy_load()
        end_time = time.time()

        performance_monitor.record_measurement("heavy_load_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate load metrics
        total_time = end_time - start_time
        total_requests = sum(len(v) for v in result.values())
        requests_per_second = total_requests / total_time

        # Assertions
        assert total_requests >= scenario["concurrent_users"]
        assert total_time < scenario["duration_seconds"]
        assert requests_per_second >= scenario["requests_per_second"]
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB
        assert metrics["average_cpu_percent"] < 85  # Should not exceed 85% CPU

        # Log load test results
        print("Heavy load test results:")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Concurrent users: {scenario['concurrent_users']}")
        print(f"  Total requests: {total_requests}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.load
    @pytest.mark.performance
    def test_stress_load_scenario(self, benchmark_data, performance_monitor, load_test_scenarios):
        """Test system under stress load conditions."""
        scenario = load_test_scenarios["stress_load"]

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=12)

        # Prepare portfolio data for risk calculations
        portfolio_returns = {}
        for symbol in test_data["symbol"].unique()[:50]:
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
            n_simulations=500,
            confidence_level=0.05,
            time_horizon=1,
            lookback_period=252,
            use_parallel=True,
            n_workers=6,
        )

        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_df)

        performance_monitor.start_monitoring()

        # Simulate stress load
        def simulate_stress_load():
            results = {
                "feature_engineering": [],
                "data_fetching": [],
                "risk_calculations": [],
                "portfolio_management": [],
                "concurrent_operations": [],
            }

            # Simulate maximum concurrent users
            with concurrent.futures.ThreadPoolExecutor(max_workers=scenario["concurrent_users"]) as executor:
                futures = []

                # Feature engineering requests (40% of load)
                for i in range(int(scenario["concurrent_users"] * 0.4)):
                    chunk = test_data.iloc[i * 3000 : (i + 1) * 3000]
                    if len(chunk) > 0:
                        future = executor.submit(feature_engineer.calculate_all_features, chunk)
                        futures.append(("feature", future))

                # Data fetching requests (30% of load)
                symbols = test_data["symbol"].unique()[:60]
                for i in range(int(scenario["concurrent_users"] * 0.3)):
                    symbol_chunk = symbols[i * 10 : (i + 1) * 10]
                    future = executor.submit(
                        data_manager.fetch_multiple_symbols,
                        symbols=symbol_chunk,
                        start_date="2020-01-01",
                        end_date="2023-01-01",
                        interval="1d",
                        show_progress=False,
                    )
                    futures.append(("fetch", future))

                # Risk calculation requests (20% of load)
                for i in range(int(scenario["concurrent_users"] * 0.2)):
                    chunk = test_data.iloc[i * 2000 : (i + 1) * 2000]
                    if len(chunk) > 0:
                        future = executor.submit(self._calculate_risk_metrics, chunk)
                        futures.append(("risk", future))

                # Portfolio management requests (10% of load)
                for i in range(int(scenario["concurrent_users"] * 0.1)):
                    weights = np.random.dirichlet(np.ones(len(returns_df.columns)))
                    portfolio_weights = dict(zip(returns_df.columns, weights, strict=False))
                    future = executor.submit(var_calculator.monte_carlo_var, portfolio_weights)
                    futures.append(("portfolio", future))

                # Collect results with timeout
                for op_type, future in futures:
                    try:
                        result = future.result(timeout=90)
                        results[op_type].append(result)
                    except Exception as e:
                        results[op_type].append({"error": str(e)})

            return results

        # Measure performance under stress load
        start_time = time.time()
        result = simulate_stress_load()
        end_time = time.time()

        performance_monitor.record_measurement("stress_load_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate load metrics
        total_time = end_time - start_time
        total_requests = sum(len(v) for v in result.values())
        requests_per_second = total_requests / total_time

        # Calculate success rate
        successful_requests = sum(len([r for r in v if "error" not in r]) for v in result.values())
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        # Assertions
        assert total_requests >= scenario["concurrent_users"]
        assert total_time < scenario["duration_seconds"]
        assert success_rate >= 0.8  # At least 80% success rate
        assert metrics["peak_memory_mb"] < 6144  # Should use less than 6GB
        assert metrics["average_cpu_percent"] < 95  # Should not exceed 95% CPU

        # Log load test results
        print("Stress load test results:")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Concurrent users: {scenario['concurrent_users']}")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.load
    @pytest.mark.performance
    def test_scalability_testing(self, benchmark_data, performance_monitor):
        """Test system scalability with increasing load."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()

        # Test different load levels
        load_levels = [10, 25, 50, 100, 200]
        scalability_results = []

        for load_level in load_levels:
            performance_monitor.start_monitoring()

            # Simulate load level
            def simulate_load_level(load_level=load_level):
                results = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=load_level) as executor:
                    futures = []

                    # Submit requests
                    for i in range(load_level):
                        chunk = test_data.iloc[i * 100 : (i + 1) * 100]
                        if len(chunk) > 0:
                            future = executor.submit(feature_engineer.calculate_all_features, chunk)
                            futures.append(future)

                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results.append(max(0, len(result)))

                return results

            # Measure performance
            start_time = time.time()
            result = simulate_load_level()
            end_time = time.time()

            performance_monitor.record_measurement(f"load_level_{load_level}_complete")
            metrics = performance_monitor.stop_monitoring()

            # Calculate metrics
            total_time = end_time - start_time
            requests_per_second = len(result) / total_time

            scalability_results.append(
                {
                    "load_level": load_level,
                    "total_time": total_time,
                    "requests_per_second": requests_per_second,
                    "memory_peak": metrics["peak_memory_mb"],
                    "cpu_usage": metrics["average_cpu_percent"],
                    "successful_requests": len(result),
                }
            )

        # Analyze scalability
        baseline_rps = scalability_results[0]["requests_per_second"]
        scalability_metrics = []

        for result in scalability_results[1:]:
            rps_ratio = result["requests_per_second"] / baseline_rps
            load_ratio = result["load_level"] / load_levels[0]
            efficiency = rps_ratio / load_ratio
            scalability_metrics.append(
                {
                    "load_level": result["load_level"],
                    "efficiency": efficiency,
                    "rps_ratio": rps_ratio,
                    "load_ratio": load_ratio,
                }
            )

        # Assertions
        assert len(scalability_results) == len(load_levels)
        assert all(r["successful_requests"] > 0 for r in scalability_results)

        # Check for reasonable scalability (efficiency should not drop below 0.5)
        for metric in scalability_metrics:
            assert metric["efficiency"] >= 0.5, f"Poor scalability at load level {metric['load_level']}"

        # Log scalability results
        print("Scalability test results:")
        for result in scalability_results:
            print(f"  Load level {result['load_level']}:")
            print(f"    Time: {result['total_time']:.2f}s")
            print(f"    RPS: {result['requests_per_second']:.2f}")
            print(f"    Memory: {result['memory_peak']:.2f} MB")
            print(f"    CPU: {result['cpu_usage']:.1f}%")

        for metric in scalability_metrics:
            print(f"  Load {metric['load_level']} efficiency: {metric['efficiency']:.2f}")

    @pytest.mark.load
    @pytest.mark.performance
    def test_concurrent_user_simulation(self, benchmark_data, performance_monitor):
        """Test system with realistic concurrent user simulation."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        data_manager = ParallelDataManager(max_workers=6)

        # Simulate different user types
        user_types = {
            "data_analyst": {
                "weight": 0.3,
                "operations": ["feature_engineering", "data_fetching"],
            },
            "risk_manager": {
                "weight": 0.2,
                "operations": ["risk_calculation", "portfolio_management"],
            },
            "trader": {
                "weight": 0.4,
                "operations": ["feature_engineering", "risk_calculation"],
            },
            "researcher": {
                "weight": 0.1,
                "operations": ["data_fetching", "feature_engineering"],
            },
        }

        performance_monitor.start_monitoring()

        # Simulate realistic user behavior
        def simulate_realistic_users():
            results = {
                "user_sessions": [],
                "operation_results": {
                    "feature_engineering": [],
                    "data_fetching": [],
                    "risk_calculation": [],
                    "portfolio_management": [],
                },
            }

            # Simulate 50 concurrent users
            total_users = 50
            user_distribution = {
                user_type: int(total_users * config["weight"]) for user_type, config in user_types.items()
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=total_users) as executor:
                futures = []

                # Create user sessions
                for user_type, count in user_distribution.items():
                    for i in range(count):
                        future = executor.submit(
                            self._simulate_user_session,
                            user_type,
                            test_data,
                            feature_engineer,
                            data_manager,
                        )
                        futures.append((user_type, future))

                # Collect results
                for user_type, future in futures:
                    try:
                        session_result = future.result(timeout=60)
                        results["user_sessions"].append({"user_type": user_type, "operations": session_result})

                        # Aggregate operation results
                        for op_type, op_result in session_result.items():
                            results["operation_results"][op_type].append(op_result)
                    except Exception as e:
                        results["user_sessions"].append({"user_type": user_type, "error": str(e)})

            return results

        # Measure performance
        start_time = time.time()
        result = simulate_realistic_users()
        end_time = time.time()

        performance_monitor.record_measurement("concurrent_users_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate metrics
        total_time = end_time - start_time
        total_sessions = len(result["user_sessions"])
        successful_sessions = len([s for s in result["user_sessions"] if "error" not in s])
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0

        # Calculate operations per second
        total_operations = sum(len(s["operations"]) for s in result["user_sessions"] if "error" not in s)
        operations_per_second = total_operations / total_time

        # Assertions
        assert total_sessions == 50
        assert success_rate >= 0.9  # At least 90% success rate
        assert total_time < 120  # Should complete within 2 minutes
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB
        assert metrics["average_cpu_percent"] < 80  # Should not exceed 80% CPU

        # Log results
        print("Concurrent user simulation results:")
        print(f"  Duration: {total_time:.2f} seconds")
        print(f"  Total sessions: {total_sessions}")
        print(f"  Successful sessions: {successful_sessions}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Operations per second: {operations_per_second:.2f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

        # Log user type distribution
        user_distribution = {}
        for user_type, count in user_distribution.items():
            user_sessions = [s for s in result["user_sessions"] if s.get("user_type") == user_type]
            user_success_rate = len([s for s in user_sessions if "error" not in s]) / len(user_sessions)
            print(f"  {user_type}: {count} users, {user_success_rate:.2%} success rate")

    def _calculate_risk_metrics(self, data):
        """Calculate basic risk metrics for load testing."""
        if len(data) == 0:
            return {"error": "No data provided"}

        try:
            returns = data["close"].pct_change().dropna()
            return {
                "volatility": returns.std(),
                "var_95": np.percentile(returns, 5),
                "max_drawdown": (returns.cumsum() - returns.cumsum().expanding().max()).min(),
                "sharpe_ratio": (returns.mean() / returns.std() if returns.std() > 0 else 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _simulate_user_session(self, user_type, test_data, feature_engineer, data_manager):
        """Simulate a user session with realistic behavior."""
        operations = {}

        try:
            # Simulate user-specific operations
            if user_type == "data_analyst":
                # Data analysts focus on feature engineering and data fetching
                chunk = test_data.iloc[:1000]
                operations["feature_engineering"] = len(feature_engineer.calculate_all_features(chunk))

                symbols = test_data["symbol"].unique()[:5]
                fetch_result = data_manager.fetch_multiple_symbols(
                    symbols=symbols,
                    start_date="2020-01-01",
                    end_date="2023-01-01",
                    interval="1d",
                    show_progress=False,
                )
                operations["data_fetching"] = len(fetch_result)

            elif user_type == "risk_manager":
                # Risk managers focus on risk calculations
                chunk = test_data.iloc[:500]
                operations["risk_calculation"] = self._calculate_risk_metrics(chunk)

                # Simulate portfolio management
                operations["portfolio_management"] = {
                    "portfolio_size": 10,
                    "risk_metrics": ["var", "cvar", "volatility"],
                }

            elif user_type == "trader":
                # Traders need quick feature engineering and risk assessment
                chunk = test_data.iloc[:500]
                operations["feature_engineering"] = len(feature_engineer.calculate_all_features(chunk))
                operations["risk_calculation"] = self._calculate_risk_metrics(chunk)

            elif user_type == "researcher":
                # Researchers need extensive data and feature analysis
                symbols = test_data["symbol"].unique()[:10]
                fetch_result = data_manager.fetch_multiple_symbols(
                    symbols=symbols,
                    start_date="2020-01-01",
                    end_date="2023-01-01",
                    interval="1d",
                    show_progress=False,
                )
                operations["data_fetching"] = len(fetch_result)

                chunk = test_data.iloc[:2000]
                operations["feature_engineering"] = len(feature_engineer.calculate_all_features(chunk))

            return operations

        except Exception as e:
            return {"error": str(e)}
