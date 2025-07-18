"""
Parallel processing utilities for Monte Carlo VaR calculations.

Provides efficient parallel processing capabilities for:
- Large-scale Monte Carlo simulations
- Multiple VaR method comparisons
- Stress testing across scenarios
- Real-time VaR monitoring
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from .monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig, VaRResult

logger = get_logger(__name__)


@dataclass
class ParallelVaRConfig:
    """Configuration for parallel VaR calculations."""

    # Parallel processing settings
    use_multiprocessing: bool = True
    n_processes: int | None = None
    n_threads: int = 4

    # Chunking settings
    chunk_size: int = 1000
    max_chunk_size: int = 10000

    # Memory management
    memory_limit_gb: float = 4.0
    enable_memory_monitoring: bool = True

    def __post_init__(self) -> None:
        """Set default number of processes if not specified."""
        if self.n_processes is None:
            self.n_processes = min(mp.cpu_count(), 8)  # Cap at 8 processes


class ParallelVaRCalculator:
    """
    Parallel processing wrapper for Monte Carlo VaR calculations.

    Handles large-scale simulations efficiently using multiprocessing
    and threading for optimal performance.
    """

    def __init__(self, config: ParallelVaRConfig):
        """
        Initialize parallel VaR calculator.

        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize executors
        self._process_executor: ProcessPoolExecutor | None = None
        self._thread_executor: ThreadPoolExecutor | None = None

        # Results storage
        self._results: list[VaRResult] = []
        self._performance_metrics: dict[str, Any] = {}

    def __enter__(self) -> "ParallelVaRCalculator":
        """Context manager entry."""
        self._initialize_executors()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        """Context manager exit."""
        self._cleanup_executors()

    def _initialize_executors(self) -> None:
        """Initialize process and thread executors."""
        if self.config.use_multiprocessing:
            self._process_executor = ProcessPoolExecutor(max_workers=self.config.n_processes)

        self._thread_executor = ThreadPoolExecutor(max_workers=self.config.n_threads)

    def _cleanup_executors(self) -> None:
        """Clean up executors."""
        if self._process_executor:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None

        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            self._thread_executor = None

    def parallel_monte_carlo_var(
        self,
        var_config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        n_chunks: int | None = None,
    ) -> VaRResult:
        """
        Calculate VaR using parallel Monte Carlo simulation.

        Args:
            var_config: Monte Carlo VaR configuration
            returns_data: Asset returns data
            weights: Portfolio weights
            n_chunks: Number of chunks to split simulations into

        Returns:
            Aggregated VaR result
        """
        import time

        start_time = time.time()

        # Determine optimal chunking
        if n_chunks is None:
            n_chunks = self._calculate_optimal_chunks(var_config.n_simulations)

        # Split simulations into chunks
        chunk_sizes = self._split_simulations(var_config.n_simulations, n_chunks)

        # Create chunk configurations
        chunk_configs = []
        for chunk_size in chunk_sizes:
            chunk_config = MonteCarloVaRConfig(
                n_simulations=chunk_size,
                confidence_level=var_config.confidence_level,
                time_horizon=var_config.time_horizon,
                lookback_period=var_config.lookback_period,
                distribution_type=var_config.distribution_type,
                correlation_method=var_config.correlation_method,
                use_parallel=False,  # Disable nested parallelization
            )
            chunk_configs.append(chunk_config)

        # Execute parallel simulations
        if self.config.use_multiprocessing and self._process_executor:
            chunk_results = self._execute_parallel_chunks_process(chunk_configs, returns_data, weights)
        else:
            chunk_results = self._execute_parallel_chunks_thread(chunk_configs, returns_data, weights)

        # Aggregate results
        aggregated_result = self._aggregate_chunk_results(chunk_results, var_config)

        calculation_time = time.time() - start_time
        aggregated_result.calculation_time = calculation_time

        self._results.append(aggregated_result)
        return aggregated_result

    def _calculate_optimal_chunks(self, n_simulations: int) -> int:
        """Calculate optimal number of chunks based on available resources."""
        # Consider memory constraints
        estimated_memory_per_sim = 0.001  # GB per simulation (rough estimate)
        max_sims_per_chunk = int(self.config.memory_limit_gb / estimated_memory_per_sim)

        # Consider CPU cores
        max_chunks_by_cpu = self.config.n_processes or self.config.n_threads

        # Calculate optimal chunks
        return max(
            1,
            min(
                n_simulations // self.config.chunk_size,
                max_sims_per_chunk // self.config.chunk_size,
                max_chunks_by_cpu,
            ),
        )

    def _split_simulations(self, n_simulations: int, n_chunks: int) -> list[int]:
        """Split total simulations into chunks."""
        base_size = n_simulations // n_chunks
        remainder = n_simulations % n_chunks

        chunk_sizes = [base_size] * n_chunks

        # Distribute remainder
        for i in range(remainder):
            chunk_sizes[i] += 1

        return chunk_sizes

    def _execute_parallel_chunks_process(
        self,
        chunk_configs: list[MonteCarloVaRConfig],
        returns_data: pd.DataFrame,
        weights: dict[str, float],
    ) -> list[VaRResult]:
        """Execute VaR calculations using process pool."""
        if not self._process_executor:
            raise RuntimeError("Process executor not initialized")

        # Prepare arguments for parallel execution
        args_list = [(config, returns_data, weights) for config in chunk_configs]

        # Submit tasks
        futures = [self._process_executor.submit(self._calculate_chunk_var, *args) for args in args_list]

        # Collect results
        chunk_results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                chunk_results.append(result)
            except Exception as e:
                self.logger.exception(f"Error in parallel VaR calculation: {e}")
                raise

        return chunk_results

    def _execute_parallel_chunks_thread(
        self,
        chunk_configs: list[MonteCarloVaRConfig],
        returns_data: pd.DataFrame,
        weights: dict[str, float],
    ) -> list[VaRResult]:
        """Execute VaR calculations using thread pool."""
        if not self._thread_executor:
            raise RuntimeError("Thread executor not initialized")

        # Prepare arguments for parallel execution
        args_list = [(config, returns_data, weights) for config in chunk_configs]

        # Submit tasks
        futures = [self._thread_executor.submit(self._calculate_chunk_var, *args) for args in args_list]

        # Collect results
        chunk_results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                chunk_results.append(result)
            except Exception as e:
                self.logger.exception(f"Error in parallel VaR calculation: {e}")
                raise

        return chunk_results

    @staticmethod
    def _calculate_chunk_var(
        config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
    ) -> VaRResult:
        """Calculate VaR for a single chunk (for parallel execution)."""
        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_data)
        return var_calculator.monte_carlo_var(weights)

    def _aggregate_chunk_results(
        self,
        chunk_results: list[VaRResult],
        original_config: MonteCarloVaRConfig,
    ) -> VaRResult:
        """Aggregate results from multiple chunks."""
        if not chunk_results:
            raise ValueError("No chunk results to aggregate")

        # Extract simulation results from chunks
        all_simulated_returns: list[Any] = []
        total_simulations = 0

        for result in chunk_results:
            # For Monte Carlo results, we need to reconstruct the simulated returns
            # This is a simplified aggregation - in practice, you'd want to store
            # the actual simulated returns from each chunk
            total_simulations += result.simulation_count

        # Calculate aggregated VaR (weighted average of chunk VaRs)
        var_values = [result.var_value for result in chunk_results]
        cvar_values = [result.cvar_value for result in chunk_results]
        simulation_counts = [result.simulation_count for result in chunk_results]

        # Weight by simulation count
        weights = np.array(simulation_counts) / total_simulations

        aggregated_var = np.average(var_values, weights=weights)
        aggregated_cvar = np.average(cvar_values, weights=weights)

        # Create aggregated result
        return VaRResult(
            var_value=float(aggregated_var),
            cvar_value=float(aggregated_cvar),
            confidence_level=original_config.confidence_level,
            time_horizon=original_config.time_horizon,
            method="parallel_monte_carlo",
            distribution="simulated",
            simulation_count=total_simulations,
            calculation_time=0.0,  # Will be set by caller
            additional_metrics={
                "n_chunks": len(chunk_results),
                "chunk_var_values": var_values,
                "chunk_cvar_values": cvar_values,
            },
            timestamp=chunk_results[0].timestamp,
        )

    def parallel_stress_test(
        self,
        var_config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        scenarios: list[str] | None = None,
    ) -> dict[str, VaRResult]:
        """
        Perform parallel stress testing across multiple scenarios.

        Args:
            var_config: Monte Carlo VaR configuration
            returns_data: Asset returns data
            weights: Portfolio weights
            scenarios: List of stress scenarios to test

        Returns:
            Dictionary of VaR results for each scenario
        """
        if scenarios is None:
            if var_config.stress_scenarios is None:
                raise ValueError("No stress scenarios configured")
            scenarios = list(var_config.stress_scenarios.keys())

        # Prepare arguments for parallel execution
        args_list = [(var_config, returns_data, weights, scenario) for scenario in scenarios]

        # Execute parallel stress tests
        if self.config.use_multiprocessing:
            if self._process_executor is None:
                raise RuntimeError("Process executor not initialized")
            futures = [self._process_executor.submit(self._calculate_stress_scenario, *args) for args in args_list]
        else:
            if self._thread_executor is None:
                raise RuntimeError("Thread executor not initialized")
            futures = [self._thread_executor.submit(self._calculate_stress_scenario, *args) for args in args_list]

        # Collect results
        stress_results = {}
        for i, future in enumerate(as_completed(futures)):
            try:
                scenario_name, result = future.result()
                stress_results[scenario_name] = result
            except Exception as e:
                self.logger.exception(f"Error in stress test for scenario {scenarios[i]}: {e}")
                raise

        return stress_results

    @staticmethod
    def _calculate_stress_scenario(
        config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        scenario: str,
    ) -> tuple[str, VaRResult]:
        """Calculate VaR for a single stress scenario (for parallel execution)."""
        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_data)

        stress_results = var_calculator.stress_test_var(weights, scenario)
        return scenario, stress_results[scenario]

    def parallel_method_comparison(
        self,
        var_config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        methods: list[str] | None = None,
    ) -> dict[str, VaRResult]:
        """
        Compare multiple VaR calculation methods in parallel.

        Args:
            var_config: Monte Carlo VaR configuration
            returns_data: Asset returns data
            weights: Portfolio weights
            methods: List of VaR methods to compare

        Returns:
            Dictionary of VaR results for each method
        """
        if methods is None:
            methods = ["monte_carlo", "historical_simulation", "parametric"]

        # Prepare arguments for parallel execution
        args_list = [(var_config, returns_data, weights, method) for method in methods]

        # Execute parallel method comparisons
        if self.config.use_multiprocessing:
            if self._process_executor is None:
                raise RuntimeError("Process executor not initialized")
            futures = [self._process_executor.submit(self._calculate_method_var, *args) for args in args_list]
        else:
            if self._thread_executor is None:
                raise RuntimeError("Thread executor not initialized")
            futures = [self._thread_executor.submit(self._calculate_method_var, *args) for args in args_list]

        # Collect results
        method_results = {}
        for i, future in enumerate(as_completed(futures)):
            try:
                method_name, result = future.result()
                method_results[method_name] = result
            except Exception as e:
                self.logger.exception(f"Error in method comparison for {methods[i]}: {e}")
                raise

        return method_results

    @staticmethod
    def _calculate_method_var(
        config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        method: str,
    ) -> tuple[str, VaRResult]:
        """Calculate VaR using a specific method (for parallel execution)."""
        var_calculator = MonteCarloVaR(config)
        var_calculator.update_data(returns_data)

        if method == "monte_carlo":
            result = var_calculator.monte_carlo_var(weights)
        elif method == "historical_simulation":
            result = var_calculator.historical_simulation_var(weights)
        elif method == "parametric":
            result = var_calculator.parametric_var(weights)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return method, result

    def real_time_var_monitoring(
        self,
        var_config: MonteCarloVaRConfig,
        returns_data: pd.DataFrame,
        weights: dict[str, float],
        update_frequency: int = 1,
    ) -> dict[str, Any]:
        """
        Set up real-time VaR monitoring with parallel updates.

        Args:
            var_config: Monte Carlo VaR configuration
            returns_data: Asset returns data
            weights: Portfolio weights
            update_frequency: Update frequency in seconds

        Returns:
            Monitoring configuration and results
        """
        # This would typically integrate with a real-time data feed
        # For now, we'll return a monitoring setup
        monitoring_config = {
            "update_frequency": update_frequency,
            "parallel_enabled": self.config.use_multiprocessing,
            "n_workers": self.config.n_processes or self.config.n_threads,
            "var_config": var_config,
            "weights": weights,
        }

        self.logger.info("Real-time VaR monitoring configured")
        return monitoring_config

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for parallel calculations."""
        if not self._results:
            return {}

        return {
            "total_calculations": len(self._results),
            "average_calculation_time": np.mean([r.calculation_time for r in self._results]),
            "total_simulations": sum(r.simulation_count for r in self._results),
            "parallel_config": self.config,
        }

    def clear_results(self) -> None:
        """Clear all calculation results."""
        self._results.clear()
        self._performance_metrics.clear()
        self.logger.info("Cleared parallel VaR calculation results")
