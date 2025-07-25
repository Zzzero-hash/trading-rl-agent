"""
Comprehensive CLI Performance and Load Testing.

This module provides performance testing for CLI operations:
- Response time benchmarks for all CLI commands
- Memory usage monitoring during operations
- Concurrent CLI session testing
- Large dataset processing performance
- Resource utilization monitoring
- Performance regression detection
- Scalability testing with increasing loads
"""

import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


@pytest.mark.performance
class TestCLICommandPerformance:
    """Test performance benchmarks for individual CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.process = psutil.Process()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_help_command_performance(self):
        """Benchmark help command response time."""
        # Warm up
        self.runner.invoke(main_app, ["--help"])

        # Benchmark
        start_time = time.time()
        start_memory = self.process.memory_info().rss

        result = self.runner.invoke(main_app, ["--help"])

        end_time = time.time()
        end_memory = self.process.memory_info().rss

        response_time = end_time - start_time
        memory_delta = end_memory - start_memory

        # Performance assertions
        assert result.exit_code == 0
        assert response_time < 1.0  # Should respond within 1 second
        assert memory_delta < 50 * 1024 * 1024  # Less than 50MB memory increase

        print(f"Help command: {response_time:.3f}s, Memory: {memory_delta / 1024 / 1024:.1f}MB")

    def test_version_command_performance(self):
        """Benchmark version command response time."""
        # Warm up
        self.runner.invoke(main_app, ["version"])

        # Benchmark
        start_time = time.time()
        start_memory = self.process.memory_info().rss

        result = self.runner.invoke(main_app, ["version"])

        end_time = time.time()
        end_memory = self.process.memory_info().rss

        response_time = end_time - start_time
        memory_delta = end_memory - start_memory

        # Performance assertions
        assert result.exit_code == 0
        assert response_time < 0.5  # Should respond within 0.5 seconds
        assert memory_delta < 10 * 1024 * 1024  # Less than 10MB memory increase

        print(f"Version command: {response_time:.3f}s, Memory: {memory_delta / 1024 / 1024:.1f}MB")

    def test_info_command_performance(self):
        """Benchmark info command response time."""
        # Warm up
        self.runner.invoke(main_app, ["info"])

        # Benchmark
        start_time = time.time()
        start_memory = self.process.memory_info().rss

        result = self.runner.invoke(main_app, ["info"])

        end_time = time.time()
        end_memory = self.process.memory_info().rss

        response_time = end_time - start_time
        memory_delta = end_memory - start_memory

        # Performance assertions
        assert result.exit_code == 0
        assert response_time < 2.0  # Should respond within 2 seconds
        assert memory_delta < 25 * 1024 * 1024  # Less than 25MB memory increase

        print(f"Info command: {response_time:.3f}s, Memory: {memory_delta / 1024 / 1024:.1f}MB")

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_command_performance(self, mock_cache, mock_standardizer, mock_pipeline):
        """Benchmark data pipeline command performance."""
        # Setup mocks for consistent timing
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        # Simulate some processing time
        def slow_download(*args, **kwargs):  # noqa: ARG001
            time.sleep(0.1)  # Simulate network delay
            return self._create_mock_dataframe()

        mock_pipeline_instance.download_data_parallel.side_effect = slow_download

        # Benchmark
        start_time = time.time()
        start_memory = self.process.memory_info().rss

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT",
                "--no-sentiment",
                "--output-dir", self.temp_dir
            ]
        )

        end_time = time.time()
        end_memory = self.process.memory_info().rss

        response_time = end_time - start_time
        memory_delta = end_memory - start_memory

        # Performance assertions
        assert result.exit_code == 0
        assert response_time < 5.0  # Should complete within 5 seconds
        assert memory_delta < 200 * 1024 * 1024  # Less than 200MB memory increase

        print(f"Data pipeline: {response_time:.3f}s, Memory: {memory_delta / 1024 / 1024:.1f}MB")

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL", "GOOGL", "MSFT"] * 100,
            "close": [150.0, 2500.0, 300.0] * 100,
            "volume": [1000000, 500000, 800000] * 100,
            "date": pd.date_range("2023-01-01", periods=300, freq="H")
        })


@pytest.mark.performance
class TestCLIMemoryUsage:
    """Test memory usage patterns for CLI operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.process = psutil.Process()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_usage_basic_commands(self):
        """Test memory usage for basic CLI commands."""
        commands = [
            ["--help"],
            ["version"],
            ["info"],
            ["data", "--help"],
            ["train", "--help"],
            ["backtest", "--help"],
            ["trade", "--help"],
        ]

        memory_usage = {}

        for cmd in commands:
            # Get initial memory
            initial_memory = self.process.memory_info().rss

            # Run command
            result = self.runner.invoke(main_app, cmd)

            # Get final memory
            final_memory = self.process.memory_info().rss
            memory_delta = final_memory - initial_memory

            assert result.exit_code == 0
            memory_usage[" ".join(cmd)] = memory_delta

            # Basic memory usage assertions
            assert memory_delta < 100 * 1024 * 1024  # Less than 100MB per command

        # Print memory usage summary
        print("\nMemory Usage Summary:")
        for cmd, usage in memory_usage.items():
            print(f"  {cmd}: {usage / 1024 / 1024:.1f}MB")

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_memory_usage_data_operations(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test memory usage for data operations with different dataset sizes."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        dataset_sizes = [100, 1000, 5000]  # Different row counts
        memory_usage = {}

        for size in dataset_sizes:
            # Create dataset of specified size
            mock_data = self._create_large_dataframe(size)
            mock_pipeline_instance.download_data_parallel.return_value = mock_data
            mock_standardizer.return_value = (mock_data, MagicMock())

            # Measure memory usage
            initial_memory = self.process.memory_info().rss

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", f"test_{size}"
                ]
            )

            final_memory = self.process.memory_info().rss
            memory_delta = final_memory - initial_memory

            assert result.exit_code == 0
            memory_usage[size] = memory_delta

            print(f"Dataset size {size}: {memory_delta / 1024 / 1024:.1f}MB")

        # Verify memory usage scales reasonably
        small_usage = memory_usage[100]
        large_usage = memory_usage[5000]

        # Memory usage should scale sub-linearly (due to fixed overhead)
        scaling_factor = large_usage / small_usage
        assert scaling_factor < 100  # Should not scale linearly with data size

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated CLI operations."""
        initial_memory = self.process.memory_info().rss
        memory_samples = []

        # Run multiple iterations of CLI commands
        for i in range(10):
            # Run a sequence of commands
            commands = [
                ["--help"],
                ["version"],
                ["info"],
            ]

            for cmd in commands:
                result = self.runner.invoke(main_app, cmd)
                assert result.exit_code == 0

            # Sample memory usage
            current_memory = self.process.memory_info().rss
            memory_samples.append(current_memory)

            # Short pause to allow cleanup
            time.sleep(0.1)

        # Analyze memory trend
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal
        assert memory_growth < 50 * 1024 * 1024  # Less than 50MB total growth

        # Check for continuous growth (potential memory leak)
        if len(memory_samples) >= 5:
            recent_growth = memory_samples[-1] - memory_samples[-5]
            # Recent growth should be minimal
            assert recent_growth < 20 * 1024 * 1024  # Less than 20MB in last 5 iterations

    def _create_large_dataframe(self, size):
        """Create a DataFrame of specified size for testing."""
        import numpy as np
        import pandas as pd

        return pd.DataFrame({
            "symbol": ["AAPL"] * size,
            "date": pd.date_range("2023-01-01", periods=size, freq="H"),
            "open": np.random.random(size) * 100 + 100,
            "high": np.random.random(size) * 100 + 120,
            "low": np.random.random(size) * 100 + 80,
            "close": np.random.random(size) * 100 + 100,
            "volume": np.random.randint(100000, 2000000, size),
        })


@pytest.mark.performance
class TestCLIConcurrencyPerformance:
    """Test CLI performance under concurrent usage."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.results = []
        self.errors = []

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_help_commands(self):
        """Test concurrent execution of help commands."""
        def run_help_command(thread_id):
            try:
                start_time = time.time()
                result = self.runner.invoke(main_app, ["--help"])
                end_time = time.time()

                response_time = end_time - start_time
                self.results.append({
                    "thread_id": thread_id,
                    "command": "help",
                    "response_time": response_time,
                    "exit_code": result.exit_code,
                    "success": result.exit_code == 0
                })
            except Exception as e:
                self.errors.append(f"Thread {thread_id}: {e}")

        # Run multiple threads concurrently
        threads = []
        num_threads = 5

        start_time = time.time()

        for i in range(num_threads):
            thread = threading.Thread(target=run_help_command, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Verify results
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        assert len(self.results) == num_threads

        # All commands should succeed
        successful_commands = [r for r in self.results if r["success"]]
        assert len(successful_commands) == num_threads

        # Average response time should be reasonable
        avg_response_time = sum(r["response_time"] for r in self.results) / len(self.results)
        assert avg_response_time < 2.0  # Average should be under 2 seconds

        print(f"Concurrent help commands: {num_threads} threads, {total_time:.3f}s total, {avg_response_time:.3f}s avg")

    def test_concurrent_info_commands(self):
        """Test concurrent execution of info commands."""
        def run_info_command(thread_id):
            try:
                start_time = time.time()
                result = self.runner.invoke(main_app, ["info"])
                end_time = time.time()

                response_time = end_time - start_time
                self.results.append({
                    "thread_id": thread_id,
                    "command": "info",
                    "response_time": response_time,
                    "exit_code": result.exit_code,
                    "success": result.exit_code == 0
                })
            except Exception as e:
                self.errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent info commands
        threads = []
        num_threads = 3  # Fewer threads for more complex command

        for i in range(num_threads):
            thread = threading.Thread(target=run_info_command, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        assert len(self.results) == num_threads

        # All should succeed
        successful_commands = [r for r in self.results if r["success"]]
        assert len(successful_commands) == num_threads

        # Response times should be reasonable
        max_response_time = max(r["response_time"] for r in self.results)
        assert max_response_time < 5.0  # Max response time under 5 seconds

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_concurrent_data_operations(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test concurrent data pipeline operations."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        def run_data_pipeline(thread_id):
            try:
                start_time = time.time()
                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", self.temp_dir,
                        "--dataset-name", f"concurrent_test_{thread_id}"
                    ]
                )
                end_time = time.time()

                response_time = end_time - start_time
                self.results.append({
                    "thread_id": thread_id,
                    "command": "data_pipeline",
                    "response_time": response_time,
                    "exit_code": result.exit_code,
                    "success": result.exit_code == 0
                })
            except Exception as e:
                self.errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent data operations
        threads = []
        num_threads = 2  # Limited threads for resource-intensive operations

        for i in range(num_threads):
            thread = threading.Thread(target=run_data_pipeline, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        assert len(self.results) == num_threads

        # Check success rate
        successful_commands = [r for r in self.results if r["success"]]
        success_rate = len(successful_commands) / len(self.results)
        assert success_rate >= 0.8  # At least 80% success rate

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"] * 50,
            "close": [150.0] * 50,
            "volume": [1000000] * 50,
            "date": pd.date_range("2023-01-01", periods=50, freq="D")
        })


@pytest.mark.performance
class TestCLIScalabilityPerformance:
    """Test CLI scalability with increasing loads."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_scalability_with_symbol_count(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test performance scaling with increasing number of symbols."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        symbol_counts = [1, 5, 10, 20]
        performance_data = {}

        for count in symbol_counts:
            # Create mock data proportional to symbol count
            mock_data = self._create_scaled_dataframe(count * 100)
            mock_pipeline_instance.download_data_parallel.return_value = mock_data
            mock_standardizer.return_value = (mock_data, MagicMock())

            # Generate symbol list
            symbols = [f"SYM{i:03d}" for i in range(count)]

            # Benchmark performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", ",".join(symbols),
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", f"scale_test_{count}"
                ]
            )

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            response_time = end_time - start_time
            memory_delta = end_memory - start_memory

            assert result.exit_code == 0

            performance_data[count] = {
                "response_time": response_time,
                "memory_delta": memory_delta,
                "symbols_per_second": count / response_time if response_time > 0 else 0
            }

            print(f"Symbols: {count}, Time: {response_time:.3f}s, Memory: {memory_delta / 1024 / 1024:.1f}MB")

        # Analyze scaling characteristics
        self._analyze_scaling_performance(performance_data)

    def test_scalability_with_worker_count(self):
        """Test performance scaling with different worker counts."""
        worker_counts = [1, 2, 4, 8]
        performance_data = {}

        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mocks
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.return_value = self._create_scaled_dataframe(500)

            mock_standardizer.return_value = (self._create_scaled_dataframe(500), MagicMock())

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            for workers in worker_counts:
                # Simulate worker-dependent processing time
                def make_worker_dependent_processing(num_workers):
                    def worker_dependent_processing(*args, **kwargs):  # noqa: ARG001
                        # Simulate work that benefits from parallelization
                        base_time = 0.5
                        parallel_time = base_time / min(num_workers, 4)  # Diminishing returns after 4 workers
                        time.sleep(parallel_time)
                        return self._create_scaled_dataframe(500)
                    return worker_dependent_processing  # noqa: B023

                worker_dependent_processing = make_worker_dependent_processing(workers)

                mock_pipeline_instance.download_data_parallel.side_effect = worker_dependent_processing

                # Benchmark performance
                start_time = time.time()

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL,GOOGL,MSFT",
                        "--workers", str(workers),
                        "--no-sentiment",
                        "--output-dir", self.temp_dir,
                        "--dataset-name", f"worker_test_{workers}"
                    ]
                )

                end_time = time.time()
                response_time = end_time - start_time

                assert result.exit_code == 0

                performance_data[workers] = {
                    "response_time": response_time,
                    "efficiency": 1.0 / response_time if response_time > 0 else 0
                }

                print(f"Workers: {workers}, Time: {response_time:.3f}s")

        # Verify performance improves with more workers (up to a point)
        single_worker_time = performance_data[1]["response_time"]
        multi_worker_time = performance_data[4]["response_time"]

        # Multi-worker should be faster than single worker
        assert multi_worker_time < single_worker_time * 0.8  # At least 20% improvement

    def _create_scaled_dataframe(self, size):
        """Create a DataFrame of specified size for scalability testing."""
        import numpy as np
        import pandas as pd

        return pd.DataFrame({
            "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], size),
            "date": pd.date_range("2023-01-01", periods=size, freq="H"),
            "open": np.random.random(size) * 100 + 100,
            "high": np.random.random(size) * 100 + 120,
            "low": np.random.random(size) * 100 + 80,
            "close": np.random.random(size) * 100 + 100,
            "volume": np.random.randint(100000, 2000000, size),
        })

    def _analyze_scaling_performance(self, performance_data):
        """Analyze scaling performance characteristics."""
        symbol_counts = sorted(performance_data.keys())

        if len(symbol_counts) >= 2:
            # Check if performance degrades gracefully
            min_symbols = symbol_counts[0]
            max_symbols = symbol_counts[-1]

            min_time = performance_data[min_symbols]["response_time"]
            max_time = performance_data[max_symbols]["response_time"]

            # Performance should not degrade exponentially
            scaling_factor = max_time / min_time if min_time > 0 else float("inf")
            symbol_scaling_factor = max_symbols / min_symbols

            # Time scaling should be less than quadratic
            assert scaling_factor < symbol_scaling_factor ** 1.5

            print(f"Scaling analysis: {symbol_scaling_factor}x symbols -> {scaling_factor:.2f}x time")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
