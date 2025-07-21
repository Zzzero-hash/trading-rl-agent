"""
Performance tests for data processing components.

Tests include:
- Parallel data fetching performance
- Feature engineering speed
- Data pipeline throughput
- Memory usage optimization
- Cache performance
- Data standardization speed
"""

import time

import pytest

from trading_rl_agent.data.data_standardizer import DataStandardizer
from trading_rl_agent.data.features import FeatureEngineer
from trading_rl_agent.data.optimized_dataset_builder import OptimizedDatasetBuilder
from trading_rl_agent.data.parallel_data_fetcher import ParallelDataManager


class TestDataProcessingPerformance:
    """Performance tests for data processing components."""

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_parallel_data_fetching_performance(self, benchmark_data, performance_monitor):
        """Test parallel data fetching performance."""
        # Prepare test data
        symbols = benchmark_data["symbol"].unique()[:50]  # Test with 50 symbols
        start_date = "2020-01-01"
        end_date = "2023-01-01"

        # Initialize parallel data manager
        data_manager = ParallelDataManager(max_workers=4)

        performance_monitor.start_monitoring()

        # Benchmark parallel fetching
        def fetch_data():
            return data_manager.fetch_multiple_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
                show_progress=False,
            )

        # Measure performance
        start_time = time.time()
        result = fetch_data()
        end_time = time.time()

        performance_monitor.record_measurement("data_fetching_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("Data fetching performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Symbols processed: {len(result)}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_feature_engineering_performance(self, benchmark_data, performance_monitor):
        """Test feature engineering performance."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize feature engineer
        feature_engineer = FeatureEngineer()

        performance_monitor.start_monitoring()

        # Benchmark feature engineering
        def engineer_features():
            return feature_engineer.calculate_all_features(test_data)

        # Measure performance
        start_time = time.time()
        result = engineer_features()
        end_time = time.time()

        performance_monitor.record_measurement("feature_engineering_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 30  # Should complete within 30 seconds
        assert metrics["peak_memory_mb"] < 1024  # Should use less than 1GB

        # Log performance metrics
        print("Feature engineering performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Features calculated: {len(result.columns)}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_data_standardization_performance(self, benchmark_data, performance_monitor):
        """Test data standardization performance."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize data standardizer
        standardizer = DataStandardizer()

        performance_monitor.start_monitoring()

        # Benchmark standardization
        def standardize_data():
            return standardizer.standardize_dataset(test_data)

        # Measure performance
        start_time = time.time()
        result = standardize_data()
        end_time = time.time()

        performance_monitor.record_measurement("standardization_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 10  # Should complete within 10 seconds
        assert metrics["peak_memory_mb"] < 512  # Should use less than 512MB

        # Log performance metrics
        print("Data standardization performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_optimized_dataset_builder_performance(self, benchmark_data, performance_monitor):
        """Test optimized dataset builder performance."""
        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize dataset builder
        dataset_builder = OptimizedDatasetBuilder()

        performance_monitor.start_monitoring()

        # Benchmark dataset building
        def build_dataset():
            return dataset_builder.build_optimized_dataset(
                data=test_data,
                sequence_length=60,
                prediction_horizon=5,
                target_column="close",
            )

        # Measure performance
        start_time = time.time()
        result = build_dataset()
        end_time = time.time()

        performance_monitor.record_measurement("dataset_building_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 45  # Should complete within 45 seconds
        assert metrics["peak_memory_mb"] < 1536  # Should use less than 1.5GB

        # Log performance metrics
        print("Dataset builder performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.memory
    def test_memory_usage_optimization(self, large_portfolio_data, memory_profiler):
        """Test memory usage optimization in data processing."""
        # Test memory usage for large dataset processing
        test_data = large_portfolio_data.copy()

        # Profile memory usage for feature engineering
        def process_large_dataset():
            feature_engineer = FeatureEngineer()
            return feature_engineer.calculate_all_features(test_data)

        memory_metrics = memory_profiler(process_large_dataset)

        # Assertions
        assert memory_metrics["max_memory_mb"] < 4096  # Should use less than 4GB
        assert memory_metrics["avg_memory_mb"] < 2048  # Average should be less than 2GB

        # Log memory metrics
        print("Memory usage for large dataset:")
        print(f"  Max memory: {memory_metrics['max_memory_mb']:.2f} MB")
        print(f"  Avg memory: {memory_metrics['avg_memory_mb']:.2f} MB")
        print(f"  Min memory: {memory_metrics['min_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark_data, performance_monitor):
        """Test cache performance for data fetching."""
        from trading_rl_agent.data.parallel_data_fetcher import SmartDataCache

        # Initialize cache
        cache = SmartDataCache(ttl_hours=24)

        # Prepare test data
        symbols = benchmark_data["symbol"].unique()[:10]

        performance_monitor.start_monitoring()

        # Benchmark cache operations
        def cache_operations():
            results = []
            for symbol in symbols:
                # Simulate cache get/fetch operations
                data = cache.get_or_fetch(symbol=symbol, start_date="2020-01-01", end_date="2023-01-01")
                results.append(data)
            return results

        # Measure performance
        start_time = time.time()
        result = cache_operations()
        end_time = time.time()

        performance_monitor.record_measurement("cache_operations_complete")
        metrics = performance_monitor.stop_monitoring()

        # Get cache statistics
        cache_stats = cache.get_stats()

        # Assertions
        assert len(result) == len(symbols)
        assert end_time - start_time < 5  # Should complete within 5 seconds
        assert metrics["peak_memory_mb"] < 256  # Should use less than 256MB

        # Log performance metrics
        print("Cache performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_high_frequency_data_processing(self, high_frequency_data, performance_monitor):
        """Test performance with high-frequency data."""
        # Prepare high-frequency data
        test_data = high_frequency_data.copy()

        # Initialize components
        feature_engineer = FeatureEngineer()
        standardizer = DataStandardizer()

        performance_monitor.start_monitoring()

        # Benchmark high-frequency processing
        def process_high_frequency():
            # Calculate features
            features = feature_engineer.calculate_all_features(test_data)
            # Standardize data
            return standardizer.standardize_dataset(features)

        # Measure performance
        start_time = time.time()
        result = process_high_frequency()
        end_time = time.time()

        performance_monitor.record_measurement("high_frequency_processing_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("High-frequency data processing:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Records processed: {len(result)}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_concurrent_data_operations(self, benchmark_data, performance_monitor):
        """Test performance with concurrent data operations."""
        import concurrent.futures

        # Prepare test data
        test_data = benchmark_data.copy()
        symbols = test_data["symbol"].unique()[:20]

        # Split data by symbols
        data_by_symbol = {}
        for symbol in symbols:
            data_by_symbol[symbol] = test_data[test_data["symbol"] == symbol].copy()

        performance_monitor.start_monitoring()

        # Benchmark concurrent operations
        def process_symbol_data(symbol_data):
            feature_engineer = FeatureEngineer()
            return feature_engineer.calculate_all_features(symbol_data)

        def concurrent_processing():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_symbol_data, data) for data in data_by_symbol.values()]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

        # Measure performance
        start_time = time.time()
        result = concurrent_processing()
        end_time = time.time()

        performance_monitor.record_measurement("concurrent_processing_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert len(result) == len(symbols)
        assert end_time - start_time < 30  # Should complete within 30 seconds
        assert metrics["peak_memory_mb"] < 1536  # Should use less than 1.5GB

        # Log performance metrics
        print("Concurrent data operations:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Symbols processed: {len(result)}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_data_pipeline_throughput(self, benchmark_data, performance_monitor):
        """Test end-to-end data pipeline throughput."""
        from trading_rl_agent.data.pipeline import DataPipeline

        # Prepare test data
        test_data = benchmark_data.copy()

        # Initialize pipeline
        pipeline = DataPipeline()

        performance_monitor.start_monitoring()

        # Benchmark pipeline throughput
        def run_pipeline():
            return pipeline.process_data(test_data)

        # Measure performance
        start_time = time.time()
        result = run_pipeline()
        end_time = time.time()

        performance_monitor.record_measurement("pipeline_complete")
        metrics = performance_monitor.stop_monitoring()

        # Calculate throughput
        records_per_second = len(result) / (end_time - start_time)

        # Assertions
        assert len(result) > 0
        assert end_time - start_time < 90  # Should complete within 90 seconds
        assert records_per_second > 100  # Should process at least 100 records per second
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("Data pipeline throughput:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Records processed: {len(result)}")
        print(f"  Throughput: {records_per_second:.2f} records/second")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
