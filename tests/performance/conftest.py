"""
Performance testing configuration and fixtures for Trading RL Agent.

Provides fixtures for:
- Performance monitoring
- Memory profiling
- CPU profiling
- Benchmark data generation
- Load testing scenarios
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import psutil
import pytest
from memory_profiler import memory_usage

from trading_rl_agent.core.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance testing."""
    return {
        "benchmark_iterations": 100,
        "stress_test_duration": 300,  # 5 minutes
        "memory_threshold_mb": 1024,  # 1GB
        "cpu_threshold_percent": 80,
        "timeout_seconds": 600,  # 10 minutes
        "large_dataset_size": 100000,
        "concurrent_operations": 10,
        "load_test_users": 50,
    }


@pytest.fixture
def benchmark_data():
    """Generate benchmark data for performance testing."""
    np.random.seed(42)
    
    # Generate large dataset for performance testing
    n_symbols = 100
    n_days = 252 * 3  # 3 years of data
    symbols = [f"SYMBOL_{i:03d}" for i in range(n_symbols)]
    
    data = []
    base_prices = np.random.uniform(10, 500, n_symbols)
    
    for i, symbol in enumerate(symbols):
        price = base_prices[i]
        for day in range(n_days):
            # Generate realistic OHLCV data
            change = np.random.normal(0, 0.02) * price
            price = max(1, price + change)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price + np.random.normal(0, 0.005) * price
            volume = int(np.random.uniform(1000000, 5000000))
            
            data.append({
                "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def large_portfolio_data():
    """Generate large portfolio data for stress testing."""
    np.random.seed(42)
    
    # Generate data for 1000 symbols over 5 years
    n_symbols = 1000
    n_days = 252 * 5
    symbols = [f"PORTFOLIO_{i:04d}" for i in range(n_symbols)]
    
    data = []
    base_prices = np.random.uniform(5, 1000, n_symbols)
    
    for i, symbol in enumerate(symbols):
        price = base_prices[i]
        for day in range(n_days):
            change = np.random.normal(0, 0.015) * price
            price = max(0.1, price + change)
            
            data.append({
                "timestamp": pd.Timestamp("2019-01-01") + pd.Timedelta(days=day),
                "symbol": symbol,
                "close": price,
                "volume": int(np.random.uniform(100000, 10000000)),
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def high_frequency_data():
    """Generate high-frequency data for stress testing."""
    np.random.seed(42)
    
    # Generate 1-minute data for 50 symbols over 1 month
    n_symbols = 50
    n_minutes = 24 * 60 * 30  # 30 days of minute data
    symbols = [f"HF_{i:02d}" for i in range(n_symbols)]
    
    data = []
    base_prices = np.random.uniform(10, 200, n_symbols)
    
    for i, symbol in enumerate(symbols):
        price = base_prices[i]
        for minute in range(n_minutes):
            change = np.random.normal(0, 0.001) * price
            price = max(0.01, price + change)
            
            data.append({
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
                "symbol": symbol,
                "close": price,
                "volume": int(np.random.uniform(100, 10000)),
            })
    
    return pd.DataFrame(data)


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.measurements = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        self.measurements = []
    
    def record_measurement(self, label: str = None):
        """Record current performance metrics."""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_cpu = self.process.cpu_percent()
        
        measurement = {
            "timestamp": current_time,
            "memory_mb": current_memory,
            "cpu_percent": current_cpu,
            "label": label,
        }
        
        if self.start_time:
            measurement["elapsed_time"] = current_time - self.start_time
            measurement["memory_delta_mb"] = current_memory - self.start_memory
            measurement["cpu_delta"] = current_cpu - self.start_cpu
        
        self.measurements.append(measurement)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        if not self.start_time:
            return {}
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        total_time = end_time - self.start_time
        total_memory_delta = end_memory - self.start_memory
        avg_cpu = np.mean([m["cpu_percent"] for m in self.measurements]) if self.measurements else 0
        
        return {
            "total_time_seconds": total_time,
            "total_memory_delta_mb": total_memory_delta,
            "peak_memory_mb": max([m["memory_mb"] for m in self.measurements]) if self.measurements else end_memory,
            "average_cpu_percent": avg_cpu,
            "peak_cpu_percent": max([m["cpu_percent"] for m in self.measurements]) if self.measurements else end_cpu,
            "measurements": self.measurements,
        }


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring fixture."""
    return PerformanceMonitor()


@pytest.fixture
def memory_profiler():
    """Provide memory profiling fixture."""
    def profile_memory(func, *args, **kwargs):
        """Profile memory usage of a function."""
        gc.collect()  # Clean up before measurement
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=300)
        return {
            "max_memory_mb": max(mem_usage),
            "min_memory_mb": min(mem_usage),
            "avg_memory_mb": np.mean(mem_usage),
            "memory_profile": mem_usage,
        }
    
    return profile_memory


@pytest.fixture
def load_test_scenarios():
    """Provide load testing scenarios."""
    return {
        "light_load": {
            "concurrent_users": 10,
            "requests_per_second": 5,
            "duration_seconds": 60,
        },
        "medium_load": {
            "concurrent_users": 50,
            "requests_per_second": 20,
            "duration_seconds": 120,
        },
        "heavy_load": {
            "concurrent_users": 100,
            "requests_per_second": 50,
            "duration_seconds": 180,
        },
        "stress_load": {
            "concurrent_users": 200,
            "requests_per_second": 100,
            "duration_seconds": 300,
        },
    }


@pytest.fixture
def stress_test_scenarios():
    """Provide stress testing scenarios."""
    return {
        "data_overflow": {
            "description": "Test system behavior with excessive data volume",
            "data_size_multiplier": 10,
            "concurrent_operations": 20,
        },
        "memory_pressure": {
            "description": "Test system under memory pressure",
            "memory_limit_mb": 512,
            "large_objects": True,
        },
        "cpu_intensive": {
            "description": "Test system under CPU-intensive operations",
            "parallel_processes": 8,
            "complex_calculations": True,
        },
        "network_latency": {
            "description": "Test system with network latency",
            "latency_ms": 1000,
            "packet_loss_percent": 5,
        },
        "rapid_state_changes": {
            "description": "Test system with rapid state changes",
            "state_changes_per_second": 100,
            "duration_seconds": 60,
        },
    }


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield
    gc.collect()
    if hasattr(gc, 'garbage'):
        gc.garbage.clear()