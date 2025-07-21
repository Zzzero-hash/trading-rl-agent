"""
Metrics collection and tracking for the trading system.

This module provides comprehensive metrics collection capabilities for:
- Trading performance metrics (P&L, Sharpe ratio, etc.)
- System performance metrics (latency, memory usage, etc.)
- Risk metrics (VaR, drawdown, etc.)
- Model performance metrics (accuracy, loss, etc.)
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricPoint:
    """A single metric data point with timestamp and value."""

    timestamp: float
    value: float | int | str
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and tracks various metrics for the trading system."""

    def __init__(self, max_history: int = 10000):
        """Initialize the metrics collector.

        Args:
            max_history: Maximum number of data points to keep in memory
        """
        self.max_history = max_history
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def record_metric(self, name: str, value: float | str, metadata: dict[str, Any] | None = None) -> None:
        """Record a metric value with timestamp.

        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata dictionary
        """
        timestamp = time.time()
        metric_point = MetricPoint(timestamp=timestamp, value=value, metadata=metadata or {})
        self.metrics[name].append(metric_point)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            name: Counter name
            value: Value to increment by (default: 1)
        """
        self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value.

        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value

    def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram.

        Args:
            name: Histogram name
            value: Value to record
        """
        self.histograms[name].append(value)

    def get_metric_history(
        self,
        name: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[MetricPoint]:
        """Get metric history for a specific metric.

        Args:
            name: Metric name
            start_time: Start timestamp filter
            end_time: End timestamp filter

        Returns:
            List of metric points within the time range
        """
        if name not in self.metrics:
            return []

        history = list(self.metrics[name])

        if start_time is not None:
            history = [p for p in history if p.timestamp >= start_time]

        if end_time is not None:
            history = [p for p in history if p.timestamp <= end_time]

        return history

    def get_metric_summary(self, name: str) -> dict[str, Any]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with summary statistics
        """
        history = self.get_metric_history(name)

        if not history:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

        values = [p.value for p in history if isinstance(p.value, int | float)]

        if not values:
            return {
                "count": len(history),
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "latest": history[-1].value if history else None,
        }

    def get_counter_value(self, name: str) -> int:
        """Get current value of a counter.

        Args:
            name: Counter name

        Returns:
            Current counter value
        """
        return self.counters.get(name, 0)

    def get_gauge_value(self, name: str) -> float | None:
        """Get current value of a gauge.

        Args:
            name: Gauge name

        Returns:
            Current gauge value or None if not set
        """
        return self.gauges.get(name)

    def get_histogram_summary(self, name: str) -> dict[str, Any]:
        """Get summary statistics for a histogram.

        Args:
            name: Histogram name

        Returns:
            Dictionary with histogram statistics
        """
        values = self.histograms.get(name, [])

        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "percentiles": {
                "50": np.percentile(values, 50),
                "90": np.percentile(values, 90),
                "95": np.percentile(values, 95),
                "99": np.percentile(values, 99),
            },
        }

    def export_metrics(self) -> dict[str, Any]:
        """Export all metrics for external monitoring systems.

        Returns:
            Dictionary with all current metrics
        """
        return {
            "metrics": {name: self.get_metric_summary(name) for name in self.metrics},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {name: self.get_histogram_summary(name) for name in self.histograms},
            "timestamp": time.time(),
        }

    def clear_metrics(self, name: str | None = None) -> None:
        """Clear metrics data.

        Args:
            name: Specific metric to clear, or None to clear all
        """
        if name is None:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
        else:
            if name in self.metrics:
                self.metrics[name].clear()
            if name in self.counters:
                del self.counters[name]
            if name in self.gauges:
                del self.gauges[name]
            if name in self.histograms:
                del self.histograms[name]
