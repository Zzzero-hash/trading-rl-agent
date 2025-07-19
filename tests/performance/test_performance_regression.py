"""
Performance regression detection for Trading RL Agent.

Features:
- Performance baseline establishment
- Regression detection algorithms
- Historical performance tracking
- Automated regression alerts
- Performance trend analysis
- Threshold-based monitoring
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass, asdict

from trading_rl_agent.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Container for performance metrics."""
    test_name: str
    execution_time: float
    memory_peak_mb: float
    cpu_usage_percent: float
    throughput: Optional[float] = None
    success_rate: Optional[float] = None
    timestamp: datetime = None
    git_commit: Optional[str] = None
    system_info: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceBaseline:
    """Container for performance baselines."""
    test_name: str
    mean_execution_time: float
    std_execution_time: float
    mean_memory_peak_mb: float
    std_memory_peak_mb: float
    mean_cpu_usage_percent: float
    std_cpu_usage_percent: float
    mean_throughput: Optional[float] = None
    std_throughput: Optional[float] = None
    mean_success_rate: Optional[float] = None
    std_success_rate: Optional[float] = None
    sample_count: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class RegressionAlert:
    """Container for regression alerts."""
    test_name: str
    metric_type: str
    current_value: float
    baseline_value: float
    threshold: float
    severity: str  # "warning", "critical"
    timestamp: datetime
    description: str


class PerformanceRegressionDetector:
    """Detect performance regressions in the trading system."""
    
    def __init__(self, storage_path: str = "performance_metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.storage_path / "performance_metrics.json"
        self.baselines_file = self.storage_path / "performance_baselines.json"
        self.alerts_file = self.storage_path / "regression_alerts.json"
        
        # Load existing data
        self.metrics = self._load_metrics()
        self.baselines = self._load_baselines()
        self.alerts = self._load_alerts()
        
        # Regression detection thresholds
        self.thresholds = {
            "execution_time": {"warning": 1.2, "critical": 1.5},  # 20% and 50% increase
            "memory_peak_mb": {"warning": 1.15, "critical": 1.3},  # 15% and 30% increase
            "cpu_usage_percent": {"warning": 1.1, "critical": 1.25},  # 10% and 25% increase
            "throughput": {"warning": 0.9, "critical": 0.8},  # 10% and 20% decrease
            "success_rate": {"warning": 0.95, "critical": 0.9},  # 5% and 10% decrease
        }
    
    def _load_metrics(self) -> List[PerformanceMetric]:
        """Load performance metrics from storage."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    return [PerformanceMetric(**metric) for metric in data]
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
        return []
    
    def _save_metrics(self) -> None:
        """Save performance metrics to storage."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump([asdict(metric) for metric in self.metrics], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from storage."""
        if self.baselines_file.exists():
            try:
                with open(self.baselines_file, 'r') as f:
                    data = json.load(f)
                    return {
                        test_name: PerformanceBaseline(**baseline)
                        for test_name, baseline in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")
        return {}
    
    def _save_baselines(self) -> None:
        """Save performance baselines to storage."""
        try:
            with open(self.baselines_file, 'w') as f:
                json.dump(
                    {name: asdict(baseline) for name, baseline in self.baselines.items()},
                    f, indent=2, default=str
                )
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")
    
    def _load_alerts(self) -> List[RegressionAlert]:
        """Load regression alerts from storage."""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    return [RegressionAlert(**alert) for alert in data]
            except Exception as e:
                logger.warning(f"Failed to load alerts: {e}")
        return []
    
    def _save_alerts(self) -> None:
        """Save regression alerts to storage."""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump([asdict(alert) for alert in self.alerts], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a new performance metric."""
        self.metrics.append(metric)
        self._save_metrics()
        
        # Check for regressions
        alerts = self._check_regressions(metric)
        if alerts:
            self.alerts.extend(alerts)
            self._save_alerts()
    
    def _check_regressions(self, metric: PerformanceMetric) -> List[RegressionAlert]:
        """Check for performance regressions."""
        alerts = []
        
        if metric.test_name not in self.baselines:
            return alerts
        
        baseline = self.baselines[metric.test_name]
        
        # Check execution time regression
        if baseline.mean_execution_time > 0:
            ratio = metric.execution_time / baseline.mean_execution_time
            if ratio > self.thresholds["execution_time"]["critical"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="execution_time",
                    current_value=metric.execution_time,
                    baseline_value=baseline.mean_execution_time,
                    threshold=self.thresholds["execution_time"]["critical"],
                    severity="critical",
                    timestamp=metric.timestamp,
                    description=f"Execution time increased by {(ratio-1)*100:.1f}%"
                ))
            elif ratio > self.thresholds["execution_time"]["warning"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="execution_time",
                    current_value=metric.execution_time,
                    baseline_value=baseline.mean_execution_time,
                    threshold=self.thresholds["execution_time"]["warning"],
                    severity="warning",
                    timestamp=metric.timestamp,
                    description=f"Execution time increased by {(ratio-1)*100:.1f}%"
                ))
        
        # Check memory regression
        if baseline.mean_memory_peak_mb > 0:
            ratio = metric.memory_peak_mb / baseline.mean_memory_peak_mb
            if ratio > self.thresholds["memory_peak_mb"]["critical"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="memory_peak_mb",
                    current_value=metric.memory_peak_mb,
                    baseline_value=baseline.mean_memory_peak_mb,
                    threshold=self.thresholds["memory_peak_mb"]["critical"],
                    severity="critical",
                    timestamp=metric.timestamp,
                    description=f"Memory usage increased by {(ratio-1)*100:.1f}%"
                ))
            elif ratio > self.thresholds["memory_peak_mb"]["warning"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="memory_peak_mb",
                    current_value=metric.memory_peak_mb,
                    baseline_value=baseline.mean_memory_peak_mb,
                    threshold=self.thresholds["memory_peak_mb"]["warning"],
                    severity="warning",
                    timestamp=metric.timestamp,
                    description=f"Memory usage increased by {(ratio-1)*100:.1f}%"
                ))
        
        # Check CPU usage regression
        if baseline.mean_cpu_usage_percent > 0:
            ratio = metric.cpu_usage_percent / baseline.mean_cpu_usage_percent
            if ratio > self.thresholds["cpu_usage_percent"]["critical"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="cpu_usage_percent",
                    current_value=metric.cpu_usage_percent,
                    baseline_value=baseline.mean_cpu_usage_percent,
                    threshold=self.thresholds["cpu_usage_percent"]["critical"],
                    severity="critical",
                    timestamp=metric.timestamp,
                    description=f"CPU usage increased by {(ratio-1)*100:.1f}%"
                ))
            elif ratio > self.thresholds["cpu_usage_percent"]["warning"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="cpu_usage_percent",
                    current_value=metric.cpu_usage_percent,
                    baseline_value=baseline.mean_cpu_usage_percent,
                    threshold=self.thresholds["cpu_usage_percent"]["warning"],
                    severity="warning",
                    timestamp=metric.timestamp,
                    description=f"CPU usage increased by {(ratio-1)*100:.1f}%"
                ))
        
        # Check throughput regression
        if metric.throughput is not None and baseline.mean_throughput is not None:
            ratio = metric.throughput / baseline.mean_throughput
            if ratio < self.thresholds["throughput"]["critical"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="throughput",
                    current_value=metric.throughput,
                    baseline_value=baseline.mean_throughput,
                    threshold=self.thresholds["throughput"]["critical"],
                    severity="critical",
                    timestamp=metric.timestamp,
                    description=f"Throughput decreased by {(1-ratio)*100:.1f}%"
                ))
            elif ratio < self.thresholds["throughput"]["warning"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="throughput",
                    current_value=metric.throughput,
                    baseline_value=baseline.mean_throughput,
                    threshold=self.thresholds["throughput"]["warning"],
                    severity="warning",
                    timestamp=metric.timestamp,
                    description=f"Throughput decreased by {(1-ratio)*100:.1f}%"
                ))
        
        # Check success rate regression
        if metric.success_rate is not None and baseline.mean_success_rate is not None:
            ratio = metric.success_rate / baseline.mean_success_rate
            if ratio < self.thresholds["success_rate"]["critical"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="success_rate",
                    current_value=metric.success_rate,
                    baseline_value=baseline.mean_success_rate,
                    threshold=self.thresholds["success_rate"]["critical"],
                    severity="critical",
                    timestamp=metric.timestamp,
                    description=f"Success rate decreased by {(1-ratio)*100:.1f}%"
                ))
            elif ratio < self.thresholds["success_rate"]["warning"]:
                alerts.append(RegressionAlert(
                    test_name=metric.test_name,
                    metric_type="success_rate",
                    current_value=metric.success_rate,
                    baseline_value=baseline.mean_success_rate,
                    threshold=self.thresholds["success_rate"]["warning"],
                    severity="warning",
                    timestamp=metric.timestamp,
                    description=f"Success rate decreased by {(1-ratio)*100:.1f}%"
                ))
        
        return alerts
    
    def update_baselines(self, min_samples: int = 5, max_age_days: int = 30) -> None:
        """Update performance baselines from recent metrics."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Group metrics by test name
        test_metrics = {}
        for metric in self.metrics:
            if metric.timestamp >= cutoff_date:
                if metric.test_name not in test_metrics:
                    test_metrics[metric.test_name] = []
                test_metrics[metric.test_name].append(metric)
        
        # Calculate baselines for each test
        for test_name, metrics in test_metrics.items():
            if len(metrics) >= min_samples:
                execution_times = [m.execution_time for m in metrics]
                memory_peaks = [m.memory_peak_mb for m in metrics]
                cpu_usages = [m.cpu_usage_percent for m in metrics]
                throughputs = [m.throughput for m in metrics if m.throughput is not None]
                success_rates = [m.success_rate for m in metrics if m.success_rate is not None]
                
                baseline = PerformanceBaseline(
                    test_name=test_name,
                    mean_execution_time=np.mean(execution_times),
                    std_execution_time=np.std(execution_times),
                    mean_memory_peak_mb=np.mean(memory_peaks),
                    std_memory_peak_mb=np.std(memory_peaks),
                    mean_cpu_usage_percent=np.mean(cpu_usages),
                    std_cpu_usage_percent=np.std(cpu_usages),
                    mean_throughput=np.mean(throughputs) if throughputs else None,
                    std_throughput=np.std(throughputs) if throughputs else None,
                    mean_success_rate=np.mean(success_rates) if success_rates else None,
                    std_success_rate=np.std(success_rates) if success_rates else None,
                    sample_count=len(metrics),
                    last_updated=datetime.now()
                )
                
                self.baselines[test_name] = baseline
        
        self._save_baselines()
    
    def get_performance_trends(self, test_name: str, days: int = 30) -> Dict:
        """Get performance trends for a specific test."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter metrics for the test
        test_metrics = [
            m for m in self.metrics
            if m.test_name == test_name and m.timestamp >= cutoff_date
        ]
        
        if not test_metrics:
            return {"error": "No metrics found for the specified test and time period"}
        
        # Sort by timestamp
        test_metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate trends
        execution_times = [m.execution_time for m in test_metrics]
        memory_peaks = [m.memory_peak_mb for m in test_metrics]
        cpu_usages = [m.cpu_usage_percent for m in test_metrics]
        
        # Linear regression for trend analysis
        x = np.arange(len(test_metrics))
        
        execution_trend = np.polyfit(x, execution_times, 1)[0] if len(execution_times) > 1 else 0
        memory_trend = np.polyfit(x, memory_peaks, 1)[0] if len(memory_peaks) > 1 else 0
        cpu_trend = np.polyfit(x, cpu_usages, 1)[0] if len(cpu_usages) > 1 else 0
        
        return {
            "test_name": test_name,
            "period_days": days,
            "metric_count": len(test_metrics),
            "execution_time": {
                "current": execution_times[-1],
                "mean": np.mean(execution_times),
                "trend": execution_trend,
                "trend_direction": "increasing" if execution_trend > 0 else "decreasing"
            },
            "memory_peak_mb": {
                "current": memory_peaks[-1],
                "mean": np.mean(memory_peaks),
                "trend": memory_trend,
                "trend_direction": "increasing" if memory_trend > 0 else "decreasing"
            },
            "cpu_usage_percent": {
                "current": cpu_usages[-1],
                "mean": np.mean(cpu_usages),
                "trend": cpu_trend,
                "trend_direction": "increasing" if cpu_trend > 0 else "decreasing"
            }
        }
    
    def get_recent_alerts(self, days: int = 7) -> List[RegressionAlert]:
        """Get recent regression alerts."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_date]
    
    def generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_metrics": len(self.metrics),
                "total_baselines": len(self.baselines),
                "total_alerts": len(self.alerts),
                "recent_alerts": len(self.get_recent_alerts())
            },
            "baselines": {},
            "recent_trends": {},
            "critical_alerts": []
        }
        
        # Add baseline information
        for test_name, baseline in self.baselines.items():
            report["baselines"][test_name] = {
                "sample_count": baseline.sample_count,
                "last_updated": baseline.last_updated.isoformat(),
                "execution_time": {
                    "mean": baseline.mean_execution_time,
                    "std": baseline.std_execution_time
                },
                "memory_peak_mb": {
                    "mean": baseline.mean_memory_peak_mb,
                    "std": baseline.std_memory_peak_mb
                },
                "cpu_usage_percent": {
                    "mean": baseline.mean_cpu_usage_percent,
                    "std": baseline.std_cpu_usage_percent
                }
            }
        
        # Add recent trends for each test
        for test_name in self.baselines.keys():
            trends = self.get_performance_trends(test_name, days=7)
            if "error" not in trends:
                report["recent_trends"][test_name] = trends
        
        # Add critical alerts
        critical_alerts = [alert for alert in self.alerts if alert.severity == "critical"]
        report["critical_alerts"] = [
            {
                "test_name": alert.test_name,
                "metric_type": alert.metric_type,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in critical_alerts[-10:]  # Last 10 critical alerts
        ]
        
        return report


class TestPerformanceRegression:
    """Tests for performance regression detection."""
    
    @pytest.fixture
    def regression_detector(self, tmp_path):
        """Create a performance regression detector for testing."""
        storage_path = tmp_path / "performance_metrics"
        return PerformanceRegressionDetector(str(storage_path))
    
    @pytest.mark.performance
    def test_metric_recording(self, regression_detector):
        """Test recording performance metrics."""
        # Create test metrics
        metric1 = PerformanceMetric(
            test_name="test_data_processing",
            execution_time=1.5,
            memory_peak_mb=512,
            cpu_usage_percent=25.0,
            throughput=1000.0,
            success_rate=0.95
        )
        
        metric2 = PerformanceMetric(
            test_name="test_risk_calculation",
            execution_time=2.0,
            memory_peak_mb=1024,
            cpu_usage_percent=50.0,
            throughput=500.0,
            success_rate=0.98
        )
        
        # Record metrics
        regression_detector.record_metric(metric1)
        regression_detector.record_metric(metric2)
        
        # Verify metrics were recorded
        assert len(regression_detector.metrics) == 2
        assert regression_detector.metrics[0].test_name == "test_data_processing"
        assert regression_detector.metrics[1].test_name == "test_risk_calculation"
    
    @pytest.mark.performance
    def test_baseline_establishment(self, regression_detector):
        """Test establishing performance baselines."""
        # Record multiple metrics for the same test
        for i in range(10):
            metric = PerformanceMetric(
                test_name="test_feature_engineering",
                execution_time=1.0 + np.random.normal(0, 0.1),
                memory_peak_mb=500 + np.random.normal(0, 50),
                cpu_usage_percent=30 + np.random.normal(0, 5),
                throughput=800 + np.random.normal(0, 100),
                success_rate=0.95 + np.random.normal(0, 0.02)
            )
            regression_detector.record_metric(metric)
        
        # Update baselines
        regression_detector.update_baselines(min_samples=5)
        
        # Verify baseline was created
        assert "test_feature_engineering" in regression_detector.baselines
        baseline = regression_detector.baselines["test_feature_engineering"]
        assert baseline.sample_count == 10
        assert baseline.mean_execution_time > 0
        assert baseline.mean_memory_peak_mb > 0
        assert baseline.mean_cpu_usage_percent > 0
    
    @pytest.mark.performance
    def test_regression_detection(self, regression_detector):
        """Test regression detection."""
        # Establish baseline
        for i in range(5):
            metric = PerformanceMetric(
                test_name="test_model_training",
                execution_time=10.0,
                memory_peak_mb=1000,
                cpu_usage_percent=40.0,
                throughput=100.0,
                success_rate=0.95
            )
            regression_detector.record_metric(metric)
        
        regression_detector.update_baselines(min_samples=3)
        
        # Record a metric that should trigger regression alerts
        regression_metric = PerformanceMetric(
            test_name="test_model_training",
            execution_time=20.0,  # 100% increase (should trigger critical)
            memory_peak_mb=1200,  # 20% increase (should trigger warning)
            cpu_usage_percent=60.0,  # 50% increase (should trigger critical)
            throughput=80.0,  # 20% decrease (should trigger warning)
            success_rate=0.85  # 10.5% decrease (should trigger critical)
        )
        
        # Record the metric and check for alerts
        regression_detector.record_metric(regression_metric)
        
        # Verify alerts were generated
        assert len(regression_detector.alerts) > 0
        
        # Check for specific alert types
        alert_types = [alert.metric_type for alert in regression_detector.alerts]
        assert "execution_time" in alert_types
        assert "memory_peak_mb" in alert_types
        assert "cpu_usage_percent" in alert_types
        assert "throughput" in alert_types
        assert "success_rate" in alert_types
        
        # Check for critical alerts
        critical_alerts = [alert for alert in regression_detector.alerts if alert.severity == "critical"]
        assert len(critical_alerts) >= 3  # execution_time, cpu_usage_percent, success_rate
    
    @pytest.mark.performance
    def test_performance_trends(self, regression_detector):
        """Test performance trend analysis."""
        # Record metrics with a clear trend
        for i in range(10):
            metric = PerformanceMetric(
                test_name="test_data_pipeline",
                execution_time=1.0 + i * 0.1,  # Increasing trend
                memory_peak_mb=500 - i * 5,    # Decreasing trend
                cpu_usage_percent=30 + i * 2,  # Increasing trend
                throughput=1000 - i * 10,      # Decreasing trend
                success_rate=0.95
            )
            regression_detector.record_metric(metric)
        
        # Get trends
        trends = regression_detector.get_performance_trends("test_data_pipeline", days=30)
        
        # Verify trends were detected
        assert trends["execution_time"]["trend_direction"] == "increasing"
        assert trends["memory_peak_mb"]["trend_direction"] == "decreasing"
        assert trends["cpu_usage_percent"]["trend_direction"] == "increasing"
        assert trends["metric_count"] == 10
    
    @pytest.mark.performance
    def test_performance_report_generation(self, regression_detector):
        """Test performance report generation."""
        # Record some metrics and establish baselines
        for i in range(5):
            metric = PerformanceMetric(
                test_name="test_end_to_end",
                execution_time=5.0 + np.random.normal(0, 0.5),
                memory_peak_mb=800 + np.random.normal(0, 100),
                cpu_usage_percent=60 + np.random.normal(0, 10),
                throughput=200 + np.random.normal(0, 20),
                success_rate=0.90 + np.random.normal(0, 0.05)
            )
            regression_detector.record_metric(metric)
        
        regression_detector.update_baselines(min_samples=3)
        
        # Generate report
        report = regression_detector.generate_performance_report()
        
        # Verify report structure
        assert "summary" in report
        assert "baselines" in report
        assert "recent_trends" in report
        assert "critical_alerts" in report
        
        # Verify summary
        summary = report["summary"]
        assert summary["total_metrics"] == 5
        assert summary["total_baselines"] == 1
        assert "test_end_to_end" in report["baselines"]
    
    @pytest.mark.performance
    def test_alert_filtering(self, regression_detector):
        """Test alert filtering by time period."""
        # Create old and recent alerts
        old_alert = RegressionAlert(
            test_name="test_old",
            metric_type="execution_time",
            current_value=20.0,
            baseline_value=10.0,
            threshold=1.5,
            severity="critical",
            timestamp=datetime.now() - timedelta(days=10),
            description="Old regression"
        )
        
        recent_alert = RegressionAlert(
            test_name="test_recent",
            metric_type="memory_peak_mb",
            current_value=1500,
            baseline_value=1000,
            threshold=1.3,
            severity="warning",
            timestamp=datetime.now() - timedelta(days=2),
            description="Recent regression"
        )
        
        regression_detector.alerts = [old_alert, recent_alert]
        
        # Get recent alerts (last 7 days)
        recent_alerts = regression_detector.get_recent_alerts(days=7)
        
        # Verify only recent alerts are returned
        assert len(recent_alerts) == 1
        assert recent_alerts[0].test_name == "test_recent"
    
    @pytest.mark.performance
    def test_threshold_configuration(self, regression_detector):
        """Test threshold configuration for regression detection."""
        # Modify thresholds
        regression_detector.thresholds["execution_time"]["warning"] = 1.1  # 10% increase
        regression_detector.thresholds["execution_time"]["critical"] = 1.2  # 20% increase
        
        # Establish baseline
        for i in range(3):
            metric = PerformanceMetric(
                test_name="test_threshold",
                execution_time=10.0,
                memory_peak_mb=1000,
                cpu_usage_percent=40.0
            )
            regression_detector.record_metric(metric)
        
        regression_detector.update_baselines(min_samples=2)
        
        # Test with different performance levels
        # 15% increase - should trigger warning but not critical
        warning_metric = PerformanceMetric(
            test_name="test_threshold",
            execution_time=11.5,
            memory_peak_mb=1000,
            cpu_usage_percent=40.0
        )
        
        regression_detector.record_metric(warning_metric)
        
        # Check for warning alert
        warning_alerts = [alert for alert in regression_detector.alerts if alert.severity == "warning"]
        assert len(warning_alerts) > 0
        
        # 25% increase - should trigger critical
        critical_metric = PerformanceMetric(
            test_name="test_threshold",
            execution_time=12.5,
            memory_peak_mb=1000,
            cpu_usage_percent=40.0
        )
        
        regression_detector.record_metric(critical_metric)
        
        # Check for critical alert
        critical_alerts = [alert for alert in regression_detector.alerts if alert.severity == "critical"]
        assert len(critical_alerts) > 0