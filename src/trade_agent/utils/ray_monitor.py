"""
Ray Cluster Monitoring and Observability Module.

This module provides comprehensive monitoring capabilities for Ray clusters,
including resource utilization tracking, performance metrics, and cluster health monitoring.
"""

import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import ray

from .cluster import check_ray_cluster_status, get_optimal_worker_count, validate_cluster_health

logger = logging.getLogger(__name__)


class RayClusterMonitor:
    """Monitor Ray cluster performance and health metrics."""

    def __init__(
        self,
        history_size: int = 100,
        update_interval: float = 5.0,
        enable_detailed_metrics: bool = True
    ):
        """
        Initialize Ray cluster monitor.

        Args:
            history_size: Number of historical data points to keep
            update_interval: Interval between metric updates in seconds
            enable_detailed_metrics: Whether to collect detailed node-level metrics
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.enable_detailed_metrics = enable_detailed_metrics

        # Metric history storage
        self.metrics_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.node_metrics_history: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_size))
        )

        # Monitoring state
        self.monitoring_active = False
        self.last_update = 0.0
        self.start_time = time.time()

        # Performance counters
        self.task_counters = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "pending": 0
        }

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_utilization": 90.0,
            "memory_utilization": 85.0,
            "gpu_utilization": 95.0,
            "node_down_time": 30.0,  # seconds
            "task_failure_rate": 10.0  # percentage
        }

        # Active alerts
        self.active_alerts: list[dict[str, Any]] = []

    def start_monitoring(self) -> bool:
        """
        Start cluster monitoring.

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if not ray.is_initialized():
            logger.error("Cannot start monitoring: Ray is not initialized")
            return False

        try:
            # Initial health check
            health = validate_cluster_health()
            if not health["healthy"]:
                logger.warning(f"Starting monitoring on unhealthy cluster: {health['reason']}")

            self.monitoring_active = True
            self.start_time = time.time()
            logger.info("✅ Ray cluster monitoring started")

            # Collect initial metrics
            self._collect_metrics()

            return True

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False

    def stop_monitoring(self) -> None:
        """Stop cluster monitoring."""
        self.monitoring_active = False
        logger.info("🛑 Ray cluster monitoring stopped")

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect current cluster metrics."""
        if not ray.is_initialized():
            return {}

        try:
            current_time = time.time()
            timestamp = datetime.now()

            # Basic cluster metrics
            health = validate_cluster_health()
            ray.cluster_resources()
            ray.available_resources()
            nodes = ray.nodes()

            # Calculate metrics
            metrics = {
                "timestamp": timestamp.isoformat(),
                "uptime": current_time - self.start_time,
                "healthy": health["healthy"],
                "total_nodes": len(nodes),
                "alive_nodes": len([n for n in nodes if n.get("Alive", False)]),
                "resources": health["resources"],
                "task_stats": self._get_task_stats(),
            }

            # Store in history
            for key, value in metrics.items():
                if key != "timestamp" and isinstance(value, int | float | bool):
                    self.metrics_history[key].append(value)

            # Store resource metrics
            for resource_type in ["cpu", "gpu", "memory"]:
                if resource_type in health["resources"]:
                    resource_data = health["resources"][resource_type]
                    self.metrics_history[f"{resource_type}_total"].append(resource_data["total"])
                    self.metrics_history[f"{resource_type}_available"].append(resource_data["available"])
                    self.metrics_history[f"{resource_type}_utilization"].append(resource_data["utilization_percent"])

            # Collect detailed node metrics if enabled
            if self.enable_detailed_metrics:
                self._collect_node_metrics(nodes)

            # Check for alerts
            self._check_alerts(metrics)

            self.last_update = current_time
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}

    def _get_task_stats(self) -> dict[str, Any]:
        """Get task execution statistics."""
        try:
            # Note: In a real implementation, you would collect actual task stats
            # from Ray's internal metrics. For now, we use our counters.
            total_tasks = sum(self.task_counters.values())

            return {
                "total_submitted": self.task_counters["submitted"],
                "total_completed": self.task_counters["completed"],
                "total_failed": self.task_counters["failed"],
                "pending": self.task_counters["pending"],
                "success_rate": (self.task_counters["completed"] / max(1, total_tasks)) * 100,
                "failure_rate": (self.task_counters["failed"] / max(1, total_tasks)) * 100,
            }
        except Exception as e:
            logger.error(f"Failed to get task stats: {e}")
            return {}

    def _collect_node_metrics(self, nodes: list[dict[str, Any]]) -> None:
        """Collect detailed metrics for each node."""
        try:
            for node in nodes:
                node_id = node.get("NodeID", "unknown")
                node.get("NodeName", node_id)

                # Store node-specific metrics
                self.node_metrics_history[node_id]["alive"].append(node.get("Alive", False))

                # Resource usage per node (if available)
                resources = node.get("Resources", {})
                for resource, amount in resources.items():
                    self.node_metrics_history[node_id][f"{resource}_total"].append(amount)

        except Exception as e:
            logger.error(f"Failed to collect node metrics: {e}")

    def _check_alerts(self, current_metrics: dict[str, Any]) -> None:
        """Check for alert conditions and manage active alerts."""
        new_alerts = []
        current_time = datetime.now()

        try:
            # CPU utilization alert
            cpu_util = current_metrics.get("resources", {}).get("cpu", {}).get("utilization_percent", 0)
            if cpu_util > self.alert_thresholds["cpu_utilization"]:
                new_alerts.append({
                    "type": "high_cpu_utilization",
                    "severity": "warning",
                    "message": f"High CPU utilization: {cpu_util:.1f}%",
                    "value": cpu_util,
                    "threshold": self.alert_thresholds["cpu_utilization"],
                    "timestamp": current_time
                })

            # Memory utilization alert
            memory_resources = current_metrics.get("resources", {}).get("memory", {})
            if memory_resources.get("total", 0) > 0:
                memory_util = ((memory_resources.get("total", 0) - memory_resources.get("available", 0)) /
                              memory_resources.get("total", 1)) * 100
                if memory_util > self.alert_thresholds["memory_utilization"]:
                    new_alerts.append({
                        "type": "high_memory_utilization",
                        "severity": "warning",
                        "message": f"High memory utilization: {memory_util:.1f}%",
                        "value": memory_util,
                        "threshold": self.alert_thresholds["memory_utilization"],
                        "timestamp": current_time
                    })

            # Node health alert
            total_nodes = current_metrics.get("total_nodes", 0)
            alive_nodes = current_metrics.get("alive_nodes", 0)
            if total_nodes > 0 and alive_nodes < total_nodes:
                new_alerts.append({
                    "type": "nodes_down",
                    "severity": "critical",
                    "message": f"{total_nodes - alive_nodes} nodes are down",
                    "value": total_nodes - alive_nodes,
                    "timestamp": current_time
                })

            # Task failure rate alert
            task_stats = current_metrics.get("task_stats", {})
            failure_rate = task_stats.get("failure_rate", 0)
            if failure_rate > self.alert_thresholds["task_failure_rate"]:
                new_alerts.append({
                    "type": "high_task_failure_rate",
                    "severity": "warning",
                    "message": f"High task failure rate: {failure_rate:.1f}%",
                    "value": failure_rate,
                    "threshold": self.alert_thresholds["task_failure_rate"],
                    "timestamp": current_time
                })

            # Add new alerts and clean up old ones
            self.active_alerts.extend(new_alerts)

            # Remove alerts older than 5 minutes
            cutoff_time = current_time - timedelta(minutes=5)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert["timestamp"] > cutoff_time
            ]

            # Log new alerts
            for alert in new_alerts:
                log_level = logging.CRITICAL if alert["severity"] == "critical" else logging.WARNING
                logger.log(log_level, f"🚨 ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current cluster metrics."""
        return self._collect_metrics()

    def get_metrics_summary(self, window_minutes: int = 10) -> dict[str, Any]:
        """
        Get metrics summary for the specified time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Dictionary with summarized metrics
        """
        try:
            # Calculate how many data points to include
            points_per_minute = 60 / self.update_interval
            window_points = int(window_minutes * points_per_minute)

            summary = {
                "window_minutes": window_minutes,
                "data_points": min(window_points, len(self.metrics_history.get("uptime", []))),
                "metrics": {}
            }

            # Summarize key metrics
            for metric_name in ["cpu_utilization", "memory_utilization", "gpu_utilization"]:
                if self.metrics_history.get(metric_name):
                    values = list(self.metrics_history[metric_name])[-window_points:]
                    if values:
                        summary["metrics"][metric_name] = {  # type: ignore[index]
                            "current": values[-1],
                            "average": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable"
                        }

            # Node stability
            if "alive_nodes" in self.metrics_history:
                node_counts = list(self.metrics_history["alive_nodes"])[-window_points:]
                if node_counts:
                    summary["metrics"]["cluster_stability"] = {  # type: ignore[index]
                        "min_nodes": min(node_counts),
                        "max_nodes": max(node_counts),
                        "current_nodes": node_counts[-1],
                        "stable": len(set(node_counts)) == 1
                    }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {e}")
            return {"error": str(e)}

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get list of active alerts."""
        # Clean up old alerts first
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=5)

        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert["timestamp"] > cutoff_time
        ]

        return self.active_alerts

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            current_metrics = self.get_current_metrics()
            summary = self.get_metrics_summary()
            alerts = self.get_active_alerts()

            # Calculate overall health score (0-100)
            health_score = 100

            # Deduct points for resource utilization
            for resource in ["cpu", "memory", "gpu"]:
                util_key = f"{resource}_utilization"
                if util_key in summary.get("metrics", {}):
                    utilization = summary["metrics"][util_key]["current"]
                    if utilization > 90:
                        health_score -= 20
                    elif utilization > 80:
                        health_score -= 10
                    elif utilization > 70:
                        health_score -= 5

            # Deduct points for alerts
            critical_alerts = len([a for a in alerts if a.get("severity") == "critical"])
            warning_alerts = len([a for a in alerts if a.get("severity") == "warning"])
            health_score -= (critical_alerts * 15 + warning_alerts * 5)

            health_score = max(0, health_score)

            # Generate recommendations
            recommendations = []

            if health_score < 70:
                recommendations.append("Cluster health is degraded - investigate active alerts")

            for resource in ["cpu", "memory"]:
                util_key = f"{resource}_utilization"
                if util_key in summary.get("metrics", {}) and summary["metrics"][util_key]["current"] > 85:
                    recommendations.append(f"Consider scaling up {resource} resources")

            if critical_alerts > 0:
                recommendations.append("Address critical alerts immediately")

            return {
                "timestamp": datetime.now().isoformat(),
                "health_score": health_score,
                "cluster_status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "unhealthy",
                "current_metrics": current_metrics,
                "metrics_summary": summary,
                "active_alerts": alerts,
                "recommendations": recommendations,
                "uptime_hours": (time.time() - self.start_time) / 3600
            }

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        """
        Export collected metrics to file.

        Args:
            filepath: Path to export file
            format: Export format ("json" or "csv")

        Returns:
            True if export successful, False otherwise
        """
        try:
            if format.lower() == "json":
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "monitoring_duration_hours": (time.time() - self.start_time) / 3600,
                    "metrics_history": {k: list(v) for k, v in self.metrics_history.items()},
                    "performance_report": self.get_performance_report(),
                    "configuration": {
                        "history_size": self.history_size,
                        "update_interval": self.update_interval,
                        "alert_thresholds": self.alert_thresholds
                    }
                }

                with open(filepath, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format.lower() == "csv":
                import pandas as pd

                # Convert metrics history to DataFrame
                df_data = {}
                max_length = max(len(v) for v in self.metrics_history.values()) if self.metrics_history else 0

                for metric, values in self.metrics_history.items():
                    # Pad shorter series with None
                    padded_values = list(values) + [None] * (max_length - len(values))
                    df_data[metric] = padded_values

                if df_data:
                    df = pd.DataFrame(df_data)
                    df.to_csv(filepath, index=False)
                else:
                    logger.warning("No metrics data to export")
                    return False
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"✅ Metrics exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def update_task_counter(self, counter_type: str, increment: int = 1) -> None:
        """
        Update task counters for monitoring.

        Args:
            counter_type: Type of counter ("submitted", "completed", "failed", "pending")
            increment: Amount to increment by
        """
        if counter_type in self.task_counters:
            self.task_counters[counter_type] += increment
        else:
            logger.warning(f"Unknown counter type: {counter_type}")


# Global monitor instance
_global_monitor: RayClusterMonitor | None = None


def get_global_monitor() -> RayClusterMonitor | None:
    """Get the global cluster monitor instance."""
    return _global_monitor


def start_global_monitoring(
    history_size: int = 100,
    update_interval: float = 5.0,
    enable_detailed_metrics: bool = True
) -> bool:
    """
    Start global cluster monitoring.

    Args:
        history_size: Number of historical data points to keep
        update_interval: Interval between metric updates in seconds
        enable_detailed_metrics: Whether to collect detailed node-level metrics

    Returns:
        True if monitoring started successfully, False otherwise
    """
    global _global_monitor

    if _global_monitor is not None:
        logger.warning("Global monitoring is already active")
        return True

    _global_monitor = RayClusterMonitor(
        history_size=history_size,
        update_interval=update_interval,
        enable_detailed_metrics=enable_detailed_metrics
    )

    return _global_monitor.start_monitoring()


def stop_global_monitoring() -> None:
    """Stop global cluster monitoring."""
    global _global_monitor

    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None
    else:
        logger.warning("Global monitoring is not active")


def print_cluster_dashboard() -> None:
    """Print a comprehensive cluster dashboard."""
    if not ray.is_initialized():
        print("❌ Ray is not initialized")
        return

    try:
        # Get current metrics
        health = validate_cluster_health()
        worker_info = get_optimal_worker_count()
        check_ray_cluster_status()

        print("\n" + "="*80)
        print("🚀 RAY CLUSTER DASHBOARD")
        print("="*80)

        # Cluster status
        status_emoji = "✅" if health["healthy"] else "❌"
        print(f"\n{status_emoji} Cluster Status: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        if not health["healthy"]:
            print(f"   Issue: {health['reason']}")

        # Resource overview
        print("\n💻 Resource Overview:")
        resources = health["resources"]

        # CPU
        cpu = resources["cpu"]
        cpu_bar = "█" * int(cpu["utilization_percent"] / 5) + "░" * (20 - int(cpu["utilization_percent"] / 5))
        print(f"   CPU:    [{cpu_bar}] {cpu['utilization_percent']:.1f}% ({cpu['available']:.1f}/{cpu['total']:.1f})")

        # GPU
        gpu = resources["gpu"]
        if gpu["total"] > 0:
            gpu_bar = "█" * int(gpu["utilization_percent"] / 5) + "░" * (20 - int(gpu["utilization_percent"] / 5))
            print(f"   GPU:    [{gpu_bar}] {gpu['utilization_percent']:.1f}% ({gpu['available']:.1f}/{gpu['total']:.1f})")
        else:
            print("   GPU:    [Not Available]")

        # Memory
        memory = resources["memory"]
        if memory["total"] > 0:
            memory_util = (memory["utilized"] / memory["total"]) * 100
            memory_bar = "█" * int(memory_util / 5) + "░" * (20 - int(memory_util / 5))
            memory_gb = memory["total"] / (1024**3)
            memory_used_gb = memory["utilized"] / (1024**3)
            print(f"   Memory: [{memory_bar}] {memory_util:.1f}% ({memory_used_gb:.1f}/{memory_gb:.1f} GB)")

        # Nodes
        print(f"\n🌐 Nodes: {health['nodes']} alive")

        # Worker recommendations
        print("\n👥 Optimal Workers:")
        print(f"   CPU Workers: {worker_info['cpu_workers']}")
        print(f"   GPU Workers: {worker_info['gpu_workers']}")
        print(f"   Total: {worker_info['total_workers']}")

        # Recommendations
        if health["recommendations"]:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(health["recommendations"][:3], 1):
                print(f"   {i}. {rec}")

        # Global monitor stats
        global _global_monitor
        if _global_monitor:
            print("\n📊 Monitoring Active:")
            print(f"   Uptime: {(_global_monitor.last_update - _global_monitor.start_time)/3600:.1f}h")
            print(f"   Data Points: {len(_global_monitor.metrics_history.get('uptime', []))}")

            active_alerts = _global_monitor.get_active_alerts()
            if active_alerts:
                print(f"   Active Alerts: {len(active_alerts)}")
                for alert in active_alerts[:2]:  # Show top 2 alerts
                    print(f"     🚨 {alert['message']}")

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"❌ Failed to generate dashboard: {e}")
