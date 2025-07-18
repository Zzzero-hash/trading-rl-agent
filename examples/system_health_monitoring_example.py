#!/usr/bin/env python3
"""
Comprehensive System Health Monitoring Example

This example demonstrates how to use the SystemHealthMonitor to:
1. Monitor system latency, memory usage, and error rates
2. Track trading system performance metrics
3. Implement automated health checks and diagnostics
4. Provide system health alerts and notifications
5. Create system health dashboards and reports

Usage:
    python system_health_monitoring_example.py
"""

import asyncio
import random
import time
from datetime import datetime
from pathlib import Path

from src.trading_rl_agent.monitoring import (
    AlertManager,
    AlertSeverity,
    HealthDashboard,
    HealthStatus,
    HealthCheckType,
    MetricsCollector,
    SystemHealthMonitor,
)


def simulate_trading_activity(metrics_collector: MetricsCollector) -> None:
    """Simulate trading activity to generate metrics."""
    # Simulate P&L changes
    pnl_change = random.uniform(-1000, 2000)
    current_pnl = metrics_collector.get_metric_summary("pnl").get("latest", 0.0)
    new_pnl = current_pnl + pnl_change
    metrics_collector.record_metric("pnl", new_pnl)
    
    # Simulate daily P&L
    daily_pnl = random.uniform(-500, 1000)
    metrics_collector.record_metric("daily_pnl", daily_pnl)
    
    # Simulate Sharpe ratio
    sharpe = random.uniform(-2.0, 3.0)
    metrics_collector.record_metric("sharpe_ratio", sharpe)
    
    # Simulate drawdown
    drawdown = random.uniform(-0.15, 0.0)
    metrics_collector.record_metric("max_drawdown", drawdown)
    
    # Simulate win rate
    win_rate = random.uniform(0.4, 0.8)
    metrics_collector.record_metric("win_rate", win_rate)
    
    # Simulate execution latency
    latency = random.uniform(10, 200)
    metrics_collector.record_metric("execution_latency", latency)
    
    # Simulate trades
    if random.random() < 0.3:  # 30% chance of a trade
        metrics_collector.increment_counter("total_trades")
    
    # Simulate open positions
    open_positions = random.randint(0, 10)
    metrics_collector.set_gauge("open_positions", open_positions)


def simulate_model_activity(metrics_collector: MetricsCollector) -> None:
    """Simulate model training and prediction activity."""
    # Simulate model accuracy
    accuracy = random.uniform(0.6, 0.95)
    metrics_collector.record_metric("model_accuracy", accuracy)
    
    # Simulate model loss
    loss = random.uniform(0.1, 0.8)
    metrics_collector.record_metric("model_loss", loss)
    
    # Simulate prediction latency
    pred_latency = random.uniform(5, 50)
    metrics_collector.record_metric("prediction_latency", pred_latency)
    
    # Simulate model confidence
    confidence = random.uniform(0.5, 0.99)
    metrics_collector.record_metric("model_confidence", confidence)


def create_custom_health_check(monitor: SystemHealthMonitor) -> None:
    """Create a custom health check for demonstration."""
    
    def custom_database_check():
        """Simulate a custom database connectivity check."""
        # Simulate database response time
        db_latency = random.uniform(5, 100)
        
        if db_latency > 80:
            status = HealthStatus.CRITICAL
        elif db_latency > 50:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return monitor.HealthCheckResult(
            check_type=HealthCheckType.CUSTOM,
            status=status,
            message=f"Database latency: {db_latency:.1f}ms",
            timestamp=time.time(),
            metrics={"database_latency": db_latency}
        )
    
    monitor.add_custom_health_check("database_connectivity", custom_database_check)


def setup_alert_handlers(alert_manager: AlertManager) -> None:
    """Setup custom alert handlers."""
    
    def critical_alert_handler(alert):
        """Handle critical alerts."""
        print(f"🚨 CRITICAL ALERT: {alert.title}")
        print(f"   Message: {alert.message}")
        print(f"   Source: {alert.source}")
        print(f"   Time: {datetime.fromtimestamp(alert.timestamp)}")
        print("-" * 50)
    
    def warning_alert_handler(alert):
        """Handle warning alerts."""
        print(f"⚠️  WARNING: {alert.title}")
        print(f"   Message: {alert.message}")
        print(f"   Source: {alert.source}")
        print("-" * 30)
    
    def info_alert_handler(alert):
        """Handle info alerts."""
        print(f"ℹ️  INFO: {alert.title}")
        print(f"   Message: {alert.message}")
        print("-" * 20)
    
    # Register alert handlers
    alert_manager.add_alert_handler("health_critical", critical_alert_handler)
    alert_manager.add_alert_handler("health_degraded", warning_alert_handler)
    alert_manager.add_alert_handler("monitoring_started", info_alert_handler)
    alert_manager.add_alert_handler("monitoring_stopped", info_alert_handler)


def main():
    """Main function demonstrating system health monitoring."""
    print("🚀 Starting System Health Monitoring Example")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("health_monitoring_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize monitoring components
    print("📊 Initializing monitoring components...")
    metrics_collector = MetricsCollector(max_history=10000)
    alert_manager = AlertManager(max_alerts=1000)
    
    # Setup alert handlers
    setup_alert_handlers(alert_manager)
    
    # Initialize system health monitor
    health_monitor = SystemHealthMonitor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        check_interval=10.0,  # Check every 10 seconds
        max_history=1000,
        health_thresholds={
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "disk_percent": {"warning": 80.0, "critical": 95.0},
            "network_latency": {"warning": 50.0, "critical": 200.0},
            "error_rate": {"warning": 0.02, "critical": 0.05},
            "execution_latency": {"warning": 50.0, "critical": 150.0},
            "drawdown": {"warning": 0.03, "critical": 0.08},
        }
    )
    
    # Add custom health check
    create_custom_health_check(health_monitor)
    
    # Initialize health dashboard
    health_dashboard = HealthDashboard(
        system_health_monitor=health_monitor,
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        output_dir=str(output_dir)
    )
    
    print("✅ Monitoring components initialized")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Start monitoring
    print("\n🔍 Starting system health monitoring...")
    health_monitor.start_monitoring()
    
    try:
        # Run simulation for 2 minutes
        simulation_duration = 120  # seconds
        start_time = time.time()
        
        print(f"⏱️  Running simulation for {simulation_duration} seconds...")
        print("📈 Simulating trading and model activity...")
        
        while time.time() - start_time < simulation_duration:
            # Simulate trading activity
            simulate_trading_activity(metrics_collector)
            
            # Simulate model activity
            simulate_model_activity(metrics_collector)
            
            # Record some latency and errors
            latency = random.uniform(5, 100)
            health_monitor.record_latency(latency)
            health_monitor.record_request()
            
            # Simulate occasional errors
            if random.random() < 0.05:  # 5% error rate
                health_monitor.record_error()
            
            # Generate dashboard and reports every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and elapsed > 0:
                print(f"\n📊 Generating reports at {elapsed:.0f}s...")
                
                # Generate health report
                report_path = output_dir / f"health_report_{int(elapsed)}s.txt"
                health_monitor.generate_health_report(str(report_path))
                print(f"   📄 Health report: {report_path}")
                
                # Generate HTML dashboard
                html_path = output_dir / f"health_dashboard_{int(elapsed)}s.html"
                health_dashboard.generate_html_dashboard(str(html_path))
                print(f"   🌐 HTML dashboard: {html_path}")
                
                # Save dashboard data
                json_path = health_dashboard.save_dashboard_data(f"dashboard_data_{int(elapsed)}s.json")
                print(f"   📊 Dashboard data: {json_path}")
                
                # Show current health summary
                health_summary = health_monitor.get_health_summary()
                print(f"   💚 Health status: {health_summary['status']}")
            
            # Sleep for a short interval
            time.sleep(1)
        
        print(f"\n✅ Simulation completed after {simulation_duration} seconds")
        
        # Generate final comprehensive report
        print("\n📋 Generating final comprehensive report...")
        
        # Final health report
        final_report_path = output_dir / "final_health_report.txt"
        health_monitor.generate_health_report(str(final_report_path))
        print(f"   📄 Final health report: {final_report_path}")
        
        # Final HTML dashboard
        final_html_path = output_dir / "final_health_dashboard.html"
        health_dashboard.generate_html_dashboard(str(final_html_path))
        print(f"   🌐 Final HTML dashboard: {final_html_path}")
        
        # Final dashboard data
        final_json_path = health_dashboard.save_dashboard_data("final_dashboard_data.json")
        print(f"   📊 Final dashboard data: {final_json_path}")
        
        # Show final statistics
        print("\n📈 Final Statistics:")
        print(f"   Total requests: {health_monitor.total_requests}")
        print(f"   Total errors: {health_monitor.error_count}")
        print(f"   Error rate: {health_monitor.get_error_rate():.2%}")
        print(f"   Average latency: {health_monitor.get_average_latency():.1f}ms")
        print(f"   Health checks performed: {len(health_monitor.health_history)}")
        
        # Show alert summary
        alert_summary = alert_manager.get_alert_summary()
        print(f"\n🚨 Alert Summary:")
        for alert_type, count in alert_summary["by_type"].items():
            print(f"   {alert_type}: {count}")
        
        print(f"\n🎉 System health monitoring example completed successfully!")
        print(f"📁 Check the output directory for detailed reports: {output_dir.absolute()}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    finally:
        # Stop monitoring
        print("\n🛑 Stopping system health monitoring...")
        health_monitor.stop_monitoring()
        print("✅ Monitoring stopped")


if __name__ == "__main__":
    main()