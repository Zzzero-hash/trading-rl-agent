#!/usr/bin/env python3
"""
Example script demonstrating the Performance Dashboard.

This script shows how to:
1. Set up the performance dashboard
2. Add real-time data updates
3. Configure dashboard settings
4. Use streaming capabilities
"""

import asyncio
import json
import time
import threading
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl_agent.monitoring.dashboard import Dashboard
from trading_rl_agent.monitoring.metrics_collector import MetricsCollector
from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard
from trading_rl_agent.monitoring.streaming_dashboard import StreamingDashboard


def create_sample_data(metrics_collector: MetricsCollector) -> None:
    """Create sample data for demonstration.
    
    Args:
        metrics_collector: MetricsCollector instance
    """
    import random
    
    # Simulate realistic trading data
    base_pnl = 10000.0
    base_return = 0.15
    
    # Add trading metrics
    metrics_collector.record_metric('pnl', base_pnl + random.uniform(-500, 500))
    metrics_collector.record_metric('daily_pnl', random.uniform(-200, 300))
    metrics_collector.record_metric('total_return', base_return + random.uniform(-0.02, 0.02))
    metrics_collector.record_metric('sharpe_ratio', random.uniform(1.0, 2.5))
    metrics_collector.record_metric('max_drawdown', random.uniform(-0.15, -0.05))
    metrics_collector.record_metric('win_rate', random.uniform(0.55, 0.75))
    
    # Add risk metrics
    metrics_collector.record_metric('var_95', random.uniform(-0.03, -0.01))
    metrics_collector.record_metric('cvar_95', random.uniform(-0.04, -0.02))
    metrics_collector.record_metric('volatility', random.uniform(0.15, 0.25))
    metrics_collector.record_metric('beta', random.uniform(0.8, 1.2))
    metrics_collector.record_metric('current_exposure', random.uniform(0.6, 0.9))
    metrics_collector.record_metric('position_concentration', random.uniform(0.3, 0.6))
    
    # Add system metrics
    metrics_collector.record_metric('cpu_usage', random.uniform(30, 70))
    metrics_collector.record_metric('memory_usage', random.uniform(50, 80))
    metrics_collector.record_metric('disk_usage', random.uniform(20, 40))
    metrics_collector.record_metric('network_latency', random.uniform(5, 25))
    metrics_collector.record_metric('error_rate', random.uniform(0.0001, 0.001))
    metrics_collector.record_metric('response_time', random.uniform(30, 60))
    
    # Update counters and gauges
    metrics_collector.increment_counter('total_trades', random.randint(1, 5))
    metrics_collector.set_gauge('open_positions', random.randint(5, 15))


def run_data_simulator(metrics_collector: MetricsCollector, stop_event: threading.Event) -> None:
    """Run a data simulator to continuously update metrics.
    
    Args:
        metrics_collector: MetricsCollector instance
        stop_event: Event to signal when to stop
    """
    print("🔄 Starting data simulator...")
    
    while not stop_event.is_set():
        try:
            create_sample_data(metrics_collector)
            time.sleep(2.0)  # Update every 2 seconds
        except Exception as e:
            print(f"❌ Error in data simulator: {e}")
            time.sleep(1.0)
    
    print("🛑 Data simulator stopped")


async def run_streaming_example() -> None:
    """Run streaming dashboard example."""
    print("🚀 Starting Streaming Dashboard Example")
    print("=" * 50)
    
    # Create components
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard,
        update_interval=1.0,
        max_data_points=500
    )
    
    # Create streaming dashboard
    streaming_dashboard = StreamingDashboard(
        performance_dashboard=performance_dashboard,
        host="localhost",
        port=8765,
        update_interval=0.5
    )
    
    # Start data simulator
    stop_event = threading.Event()
    simulator_thread = threading.Thread(
        target=run_data_simulator,
        args=(metrics_collector, stop_event),
        daemon=True
    )
    simulator_thread.start()
    
    try:
        print("📡 Starting streaming server on ws://localhost:8765")
        print("🌐 Dashboard will be available at http://localhost:8501")
        print("📊 Press Ctrl+C to stop")
        
        # Start streaming server
        await streaming_dashboard.start_server()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        stop_event.set()
        await streaming_dashboard.stop_server()


def run_basic_example() -> None:
    """Run basic dashboard example."""
    print("🚀 Starting Basic Dashboard Example")
    print("=" * 50)
    
    # Create components
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    
    # Add initial sample data
    create_sample_data(metrics_collector)
    
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard,
        update_interval=2.0,
        max_data_points=200
    )
    
    # Create custom configuration
    custom_config = {
        'layout': 'grid',
        'theme': 'light',
        'auto_refresh': True,
        'refresh_interval': 2.0,
        'charts': {
            'pnl_chart': True,
            'risk_metrics': True,
            'position_overview': True,
            'performance_metrics': True,
            'system_health': True,
            'alerts': True,
        },
        'time_range': '24h',
        'metrics_display': {
            'show_percentages': True,
            'show_currency': True,
            'currency_symbol': '$',
            'decimal_places': 2,
        }
    }
    
    # Save configuration
    config_path = Path("dashboard_config.json")
    with open(config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    print(f"📋 Saved configuration to {config_path}")
    
    # Start data simulator
    stop_event = threading.Event()
    simulator_thread = threading.Thread(
        target=run_data_simulator,
        args=(metrics_collector, stop_event),
        daemon=True
    )
    simulator_thread.start()
    
    try:
        print("🌐 Starting dashboard...")
        print("📊 Dashboard will be available at http://localhost:8501")
        print("🔄 Data will be updated every 2 seconds")
        print("📊 Press Ctrl+C to stop")
        
        # Run the dashboard
        performance_dashboard.run_dashboard()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        stop_event.set()


def create_dashboard_config() -> None:
    """Create a sample dashboard configuration file."""
    config = {
        'layout': 'grid',
        'theme': 'light',
        'auto_refresh': True,
        'refresh_interval': 1.0,
        'charts': {
            'pnl_chart': True,
            'risk_metrics': True,
            'position_overview': True,
            'performance_metrics': True,
            'system_health': True,
            'alerts': True,
        },
        'time_range': '24h',
        'metrics_display': {
            'show_percentages': True,
            'show_currency': True,
            'currency_symbol': '$',
            'decimal_places': 2,
        }
    }
    
    config_path = Path("sample_dashboard_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created sample configuration: {config_path}")
    print("📋 You can use this configuration with:")
    print(f"   python -m trading_rl_agent.cli_dashboard run --config {config_path}")


def main() -> None:
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Performance Dashboard Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic dashboard
  python dashboard_example.py basic

  # Run streaming dashboard
  python dashboard_example.py streaming

  # Create sample configuration
  python dashboard_example.py config
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['basic', 'streaming', 'config'],
        help='Example mode to run'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        run_basic_example()
    elif args.mode == 'streaming':
        asyncio.run(run_streaming_example())
    elif args.mode == 'config':
        create_dashboard_config()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()