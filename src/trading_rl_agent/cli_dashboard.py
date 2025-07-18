"""
CLI for running the performance dashboard.

This module provides command-line interface for:
- Running the performance dashboard
- Configuring dashboard settings
- Starting streaming services
- Managing dashboard instances
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import streamlit.web.cli as stcli
from streamlit.web.server import Server

from .monitoring.dashboard import Dashboard
from .monitoring.metrics_collector import MetricsCollector
from .monitoring.performance_dashboard import PerformanceDashboard, run_performance_dashboard
from .monitoring.streaming_dashboard import StreamingDashboard, run_streaming_dashboard


def run_dashboard_cli() -> None:
    """Run the dashboard CLI."""
    parser = argparse.ArgumentParser(
        description="Trading Performance Dashboard CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic dashboard
  python -m trading_rl_agent.cli_dashboard run

  # Run dashboard with custom config
  python -m trading_rl_agent.cli_dashboard run --config dashboard_config.json

  # Run dashboard with streaming
  python -m trading_rl_agent.cli_dashboard run --streaming --port 8765

  # Run streaming server only
  python -m trading_rl_agent.cli_dashboard stream --host 0.0.0.0 --port 8765

  # Export dashboard data
  python -m trading_rl_agent.cli_dashboard export --output dashboard_data.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run dashboard command
    run_parser = subparsers.add_parser('run', help='Run the performance dashboard')
    run_parser.add_argument(
        '--config',
        type=str,
        help='Path to dashboard configuration file'
    )
    run_parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable WebSocket streaming'
    )
    run_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Dashboard host (default: localhost)'
    )
    run_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Dashboard port (default: 8501)'
    )
    run_parser.add_argument(
        '--streaming-host',
        type=str,
        default='localhost',
        help='Streaming server host (default: localhost)'
    )
    run_parser.add_argument(
        '--streaming-port',
        type=int,
        default=8765,
        help='Streaming server port (default: 8765)'
    )
    run_parser.add_argument(
        '--update-interval',
        type=float,
        default=1.0,
        help='Update interval in seconds (default: 1.0)'
    )
    run_parser.add_argument(
        '--max-data-points',
        type=int,
        default=1000,
        help='Maximum data points to keep in memory (default: 1000)'
    )
    
    # Stream command
    stream_parser = subparsers.add_parser('stream', help='Run streaming server only')
    stream_parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Streaming server host (default: localhost)'
    )
    stream_parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Streaming server port (default: 8765)'
    )
    stream_parser.add_argument(
        '--update-interval',
        type=float,
        default=0.1,
        help='Update interval in seconds (default: 0.1)'
    )
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export dashboard data')
    export_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path'
    )
    export_parser.add_argument(
        '--config',
        type=str,
        help='Path to dashboard configuration file'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage dashboard configuration')
    config_parser.add_argument(
        '--create',
        type=str,
        help='Create new configuration file'
    )
    config_parser.add_argument(
        '--validate',
        type=str,
        help='Validate configuration file'
    )
    config_parser.add_argument(
        '--show',
        type=str,
        help='Show configuration file contents'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show dashboard status')
    status_parser.add_argument(
        '--streaming-host',
        type=str,
        default='localhost',
        help='Streaming server host (default: localhost)'
    )
    status_parser.add_argument(
        '--streaming-port',
        type=int,
        default=8765,
        help='Streaming server port (default: 8765)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'run':
            run_dashboard_command(args)
        elif args.command == 'stream':
            run_streaming_command(args)
        elif args.command == 'export':
            run_export_command(args)
        elif args.command == 'config':
            run_config_command(args)
        elif args.command == 'status':
            run_status_command(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def run_dashboard_command(args: argparse.Namespace) -> None:
    """Run the dashboard command.

    Args:
        args: Command line arguments
    """
    print("🚀 Starting Trading Performance Dashboard...")
    
    # Create dashboard components
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    
    # Add some sample data for demonstration
    _add_sample_data(metrics_collector)
    
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard,
        update_interval=args.update_interval,
        max_data_points=args.max_data_points
    )
    
    # Load configuration if provided
    if args.config:
        try:
            performance_dashboard.load_configuration(args.config)
            print(f"📋 Loaded configuration from {args.config}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load configuration: {e}")
    
    # Start streaming if enabled
    if args.streaming:
        print(f"📡 Starting streaming server on ws://{args.streaming_host}:{args.streaming_port}")
        
        # Start streaming in background
        async def start_streaming():
            streaming_dashboard = StreamingDashboard(
                performance_dashboard=performance_dashboard,
                host=args.streaming_host,
                port=args.streaming_port,
                update_interval=args.update_interval
            )
            await streaming_dashboard.start_server()
        
        # Run streaming in background thread
        import threading
        streaming_thread = threading.Thread(
            target=lambda: asyncio.run(start_streaming()),
            daemon=True
        )
        streaming_thread.start()
    
    # Run the dashboard
    print(f"🌐 Dashboard will be available at http://{args.host}:{args.port}")
    print("📊 Press Ctrl+C to stop the dashboard")
    
    # Set Streamlit arguments
    sys.argv = [
        'streamlit',
        'run',
        str(Path(__file__).parent / 'monitoring' / 'performance_dashboard.py'),
        '--server.port', str(args.port),
        '--server.address', args.host,
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    # Run Streamlit
    stcli.main()


def run_streaming_command(args: argparse.Namespace) -> None:
    """Run the streaming command.

    Args:
        args: Command line arguments
    """
    print(f"📡 Starting streaming server on ws://{args.host}:{args.port}")
    
    # Create dashboard components
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    
    # Add sample data
    _add_sample_data(metrics_collector)
    
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard
    )
    
    # Run streaming server
    asyncio.run(run_streaming_dashboard(
        performance_dashboard=performance_dashboard,
        host=args.host,
        port=args.port
    ))


def run_export_command(args: argparse.Namespace) -> None:
    """Run the export command.

    Args:
        args: Command line arguments
    """
    print(f"📤 Exporting dashboard data to {args.output}")
    
    # Create dashboard components
    metrics_collector = MetricsCollector()
    dashboard = Dashboard(metrics_collector)
    
    # Add sample data
    _add_sample_data(metrics_collector)
    
    performance_dashboard = PerformanceDashboard(
        metrics_collector=metrics_collector,
        dashboard=dashboard
    )
    
    # Load configuration if provided
    if args.config:
        try:
            performance_dashboard.load_configuration(args.config)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load configuration: {e}")
    
    # Export data
    try:
        performance_dashboard.export_data(args.output)
        print(f"✅ Data exported successfully to {args.output}")
    except Exception as e:
        print(f"❌ Failed to export data: {e}")
        sys.exit(1)


def run_config_command(args: argparse.Namespace) -> None:
    """Run the config command.

    Args:
        args: Command line arguments
    """
    if args.create:
        _create_config_file(args.create)
    elif args.validate:
        _validate_config_file(args.validate)
    elif args.show:
        _show_config_file(args.show)
    else:
        print("Please specify --create, --validate, or --show")


def run_status_command(args: argparse.Namespace) -> None:
    """Run the status command.

    Args:
        args: Command line arguments
    """
    print("📊 Dashboard Status")
    print("=" * 50)
    
    # Check if streaming server is running
    try:
        import websockets
        import asyncio
        
        async def check_streaming():
            try:
                uri = f"ws://{args.streaming_host}:{args.streaming_port}"
                websocket = await websockets.connect(uri, timeout=2)
                await websocket.close()
                return True
            except:
                return False
        
        is_streaming = asyncio.run(check_streaming())
        print(f"📡 Streaming Server: {'🟢 Running' if is_streaming else '🔴 Not Running'}")
        if is_streaming:
            print(f"   URL: ws://{args.streaming_host}:{args.streaming_port}")
    except ImportError:
        print("📡 Streaming Server: ⚠️ websockets not installed")
    
    # Check if dashboard is running
    try:
        import requests
        response = requests.get(f"http://localhost:8501", timeout=2)
        print(f"🌐 Dashboard: {'🟢 Running' if response.status_code == 200 else '🔴 Not Running'}")
        if response.status_code == 200:
            print("   URL: http://localhost:8501")
    except:
        print("🌐 Dashboard: 🔴 Not Running")
    
    print("\n📋 System Information:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {sys.platform}")


def _add_sample_data(metrics_collector: MetricsCollector) -> None:
    """Add sample data to metrics collector.

    Args:
        metrics_collector: MetricsCollector instance
    """
    import time
    import random
    
    # Add sample trading metrics
    metrics_collector.record_metric('pnl', 1250.50)
    metrics_collector.record_metric('daily_pnl', 125.75)
    metrics_collector.record_metric('total_return', 0.125)
    metrics_collector.record_metric('sharpe_ratio', 1.85)
    metrics_collector.record_metric('max_drawdown', -0.08)
    metrics_collector.record_metric('win_rate', 0.65)
    
    # Add sample risk metrics
    metrics_collector.record_metric('var_95', -0.025)
    metrics_collector.record_metric('cvar_95', -0.035)
    metrics_collector.record_metric('volatility', 0.18)
    metrics_collector.record_metric('beta', 0.95)
    metrics_collector.record_metric('current_exposure', 0.75)
    metrics_collector.record_metric('position_concentration', 0.45)
    
    # Add sample system metrics
    metrics_collector.record_metric('cpu_usage', 45.2)
    metrics_collector.record_metric('memory_usage', 62.8)
    metrics_collector.record_metric('disk_usage', 23.1)
    metrics_collector.record_metric('network_latency', 12.5)
    metrics_collector.record_metric('error_rate', 0.001)
    metrics_collector.record_metric('response_time', 45.0)
    
    # Add counters and gauges
    metrics_collector.increment_counter('total_trades', 156)
    metrics_collector.set_gauge('open_positions', 8)
    metrics_collector.set_gauge('training_status', 0)


def _create_config_file(filepath: str) -> None:
    """Create a new configuration file.

    Args:
        filepath: Path to create configuration file
    """
    default_config = {
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
    
    try:
        with open(filepath, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ Configuration file created: {filepath}")
    except Exception as e:
        print(f"❌ Failed to create configuration file: {e}")


def _validate_config_file(filepath: str) -> None:
    """Validate a configuration file.

    Args:
        filepath: Path to configuration file
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_keys = ['layout', 'theme', 'auto_refresh', 'charts']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key: {key}")
                return
        
        print(f"✅ Configuration file is valid: {filepath}")
        
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format: {filepath}")
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Validation error: {e}")


def _show_config_file(filepath: str) -> None:
    """Show configuration file contents.

    Args:
        filepath: Path to configuration file
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        print(f"📋 Configuration: {filepath}")
        print("=" * 50)
        print(json.dumps(config, indent=2))
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


if __name__ == "__main__":
    run_dashboard_cli()