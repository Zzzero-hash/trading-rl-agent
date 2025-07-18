#!/usr/bin/env python3
"""
Test script for the Performance Dashboard.

This script tests the basic functionality of the dashboard components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all dashboard components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from trading_rl_agent.monitoring.dashboard import Dashboard
        from trading_rl_agent.monitoring.metrics_collector import MetricsCollector
        from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard
        from trading_rl_agent.monitoring.streaming_dashboard import StreamingDashboard, WebSocketClient
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic dashboard functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        from trading_rl_agent.monitoring.dashboard import Dashboard
        from trading_rl_agent.monitoring.metrics_collector import MetricsCollector
        from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard
        
        # Create components
        metrics_collector = MetricsCollector()
        dashboard = Dashboard(metrics_collector)
        performance_dashboard = PerformanceDashboard(
            metrics_collector=metrics_collector,
            dashboard=dashboard
        )
        
        # Test metrics collection
        metrics_collector.record_metric('pnl', 1000.0)
        metrics_collector.record_metric('daily_pnl', 100.0)
        metrics_collector.increment_counter('total_trades', 1)
        metrics_collector.set_gauge('open_positions', 5)
        
        # Test dashboard methods
        trading_metrics = dashboard.get_trading_metrics()
        risk_metrics = dashboard.get_risk_metrics()
        system_health = dashboard.get_system_health()
        
        print(f"✅ Trading metrics: {len(trading_metrics)} items")
        print(f"✅ Risk metrics: {len(risk_metrics)} items")
        print(f"✅ System health: {len(system_health)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_configuration():
    """Test dashboard configuration."""
    print("🧪 Testing configuration...")
    
    try:
        from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard
        
        # Test default configuration
        dashboard = PerformanceDashboard()
        config = dashboard._get_default_config()
        
        required_keys = ['layout', 'theme', 'auto_refresh', 'charts']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        return False

def test_cli_imports():
    """Test CLI imports."""
    print("🧪 Testing CLI imports...")
    
    try:
        from trading_rl_agent.cli_dashboard import run_dashboard_cli
        print("✅ CLI imports successful")
        return True
    except ImportError as e:
        print(f"❌ CLI import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Dashboard Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
        ("CLI Imports", test_cli_imports),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n📋 Quick start commands:")
        print("  python -m trading_rl_agent.cli_dashboard run")
        print("  python -m trading_rl_agent.cli_dashboard run --streaming")
        print("  python examples/dashboard_example.py basic")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())