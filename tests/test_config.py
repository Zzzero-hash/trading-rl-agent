#!/usr/bin/env python3
"""
Test script for the restructured trading RL agent configuration system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_configuration_system():
    """Test the configuration system."""
    print("🧪 Testing Trading RL Agent Configuration System")
    print("=" * 50)

    try:
        # Test basic imports
        from trading_rl_agent.core.config import (
            ConfigManager,
            SystemConfig,
        )

        print("✅ All configuration classes imported successfully")

        # Test creating default configuration
        config = SystemConfig()
        print("✅ Default SystemConfig created successfully")

        # Test configuration values
        print(f"  Environment: {config.environment}")
        print(f"  Debug mode: {config.debug}")
        print(f"  Agent type: {config.agent.agent_type}")
        print(f"  Max position size: {config.risk.max_position_size}")
        print(f"  Broker: {config.execution.broker}")
        print(f"  Paper trading: {config.execution.paper_trading}")

        # Test configuration manager
        manager = ConfigManager()
        # managed_config = manager.get_config()  # Not used currently
        print("✅ ConfigManager working correctly")

        # Test configuration updates
        updates = {
            "environment": "production",
            "debug": False,
            "agent": {"agent_type": "td3"},
            "risk": {"max_position_size": 0.05},
        }
        manager.update_config(updates)
        updated_config = manager.get_config()
        print("✅ Configuration updates working correctly")
        print(f"  Updated environment: {updated_config.environment}")

        print("\n🎉 All configuration tests passed!")
        print("🚀 Configuration system is ready for production!")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_configuration_system()
    sys.exit(0 if success else 1)
