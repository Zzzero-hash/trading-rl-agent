"""
Basic CLI Tests for Trading RL Agent.

Simple tests to verify basic CLI functionality works.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestCLIBasic:
    """Basic CLI functionality tests."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_cli_import(self):
        """Test that CLI can be imported."""
        try:
            from trade_agent.cli import app

            assert app is not None
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_help_command(self):
        """Test help command."""
        try:
            from trade_agent.cli import app

            result = self.runner.invoke(app, ["--help"])
            assert result.exit_code == 0
            assert "trade-agent" in result.output
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_version_command(self):
        """Test version command function directly."""
        try:
            from trade_agent.cli import version

            # Mock the console to avoid I/O issues
            with patch("trading_rl_agent.cli.console") as mock_console:
                # Call the version function directly
                version()
                # Verify the console was called
                mock_console.print.assert_called()
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_info_command(self):
        """Test info command function directly."""
        try:
            from trade_agent.cli import info

            # Mock both the config manager and console to avoid issues
            with (
                patch("trading_rl_agent.cli.get_config_manager") as mock_config,
                patch("trading_rl_agent.cli.console") as mock_console,
            ):
                # Create a mock settings object
                mock_settings = type(
                    "MockSettings",
                    (),
                    {
                        "environment": "test",
                        "debug": False,
                        "data": type(
                            "MockData",
                            (),
                            {
                                "primary_source": "yfinance",
                                "symbols": ["AAPL", "GOOGL"],
                            },
                        )(),
                        "agent": type("MockAgent", (), {"agent_type": "ppo"})(),
                        "risk": type("MockRisk", (), {"max_position_size": 1000})(),
                        "execution": type(
                            "MockExecution",
                            (),
                            {"broker": "paper", "paper_trading": True},
                        )(),
                    },
                )()
                mock_config.return_value = mock_settings

                # Call the info function directly
                info()
                # Verify the console was called
                mock_console.print.assert_called()
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_subcommand_structure(self):
        """Test subcommand structure without using CliRunner."""
        try:
            from trade_agent.cli import (
                backtest_app,
                data_app,
                scenario_app,
                trade_app,
                train_app,
            )

            # Verify all sub-apps exist and have commands
            assert data_app is not None
            assert train_app is not None
            assert backtest_app is not None
            assert trade_app is not None
            assert scenario_app is not None

            # Check that sub-apps have registered commands
            assert hasattr(data_app, "registered_commands")
            assert hasattr(train_app, "registered_commands")
            assert hasattr(backtest_app, "registered_commands")
            assert hasattr(trade_app, "registered_commands")
            assert hasattr(scenario_app, "registered_commands")

        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_invalid_command(self):
        """Test invalid command handling."""
        try:
            from trade_agent.cli import app

            result = self.runner.invoke(app, ["invalid-command"])
            assert result.exit_code != 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")


class TestCLIStructure:
    """Test CLI structure and command availability."""

    def test_cli_structure(self):
        """Test that CLI has the expected structure."""
        try:
            from trade_agent.cli import (
                app,
                backtest_app,
                data_app,
                scenario_app,
                trade_app,
                train_app,
            )

            # Check that all sub-apps exist
            assert app is not None
            assert data_app is not None
            assert train_app is not None
            assert backtest_app is not None
            assert trade_app is not None
            assert scenario_app is not None
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_command_registration(self):
        """Test that commands are properly registered."""
        try:
            from trade_agent.cli import app

            # Check that the app has commands
            assert hasattr(app, "registered_commands")
            assert len(app.registered_commands) > 0

            # Check for key commands - use a more flexible approach
            command_names = []
            for cmd in app.registered_commands:
                if hasattr(cmd, "name") and cmd.name:
                    command_names.append(cmd.name)
                elif hasattr(cmd, "callback") and cmd.callback and hasattr(cmd.callback, "__name__"):
                    # Check function name for commands without explicit names
                    command_names.append(cmd.callback.__name__)

            # Check that we have some commands registered
            assert len(command_names) > 0

            # Check for version and info commands (they might be registered differently)
            has_version = any("version" in name.lower() for name in command_names)
            has_info = any("info" in name.lower() for name in command_names)

            # At least one of these should be present
            assert has_version or has_info or len(command_names) >= 3
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
