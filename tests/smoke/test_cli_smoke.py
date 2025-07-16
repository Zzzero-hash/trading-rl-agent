"""
Smoke tests for the Trading RL Agent CLI.

Fast tests that verify basic CLI functionality is working.
"""

import pytest


class TestCLISmoke:
    """Smoke tests for CLI commands."""

    @pytest.mark.smoke
    def test_cli_import(self):
        """Test that CLI can be imported (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        # Test basic imports
        from trading_rl_agent.cli import app

        assert app is not None

        # Test that registered commands exist
        assert hasattr(app, "registered_commands")
        assert len(app.registered_commands) > 0

    @pytest.mark.smoke
    def test_cli_structure(self):
        """Test CLI structure (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import backtest_app, data_app, trade_app, train_app

        # Test that all sub-apps exist
        assert data_app is not None
        assert train_app is not None
        assert backtest_app is not None
        assert trade_app is not None

    @pytest.mark.smoke
    def test_version_command_exists(self):
        """Test that version command exists (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import app

        # Check if version command exists - try different ways to access command info
        version_command = None
        for command in app.registered_commands:
            # Try different attributes that might contain the command name
            cmd_name = getattr(command, "name", None)
            if cmd_name == "version":
                version_command = command
                break

            # If name is None, try to get it from the callback function name
            callback = getattr(command, "callback", None)
            if callback and hasattr(callback, "__name__") and callback.__name__ == "version":
                version_command = command
                break

        assert version_command is not None

    @pytest.mark.smoke
    def test_data_commands_exist(self):
        """Test that data commands exist (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import data_app

        # Check if key data commands exist
        expected_commands = ["all", "symbols", "refresh", "download", "process", "standardize", "pipeline"]
        existing_commands = []

        for cmd in data_app.registered_commands:
            cmd_name = getattr(cmd, "name", None)
            if cmd_name:
                existing_commands.append(cmd_name)
            else:
                # Try to get name from callback function
                callback = getattr(cmd, "callback", None)
                if callback and hasattr(callback, "__name__"):
                    existing_commands.append(callback.__name__)

        for cmd in expected_commands:
            assert cmd in existing_commands, f"Data command '{cmd}' not found in {existing_commands}"

    @pytest.mark.smoke
    def test_train_commands_exist(self):
        """Test that train commands exist (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import train_app

        # Check if key train commands exist - using actual function names
        expected_commands = ["cnn_lstm", "rl", "hybrid", "hyperopt"]
        existing_commands = []

        for cmd in train_app.registered_commands:
            cmd_name = getattr(cmd, "name", None)
            if cmd_name:
                existing_commands.append(cmd_name)
            else:
                # Try to get name from callback function
                callback = getattr(cmd, "callback", None)
                if callback and hasattr(callback, "__name__"):
                    existing_commands.append(callback.__name__)

        for cmd in expected_commands:
            assert cmd in existing_commands, f"Train command '{cmd}' not found in {existing_commands}"

    @pytest.mark.smoke
    def test_backtest_commands_exist(self):
        """Test that backtest commands exist (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import backtest_app

        # Check if key backtest commands exist
        expected_commands = ["strategy", "evaluate", "compare", "report"]
        existing_commands = []

        for cmd in backtest_app.registered_commands:
            cmd_name = getattr(cmd, "name", None)
            if cmd_name:
                existing_commands.append(cmd_name)
            else:
                # Try to get name from callback function
                callback = getattr(cmd, "callback", None)
                if callback and hasattr(callback, "__name__"):
                    existing_commands.append(callback.__name__)

        for cmd in expected_commands:
            assert cmd in existing_commands, f"Backtest command '{cmd}' not found in {existing_commands}"

    @pytest.mark.smoke
    def test_trade_commands_exist(self):
        """Test that trade commands exist (smoke test)."""
        import sys
        from pathlib import Path

        # Ensure src is in Python path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from trading_rl_agent.cli import trade_app

        # Check if key trade commands exist
        expected_commands = ["start", "stop", "status", "monitor", "paper"]
        existing_commands = []

        for cmd in trade_app.registered_commands:
            cmd_name = getattr(cmd, "name", None)
            if cmd_name:
                existing_commands.append(cmd_name)
            else:
                # Try to get name from callback function
                callback = getattr(cmd, "callback", None)
                if callback and hasattr(callback, "__name__"):
                    existing_commands.append(callback.__name__)

        for cmd in expected_commands:
            assert cmd in existing_commands, f"Trade command '{cmd}' not found in {existing_commands}"
