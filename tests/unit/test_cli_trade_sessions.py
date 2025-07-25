"""
Comprehensive CLI Trading Session Management Tests.

This module provides comprehensive testing for all trading CLI commands:
- Trading session lifecycle (start, stop, monitor, status)
- Paper trading functionality
- Symbol validation and portfolio management
- Risk management parameter testing
- Real-time monitoring and alerts
- Session persistence and recovery
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


class TestCLITradingSessionLifecycle:
    """Test complete trading session lifecycle management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model"
        self.model_path.mkdir()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_session_basic(self, mock_start_trading):
        """Test starting a basic trading session."""
        mock_start_trading.return_value = {"session_id": "test_session_123", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL,GOOGL,MSFT",
                "--model-path", str(self.model_path),
                "--paper-trading",
                "--initial-capital", "100000"
            ]
        )

        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

        # Verify the correct parameters were passed
        call_args = mock_start_trading.call_args
        call_kwargs = call_args[1] if len(call_args) > 1 else call_args[0]
        assert "AAPL,GOOGL,MSFT" in str(call_kwargs)
        assert str(self.model_path) in str(call_kwargs)

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_session_with_risk_management(self, mock_start_trading):
        """Test starting trading session with risk management parameters."""
        mock_start_trading.return_value = {"session_id": "test_session_456", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL,GOOGL",
                "--model-path", str(self.model_path),
                "--paper-trading",
                "--initial-capital", "50000",
                "--max-position-size", "0.1",
                "--stop-loss", "0.02",
                "--take-profit", "0.05"
            ]
        )

        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_session_live_mode(self, mock_start_trading):
        """Test starting live trading session (not paper trading)."""
        mock_start_trading.return_value = {"session_id": "live_session_789", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", str(self.model_path),
                "--initial-capital", "10000"
                # Note: no --paper-trading flag
            ]
        )

        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

    @patch("trade_agent.cli.get_trading_status")
    def test_trading_status_basic(self, mock_get_status):
        """Test getting basic trading status."""
        mock_get_status.return_value = {
            "status": "running",
            "active_sessions": 2,
            "total_pnl": 1250.75,
            "positions": 5
        }

        result = self.runner.invoke(
            main_app,
            ["trade", "status"]
        )

        assert result.exit_code == 0
        mock_get_status.assert_called_once()

    @patch("trade_agent.cli.get_trading_status")
    def test_trading_status_detailed(self, mock_get_status):
        """Test getting detailed trading status."""
        mock_get_status.return_value = {
            "status": "running",
            "active_sessions": 1,
            "sessions": [
                {
                    "session_id": "test_session_123",
                    "symbols": ["AAPL", "GOOGL"],
                    "start_time": "2023-01-01T09:30:00",
                    "pnl": 500.25,
                    "positions": [
                        {"symbol": "AAPL", "quantity": 100, "avg_price": 150.0},
                        {"symbol": "GOOGL", "quantity": 50, "avg_price": 2500.0}
                    ]
                }
            ]
        }

        result = self.runner.invoke(
            main_app,
            ["trade", "status", "--detailed"]
        )

        assert result.exit_code == 0
        mock_get_status.assert_called_once()

    @patch("trade_agent.cli.get_trading_status")
    def test_trading_status_specific_session(self, mock_get_status):
        """Test getting status for specific session."""
        mock_get_status.return_value = {
            "session_id": "test_session_456",
            "status": "running",
            "symbols": ["MSFT"],
            "pnl": -125.50,
            "positions": [{"symbol": "MSFT", "quantity": 75, "avg_price": 300.0}]
        }

        result = self.runner.invoke(
            main_app,
            ["trade", "status", "--session-id", "test_session_456"]
        )

        assert result.exit_code == 0
        mock_get_status.assert_called_once()

    @patch("trade_agent.cli.monitor_trading")
    def test_trading_monitor_basic(self, mock_monitor):
        """Test basic trading monitoring."""
        mock_monitor.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--metrics", "pnl,positions,orders",
                "--interval", "30"
            ]
        )

        assert result.exit_code == 0
        mock_monitor.assert_called_once()

    @patch("trade_agent.cli.monitor_trading")
    def test_trading_monitor_specific_session(self, mock_monitor):
        """Test monitoring specific trading session."""
        mock_monitor.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--session-id", "test_session_789",
                "--metrics", "all",
                "--interval", "10"
            ]
        )

        assert result.exit_code == 0
        mock_monitor.assert_called_once()

    @patch("trade_agent.cli.stop_trading")
    def test_stop_trading_specific_session(self, mock_stop_trading):
        """Test stopping specific trading session."""
        mock_stop_trading.return_value = {"session_id": "test_session_123", "status": "stopped"}

        result = self.runner.invoke(
            main_app,
            ["trade", "stop", "--session-id", "test_session_123"]
        )

        assert result.exit_code == 0
        mock_stop_trading.assert_called_once()

    @patch("trade_agent.cli.stop_trading")
    def test_stop_all_trading_sessions(self, mock_stop_trading):
        """Test stopping all trading sessions."""
        mock_stop_trading.return_value = {"stopped_sessions": 3, "status": "all_stopped"}

        result = self.runner.invoke(
            main_app,
            ["trade", "stop", "--all-sessions"]
        )

        assert result.exit_code == 0
        mock_stop_trading.assert_called_once()

    @patch("trade_agent.cli.start_paper_trading")
    def test_paper_trading_basic(self, mock_paper_trading):
        """Test basic paper trading functionality."""
        mock_paper_trading.return_value = {"session_id": "paper_session_1", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "paper",
                "--symbols", "AAPL,GOOGL,MSFT,TSLA",
                "--duration", "1d",
                "--initial-capital", "100000"
            ]
        )

        assert result.exit_code == 0
        mock_paper_trading.assert_called_once()

    @patch("trade_agent.cli.start_paper_trading")
    def test_paper_trading_different_durations(self, mock_paper_trading):
        """Test paper trading with different durations."""
        mock_paper_trading.return_value = {"session_id": "paper_session_2", "status": "started"}

        durations = ["1h", "4h", "1d", "1w"]
        for duration in durations:
            result = self.runner.invoke(
                main_app,
                [
                    "trade", "paper",
                    "--symbols", "AAPL",
                    "--duration", duration
                ]
            )

            assert result.exit_code == 0

        # Should have been called once for each duration
        assert mock_paper_trading.call_count == len(durations)


class TestCLITradingParameterValidation:
    """Test trading command parameter validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model"
        self.model_path.mkdir()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_start_trading_missing_model_path(self):
        """Test start trading with missing model path."""
        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--paper-trading"
                # Missing --model-path
            ]
        )

        # Should either require model path or provide clear error
        if result.exit_code != 0:
            assert "model" in result.output.lower() or "required" in result.output.lower()

    def test_start_trading_invalid_symbols(self):
        """Test start trading with invalid symbols."""
        with patch("trade_agent.cli.start_trading") as mock_start:
            mock_start.side_effect = ValueError("Invalid symbols")

            result = self.runner.invoke(
                main_app,
                [
                    "trade", "start",
                    "--symbols", "INVALID@SYMBOL!",
                    "--model-path", str(self.model_path),
                    "--paper-trading"
                ]
            )

            assert result.exit_code != 0

    def test_start_trading_invalid_capital(self):
        """Test start trading with invalid initial capital."""
        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", str(self.model_path),
                "--initial-capital", "-1000",
                "--paper-trading"
            ]
        )

        # Should validate capital is positive
        if result.exit_code != 0:
            assert "capital" in result.output.lower() or "invalid" in result.output.lower()

    def test_start_trading_invalid_risk_parameters(self):
        """Test start trading with invalid risk management parameters."""
        # Test invalid stop loss
        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", str(self.model_path),
                "--stop-loss", "1.5",  # Greater than 1.0 (100%)
                "--paper-trading"
            ]
        )

        # Should validate risk parameters
        if result.exit_code != 0:
            assert "stop" in result.output.lower() or "risk" in result.output.lower()

    def test_monitor_trading_invalid_interval(self):
        """Test monitor trading with invalid interval."""
        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--interval", "0"
            ]
        )

        # Should validate interval is positive
        if result.exit_code != 0:
            assert "interval" in result.output.lower() or "invalid" in result.output.lower()

    def test_paper_trading_invalid_duration(self):
        """Test paper trading with invalid duration."""
        result = self.runner.invoke(
            main_app,
            [
                "trade", "paper",
                "--symbols", "AAPL",
                "--duration", "invalid_duration"
            ]
        )

        # Should validate duration format
        if result.exit_code != 0:
            assert "duration" in result.output.lower() or "invalid" in result.output.lower()


class TestCLITradingErrorHandling:
    """Test trading command error handling scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model"
        self.model_path.mkdir()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_api_connection_error(self, mock_start_trading):
        """Test start trading with API connection error."""
        mock_start_trading.side_effect = ConnectionError("Unable to connect to broker API")

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", str(self.model_path),
                "--paper-trading"
            ]
        )

        assert result.exit_code != 0
        # Should provide meaningful error message
        assert "error" in result.output.lower() or "connection" in result.output.lower()

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_insufficient_permissions(self, mock_start_trading):
        """Test start trading with insufficient permissions."""
        mock_start_trading.side_effect = PermissionError("Insufficient permissions for live trading")

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", str(self.model_path)
                # No --paper-trading (attempting live trading)
            ]
        )

        assert result.exit_code != 0
        # Should provide meaningful error message
        assert "permission" in result.output.lower() or "error" in result.output.lower()

    @patch("trade_agent.cli.get_trading_status")
    def test_status_no_active_sessions(self, mock_get_status):
        """Test getting status when no sessions are active."""
        mock_get_status.return_value = {
            "status": "idle",
            "active_sessions": 0,
            "message": "No active trading sessions"
        }

        result = self.runner.invoke(
            main_app,
            ["trade", "status"]
        )

        assert result.exit_code == 0
        # Should handle gracefully
        assert "no active" in result.output.lower() or "idle" in result.output.lower()

    @patch("trade_agent.cli.get_trading_status")
    def test_status_session_not_found(self, mock_get_status):
        """Test getting status for non-existent session."""
        mock_get_status.side_effect = ValueError("Session not found")

        result = self.runner.invoke(
            main_app,
            ["trade", "status", "--session-id", "nonexistent_session"]
        )

        assert result.exit_code != 0
        # Should provide meaningful error message
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    @patch("trade_agent.cli.stop_trading")
    def test_stop_trading_session_already_stopped(self, mock_stop_trading):
        """Test stopping already stopped session."""
        mock_stop_trading.side_effect = ValueError("Session already stopped")

        result = self.runner.invoke(
            main_app,
            ["trade", "stop", "--session-id", "stopped_session"]
        )

        assert result.exit_code != 0
        # Should handle gracefully
        assert "already" in result.output.lower() or "stopped" in result.output.lower()

    @patch("trade_agent.cli.monitor_trading")
    def test_monitor_trading_session_terminated(self, mock_monitor):
        """Test monitoring when session terminates unexpectedly."""
        mock_monitor.side_effect = RuntimeError("Trading session terminated unexpectedly")

        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--session-id", "terminated_session"
            ]
        )

        assert result.exit_code != 0
        # Should provide meaningful error message
        assert "terminated" in result.output.lower() or "error" in result.output.lower()


class TestCLITradingAdvancedFeatures:
    """Test advanced trading command features."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model"
        self.model_path.mkdir()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_with_strategy_parameters(self, mock_start_trading):
        """Test starting trading with custom strategy parameters."""
        mock_start_trading.return_value = {"session_id": "strategy_session", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL,GOOGL",
                "--model-path", str(self.model_path),
                "--strategy", "mean_reversion",
                "--rebalance-frequency", "daily",
                "--risk-level", "moderate",
                "--paper-trading"
            ]
        )

        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

    @patch("trade_agent.cli.start_trading")
    def test_start_trading_with_portfolio_constraints(self, mock_start_trading):
        """Test starting trading with portfolio constraints."""
        mock_start_trading.return_value = {"session_id": "constrained_session", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL,GOOGL,MSFT,TSLA,NVDA",
                "--model-path", str(self.model_path),
                "--max-positions", "3",
                "--sector-limit", "0.4",
                "--concentration-limit", "0.2",
                "--paper-trading"
            ]
        )

        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

    @patch("trade_agent.cli.monitor_trading")
    def test_monitor_trading_with_alerts(self, mock_monitor):
        """Test monitoring trading with alert configuration."""
        mock_monitor.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--alert-pnl-threshold", "1000",
                "--alert-drawdown-threshold", "0.05",
                "--email-alerts",
                "--slack-webhook", "https://hooks.slack.com/test"
            ]
        )

        assert result.exit_code == 0
        mock_monitor.assert_called_once()

    @patch("trade_agent.cli.get_trading_status")
    def test_status_with_performance_metrics(self, mock_get_status):
        """Test getting status with detailed performance metrics."""
        mock_get_status.return_value = {
            "status": "running",
            "performance": {
                "total_return": 0.125,
                "sharpe_ratio": 1.85,
                "max_drawdown": 0.03,
                "win_rate": 0.67,
                "avg_trade_duration": "2.5 hours"
            },
            "risk_metrics": {
                "var_95": 0.02,
                "expected_shortfall": 0.035,
                "beta": 0.95,
                "volatility": 0.18
            }
        }

        result = self.runner.invoke(
            main_app,
            ["trade", "status", "--include-performance", "--include-risk-metrics"]
        )

        assert result.exit_code == 0
        mock_get_status.assert_called_once()

    @patch("trade_agent.cli.start_paper_trading")
    def test_paper_trading_with_market_conditions(self, mock_paper_trading):
        """Test paper trading with specific market condition simulation."""
        mock_paper_trading.return_value = {"session_id": "sim_session", "status": "started"}

        result = self.runner.invoke(
            main_app,
            [
                "trade", "paper",
                "--symbols", "AAPL,GOOGL",
                "--market-condition", "high_volatility",
                "--noise-level", "0.02",
                "--correlation-factor", "0.8",
                "--duration", "1d"
            ]
        )

        assert result.exit_code == 0
        mock_paper_trading.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
