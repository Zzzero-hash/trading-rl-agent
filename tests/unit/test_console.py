"""
Unit tests for console utility functions.

Tests the table printing and formatting functions in console.py.
"""

import io
from unittest.mock import patch

from trade_agent.console import (
    console,
    print_error_summary,
    print_metrics_table,
    print_status_table,
    print_table,
)


class TestConsoleFunctions:
    """Test console utility functions."""

    def test_print_table_basic(self):
        """Test basic table printing."""
        rows = [[1, "AAPL", 150.25], [2, "GOOGL", 2750.50], [3, "MSFT", 300.75]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers, title="Stock Prices")
            output = fake_out.getvalue()

            assert "Stock Prices" in output
            assert "ID" in output
            assert "Symbol" in output
            assert "Price" in output
            assert "AAPL" in output
            assert "GOOGL" in output
            assert "MSFT" in output

    def test_print_table_no_headers(self):
        """Test table printing without headers."""
        rows = [[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows)
            output = fake_out.getvalue()

            assert "Col 1" in output
            assert "Col 2" in output
            assert "Col 3" in output
            assert "AAPL" in output
            assert "GOOGL" in output

    def test_print_table_single_row(self):
        """Test table printing with single row (should fall back to print)."""
        rows = [[1, "AAPL", 150.25]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # Should use simple print format for single row
            assert "ID: 1" in output
            assert "Symbol: AAPL" in output
            assert "Price: 150.25" in output

    def test_print_table_empty_rows(self):
        """Test table printing with empty rows."""
        rows = []
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            assert "No data to display" in output

    def test_print_table_tsv_output(self):
        """Test table printing with TSV output."""
        rows = [[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers, no_style=True)
            output = fake_out.getvalue()

            # Should be tab-separated
            lines = output.strip().split("\n")
            assert len(lines) == 3  # headers + 2 rows
            assert "\t" in lines[0]  # headers should be tab-separated
            assert "\t" in lines[1]  # first row should be tab-separated

    def test_print_table_with_none_values(self):
        """Test table printing with None values."""
        rows = [[1, "AAPL", None], [2, None, 2750.50], [None, "MSFT", 300.75]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # None values should be replaced with "-"
            assert "-" in output
            assert "AAPL" in output
            assert "MSFT" in output

    def test_print_table_different_styles(self):
        """Test table printing with different styles."""
        rows = [[1, "AAPL", 150.25]]
        headers = ["ID", "Symbol", "Price"]

        styles = ["ascii", "simple", "grid", "minimal"]

        for style in styles:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                print_table(rows, headers, style=style)
                output = fake_out.getvalue()

                # Should contain the data regardless of style
                assert "AAPL" in output
                assert "150.25" in output

    def test_print_metrics_table(self):
        """Test metrics table printing."""
        results = [
            {
                "strategy": "momentum",
                "total_return": 0.15,
                "sharpe_ratio": 1.25,
                "max_drawdown": -0.05,
                "win_rate": 0.65,
                "num_trades": 100,
            },
            {
                "strategy": "mean_reversion",
                "total_return": 0.08,
                "sharpe_ratio": 0.85,
                "max_drawdown": -0.03,
                "win_rate": 0.55,
                "num_trades": 75,
            },
        ]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_metrics_table(results, title="Strategy Comparison")
            output = fake_out.getvalue()

            assert "Strategy Comparison" in output
            assert "momentum" in output
            assert "mean_reversi" in output  # Accept partial match due to truncation
            assert "15.00%" in output  # formatted percentage
            assert "8.00%" in output  # formatted percentage
            assert "1.25" in output
            assert "0.85" in output

    def test_print_metrics_table_empty(self):
        """Test metrics table printing with empty results."""
        results = []

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_metrics_table(results)
            output = fake_out.getvalue()

            assert "No results to display" in output

    def test_print_metrics_table_single_result(self):
        """Test metrics table printing with single result."""
        results = [{"strategy": "momentum", "total_return": 0.15, "sharpe_ratio": 1.25}]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_metrics_table(results)
            output = fake_out.getvalue()

            # Should use simple print format for single result
            assert "strategy: momentum" in output
            assert "total_return: 15.00%" in output
            assert "sharpe_ratio: 1.25" in output

    def test_print_status_table(self):
        """Test status table printing."""
        status_data = {
            "Python Version": "3.9.0",
            "Trading RL Agent": "1.0.0",
            "PyTorch": "1.12.0",
            "CUDA Available": "False",
        }

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_status_table(status_data, title="System Status")
            output = fake_out.getvalue()

            assert "System Status" in output
            assert "Python Version" in output
            assert "3.9.0" in output
            assert "Trading RL Agent" in output
            assert "1.0.0" in output

    def test_print_status_table_empty(self):
        """Test status table printing with empty data."""
        status_data = {}

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_status_table(status_data)
            output = fake_out.getvalue()

            assert "No status data to display" in output

    def test_print_error_summary(self):
        """Test error summary printing."""
        errors = [
            "Failed to download data for AAPL",
            "Model training failed due to insufficient data",
            "Invalid configuration parameter: learning_rate",
        ]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_error_summary(errors, title="Error Summary")
            output = fake_out.getvalue()

            assert "Error Summary" in output
            assert "Failed to download data for AAPL" in output
            assert "Model training failed due to insufficient data" in output
            assert "Invalid configuration parameter: learning_rate" in output

    def test_print_error_summary_empty(self):
        """Test error summary printing with empty errors."""
        errors = []

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_error_summary(errors)
            output = fake_out.getvalue()

            # Should not print anything for empty errors
            assert output == ""

    def test_print_table_with_float_formatting(self):
        """Test table printing with float formatting."""
        rows = [[1, "AAPL", 150.25], [2, "GOOGL", 2750.50], [3, "MSFT", 300.75]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # Floats should be formatted to 2 decimal places
            assert "150.25" in output
            assert "2750.50" in output
            assert "300.75" in output

    def test_print_table_with_int_formatting(self):
        """Test table printing with integer formatting."""
        rows = [[1, "AAPL", 150], [2, "GOOGL", 2750], [3, "MSFT", 300]]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # Integers should be displayed as-is
            assert "150" in output
            assert "2750" in output
            assert "300" in output

    def test_print_metrics_table_with_mixed_types(self):
        """Test metrics table with mixed data types."""
        results = [
            {
                "strategy": "momentum",
                "total_return": 0.15,
                "sharpe_ratio": 1.25,
                "num_trades": 100,
                "is_active": True,
                "last_update": "2024-01-01",
            }
        ]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_metrics_table(results)
            output = fake_out.getvalue()

            # Should handle mixed types correctly
            assert "strategy: momentum" in output
            assert "total_return: 15.00%" in output
            assert "sharpe_ratio: 1.25" in output
            assert "num_trades: 100" in output
            assert "is_active: True" in output
            assert "last_update: 2024-01-01" in output

    def test_print_table_column_width_limits(self):
        """Test table printing with column width limits."""
        rows = [
            [1, "This is a very long symbol name that should be truncated", 150.25],
            [2, "Another very long symbol name that exceeds the limit", 2750.50],
        ]
        headers = ["ID", "Symbol", "Price"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # Should handle long text gracefully
            assert "ID" in output
            assert "Symbol" in output
            assert "Price" in output

    def test_console_initialization(self):
        """Test that console is properly initialized."""
        assert console is not None
        assert hasattr(console, "print")
        assert callable(console.print)

    def test_print_table_with_unicode_characters(self):
        """Test table printing with unicode characters."""
        rows = [[1, "AAPL", "📈"], [2, "GOOGL", "📉"], [3, "MSFT", "📊"]]
        headers = ["ID", "Symbol", "Status"]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_table(rows, headers)
            output = fake_out.getvalue()

            # Should handle unicode characters
            assert "📈" in output
            assert "📉" in output
            assert "📊" in output

    def test_print_metrics_table_percentage_formatting(self):
        """Test metrics table percentage formatting."""
        results = [
            {
                "strategy": "momentum",
                "return_rate": 0.15,
                "sharpe_ratio": 1.25,
                "win_rate": 0.65,
            }
        ]

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            print_metrics_table(results)
            output = fake_out.getvalue()

            # Should format percentages correctly
            assert "return_rate: 15.00%" in output
            assert "win_rate: 65.00%" in output
            assert "sharpe_ratio: 1.25" in output  # Not a percentage
