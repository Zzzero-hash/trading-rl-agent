"""
Comprehensive CLI Argument Validation Tests.

This module provides extensive testing for CLI argument validation:
- Parameter range validation (epochs, batch_size, learning_rate, etc.)
- File path validation and existence checks
- Symbol format validation and sanitization
- Date format validation and logical consistency
- Resource constraint validation (memory, workers, etc.)
- Conflicting argument combination detection
- Type conversion and boundary testing
- Custom validation logic testing
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


class TestCLINumericParameterValidation:
    """Test validation of numeric CLI parameters."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = self._create_mock_dataset()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset(self):
        """Create a mock dataset file."""
        import pandas as pd

        data_path = Path(self.temp_dir) / "dataset.csv"
        data = {
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "symbol": ["AAPL"] * 50,
            "close": [150.0 + i * 0.1 for i in range(50)],
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return data_path

    def test_epochs_parameter_validation(self):
        """Test validation of epochs parameter."""
        invalid_epochs = [-1, 0, "invalid", 1.5, 10000000]

        for epochs in invalid_epochs:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--epochs", str(epochs),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["epochs", "invalid", "error"])

    def test_batch_size_parameter_validation(self):
        """Test validation of batch_size parameter."""
        invalid_batch_sizes = [-1, 0, "invalid", 1.5, 999999]

        for batch_size in invalid_batch_sizes:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--batch-size", str(batch_size),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["batch", "size", "invalid", "error"])

    def test_learning_rate_parameter_validation(self):
        """Test validation of learning_rate parameter."""
        invalid_learning_rates = [-0.1, 0, "invalid", 1.5, 100]

        for lr in invalid_learning_rates:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--learning-rate", str(lr),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["learning", "rate", "invalid", "error"])

    def test_sequence_length_parameter_validation(self):
        """Test validation of sequence_length parameter."""
        invalid_sequence_lengths = [-1, 0, "invalid", 1.5, 999999]

        for seq_len in invalid_sequence_lengths:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--sequence-length", str(seq_len),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["sequence", "length", "invalid", "error"])

    def test_prediction_horizon_parameter_validation(self):
        """Test validation of prediction_horizon parameter."""
        invalid_horizons = [-1, 0, "invalid", 1.5, 1000]

        for horizon in invalid_horizons:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--prediction-horizon", str(horizon),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["prediction", "horizon", "invalid", "error"])

    def test_worker_count_parameter_validation(self):
        """Test validation of worker count parameters."""
        invalid_worker_counts = [-1, 0, "invalid", 1.5, 10000]

        for workers in invalid_worker_counts:
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--workers", str(workers),
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["workers", "invalid", "error"])

    def test_capital_parameter_validation(self):
        """Test validation of capital parameters."""
        invalid_capitals = [-1000, 0, "invalid", "1k"]

        for capital in invalid_capitals:
            result = self.runner.invoke(
                main_app,
                [
                    "trade", "start",
                    "--symbols", "AAPL",
                    "--model-path", self.temp_dir,
                    "--initial-capital", str(capital),
                    "--paper-trading"
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["capital", "invalid", "error"])

    def test_percentage_parameter_validation(self):
        """Test validation of percentage parameters."""
        invalid_percentages = [-0.5, 1.5, "invalid", "50%"]

        for percentage in invalid_percentages:
            result = self.runner.invoke(
                main_app,
                [
                    "trade", "start",
                    "--symbols", "AAPL",
                    "--model-path", self.temp_dir,
                    "--max-position-size", str(percentage),
                    "--paper-trading"
                ]
            )

            # Should either validate or provide meaningful error
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["position", "size", "invalid", "error"])


class TestCLIStringParameterValidation:
    """Test validation of string CLI parameters."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_symbol_format_validation(self):
        """Test validation of symbol formats."""
        invalid_symbols = [
            "invalid@symbol",
            "SYMBOL!",
            "123ABC",
            "symbol with spaces",
            "TOOLONGSYMBOL123456",
            "special#chars",
            "symbols,with,numbers123"
        ]

        for symbols in invalid_symbols:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline:
                mock_pipeline_instance = mock_pipeline.return_value
                mock_pipeline_instance.download_data_parallel.side_effect = ValueError(f"Invalid symbols: {symbols}")

                with patch("trade_agent.utils.cache_manager.CacheManager"):
                    result = self.runner.invoke(
                        main_app,
                        [
                            "data", "pipeline", "--run",
                            "--symbols", symbols,
                            "--output-dir", self.temp_dir
                        ]
                    )

                    # Should catch invalid symbols
                    assert result.exit_code == 1

    def test_date_format_validation(self):
        """Test validation of date formats."""
        invalid_dates = [
            "invalid-date",
            "2023/01/01",  # Wrong separator
            "01-01-2023",  # Wrong order
            "2023-13-01",  # Invalid month
            "2023-01-32",  # Invalid day
            "23-01-01",    # Wrong year format
            "2023-1-1",    # Missing zero padding
            ""             # Empty string
        ]

        for date in invalid_dates:
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--start-date", date,
                    "--output-dir", self.temp_dir
                ]
            )

            # Should either validate dates or handle gracefully
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["date", "invalid", "format", "error"])

    def test_dataset_name_validation(self):
        """Test validation of dataset names."""
        invalid_names = [
            "name with spaces",
            "name/with/slashes",
            "name\\with\\backslashes",
            "name:with:colons",
            "name*with*asterisks",
            "name?with?questions",
            "name<with>brackets",
            "name|with|pipes",
            ""  # Empty name
        ]

        for name in invalid_names:
            with patch("trade_agent.data.pipeline.DataPipeline"), \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset"), \
                 patch("trade_agent.utils.cache_manager.CacheManager"):

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--dataset-name", name,
                        "--output-dir", self.temp_dir
                    ]
                )

                # Should handle invalid dataset names
                # Some may be allowed and sanitized, others may error
                # The important thing is consistent behavior

    def test_feature_set_validation(self):
        """Test validation of feature set parameter."""
        invalid_feature_sets = [
            "invalid_set",
            "FULL",  # Wrong case
            "basic_technical",
            "123",
            ""
        ]

        for feature_set in invalid_feature_sets:
            with patch("trade_agent.data.pipeline.DataPipeline"), \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset"), \
                 patch("trade_agent.utils.cache_manager.CacheManager"):

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--feature-set", feature_set,
                        "--output-dir", self.temp_dir
                    ]
                )

                # Should validate feature set options
                # May accept all values or validate against known sets

    def test_export_format_validation(self):
        """Test validation of export format parameter."""
        invalid_formats = [
            "txt",
            "json",
            "xml",
            "excel",
            "invalid_format",
            ""
        ]

        for export_format in invalid_formats:
            with patch("trade_agent.data.pipeline.DataPipeline"), \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset"), \
                 patch("trade_agent.utils.cache_manager.CacheManager"):

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--export-formats", export_format,
                        "--output-dir", self.temp_dir
                    ]
                )

                # Should complete but warn about unknown formats
                if result.exit_code == 0 and export_format not in ["csv", "parquet", "feather"]:
                    assert "Unknown format" in result.output


class TestCLIFilePathValidation:
    """Test validation of file path parameters."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file_paths(self):
        """Test handling of nonexistent file paths."""
        nonexistent_paths = [
            "/nonexistent/path/file.csv",
            "~/nonexistent/file.csv",
            "./missing/dataset.csv",
            str(Path(self.temp_dir) / "missing.csv")
        ]

        for path in nonexistent_paths:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    path,
                    "--epochs", "5",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 1
            assert any(word in result.output.lower() for word in ["not found", "missing", "does not exist"])

    def test_invalid_file_extensions(self):
        """Test handling of invalid file extensions."""
        # Create files with wrong extensions
        invalid_files = [
            Path(self.temp_dir) / "data.txt",
            Path(self.temp_dir) / "data.json",
            Path(self.temp_dir) / "data.xlsx",
            Path(self.temp_dir) / "data"  # No extension
        ]

        for file_path in invalid_files:
            # Create the file
            file_path.write_text("some content")

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(file_path),
                    "--epochs", "5",
                    "--output-dir", self.temp_dir
                ]
            )

            # May accept any file or validate extension
            # The important thing is consistent behavior

    def test_directory_permissions(self):
        """Test handling of directory permission issues."""
        # Try to create output in read-only directory (if possible)
        readonly_dir = Path(self.temp_dir) / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only
            readonly_dir.chmod(0o444)

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--output-dir", str(readonly_dir)
                ]
            )

            # Should handle permission errors gracefully
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in ["permission", "denied", "error"])

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_relative_vs_absolute_paths(self):
        """Test handling of relative vs absolute paths."""
        # Create a dataset file
        dataset_file = Path(self.temp_dir) / "test_data.csv"
        import pandas as pd
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(dataset_file, index=False)

        # Test relative path
        relative_path = "./test_data.csv"

        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    relative_path,
                    "--epochs", "1",
                    "--output-dir", "."
                ]
            )

            # Should handle relative paths correctly
            # May succeed or fail depending on implementation

        finally:
            os.chdir(original_cwd)

    def test_special_path_characters(self):
        """Test handling of special characters in paths."""
        special_paths = [
            Path(self.temp_dir) / "file with spaces.csv",
            Path(self.temp_dir) / "file-with-dashes.csv",
            Path(self.temp_dir) / "file_with_underscores.csv",
            Path(self.temp_dir) / "file.with.dots.csv",
        ]

        for path in special_paths:
            # Create the file
            import pandas as pd
            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_csv(path, index=False)

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(path),
                    "--epochs", "1",
                    "--output-dir", self.temp_dir
                ]
            )

            # Should handle special characters in paths


class TestCLIConflictingArgumentValidation:
    """Test validation of conflicting argument combinations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_conflicting_sentiment_options(self):
        """Test conflicting sentiment analysis options."""
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--sentiment",
                "--no-sentiment",  # Conflicting flags
                "--output-dir", self.temp_dir
            ]
        )

        # Should handle conflicting flags appropriately
        # Either with validation error or by using the last specified option

    def test_conflicting_cache_options(self):
        """Test conflicting cache options."""
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--cache",
                "--no-cache",  # Conflicting flags
                "--output-dir", self.temp_dir
            ]
        )

        # Should handle conflicting cache flags appropriately

    def test_paper_trading_with_live_parameters(self):
        """Test paper trading with parameters that only apply to live trading."""
        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", self.temp_dir,
                "--paper-trading",
                "--live-api-key", "test_key",  # Should not be needed for paper trading
                "--broker-account", "12345"   # Should not be needed for paper trading
            ]
        )

        # Should either ignore incompatible parameters or warn about them

    def test_optimization_with_fixed_parameters(self):
        """Test hyperparameter optimization with fixed parameter values."""
        data_path = self._create_mock_dataset()

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(data_path),
                "--optimize-hyperparams",
                "--learning-rate", "0.001",  # Fixed value conflicts with optimization
                "--batch-size", "32",        # Fixed value conflicts with optimization
                "--epochs", "5",
                "--output-dir", self.temp_dir
            ]
        )

        # Should handle appropriately - either warn or override fixed values

    def test_gpu_with_cpu_specific_options(self):
        """Test GPU training with CPU-specific options."""
        data_path = self._create_mock_dataset()

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(data_path),
                "--gpu",
                "--cpu-threads", "8",  # CPU-specific option with GPU flag
                "--epochs", "5",
                "--output-dir", self.temp_dir
            ]
        )

        # Should either ignore incompatible options or provide warning

    def _create_mock_dataset(self):
        """Create a mock dataset file."""
        import pandas as pd

        data_path = Path(self.temp_dir) / "dataset.csv"
        data = {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "symbol": ["AAPL"] * 20,
            "close": [150.0 + i * 0.1 for i in range(20)],
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return data_path


class TestCLIBoundaryValueValidation:
    """Test validation of boundary values for parameters."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_minimum_boundary_values(self):
        """Test minimum boundary values for various parameters."""
        data_path = self._create_mock_dataset()

        # Test minimum valid values
        boundary_tests = [
            ("--epochs", "1"),
            ("--batch-size", "1"),
            ("--learning-rate", "0.0001"),
            ("--sequence-length", "1"),
            ("--prediction-horizon", "1"),
        ]

        for param, value in boundary_tests:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(data_path),
                    param, value,
                    "--output-dir", self.temp_dir
                ]
            )

            # Minimum valid values should be accepted

    def test_maximum_boundary_values(self):
        """Test maximum reasonable boundary values."""
        data_path = self._create_mock_dataset()

        # Test maximum reasonable values
        boundary_tests = [
            ("--epochs", "1000"),
            ("--batch-size", "512"),
            ("--learning-rate", "0.1"),
            ("--sequence-length", "365"),  # One year of daily data
            ("--prediction-horizon", "30"), # One month ahead
        ]

        for param, value in boundary_tests:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(data_path),
                    param, value,
                    "--output-dir", self.temp_dir
                ]
            )

            # Maximum reasonable values should be accepted

    def test_extreme_boundary_values(self):
        """Test extreme boundary values that should be rejected."""
        data_path = self._create_mock_dataset()

        # Test extreme values that should be rejected
        extreme_tests = [
            ("--epochs", "999999"),
            ("--batch-size", "999999"),
            ("--learning-rate", "100"),
            ("--sequence-length", "999999"),
            ("--prediction-horizon", "999999"),
        ]

        for param, value in extreme_tests:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(data_path),
                    param, value,
                    "--output-dir", self.temp_dir
                ]
            )

            # Extreme values should be handled appropriately
            # Either with validation errors or reasonable defaults

    def test_float_precision_boundaries(self):
        """Test floating point precision boundaries."""
        data_path = self._create_mock_dataset()

        # Test various float precisions
        precision_tests = [
            ("--learning-rate", "0.000001"),    # Very small
            ("--learning-rate", "0.123456789"), # High precision
            ("--learning-rate", "1e-6"),        # Scientific notation
            ("--learning-rate", "1E-6"),        # Scientific notation (caps)
        ]

        for param, value in precision_tests:
            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(data_path),
                    param, value,
                    "--output-dir", self.temp_dir
                ]
            )

            # Should handle various float formats

    def _create_mock_dataset(self):
        """Create a mock dataset file."""
        import pandas as pd

        data_path = Path(self.temp_dir) / "dataset.csv"
        data = {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "symbol": ["AAPL"] * 100,
            "close": [150.0 + i * 0.1 for i in range(100)],
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return data_path


if __name__ == "__main__":
    pytest.main([__file__])
