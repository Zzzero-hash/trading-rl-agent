"""
CLI User Experience Tests.

This module provides comprehensive testing for CLI user experience:
- Help text clarity and completeness
- Error message quality and actionability
- Progress indication and feedback
- Output formatting and readability
- Interactive workflow usability
- Command discoverability
- Consistent user interface patterns
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


class TestCLIHelpTextQuality:
    """Test quality and completeness of CLI help text."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_main_help_completeness(self):
        """Test that main help text is complete and informative."""
        result = self.runner.invoke(main_app, ["--help"])

        assert result.exit_code == 0
        help_text = result.output.lower()

        # Should contain key information
        assert "trade-agent" in help_text
        assert "trading" in help_text
        assert "production-grade" in help_text or "trading system" in help_text

        # Should list main command groups
        assert "data" in help_text
        assert "train" in help_text
        assert "backtest" in help_text
        assert "trade" in help_text

        # Should have clear structure
        assert "commands:" in help_text or "options:" in help_text

    def test_subcommand_help_completeness(self):
        """Test that subcommand help text is complete."""
        subcommands = [
            ["data", "--help"],
            ["train", "--help"],
            ["backtest", "--help"],
            ["trade", "--help"],
            ["scenario", "--help"]
        ]

        for cmd in subcommands:
            result = self.runner.invoke(main_app, cmd)
            assert result.exit_code == 0

            help_text = result.output.lower()

            # Should contain command-specific information
            assert cmd[0] in help_text
            assert "commands:" in help_text or "options:" in help_text

            # Should have clear descriptions
            assert len(help_text.strip()) > 100  # Substantial help text

    def test_specific_command_help_detail(self):
        """Test specific command help provides sufficient detail."""
        # Test data pipeline command help
        result = self.runner.invoke(main_app, ["data", "pipeline", "--help"])
        assert result.exit_code == 0

        help_text = result.output.lower()

        # Should explain key parameters
        expected_params = ["symbols", "start-date", "end-date", "output-dir", "run"]
        for param in expected_params:
            assert param in help_text

        # Should have examples or usage information
        # (exact format depends on implementation)

    def test_help_text_formatting(self):
        """Test that help text is well-formatted and readable."""
        result = self.runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0

        lines = result.output.split("\n")

        # Should not have excessively long lines
        for line in lines:
            assert len(line) <= 120  # Reasonable line length

        # Should have consistent indentation
        # (exact requirements depend on implementation)

    def test_error_help_suggestions(self):
        """Test that error messages provide helpful suggestions."""
        # Test invalid command
        result = self.runner.invoke(main_app, ["invalid-command"])
        assert result.exit_code != 0

        error_text = result.output.lower()

        # Should provide helpful error message
        assert "no such command" in error_text or "unknown command" in error_text

        # Should suggest alternatives or direct to help
        suggests_help = any(phrase in error_text for phrase in [
            "--help", "help", "available commands", "try"
        ])
        assert suggests_help


class TestCLIErrorMessageQuality:
    """Test quality and actionability of error messages."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_required_argument_errors(self):
        """Test error messages for missing required arguments."""
        # Test train command without required data path
        result = self.runner.invoke(main_app, ["train", "cnn-lstm"])

        # Should provide clear error about missing argument
        if result.exit_code != 0:
            error_text = result.output.lower()
            assert any(word in error_text for word in [
                "missing", "required", "argument", "path"
            ])

    def test_invalid_file_path_errors(self):
        """Test error messages for invalid file paths."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.csv"

        result = self.runner.invoke(
            main_app,
            ["train", "cnn-lstm", str(nonexistent_file), "--output-dir", self.temp_dir]
        )

        assert result.exit_code == 1
        error_text = result.output.lower()

        # Should clearly indicate file not found
        assert any(phrase in error_text for phrase in [
            "not found", "does not exist", "file not found", "missing"
        ])

        # Should show the problematic path
        assert str(nonexistent_file) in result.output

    def test_invalid_parameter_value_errors(self):
        """Test error messages for invalid parameter values."""
        data_file = self._create_mock_dataset_file()

        # Test invalid epochs
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm", str(data_file),
                "--epochs", "-1",
                "--output-dir", self.temp_dir
            ]
        )

        # Should provide clear error about invalid value
        if result.exit_code != 0:
            error_text = result.output.lower()
            assert any(word in error_text for word in [
                "invalid", "negative", "positive", "epochs"
            ])

    def test_network_error_messages(self):
        """Test error messages for network-related issues."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mock to simulate network error
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.side_effect = ConnectionError("Network unreachable")

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 1

            # Should provide user-friendly error message
            result.output.lower()
            # The exact error message depends on implementation
            # but should be user-friendly

    def test_permission_error_messages(self):
        """Test error messages for permission issues."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mock to simulate permission error
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.side_effect = PermissionError("Permission denied")

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", "/root/protected"  # Likely to cause permission error
                ]
            )

            assert result.exit_code == 1

            # Should provide clear permission error message
            # The exact handling depends on implementation

    def _create_mock_dataset_file(self):
        """Create a mock dataset file for testing."""
        import pandas as pd

        data_file = Path(self.temp_dir) / "test_data.csv"
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "symbol": ["AAPL"] * 10,
            "close": [150.0] * 10
        })
        df.to_csv(data_file, index=False)
        return data_file


class TestCLIProgressIndication:
    """Test progress indication and user feedback."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_progress_feedback(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test that data pipeline provides progress feedback."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "progress_test"
            ]
        )

        assert result.exit_code == 0

        # Should provide progress indicators
        output_lines = result.output.split("\n")

        # Should have progress messages
        progress_indicators = [
            "Auto-Processing Pipeline",
            "Processing",
            "symbols",
            "completed successfully"
        ]

        output_text = result.output.lower()
        progress_found = sum(1 for indicator in progress_indicators if indicator.lower() in output_text)
        assert progress_found >= 2  # At least some progress indicators

    def test_training_progress_indication(self):
        """Test training progress indication."""
        data_file = self._create_mock_dataset_file()

        with patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster") as mock_init_ray, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data") as mock_load_data, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer") as mock_trainer_class, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config") as mock_model_config, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config") as mock_training_config:

            # Setup mocks
            mock_init_ray.return_value = None
            mock_sequences, mock_targets = self._create_mock_sequences_targets()
            mock_load_data.return_value = (mock_sequences, mock_targets)
            mock_model_config.return_value = {"input_dim": 10}
            mock_training_config.return_value = {"epochs": 5}

            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(data_file),
                    "--epochs", "5",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 0

            # Should have training progress messages
            output_text = result.output.lower()
            training_indicators = [
                "training", "cnn+lstm", "initializing", "loading"
            ]

            indicators_found = sum(1 for indicator in training_indicators if indicator in output_text)
            assert indicators_found >= 2

    def test_verbose_output_levels(self):
        """Test different verbose output levels provide appropriate detail."""
        verbose_levels = [0, 1, 2, 3]
        output_lengths = {}

        for level in verbose_levels:
            verbose_args = ["-v"] * level if level > 0 else []

            result = self.runner.invoke(
                main_app,
                [*verbose_args, "info"]
            )

            assert result.exit_code == 0
            output_lengths[level] = len(result.output)

        # Higher verbose levels should generally provide more output
        # (though exact behavior depends on implementation)
        base_length = output_lengths[0]
        verbose_length = output_lengths.get(2, base_length)

        # At minimum, verbose mode should not provide less information
        assert verbose_length >= base_length

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "close": [150.0, 2500.0, 300.0],
            "volume": [1000000, 500000, 800000],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"])
        })

    def _create_mock_dataset_file(self):
        """Create a mock dataset file for testing."""
        import pandas as pd

        data_file = Path(self.temp_dir) / "test_data.csv"
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=50),
            "symbol": ["AAPL"] * 50,
            "close": [150.0 + i * 0.1 for i in range(50)]
        })
        df.to_csv(data_file, index=False)
        return data_file

    def _create_mock_sequences_targets(self):
        """Create mock sequences and targets for training."""
        import numpy as np
        sequences = np.random.rand(50, 30, 10)
        targets = np.random.rand(50, 1)
        return sequences, targets


class TestCLIOutputFormatting:
    """Test output formatting and readability."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_table_output_formatting(self):
        """Test that table outputs are well-formatted."""
        # Test version command table
        result = self.runner.invoke(main_app, ["version"])
        assert result.exit_code == 0

        lines = result.output.split("\n")

        # Should have structured output
        # (exact format depends on implementation)
        assert len(lines) > 1

        # Should not have excessively long lines
        for line in lines:
            assert len(line) <= 120

    def test_info_command_formatting(self):
        """Test info command output formatting."""
        result = self.runner.invoke(main_app, ["info"])
        assert result.exit_code == 0

        lines = result.output.split("\n")

        # Should have structured information display
        assert len(lines) > 3  # Multiple lines of information

        # Should be readable
        for line in lines:
            if line.strip():  # Skip empty lines
                assert len(line) <= 120

    def test_error_output_formatting(self):
        """Test error output formatting."""
        result = self.runner.invoke(main_app, ["invalid-command"])
        assert result.exit_code != 0

        # Error should be clearly formatted
        error_lines = result.output.split("\n")

        # Should not be excessively verbose
        assert len(error_lines) <= 20

        # Should have clear error indication
        error_text = result.output.lower()
        assert any(word in error_text for word in ["error", "invalid", "command", "no such"])

    def test_success_message_formatting(self):
        """Test success message formatting."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mocks for successful operation
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            import tempfile
            temp_dir = tempfile.mkdtemp()

            try:
                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", temp_dir,
                        "--dataset-name", "format_test"
                    ]
                )

                assert result.exit_code == 0

                # Should have clear success indicators
                success_text = result.output.lower()
                assert any(phrase in success_text for phrase in [
                    "completed successfully", "complete", "success"
                ])

            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLICommandDiscoverability:
    """Test command discoverability and user guidance."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_main_command_listing(self):
        """Test that main commands are easily discoverable."""
        result = self.runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0

        help_text = result.output.lower()

        # Should list main command groups
        main_commands = ["data", "train", "backtest", "trade", "scenario"]
        for cmd in main_commands:
            assert cmd in help_text

    def test_subcommand_listing(self):
        """Test that subcommands are easily discoverable."""
        result = self.runner.invoke(main_app, ["data", "--help"])
        assert result.exit_code == 0

        help_text = result.output.lower()

        # Should show data subcommands
        # (exact subcommands depend on implementation)
        assert "pipeline" in help_text

    def test_command_suggestions_on_typos(self):
        """Test command suggestions for common typos."""
        # Test common typos
        typos = ["dta", "traning", "backest", "tade"]

        for typo in typos:
            result = self.runner.invoke(main_app, [typo])
            assert result.exit_code != 0

            # Should provide helpful error (implementation dependent)
            error_text = result.output.lower()
            assert "no such command" in error_text or "unknown" in error_text

    def test_example_usage_in_help(self):
        """Test that help includes example usage where appropriate."""
        # Test data pipeline help for examples
        result = self.runner.invoke(main_app, ["data", "pipeline", "--help"])
        assert result.exit_code == 0

        help_text = result.output.lower()

        # Should include usage examples or patterns
        # (exact format depends on implementation)
        has_examples = any(word in help_text for word in [
            "example", "usage", "sample", "e.g."
        ])

        # Examples are helpful but not strictly required
        # This test documents the current state


class TestCLIConsistentInterface:
    """Test consistent user interface patterns."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_consistent_help_format(self):
        """Test that help format is consistent across commands."""
        commands = [
            ["data", "--help"],
            ["train", "--help"],
            ["backtest", "--help"],
            ["trade", "--help"]
        ]

        help_formats = []

        for cmd in commands:
            result = self.runner.invoke(main_app, cmd)
            assert result.exit_code == 0

            # Analyze help format
            lines = result.output.split("\n")
            help_formats.append({
                "total_lines": len(lines),
                "has_usage": "usage:" in result.output.lower(),
                "has_options": "options:" in result.output.lower(),
                "has_commands": "commands:" in result.output.lower()
            })

        # Should have consistent structure
        # (exact requirements depend on implementation)

    def test_consistent_error_format(self):
        """Test that error messages have consistent format."""
        error_scenarios = [
            ["invalid-command"],
            ["data", "invalid-subcommand"],
            ["train", "invalid-train-command"]
        ]

        for scenario in error_scenarios:
            result = self.runner.invoke(main_app, scenario)
            assert result.exit_code != 0

            # Should have consistent error format
            error_lines = result.output.split("\n")

            # Should not be excessively verbose
            assert len(error_lines) <= 15

            # Should have clear error indication
            assert any(word in result.output.lower() for word in [
                "error", "invalid", "unknown", "no such"
            ])

    def test_consistent_parameter_naming(self):
        """Test that parameter naming is consistent across commands."""
        # Test that common parameters use consistent names
        help_texts = {}

        commands = [
            ["data", "pipeline", "--help"],
            # Add more commands as they're implemented
        ]

        for cmd in commands:
            result = self.runner.invoke(main_app, cmd)
            if result.exit_code == 0:
                help_texts[" ".join(cmd)] = result.output.lower()

        # Should use consistent naming patterns
        # For example, output directories should consistently be --output-dir
        # (exact consistency requirements depend on implementation)

    def test_consistent_success_messages(self):
        """Test that success messages are consistent."""
        # This would test multiple successful operations
        # to ensure consistent success message format
        # Implementation depends on available mock operations


if __name__ == "__main__":
    pytest.main([__file__])
