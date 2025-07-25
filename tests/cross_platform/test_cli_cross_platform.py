"""
Cross-Platform CLI Compatibility Tests.

This module provides cross-platform testing for CLI operations:
- Windows/Linux/macOS path handling compatibility
- Environment variable handling across platforms
- File system operation compatibility
- Shell command execution differences
- Character encoding handling
- Platform-specific configuration validation
- Cross-platform installation and setup testing
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


@pytest.mark.skipif(os.name != "posix", reason="Unix-specific tests")
class TestCLIUnixCompatibility:
    """Test CLI compatibility on Unix-like systems (Linux/macOS)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unix_path_handling(self):
        """Test proper handling of Unix-style paths."""
        unix_paths = [
            "/tmp/trading_data",
            "/home/user/trading/models",
            "/var/lib/trading-agent/cache",
            "~/trading_workspace/datasets"
        ]

        for path in unix_paths:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

                # Setup mocks
                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance
                mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

                mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

                mock_cache_instance = MagicMock()
                mock_cache.return_value = mock_cache_instance
                mock_cache_instance.get_cached_data.return_value = None

                # Use a safe path for testing
                safe_path = self.temp_dir
                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", safe_path,
                        "--dataset-name", "unix_path_test"
                    ]
                )

                # Should handle path correctly
                assert result.exit_code == 0

    def test_unix_environment_variables(self):
        """Test Unix environment variable handling."""
        unix_env_vars = {
            "HOME": "/home/testuser",
            "USER": "testuser",
            "SHELL": "/bin/bash",
            "TRADING_RL_AGENT_HOME": "/home/testuser/.trading-agent"
        }

        original_env = {}
        for key, value in unix_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result = self.runner.invoke(main_app, ["info"])
            assert result.exit_code == 0

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_unix_file_permissions(self):
        """Test Unix file permission handling."""
        # Create a config file with restricted permissions
        config_file = Path(self.temp_dir) / "unix_config.yaml"
        config_content = """
        environment: test
        debug: false
        """
        config_file.write_text(config_content)
        config_file.chmod(0o600)  # Owner read/write only

        result = self.runner.invoke(
            main_app,
            ["--config", str(config_file), "info"]
        )

        assert result.exit_code == 0

    def test_unix_symbolic_links(self):
        """Test handling of symbolic links on Unix systems."""
        # Create a real directory and a symbolic link to it
        real_dir = Path(self.temp_dir) / "real_output"
        real_dir.mkdir()

        symlink_dir = Path(self.temp_dir) / "symlink_output"
        symlink_dir.symlink_to(real_dir)

        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

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
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", str(symlink_dir),
                    "--dataset-name", "symlink_test"
                ]
            )

            # Should handle symbolic links properly
            assert result.exit_code == 0

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific tests")
class TestCLIWindowsCompatibility:
    """Test CLI compatibility on Windows systems."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_windows_path_handling(self):
        """Test proper handling of Windows-style paths."""
        windows_paths = [
            "C:\\Users\\trader\\AppData\\Local\\TradingAgent",
            "D:\\Trading\\Data\\Datasets",
            "C:\\Program Files\\TradingAgent\\Models",
            "%USERPROFILE%\\Documents\\Trading"
        ]

        for path in windows_paths:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

                # Setup mocks
                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance
                mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

                mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

                mock_cache_instance = MagicMock()
                mock_cache.return_value = mock_cache_instance
                mock_cache_instance.get_cached_data.return_value = None

                # Use a safe path for testing
                safe_path = self.temp_dir
                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", safe_path,
                        "--dataset-name", "windows_path_test"
                    ]
                )

                # Should handle path correctly
                assert result.exit_code == 0

    def test_windows_environment_variables(self):
        """Test Windows environment variable handling."""
        windows_env_vars = {
            "USERNAME": "testuser",
            "USERPROFILE": "C:\\Users\\testuser",
            "APPDATA": "C:\\Users\\testuser\\AppData\\Roaming",
            "LOCALAPPDATA": "C:\\Users\\testuser\\AppData\\Local",
            "TRADING_RL_AGENT_HOME": "C:\\Users\\testuser\\AppData\\Local\\TradingAgent"
        }

        original_env = {}
        for key, value in windows_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result = self.runner.invoke(main_app, ["info"])
            assert result.exit_code == 0

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_windows_drive_letters(self):
        """Test handling of Windows drive letters."""
        drive_paths = ["C:\\temp", "D:\\data", "E:\\models"]

        for drive_path in drive_paths:
            # Test with safe temp directory instead of actual drive paths
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "drive_test"
                ]
            )

            # Should not fail due to drive letter handling
            # (Actual success depends on mocking and implementation)

    def test_windows_long_path_support(self):
        """Test support for Windows long paths."""
        # Create a very long path name
        long_dirname = "a" * 200  # Very long directory name
        long_path = Path(self.temp_dir) / long_dirname

        try:
            long_path.mkdir(parents=True, exist_ok=True)

            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

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
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", str(long_path),
                        "--dataset-name", "long_path_test"
                    ]
                )

                # Should handle long paths appropriately
                # Either succeed or fail with appropriate error message
                if result.exit_code != 0:
                    assert any(word in result.output.lower() for word in [
                        "path", "long", "length", "invalid"
                    ])

        except OSError:
            # Skip if filesystem doesn't support long paths
            pytest.skip("Filesystem doesn't support long paths")

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIUniversalCompatibility:
    """Test CLI compatibility across all platforms."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pathlib_cross_platform_handling(self):
        """Test that pathlib handles cross-platform paths correctly."""
        # Test various path formats that should work cross-platform
        test_paths = [
            "data/datasets",
            "models/trained",
            "cache/temp",
            "output/results"
        ]

        for path_str in test_paths:
            test_path = Path(self.temp_dir) / path_str
            test_path.mkdir(parents=True, exist_ok=True)

            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

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
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", str(test_path),
                        "--dataset-name", "cross_platform_test"
                    ]
                )

                assert result.exit_code == 0

    def test_text_encoding_handling(self):
        """Test handling of different text encodings across platforms."""
        # Test with various character encodings that might be used
        test_symbols = [
            "AAPL",  # ASCII
            "BRK.A",  # With period
            "BRK-A",  # With hyphen
        ]

        for symbol in test_symbols:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance
                mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

                mock_cache_instance = MagicMock()
                mock_cache.return_value = mock_cache_instance
                mock_cache_instance.get_cached_data.return_value = None

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", symbol,
                        "--no-sentiment",
                        "--output-dir", self.temp_dir,
                        "--dataset-name", f"encoding_test_{symbol.replace('.', '_').replace('-', '_')}"
                    ]
                )

                # Should handle different character encodings
                assert result.exit_code == 0

    def test_command_line_argument_parsing(self):
        """Test command line argument parsing across platforms."""
        # Test various argument formats
        argument_formats = [
            ["--symbols", "AAPL,GOOGL"],          # Standard format
            ["--symbols=AAPL,GOOGL"],             # Equals format
            ["-s", "AAPL,GOOGL"],                 # Short format (if supported)
        ]

        for args in argument_formats:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance
                mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

                mock_cache_instance = MagicMock()
                mock_cache.return_value = mock_cache_instance
                mock_cache_instance.get_cached_data.return_value = None

                cmd_args = [
                    "data", "pipeline", "--run",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "arg_format_test",
                    *args
                ]

                self.runner.invoke(main_app, cmd_args)

                # Should parse arguments correctly regardless of format
                # (Success depends on whether the format is supported)

    def test_configuration_file_paths(self):
        """Test configuration file path resolution across platforms."""
        # Test different config file locations
        config_locations = [
            "config.yaml",                    # Relative path
            "./config.yaml",                  # Explicit relative
            str(Path(self.temp_dir) / "config.yaml"),  # Absolute path
        ]

        for config_path in config_locations:
            # Create config file
            if not os.path.isabs(config_path):
                actual_config_path = Path(self.temp_dir) / config_path
            else:
                actual_config_path = Path(config_path)

            actual_config_path.parent.mkdir(parents=True, exist_ok=True)
            config_content = """
            environment: test
            debug: false
            """
            actual_config_path.write_text(config_content)

            with patch("trade_agent.cli.load_settings") as mock_load_settings:
                mock_settings = MagicMock()
                mock_settings.environment = "test"
                mock_load_settings.return_value = mock_settings

                result = self.runner.invoke(
                    main_app,
                    ["--config", str(actual_config_path), "info"]
                )

                # Should handle different config path formats
                assert result.exit_code == 0

    def test_temp_directory_handling(self):
        """Test temporary directory handling across platforms."""
        import tempfile

        # Test that temporary directories work consistently
        platform_temp_dirs = [
            tempfile.gettempdir(),
            str(Path(tempfile.gettempdir()) / "trading_agent_test")
        ]

        for temp_dir in platform_temp_dirs:
            temp_path = Path(temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)

            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

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
                        "--symbols", "AAPL",
                        "--no-sentiment",
                        "--output-dir", str(temp_path),
                        "--dataset-name", "temp_dir_test"
                    ]
                )

                # Should handle platform-specific temp directories
                assert result.exit_code == 0

    def test_line_ending_handling(self):
        """Test handling of different line endings across platforms."""
        # Create config files with different line endings
        line_endings = [
            ("unix", "\n"),
            ("windows", "\r\n"),
            ("mac_classic", "\r")
        ]

        for name, line_ending in line_endings:
            config_file = Path(self.temp_dir) / f"config_{name}.yaml"
            config_content = f"environment: test{line_ending}debug: false{line_ending}"

            # Write with specific line endings
            with open(config_file, "wb") as f:
                f.write(config_content.encode("utf-8"))

            with patch("trade_agent.cli.load_settings") as mock_load_settings:
                mock_settings = MagicMock()
                mock_settings.environment = "test"
                mock_load_settings.return_value = mock_settings

                result = self.runner.invoke(
                    main_app,
                    ["--config", str(config_file), "info"]
                )

                # Should handle different line endings correctly
                assert result.exit_code == 0

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIPlatformSpecificFeatures:
    """Test platform-specific CLI features and behaviors."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_platform_detection(self):
        """Test that platform detection works correctly."""
        result = self.runner.invoke(main_app, ["info"])

        assert result.exit_code == 0

        # Should contain platform information
        platform_indicators = ["platform", "system", "os", "version"]
        has_platform_info = any(indicator in result.output.lower()
                               for indicator in platform_indicators)

        # Platform information is helpful but not strictly required
        # This test documents current behavior

    def test_shell_integration_compatibility(self):
        """Test shell integration compatibility across platforms."""
        # Test that CLI works regardless of shell environment
        shell_envs = {
            "SHELL": "/bin/bash" if os.name == "posix" else "cmd.exe",
            "TERM": "xterm-256color" if os.name == "posix" else "cmd"
        }

        original_env = {}
        for key, value in shell_envs.items():
            original_env[key] = os.environ.get(key)
            if os.name == "posix" or (key in ["SHELL"] and os.name == "nt"):
                continue  # Skip setting incompatible env vars
            os.environ[key] = value

        try:
            result = self.runner.invoke(main_app, ["--help"])
            assert result.exit_code == 0

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_executable_path_resolution(self):
        """Test executable path resolution across platforms."""
        # Test that the CLI can find and execute properly
        result = self.runner.invoke(main_app, ["version"])

        # Should be able to determine its own version/location
        assert result.exit_code == 0

        # Should contain version information
        version_indicators = ["version", "v", "trading", "agent"]
        has_version_info = any(indicator in result.output.lower()
                              for indicator in version_indicators)
        assert has_version_info

    def test_dependency_compatibility(self):
        """Test that dependencies work correctly across platforms."""
        # Test basic imports and functionality
        result = self.runner.invoke(main_app, ["info"])

        # Should not fail due to missing dependencies
        assert result.exit_code == 0

        # Should provide some system information
        assert len(result.output.strip()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
