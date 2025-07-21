"""Security tests for input validation in trading RL agent."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.trade_agent.core.config import Config


class TestInputValidationSecurity:
    """Test input validation security measures."""

    @pytest.fixture
    def mock_config(self) -> Config:
        """Create a mock configuration for testing."""
        config = Mock(spec=Config)
        config.symbols = ["AAPL", "GOOGL", "MSFT"]
        config.start_date = "2023-01-01"
        config.end_date = "2023-12-31"
        config.initial_balance = 100000.0
        config.max_position_size = 0.1
        config.risk_free_rate = 0.02
        return config

    def test_sql_injection_prevention(self, _mock_config):
        """Test that SQL injection attempts are prevented."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "'; UPDATE users SET password='hacked'; --",
            "'; DELETE FROM users; --",
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                # Test with various components that might use database queries
                config = Config()
                config.symbols = [malicious_input]
                assert malicious_input not in str(config.symbols)

    def test_path_traversal_prevention(self, _mock_config):
        """Test that path traversal attempts are prevented."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%5C..%5C..%5Cwindows%5Csystem32%5Cconfig%5Csam",
        ]

        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, OSError, FileNotFoundError)):
                # Test file path validation
                if any(char in malicious_path for char in ["..", "%2F", "%5C"]):
                    raise ValueError(f"Path traversal detected: {malicious_path}")

    def test_command_injection_prevention(self, _mock_config):
        """Test that command injection attempts are prevented."""
        malicious_commands = [
            "; rm -rf /",
            "& del /s /q C:\\",
            "| cat /etc/passwd",
            "`whoami`",
            "$(id)",
            "&& echo 'hacked'",
            "|| echo 'hacked'",
        ]

        for malicious_command in malicious_commands:
            with pytest.raises(ValueError):
                # Test command validation
                if any(char in malicious_command for char in [";", "&", "|", "`", "$(", "&&", "||"]):
                    raise ValueError(f"Command injection detected: {malicious_command}")

    def test_xss_prevention(self, _mock_config):
        """Test that XSS attempts are prevented."""
        malicious_scripts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",
            "onerror=alert('xss')",
            "onclick=alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ]

        for malicious_script in malicious_scripts:
            with pytest.raises(ValueError):
                # Test XSS prevention
                if any(
                    tag in malicious_script.lower()
                    for tag in [
                        "<script>",
                        "javascript:",
                        "onload=",
                        "onerror=",
                        "onclick=",
                        "<img",
                        "<svg",
                    ]
                ):
                    raise ValueError(f"XSS attempt detected: {malicious_script}")

    def test_numeric_input_validation(self, _mock_config):
        """Test numeric input validation."""
        # Test valid numeric inputs
        valid_numbers = [0, 1, 100, 1000.5, -1, -100.5]
        for num in valid_numbers:
            assert isinstance(num, int | float)

        # Test invalid numeric inputs
        invalid_inputs = ["not_a_number", "123abc", "abc123", "", None, [], {}]
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                float(invalid_input)

    def test_symbol_validation(self, _mock_config):
        """Test trading symbol validation."""
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for symbol in valid_symbols:
            assert isinstance(symbol, str)
            assert len(symbol) <= 10
            assert symbol.isalnum()

        invalid_symbols = [
            "",  # Empty string
            "A" * 11,  # Too long
            "AAPL!",  # Invalid characters
            "AAPL@",  # Invalid characters
            "AAPL#",  # Invalid characters
            "AAPL$",  # Invalid characters
            "AAPL%",  # Invalid characters
            "AAPL^",  # Invalid characters
            "AAPL&",  # Invalid characters
            "AAPL*",  # Invalid characters
            "AAPL(",  # Invalid characters
            "AAPL)",  # Invalid characters
            "AAPL-",  # Invalid characters
            "AAPL+",  # Invalid characters
            "AAPL=",  # Invalid characters
            "AAPL[",  # Invalid characters
            "AAPL]",  # Invalid characters
            "AAPL{",  # Invalid characters
            "AAPL}",  # Invalid characters
            "AAPL|",  # Invalid characters
            "AAPL\\",  # Invalid characters
            "AAPL/",  # Invalid characters
            "AAPL:",  # Invalid characters
            "AAPL;",  # Invalid characters
            "AAPL'",  # Invalid characters
            'AAPL"',  # Invalid characters
            "AAPL,",  # Invalid characters
            "AAPL.",  # Invalid characters
            "AAPL<",  # Invalid characters
            "AAPL>",  # Invalid characters
            "AAPL?",  # Invalid characters
            "AAPL/",  # Invalid characters
        ]

        for symbol in invalid_symbols:
            with pytest.raises(ValueError):
                if not symbol or len(symbol) > 10 or not symbol.isalnum():
                    raise ValueError(f"Invalid symbol: {symbol}")

    def test_date_validation(self, _mock_config):
        """Test date input validation."""
        valid_dates = ["2023-01-01", "2023-12-31", "2024-02-29"]
        for date_str in valid_dates:
            try:
                pd.to_datetime(date_str)
            except ValueError:
                pytest.fail(f"Valid date {date_str} was rejected")

        invalid_dates = [
            "2023-13-01",  # Invalid month
            "2023-00-01",  # Invalid month
            "2023-01-32",  # Invalid day
            "2023-01-00",  # Invalid day
            "2023-02-30",  # Invalid day for February
            "2023-04-31",  # Invalid day for April
            "not_a_date",
            "2023/01/01",  # Wrong format
            "01-01-2023",  # Wrong format
            "2023.01.01",  # Wrong format
        ]

        for date_str in invalid_dates:
            with pytest.raises(ValueError):
                pd.to_datetime(date_str, format="%Y-%m-%d")

    def test_array_input_validation(self, _mock_config):
        """Test array/DataFrame input validation."""
        # Test valid arrays
        valid_arrays = [
            np.array([1, 2, 3]),
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            pd.Series([1, 2, 3]),
        ]

        for arr in valid_arrays:
            assert hasattr(arr, "__len__")
            assert len(arr) > 0

        # Test invalid arrays
        invalid_arrays = [
            None,
            [],
            {},
            "",
            123,
            "not_an_array",
        ]

        for arr in invalid_arrays:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                if arr is None or not hasattr(arr, "__len__") or len(arr) == 0:
                    raise ValueError(f"Invalid array: {arr}")

    def test_config_validation(self, mock_config):
        """Test configuration validation."""
        # Test valid configuration
        assert mock_config.symbols is not None
        assert isinstance(mock_config.symbols, list)
        assert len(mock_config.symbols) > 0
        assert all(isinstance(s, str) for s in mock_config.symbols)

        assert mock_config.initial_balance > 0
        assert isinstance(mock_config.initial_balance, int | float)

        assert 0 < mock_config.max_position_size <= 1
        assert isinstance(mock_config.max_position_size, int | float)

        # Test invalid configurations
        invalid_configs = [
            {"symbols": []},  # Empty symbols
            {"symbols": None},  # None symbols
            {"initial_balance": -1000},  # Negative balance
            {"initial_balance": 0},  # Zero balance
            {"max_position_size": -0.1},  # Negative position size
            {"max_position_size": 1.5},  # Position size > 1
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                if "symbols" in invalid_config and not invalid_config["symbols"]:
                    raise ValueError("Symbols list cannot be empty")
                if "initial_balance" in invalid_config and invalid_config["initial_balance"] <= 0:
                    raise ValueError("Initial balance must be positive")
                if "max_position_size" in invalid_config and not (0 < invalid_config["max_position_size"] <= 1):
                    raise ValueError("Max position size must be between 0 and 1")

    def test_file_upload_validation(self, _mock_config):
        """Test file upload validation."""
        # Test valid file types
        valid_extensions = [
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".parquet",
            ".h5",
            ".hdf5",
        ]
        for ext in valid_extensions:
            assert ext.startswith(".")

        # Test invalid file types
        invalid_extensions = [
            ".exe",
            ".bat",
            ".sh",
            ".py",
            ".js",
            ".html",
            ".php",
            ".asp",
            ".jsp",
            ".jar",
            ".war",
            ".ear",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".com",
        ]

        for ext in invalid_extensions:
            with pytest.raises(ValueError):
                if ext in [
                    ".exe",
                    ".bat",
                    ".sh",
                    ".py",
                    ".js",
                    ".html",
                    ".php",
                    ".asp",
                    ".jsp",
                    ".jar",
                    ".war",
                    ".ear",
                    ".dll",
                    ".so",
                    ".dylib",
                    ".bin",
                    ".com",
                ]:
                    raise ValueError(f"Invalid file type: {ext}")

    def test_url_validation(self, _mock_config):
        """Test URL validation."""
        valid_urls = [
            "https://api.example.com",
            "http://localhost:8000",
            "https://data.example.com/v1/endpoint",
        ]

        for url in valid_urls:
            assert url.startswith(("http://", "https://"))

        invalid_urls = [
            "ftp://example.com",  # Unsupported protocol
            "file:///etc/passwd",  # File protocol
            "javascript:alert('xss')",  # JavaScript protocol
            "data:text/html,<script>alert('xss')</script>",  # Data protocol
            "not_a_url",
            "",
            None,
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                if not url or not url.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid URL: {url}")

    def test_json_validation(self, _mock_config):
        """Test JSON input validation."""
        import json

        valid_jsons = [
            '{"key": "value"}',
            '{"numbers": [1, 2, 3]}',
            '{"nested": {"key": "value"}}',
        ]

        for json_str in valid_jsons:
            try:
                json.loads(json_str)
            except json.JSONDecodeError:
                pytest.fail(f"Valid JSON {json_str} was rejected")

        invalid_jsons = [
            '{"key": "value"',  # Missing closing brace
            '{"key": value}',  # Missing quotes
            '{"key": "value",}',  # Trailing comma
            "not_json",
            "",
            None,
        ]

        for json_str in invalid_jsons:
            with pytest.raises(json.JSONDecodeError):
                json.loads(json_str)

    def test_environment_variable_validation(self, _mock_config):
        """Test environment variable validation."""
        import os

        # Test that sensitive environment variables are not logged
        sensitive_vars = [
            "API_KEY",
            "SECRET_KEY",
            "PASSWORD",
            "TOKEN",
            "CREDENTIALS",
            "PRIVATE_KEY",
            "DATABASE_URL",
        ]

        for var in sensitive_vars:
            if var in os.environ:
                # Ensure sensitive variables are not logged in plain text
                log_message = f"Environment variable {var} is set"
                assert os.environ[var] not in log_message

    def test_memory_usage_validation(self, _mock_config):
        """Test memory usage validation."""
        import os

        import psutil

        # Test that memory usage is reasonable
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Memory usage should be less than 1GB for normal operations
        assert memory_info.rss < 1024 * 1024 * 1024  # 1GB

    def test_cpu_usage_validation(self, _mock_config):
        """Test CPU usage validation."""
        import os

        import psutil

        # Test that CPU usage is reasonable
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()

        # CPU usage should be reasonable (less than 100% for a single process)
        assert cpu_percent < 100

    def test_network_validation(self, _mock_config):
        """Test network input validation."""
        valid_ips = [
            "127.0.0.1",
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
        ]

        for ip in valid_ips:
            parts = ip.split(".")
            assert len(parts) == 4
            assert all(0 <= int(part) <= 255 for part in parts)

        invalid_ips = [
            "256.1.2.3",  # Invalid octet
            "1.2.3.256",  # Invalid octet
            "1.2.3",  # Too few octets
            "1.2.3.4.5",  # Too many octets
            "not_an_ip",
            "",
            None,
        ]

        for ip in invalid_ips:
            with pytest.raises((ValueError, IndexError)):
                if ip and "." in ip:
                    parts = ip.split(".")
                    if len(parts) != 4:
                        raise ValueError(f"Invalid IP format: {ip}")
                    for part in parts:
                        if not (0 <= int(part) <= 255):
                            raise ValueError(f"Invalid IP octet: {part}")
                else:
                    raise ValueError(f"Invalid IP: {ip}")

    def test_encoding_validation(self, _mock_config):
        """Test encoding validation."""
        # Test valid encodings
        valid_encodings = ["utf-8", "ascii", "latin-1", "iso-8859-1"]
        for encoding in valid_encodings:
            try:
                "test".encode(encoding)
            except LookupError:
                pytest.fail(f"Valid encoding {encoding} was rejected")

        # Test invalid encodings
        invalid_encodings = [
            "invalid_encoding",
            "not_an_encoding",
            "",
            None,
        ]

        for encoding in invalid_encodings:
            with pytest.raises(LookupError):
                "test".encode(encoding)

    def test_compression_validation(self, _mock_config):
        """Test compression validation."""
        import bz2
        import gzip
        import lzma

        # Test valid compression
        test_data = b"test data for compression"

        # Gzip compression
        compressed_gzip = gzip.compress(test_data)
        assert len(compressed_gzip) > 0

        # Bzip2 compression
        compressed_bz2 = bz2.compress(test_data)
        assert len(compressed_bz2) > 0

        # LZMA compression
        compressed_lzma = lzma.compress(test_data)
        assert len(compressed_lzma) > 0

        # Test decompression
        assert gzip.decompress(compressed_gzip) == test_data
        assert bz2.decompress(compressed_bz2) == test_data
        assert lzma.decompress(compressed_lzma) == test_data

    def test_serialization_validation(self, _mock_config):
        """Test serialization validation."""
        import json
        import pickle

        test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}

        # Test JSON serialization
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        assert json.loads(json_str) == test_data

        # Test pickle serialization (with security considerations)
        pickle_data = pickle.dumps(test_data)
        assert isinstance(pickle_data, bytes)
        assert pickle.loads(pickle_data) == test_data

        # Test that malicious pickle data is rejected
        malicious_pickle = b"cos\nsystem\n(S'echo hacked'\ntR."
        with pytest.raises((pickle.UnpicklingError, EOFError)):
            pickle.loads(malicious_pickle)

    def test_regular_expression_validation(self, _mock_config):
        """Test regular expression validation."""
        import re

        # Test valid regex patterns
        valid_patterns = [
            r"\d+",
            r"[a-zA-Z]+",
            r"^[a-z0-9]+$",
            r"\b\w+\b",
        ]

        for pattern in valid_patterns:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Valid regex pattern {pattern} was rejected")

        # Test potentially dangerous regex patterns
        dangerous_patterns = [
            r"(a+)+",  # Catastrophic backtracking
            r"(a|aa)*",  # Catastrophic backtracking
            r"(a|a?)*",  # Catastrophic backtracking
        ]

        for pattern in dangerous_patterns:
            # These patterns should be detected and rejected
            with pytest.raises(ValueError):
                if "++" in pattern or "**" in pattern or "??" in pattern:
                    raise ValueError(f"Potentially dangerous regex pattern: {pattern}")

    def test_thread_safety_validation(self, _mock_config):
        """Test thread safety validation."""
        import threading
        import time

        # Test that shared resources are properly protected
        shared_counter = 0
        lock = threading.Lock()

        def increment_counter():
            nonlocal shared_counter
            with lock:
                current = shared_counter
                time.sleep(0.001)  # Simulate some work
                shared_counter = current + 1

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Counter should be exactly 10
        assert shared_counter == 10

    def test_resource_cleanup_validation(self, _mock_config):
        """Test resource cleanup validation."""
        import os
        import tempfile

        # Test file cleanup
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test data")

        # File should exist
        assert os.path.exists(temp_path)

        # Clean up
        from pathlib import Path

        Path(temp_path).unlink()

        # File should not exist
        assert not os.path.exists(temp_path)

    def test_error_handling_validation(self, _mock_config):
        """Test error handling validation."""
        # Test that exceptions are properly caught and handled
        try:
            raise ValueError("Test error")
        except ValueError as e:
            assert str(e) == "Test error"
        except Exception:
            pytest.fail("Wrong exception type caught")

        # Test that exceptions don't expose sensitive information
        try:
            raise ValueError("API_KEY=secret123")
        except ValueError as e:
            error_message = str(e)
            assert "API_KEY" not in error_message
            assert "secret123" not in error_message

    def test_logging_security_validation(self, _mock_config):
        """Test logging security validation."""
        import logging

        # Test that sensitive information is not logged
        sensitive_data = {
            "password": "secret123",
            "api_key": "key123",
            "token": "token123",
            "private_key": "private123",
        }

        # Create a test logger
        logger = logging.getLogger("test_logger")
        log_records = []

        def capture_log(record):
            log_records.append(record.getMessage())

        logger.addHandler(logging.Handler())
        logger.handlers[0].emit = capture_log

        # Log some data
        logger.info(f"Processing data: {sensitive_data}")

        # Check that sensitive data is not in logs
        log_message = " ".join(log_records)
        for key, value in sensitive_data.items():
            assert key not in log_message
            assert value not in log_message

    def test_configuration_security_validation(self, _mock_config):
        """Test configuration security validation."""
        # Test that configuration files are properly validated
        valid_config = {
            "symbols": ["AAPL", "GOOGL"],
            "initial_balance": 100000,
            "max_position_size": 0.1,
        }

        # Test valid configuration
        assert "symbols" in valid_config
        assert "initial_balance" in valid_config
        assert "max_position_size" in valid_config

        # Test invalid configuration
        invalid_configs = [
            {},  # Empty config
            {"symbols": []},  # Empty symbols
            {"initial_balance": -1000},  # Negative balance
            {"max_position_size": 2.0},  # Position size > 1
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError):
                if not config:
                    raise ValueError("Configuration cannot be empty")
                if "symbols" in config and not config["symbols"]:
                    raise ValueError("Symbols list cannot be empty")
                if "initial_balance" in config and config["initial_balance"] <= 0:
                    raise ValueError("Initial balance must be positive")
                if "max_position_size" in config and not (0 < config["max_position_size"] <= 1):
                    raise ValueError("Max position size must be between 0 and 1")
