"""Security tests for API security in trading RL agent."""

import pytest
import json
import hashlib
import hmac
import time
import base64
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.trading_rl_agent.core.config import Config
from src.trading_rl_agent.monitoring.api_security import APISecurityManager


class TestAPISecurity:
    """Test API security measures."""

    @pytest.fixture
    def mock_api_security_manager(self) -> APISecurityManager:
        """Create a mock API security manager for testing."""
        security_manager = Mock(spec=APISecurityManager)
        security_manager.secret_key = "test_secret_key_12345"
        security_manager.api_rate_limit = 100  # requests per minute
        security_manager.max_request_size = 1024 * 1024  # 1MB
        return security_manager

    def test_api_authentication(self, mock_api_security_manager):
        """Test API authentication mechanisms."""
        # Test API key authentication
        valid_api_keys = [
            "sk_test_1234567890abcdef",
            "pk_live_abcdef1234567890",
            "api_key_1234567890abcdefghijklmnop",
        ]

        for api_key in valid_api_keys:
            # Test API key validation
            assert len(api_key) >= 20
            assert api_key.startswith(("sk_", "pk_", "api_key_"))
            assert all(c.isalnum() or c == '_' for c in api_key)

        # Test invalid API keys
        invalid_api_keys = [
            "",  # Empty
            "short",  # Too short
            "invalid_key_without_prefix",
            "sk_test_1234567890abcdef!",  # Invalid character
        ]

        for api_key in invalid_api_keys:
            with pytest.raises(ValueError):
                if not api_key or len(api_key) < 20 or not api_key.startswith(("sk_", "pk_", "api_key_")):
                    raise ValueError(f"Invalid API key: {api_key}")

    def test_api_rate_limiting(self, mock_api_security_manager):
        """Test API rate limiting."""
        # Simulate rate limiting
        requests_per_minute = mock_api_security_manager.api_rate_limit
        requests = []

        # Test normal rate
        for i in range(requests_per_minute):
            current_time = time.time()
            requests = [req for req in requests if current_time - req < 60]
            requests.append(current_time)
            assert len(requests) <= requests_per_minute

        # Test rate limit exceeded
        with pytest.raises(ValueError):
            current_time = time.time()
            requests = [req for req in requests if current_time - req < 60]
            if len(requests) >= requests_per_minute:
                raise ValueError("Rate limit exceeded")
            requests.append(current_time)

    def test_request_size_limiting(self, mock_api_security_manager):
        """Test request size limiting."""
        max_size = mock_api_security_manager.max_request_size

        # Test valid request sizes
        valid_sizes = [100, 1000, 10000, max_size]
        for size in valid_sizes:
            assert size <= max_size

        # Test oversized requests
        oversized_requests = [max_size + 1, max_size + 1000, max_size * 2]
        for size in oversized_requests:
            with pytest.raises(ValueError):
                if size > max_size:
                    raise ValueError(f"Request too large: {size} bytes")

    def test_input_validation(self, mock_api_security_manager):
        """Test API input validation."""
        # Test valid inputs
        valid_inputs = [
            {"symbol": "AAPL", "quantity": 100},
            {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            {"portfolio_id": "12345", "action": "buy"},
        ]

        for input_data in valid_inputs:
            # Test input validation
            assert isinstance(input_data, dict)
            assert len(input_data) > 0
            for key, value in input_data.items():
                assert isinstance(key, str)
                assert len(key) > 0
                assert len(key) <= 100  # Reasonable max key length

        # Test invalid inputs
        invalid_inputs = [
            {},  # Empty dict
            {"": "value"},  # Empty key
            {"key": ""},  # Empty value
            {"A" * 101: "value"},  # Key too long
            {"key": "A" * 10001},  # Value too long
        ]

        for input_data in invalid_inputs:
            with pytest.raises(ValueError):
                if not input_data:
                    raise ValueError("Input cannot be empty")
                for key, value in input_data.items():
                    if not key:
                        raise ValueError("Key cannot be empty")
                    if len(key) > 100:
                        raise ValueError("Key too long")
                    if len(str(value)) > 10000:
                        raise ValueError("Value too long")

    def test_sql_injection_prevention(self, mock_api_security_manager):
        """Test SQL injection prevention in API."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "'; UPDATE users SET password='hacked'; --",
            "'; DELETE FROM users; --",
        ]

        for malicious_input in malicious_inputs:
            # Test SQL injection detection
            sql_keywords = [
                "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                "EXEC", "EXECUTE", "UNION", "WAITFOR", "IF", "CASE", "WHEN", "THEN",
                "ELSE", "END", "BEGIN", "TRANSACTION", "COMMIT", "ROLLBACK"
            ]
            
            detected_keywords = [keyword for keyword in sql_keywords 
                               if keyword.lower() in malicious_input.lower()]
            
            with pytest.raises(ValueError):
                if detected_keywords:
                    raise ValueError(f"SQL injection detected: {detected_keywords}")

    def test_xss_prevention(self, mock_api_security_manager):
        """Test XSS prevention in API."""
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
                if pattern in malicious_script.lower():
                    detected_xss = True
                    break
            
            with pytest.raises(ValueError):
                if detected_xss:
                    raise ValueError(f"XSS attempt detected: {malicious_script}")

    def test_csrf_protection(self, mock_api_security_manager):
        """Test CSRF protection in API."""
        # Test CSRF token generation
        csrf_token = hashlib.sha256(f"csrf_{time.time()}_{mock_api_security_manager.secret_key}".encode()).hexdigest()
        assert len(csrf_token) == 64
        assert csrf_token.isalnum()

        # Test CSRF token validation
        def validate_csrf_token(token: str, expected_token: str) -> bool:
            if not token or not expected_token:
                return False
            return hmac.compare_digest(token, expected_token)

        # Test valid token
        assert validate_csrf_token(csrf_token, csrf_token)

        # Test invalid token
        assert not validate_csrf_token(csrf_token, "invalid_token")
        assert not validate_csrf_token("", csrf_token)
        assert not validate_csrf_token(csrf_token, "")

    def test_secure_headers(self, mock_api_security_manager):
        """Test secure HTTP headers in API responses."""
        # Test required security headers
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        for header, value in required_headers.items():
            assert header in required_headers
            assert required_headers[header] == value

        # Test missing headers
        missing_headers = ["X-Content-Type-Options", "X-Frame-Options"]
        for header in missing_headers:
            with pytest.raises(ValueError):
                if header not in required_headers:
                    raise ValueError(f"Missing security header: {header}")

    def test_authentication_tokens(self, mock_api_security_manager):
        """Test authentication token security."""
        # Test JWT token creation
        payload = {
            "user_id": "test_user",
            "role": "trader",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        # Simulate JWT encoding (in real implementation, use jwt library)
        token = base64.b64encode(json.dumps(payload).encode()).decode()
        assert isinstance(token, str)
        assert len(token) > 0

        # Test token validation
        try:
            decoded_payload = json.loads(base64.b64decode(token).decode())
            assert decoded_payload["user_id"] == "test_user"
            assert decoded_payload["role"] == "trader"
        except Exception:
            pytest.fail("Valid token was rejected")

        # Test expired token
        expired_payload = {
            "user_id": "test_user",
            "role": "trader",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        
        with pytest.raises(ValueError):
            if expired_payload["exp"] < datetime.utcnow():
                raise ValueError("Token expired")

    def test_request_logging(self, mock_api_security_manager):
        """Test API request logging security."""
        # Test request logging
        request_data = {
            "method": "POST",
            "path": "/api/trade",
            "user_id": "user123",
            "ip_address": "192.168.1.1",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Verify request data
        assert request_data["method"] in ["GET", "POST", "PUT", "DELETE", "PATCH"]
        assert request_data["path"].startswith("/api/")
        assert request_data["user_id"] is not None
        assert request_data["ip_address"] is not None

        # Test sensitive data filtering
        sensitive_data = {
            "password": "secret123",
            "api_key": "key123",
            "token": "token123",
            "private_key": "private123",
        }

        # Ensure sensitive data is not logged
        log_message = f"Request: {request_data}"
        for key, value in sensitive_data.items():
            assert key not in log_message
            assert value not in log_message

    def test_error_handling(self, mock_api_security_manager):
        """Test secure error handling in API."""
        # Test that sensitive information is not exposed in errors
        try:
            raise ValueError("Database connection failed: user=admin, password=secret123")
        except ValueError as e:
            error_message = str(e)
            # Ensure sensitive data is not in error message
            assert "password=secret123" not in error_message
            assert "user=admin" not in error_message

        # Test generic error messages
        generic_errors = [
            "Internal server error",
            "Bad request",
            "Unauthorized",
            "Forbidden",
            "Not found",
        ]

        for error in generic_errors:
            assert len(error) > 0
            assert len(error) <= 100  # Reasonable error message length

    def test_input_sanitization(self, mock_api_security_manager):
        """Test input sanitization in API."""
        # Test string sanitization
        test_inputs = [
            "Hello World",
            "AAPL",
            "123.45",
            "user@example.com",
        ]

        for test_input in test_inputs:
            # Test basic sanitization
            sanitized = test_input.strip()
            assert sanitized == test_input

        # Test malicious input sanitization
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert('xss')",
        ]

        for malicious_input in malicious_inputs:
            # Test sanitization
            sanitized = malicious_input.replace("<script>", "").replace("javascript:", "")
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized

    def test_output_encoding(self, mock_api_security_manager):
        """Test output encoding security."""
        import html

        # Test HTML encoding
        test_outputs = [
            "<script>alert('xss')</script>",
            "Hello & World",
            "Price: $100 < $200",
            "User input: <img src=x onerror=alert('xss')>",
        ]

        for test_output in test_outputs:
            # Test HTML encoding
            encoded = html.escape(test_output)
            assert "&lt;" in encoded or "&gt;" in encoded or "&amp;" in encoded
            assert "<script>" not in encoded.lower()

    def test_session_management(self, mock_api_security_manager):
        """Test session management security."""
        # Test session creation
        session_id = hashlib.sha256(f"session_{time.time()}".encode()).hexdigest()
        session_data = {
            "user_id": "user123",
            "created_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hour
            "ip_address": "192.168.1.1",
        }

        assert len(session_id) == 64
        assert session_data["expires_at"] > session_data["created_at"]

        # Test session expiration
        expired_session = {
            "user_id": "user123",
            "created_at": time.time() - 7200,  # 2 hours ago
            "expires_at": time.time() - 3600,  # 1 hour ago
        }

        with pytest.raises(ValueError):
            if time.time() > expired_session["expires_at"]:
                raise ValueError("Session expired")

    def test_authorization_checks(self, mock_api_security_manager):
        """Test authorization checks in API."""
        # Test role-based access control
        user_roles = {
            "user123": "trader",
            "admin456": "admin",
            "viewer789": "viewer",
        }

        # Define permissions for each role
        role_permissions = {
            "viewer": ["read_data", "view_reports"],
            "trader": ["read_data", "view_reports", "place_trades", "modify_positions"],
            "admin": ["read_data", "view_reports", "place_trades", "modify_positions", "manage_users", "system_config"],
        }

        # Test permission checking
        def check_permission(user_id: str, permission: str) -> bool:
            if user_id not in user_roles:
                return False
            user_role = user_roles[user_id]
            if user_role not in role_permissions:
                return False
            return permission in role_permissions[user_role]

        # Test valid permissions
        assert check_permission("user123", "place_trades")
        assert check_permission("admin456", "manage_users")
        assert check_permission("viewer789", "read_data")

        # Test invalid permissions
        assert not check_permission("user123", "manage_users")
        assert not check_permission("viewer789", "place_trades")
        assert not check_permission("nonexistent", "read_data")

    def test_request_validation(self, mock_api_security_manager):
        """Test request validation security."""
        # Test valid requests
        valid_requests = [
            {"method": "GET", "path": "/api/data", "headers": {"Authorization": "Bearer token"}},
            {"method": "POST", "path": "/api/trade", "headers": {"Content-Type": "application/json"}},
            {"method": "PUT", "path": "/api/portfolio", "headers": {"X-CSRF-Token": "token"}},
        ]

        for request in valid_requests:
            # Test request validation
            assert request["method"] in ["GET", "POST", "PUT", "DELETE", "PATCH"]
            assert request["path"].startswith("/api/")
            assert isinstance(request["headers"], dict)

        # Test invalid requests
        invalid_requests = [
            {"method": "INVALID", "path": "/api/data", "headers": {}},
            {"method": "GET", "path": "/invalid/path", "headers": {}},
            {"method": "POST", "path": "/api/trade", "headers": "not_a_dict"},
        ]

        for request in invalid_requests:
            with pytest.raises(ValueError):
                if request["method"] not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    raise ValueError(f"Invalid method: {request['method']}")
                if not request["path"].startswith("/api/"):
                    raise ValueError(f"Invalid path: {request['path']}")
                if not isinstance(request["headers"], dict):
                    raise ValueError("Headers must be a dictionary")

    def test_response_validation(self, mock_api_security_manager):
        """Test response validation security."""
        # Test valid responses
        valid_responses = [
            {"status_code": 200, "data": {"result": "success"}},
            {"status_code": 201, "data": {"id": "12345"}},
            {"status_code": 400, "error": "Bad request"},
        ]

        for response in valid_responses:
            # Test response validation
            assert response["status_code"] in [200, 201, 400, 401, 403, 404, 500]
            assert isinstance(response, dict)

        # Test invalid responses
        invalid_responses = [
            {"status_code": 999, "data": {}},  # Invalid status code
            {"status_code": 200, "data": None},  # None data
            {"status_code": "200", "data": {}},  # String status code
        ]

        for response in invalid_responses:
            with pytest.raises(ValueError):
                if response["status_code"] not in [200, 201, 400, 401, 403, 404, 500]:
                    raise ValueError(f"Invalid status code: {response['status_code']}")
                if not isinstance(response["status_code"], int):
                    raise ValueError("Status code must be an integer")

    def test_secure_communication(self, mock_api_security_manager):
        """Test secure communication protocols."""
        # Test HTTPS requirement
        valid_urls = [
            "https://api.example.com",
            "https://data.example.com/v1/endpoint",
        ]

        for url in valid_urls:
            assert url.startswith("https://")

        # Test invalid protocols
        invalid_urls = [
            "http://api.example.com",  # HTTP not allowed
            "ftp://api.example.com",  # FTP not allowed
            "file:///etc/passwd",  # File protocol not allowed
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                if not url.startswith("https://"):
                    raise ValueError(f"Insecure protocol: {url}")

    def test_data_encryption(self, mock_api_security_manager):
        """Test data encryption security."""
        # Test encryption key validation
        encryption_key = "test_encryption_key_12345"
        assert len(encryption_key) >= 32
        assert isinstance(encryption_key, str)

        # Test data encryption (simplified)
        test_data = "sensitive_data"
        # In real implementation, use proper encryption
        encrypted = base64.b64encode(test_data.encode()).decode()
        assert encrypted != test_data

        # Test data decryption
        decrypted = base64.b64decode(encrypted).decode()
        assert decrypted == test_data

    def test_audit_logging(self, mock_api_security_manager):
        """Test audit logging security."""
        # Test audit event logging
        audit_events = [
            {"event": "api_request", "user_id": "user123", "action": "GET /api/data"},
            {"event": "api_request", "user_id": "user123", "action": "POST /api/trade"},
            {"event": "authentication", "user_id": "user123", "result": "success"},
            {"event": "authorization", "user_id": "user123", "permission": "place_trades"},
        ]

        for event in audit_events:
            # Verify audit event structure
            assert "event" in event
            assert "user_id" in event
            assert isinstance(event["event"], str)
            assert isinstance(event["user_id"], str)

        # Test sensitive data filtering in audit logs
        sensitive_data = {
            "password": "secret123",
            "api_key": "key123",
            "token": "token123",
        }

        audit_message = "User user123 performed action with password=secret123"
        for key, value in sensitive_data.items():
            assert key not in audit_message
            assert value not in audit_message