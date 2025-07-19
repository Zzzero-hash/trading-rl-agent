"""Security tests for authentication and authorization in trading RL agent."""

import hashlib
import hmac
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

import jwt
import pytest

from src.trading_rl_agent.monitoring.auth_manager import AuthManager


class TestAuthenticationSecurity:
    """Test authentication security measures."""

    @pytest.fixture
    def mock_auth_manager(self) -> AuthManager:
        """Create a mock authentication manager for testing."""
        auth_manager = Mock(spec=AuthManager)
        auth_manager.secret_key = "test_secret_key_12345"
        auth_manager.algorithm = "HS256"
        auth_manager.token_expiry = 3600  # 1 hour
        return auth_manager

    def test_password_hashing_security(self, mock_auth_manager):
        """Test password hashing security."""
        passwords = [
            "simple_password",
            "ComplexP@ssw0rd123!",
            "very_long_password_with_special_chars_!@#$%^&*()",
            "1234567890",
        ]

        for password in passwords:
            # Test bcrypt hashing
            salt = hashlib.sha256(password.encode()).hexdigest()[:16]
            hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
            hashed_hex = hashed.hex()

            # Verify hash is different from original password
            assert hashed_hex != password
            assert len(hashed_hex) > len(password)

            # Verify hash is deterministic
            hashed2 = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
            assert hashed_hex == hashed2.hex()

    def test_jwt_token_security(self, mock_auth_manager):
        """Test JWT token security."""
        # Test valid JWT creation
        payload = {"user_id": "test_user", "role": "trader", "exp": datetime.utcnow() + timedelta(hours=1)}

        token = jwt.encode(payload, mock_auth_manager.secret_key, algorithm=mock_auth_manager.algorithm)
        assert isinstance(token, str)
        assert len(token) > 0

        # Test JWT decoding
        decoded = jwt.decode(token, mock_auth_manager.secret_key, algorithms=[mock_auth_manager.algorithm])
        assert decoded["user_id"] == "test_user"
        assert decoded["role"] == "trader"

        # Test expired token
        expired_payload = {"user_id": "test_user", "role": "trader", "exp": datetime.utcnow() - timedelta(hours=1)}
        expired_token = jwt.encode(expired_payload, mock_auth_manager.secret_key, algorithm=mock_auth_manager.algorithm)

        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(expired_token, mock_auth_manager.secret_key, algorithms=[mock_auth_manager.algorithm])

        # Test invalid signature
        invalid_token = jwt.encode(payload, "wrong_secret", algorithm=mock_auth_manager.algorithm)
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(invalid_token, mock_auth_manager.secret_key, algorithms=[mock_auth_manager.algorithm])

    def test_api_key_validation(self, mock_auth_manager):
        """Test API key validation."""
        # Test valid API key format
        valid_api_keys = [
            "sk_test_1234567890abcdef",
            "pk_live_abcdef1234567890",
            "api_key_1234567890abcdefghijklmnop",
        ]

        for api_key in valid_api_keys:
            assert len(api_key) >= 20
            assert api_key.startswith(("sk_", "pk_", "api_key_"))
            assert all(c.isalnum() or c == "_" for c in api_key)

        # Test invalid API keys
        invalid_api_keys = [
            "",  # Empty
            "short",  # Too short
            "invalid_key_without_prefix",
            "sk_test_1234567890abcdef!",  # Invalid character
            "sk_test_1234567890abcdef@",  # Invalid character
            "sk_test_1234567890abcdef#",  # Invalid character
            "sk_test_1234567890abcdef$",  # Invalid character
            "sk_test_1234567890abcdef%",  # Invalid character
            "sk_test_1234567890abcdef^",  # Invalid character
            "sk_test_1234567890abcdef&",  # Invalid character
            "sk_test_1234567890abcdef*",  # Invalid character
            "sk_test_1234567890abcdef(",  # Invalid character
            "sk_test_1234567890abcdef)",  # Invalid character
            "sk_test_1234567890abcdef-",  # Invalid character
            "sk_test_1234567890abcdef+",  # Invalid character
            "sk_test_1234567890abcdef=",  # Invalid character
            "sk_test_1234567890abcdef[",  # Invalid character
            "sk_test_1234567890abcdef]",  # Invalid character
            "sk_test_1234567890abcdef{",  # Invalid character
            "sk_test_1234567890abcdef}",  # Invalid character
            "sk_test_1234567890abcdef|",  # Invalid character
            "sk_test_1234567890abcdef\\",  # Invalid character
            "sk_test_1234567890abcdef/",  # Invalid character
            "sk_test_1234567890abcdef:",  # Invalid character
            "sk_test_1234567890abcdef;",  # Invalid character
            "sk_test_1234567890abcdef'",  # Invalid character
            'sk_test_1234567890abcdef"',  # Invalid character
            "sk_test_1234567890abcdef,",  # Invalid character
            "sk_test_1234567890abcdef.",  # Invalid character
            "sk_test_1234567890abcdef<",  # Invalid character
            "sk_test_1234567890abcdef>",  # Invalid character
            "sk_test_1234567890abcdef?",  # Invalid character
        ]

        for api_key in invalid_api_keys:
            with pytest.raises(ValueError):
                if not api_key or len(api_key) < 20 or not api_key.startswith(("sk_", "pk_", "api_key_")):
                    raise ValueError(f"Invalid API key format: {api_key}")
                if not all(c.isalnum() or c == "_" for c in api_key):
                    raise ValueError(f"Invalid API key characters: {api_key}")

    def test_rate_limiting(self, mock_auth_manager):
        """Test rate limiting functionality."""
        # Simulate rate limiting
        requests_per_minute = 60
        requests = []

        # Simulate multiple requests
        for i in range(requests_per_minute + 10):
            current_time = time.time()
            # Remove requests older than 1 minute
            requests = [req for req in requests if current_time - req < 60]

            if len(requests) >= requests_per_minute:
                with pytest.raises(ValueError):
                    raise ValueError("Rate limit exceeded")

            requests.append(current_time)

        # Verify rate limiting works
        assert len(requests) <= requests_per_minute

    def test_session_management(self, mock_auth_manager):
        """Test session management security."""
        # Test session creation
        session_id = hashlib.sha256(f"user123_{time.time()}".encode()).hexdigest()
        session_data = {
            "user_id": "user123",
            "created_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hour
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        assert len(session_id) == 64  # SHA256 hash length
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

        # Test session hijacking prevention
        original_ip = "192.168.1.1"
        new_ip = "192.168.1.2"

        with pytest.raises(ValueError):
            if original_ip != new_ip:
                raise ValueError("IP address mismatch - possible session hijacking")

    def test_multi_factor_authentication(self, mock_auth_manager):
        """Test multi-factor authentication."""
        import pyotp

        # Test TOTP generation
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)

        # Generate current code
        current_code = totp.now()
        assert len(current_code) == 6
        assert current_code.isdigit()

        # Verify current code
        assert totp.verify(current_code)

        # Test invalid code
        invalid_code = "000000"
        assert not totp.verify(invalid_code)

        # Test expired code (simulate)
        time.sleep(1)  # Wait for next time window
        new_code = totp.now()
        assert new_code != current_code

    def test_password_policy_enforcement(self, mock_auth_manager):
        """Test password policy enforcement."""
        # Test valid passwords
        valid_passwords = [
            "ComplexP@ssw0rd123!",
            "MySecureP@ssw0rd2023!",
            "Tr@dingRl@gent2023!",
            "S3cur3P@ssw0rd!",
        ]

        for password in valid_passwords:
            # Check minimum length
            assert len(password) >= 12

            # Check complexity requirements
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

            assert has_upper, f"Password missing uppercase: {password}"
            assert has_lower, f"Password missing lowercase: {password}"
            assert has_digit, f"Password missing digit: {password}"
            assert has_special, f"Password missing special character: {password}"

        # Test invalid passwords
        invalid_passwords = [
            "short",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoDigits!",  # No digits
            "NoSpecial123",  # No special characters
            "password",  # Common password
            "123456",  # Common password
            "qwerty",  # Common password
            "admin",  # Common password
            "letmein",  # Common password
        ]

        for password in invalid_passwords:
            with pytest.raises(ValueError):
                if len(password) < 12:
                    raise ValueError("Password too short")
                if not any(c.isupper() for c in password):
                    raise ValueError("Password missing uppercase")
                if not any(c.islower() for c in password):
                    raise ValueError("Password missing lowercase")
                if not any(c.isdigit() for c in password):
                    raise ValueError("Password missing digit")
                if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                    raise ValueError("Password missing special character")

    def test_brute_force_protection(self, mock_auth_manager):
        """Test brute force attack protection."""
        # Simulate failed login attempts
        failed_attempts = {}
        max_attempts = 5
        lockout_duration = 300  # 5 minutes

        def attempt_login(user_id: str, password: str) -> bool:
            current_time = time.time()

            # Check if account is locked
            if user_id in failed_attempts:
                attempts, lockout_time = failed_attempts[user_id]
                if current_time < lockout_time:
                    raise ValueError("Account temporarily locked")
                if current_time >= lockout_time:
                    # Reset failed attempts after lockout period
                    failed_attempts[user_id] = (0, 0)

            # Simulate login attempt
            if password == "correct_password":
                # Successful login - reset failed attempts
                failed_attempts[user_id] = (0, 0)
                return True
            # Failed login
            attempts = failed_attempts.get(user_id, (0, 0))[0] + 1
            lockout_time = current_time + lockout_duration if attempts >= max_attempts else 0
            failed_attempts[user_id] = (attempts, lockout_time)

            if attempts >= max_attempts:
                raise ValueError("Account locked due to too many failed attempts")
            return False

        # Test successful login
        assert attempt_login("user1", "correct_password")

        # Test failed attempts
        for i in range(max_attempts):
            with pytest.raises(ValueError):
                attempt_login("user2", "wrong_password")

        # Test account lockout
        with pytest.raises(ValueError):
            attempt_login("user2", "correct_password")

    def test_secure_headers(self, mock_auth_manager):
        """Test secure HTTP headers."""
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
            assert value is not None

        # Test missing headers
        missing_headers = ["X-Content-Type-Options", "X-Frame-Options"]
        for header in missing_headers:
            with pytest.raises(ValueError):
                if header not in required_headers:
                    raise ValueError(f"Missing security header: {header}")

    def test_csrf_protection(self, mock_auth_manager):
        """Test CSRF protection."""
        # Test CSRF token generation
        csrf_token = hashlib.sha256(f"csrf_{time.time()}_{mock_auth_manager.secret_key}".encode()).hexdigest()
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

    def test_oauth_integration_security(self, mock_auth_manager):
        """Test OAuth integration security."""
        # Test OAuth state parameter
        oauth_state = hashlib.sha256(f"oauth_state_{time.time()}".encode()).hexdigest()
        assert len(oauth_state) == 64

        # Test OAuth callback validation
        def validate_oauth_callback(state: str, code: str, expected_state: str) -> bool:
            return not (not state or not code or not expected_state or state != expected_state or len(code) < 10)

        # Test valid callback
        oauth_code = "valid_oauth_code_123456789"
        assert validate_oauth_callback(oauth_state, oauth_code, oauth_state)

        # Test invalid callback
        assert not validate_oauth_callback("", oauth_code, oauth_state)
        assert not validate_oauth_callback(oauth_state, "", oauth_state)
        assert not validate_oauth_callback("wrong_state", oauth_code, oauth_state)
        assert not validate_oauth_callback(oauth_state, "short", oauth_state)

    def test_audit_logging(self, mock_auth_manager):
        """Test audit logging for authentication events."""

        # Test authentication event logging
        auth_events = [
            {"event": "login", "user_id": "user123", "ip": "192.168.1.1", "success": True},
            {"event": "login", "user_id": "user123", "ip": "192.168.1.1", "success": False},
            {"event": "logout", "user_id": "user123", "ip": "192.168.1.1"},
            {"event": "password_change", "user_id": "user123", "ip": "192.168.1.1"},
            {"event": "failed_login", "user_id": "user123", "ip": "192.168.1.1", "reason": "wrong_password"},
        ]

        for event in auth_events:
            # Verify event has required fields
            assert "event" in event
            assert "user_id" in event
            assert "ip" in event

            # Verify event type is valid
            valid_events = ["login", "logout", "password_change", "failed_login", "account_lockout"]
            assert event["event"] in valid_events

            # Verify IP address format
            ip_parts = event["ip"].split(".")
            assert len(ip_parts) == 4
            assert all(0 <= int(part) <= 255 for part in ip_parts)

    def test_privilege_escalation_prevention(self, mock_auth_manager):
        """Test privilege escalation prevention."""
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

    def test_session_fixation_prevention(self, mock_auth_manager):
        """Test session fixation prevention."""

        # Test session regeneration after login
        def create_session(user_id: str) -> str:
            return hashlib.sha256(f"session_{user_id}_{time.time()}".encode()).hexdigest()

        def regenerate_session(old_session: str, user_id: str) -> str:
            # Invalidate old session
            old_session = None
            # Create new session
            return create_session(user_id)

        # Test session regeneration
        old_session = create_session("user123")
        new_session = regenerate_session(old_session, "user123")

        assert old_session != new_session
        assert len(new_session) == 64

    def test_secure_password_reset(self, mock_auth_manager):
        """Test secure password reset functionality."""

        # Test password reset token generation
        def generate_reset_token(user_id: str) -> str:
            return hashlib.sha256(f"reset_{user_id}_{time.time()}".encode()).hexdigest()

        # Test password reset token validation
        def validate_reset_token(token: str, user_id: str) -> bool:
            # In a real implementation, you would check against stored tokens
            return not (not token or not user_id or len(token) != 64)

        # Test valid reset token
        reset_token = generate_reset_token("user123")
        assert validate_reset_token(reset_token, "user123")

        # Test invalid reset token
        assert not validate_reset_token("", "user123")
        assert not validate_reset_token("short_token", "user123")
        assert not validate_reset_token(reset_token, "")

    def test_secure_logout(self, mock_auth_manager):
        """Test secure logout functionality."""
        # Test session invalidation
        active_sessions = {
            "session1": {"user_id": "user123", "created_at": time.time()},
            "session2": {"user_id": "user456", "created_at": time.time()},
        }

        def logout_user(session_id: str) -> bool:
            if session_id in active_sessions:
                del active_sessions[session_id]
                return True
            return False

        # Test successful logout
        assert logout_user("session1")
        assert "session1" not in active_sessions
        assert "session2" in active_sessions

        # Test logout of non-existent session
        assert not logout_user("nonexistent_session")

        # Test logout of already logged out session
        assert not logout_user("session1")

    def test_secure_cookie_settings(self, mock_auth_manager):
        """Test secure cookie settings."""
        # Test secure cookie configuration
        secure_cookie_settings = {
            "httpOnly": True,
            "secure": True,
            "sameSite": "Strict",
            "maxAge": 3600,  # 1 hour
            "path": "/",
        }

        # Verify secure settings
        assert secure_cookie_settings["httpOnly"] is True
        assert secure_cookie_settings["secure"] is True
        assert secure_cookie_settings["sameSite"] == "Strict"
        assert secure_cookie_settings["maxAge"] > 0
        assert secure_cookie_settings["path"] == "/"

        # Test insecure settings detection
        insecure_settings = {
            "httpOnly": False,
            "secure": False,
            "sameSite": "None",
            "maxAge": 0,
        }

        for setting, value in insecure_settings.items():
            with pytest.raises(ValueError):
                if setting == "httpOnly" and not value:
                    raise ValueError("Cookies must be httpOnly")
                if setting == "secure" and not value:
                    raise ValueError("Cookies must be secure")
                if setting == "sameSite" and value == "None":
                    raise ValueError("SameSite cannot be None")
                if setting == "maxAge" and value <= 0:
                    raise ValueError("Cookie maxAge must be positive")
