"""
API Security Manager for trading RL agent.

Provides comprehensive security features including rate limiting,
authentication, authorization, and input validation for API endpoints.
"""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import jwt
import structlog
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_size: int = 60  # seconds


class SecurityConfig(BaseModel):
    """Configuration for API security."""

    secret_key: str
    algorithm: str = "HS256"
    token_expiry_hours: int = 24
    rate_limit: RateLimitConfig = RateLimitConfig()
    allowed_origins: list[str] = ["*"]
    require_authentication: bool = True


class APISecurityManager:
    """
    Comprehensive API security manager.

    Provides rate limiting, authentication, authorization, and input
    validation for API endpoints.
    """

    def __init__(self, config: SecurityConfig):
        """
        Initialize the API security manager.

        Args:
            config: Security configuration
        """
        self.config = config
        self.rate_limit_store: dict[str, list[float]] = {}
        self.blacklisted_tokens: set[str] = set()
        self.active_sessions: dict[str, dict[str, Any]] = {}

        logger.info("APISecurityManager initialized")

    def authenticate_request(self, request: Request) -> dict[str, Any]:
        """
        Authenticate an incoming request.

        Args:
            request: FastAPI request object

        Returns:
            Authentication result with user info

        Raises:
            HTTPException: If authentication fails
        """
        if not self.config.require_authentication:
            return {"authenticated": True, "user": "anonymous"}

        # Extract token from headers
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
            )

        # Parse Bearer token
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )

            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id and session_id not in self.active_sessions:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired",
                )

            logger.info("Request authenticated", user_id=payload.get("user_id"))
            return {
                "authenticated": True,
                "user_id": payload.get("user_id"),
                "session_id": session_id,
                "permissions": payload.get("permissions", []),
            }

        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            ) from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            ) from exc

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if a client has exceeded rate limits.

        Args:
            client_id: Unique identifier for the client

        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()

        # Initialize rate limit tracking for this client
        if client_id not in self.rate_limit_store:
            self.rate_limit_store[client_id] = []

        # Clean old requests outside the window
        window_start = now - self.config.rate_limit.window_size
        self.rate_limit_store[client_id] = [
            req_time for req_time in self.rate_limit_store[client_id] if req_time > window_start
        ]

        # Check rate limits
        requests_in_window = len(self.rate_limit_store[client_id])

        if requests_in_window >= self.config.rate_limit.requests_per_minute:
            logger.warning("Rate limit exceeded", client_id=client_id)
            return False

        # Add current request
        self.rate_limit_store[client_id].append(now)
        return True

    def validate_input(self, data: Any, schema: BaseModel) -> BaseModel:
        """
        Validate input data against a Pydantic schema.

        Args:
            data: Input data to validate
            schema: Pydantic model for validation

        Returns:
            Validated data

        Raises:
            HTTPException: If validation fails
        """
        try:
            validated_data = schema(**data) if isinstance(data, dict) else schema(data)

            logger.debug("Input validation successful", schema=schema.__name__)
            return validated_data

        except ValidationError as exc:
            logger.warning("Input validation failed", errors=exc.errors())
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Validation error: {exc.errors()}",
            ) from exc

    def sanitize_input(self, data: str) -> str:
        """
        Sanitize input data to prevent injection attacks.

        Args:
            data: Input string to sanitize

        Returns:
            Sanitized string
        """
        # Remove potentially dangerous characters
        dangerous_chars = [
            "<",
            ">",
            '"',
            "'",
            "&",
            ";",
            "|",
            "`",
            "$",
            "(",
            ")",
            "{",
            "}",
        ]
        sanitized = data

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")

        # Remove SQL injection patterns
        sql_patterns = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "UNION",
            "EXEC",
            "EXECUTE",
            "SCRIPT",
            "JAVASCRIPT",
        ]

        sanitized_upper = sanitized.upper()
        for pattern in sql_patterns:
            if pattern in sanitized_upper:
                sanitized = sanitized.replace(pattern, "")
                sanitized = sanitized.replace(pattern.lower(), "")

        logger.debug(
            "Input sanitized",
            original_length=len(data),
            sanitized_length=len(sanitized),
        )
        return sanitized

    def generate_token(self, user_id: str, permissions: list[str] | None = None) -> str:
        """
        Generate a JWT token for a user.

        Args:
            user_id: User identifier
            permissions: List of user permissions

        Returns:
            JWT token string
        """
        session_id = self._generate_session_id()

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "permissions": permissions or [],
            "exp": datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours),
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

        # Store active session
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "permissions": permissions or [],
        }

        logger.info("Token generated", user_id=user_id, session_id=session_id)
        return str(token)

    def revoke_token(self, token: str) -> None:
        """
        Revoke a JWT token.

        Args:
            token: JWT token to revoke
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )

            session_id = payload.get("session_id")
            if session_id:
                self.active_sessions.pop(session_id, None)

            self.blacklisted_tokens.add(token)

            logger.info("Token revoked", session_id=session_id)

        except jwt.InvalidTokenError:
            logger.warning("Attempted to revoke invalid token")

    def check_permission(self, user_permissions: list[str], required_permission: str) -> bool:
        """
        Check if a user has a specific permission.

        Args:
            user_permissions: List of user permissions
            required_permission: Permission to check for

        Returns:
            True if user has the permission, False otherwise
        """
        return required_permission in user_permissions or "admin" in user_permissions

    def validate_origin(self, origin: str) -> bool:
        """
        Validate request origin against allowed origins.

        Args:
            origin: Request origin

        Returns:
            True if origin is allowed, False otherwise
        """
        if "*" in self.config.allowed_origins:
            return True

        urlparse(origin)
        return origin in self.config.allowed_origins

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = str(int(time.time() * 1000000))
        random_component = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"{timestamp}_{random_component}"

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions and tokens."""
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, session_data in self.active_sessions.items():
            session_age = now - session_data["created_at"]
            if session_age > timedelta(hours=self.config.token_expiry_hours):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.active_sessions.pop(session_id, None)

        if expired_sessions:
            logger.info("Cleaned up expired sessions", count=len(expired_sessions))

    def get_security_metrics(self) -> dict[str, Any]:
        """
        Get security metrics and statistics.

        Returns:
            Dictionary with security metrics
        """
        return {
            "active_sessions": len(self.active_sessions),
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "rate_limited_clients": len(self.rate_limit_store),
            "total_rate_limited_requests": sum(len(requests) for requests in self.rate_limit_store.values()),
        }
