"""
Alpaca Markets Configuration Management

Provides configuration management for Alpaca Markets integration,
including environment variable handling, validation, and default settings.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AlpacaConfigModel(BaseModel):
    """Pydantic model for Alpaca configuration validation."""

    # API Credentials
    api_key: str = Field(..., description="Alpaca API key")
    secret_key: str = Field(..., description="Alpaca secret key")

    # URLs
    base_url: str = Field(default="https://paper-api.alpaca.markets", description="Alpaca trading API base URL")
    data_url: str = Field(default="https://data.alpaca.markets", description="Alpaca data API base URL")

    # API Version
    use_v2_api: bool = Field(default=True, description="Whether to use Alpaca V2 SDK")

    # Trading Mode
    paper_trading: bool = Field(default=True, description="Whether to use paper trading")

    # Retry Configuration
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retry attempts in seconds")

    # Timeout Configuration
    websocket_timeout: int = Field(default=30, ge=10, le=300, description="WebSocket connection timeout in seconds")
    order_timeout: int = Field(default=60, ge=10, le=600, description="Order execution timeout in seconds")

    # Cache Configuration
    cache_dir: str = Field(default="data/alpaca_cache", description="Directory for caching data")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")

    # Data Feed Configuration
    data_feed: str = Field(default="iex", description="Data feed to use (iex, sip)")
    extended_hours: bool = Field(default=False, description="Whether to allow extended hours trading")

    # Risk Management
    max_position_size: float = Field(default=10000.0, ge=0.0, description="Maximum position size in dollars")
    max_daily_trades: int = Field(default=100, ge=1, le=1000, description="Maximum number of trades per day")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_trades: bool = Field(default=True, description="Whether to log all trades")

    @validator("api_key", "secret_key")
    def validate_credentials(cls, v: str) -> str:  # noqa: N805
        """Validate that credentials are not empty."""
        if not v or v.strip() == "":
            raise ValueError("API credentials cannot be empty")
        return v.strip()

    @validator("base_url", "data_url")
    def validate_urls(cls, v: str) -> str:  # noqa: N805
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator("data_feed")
    def validate_data_feed(cls, v: str) -> str:  # noqa: N805
        """Validate data feed selection."""
        valid_feeds = ["iex", "sip"]
        if v not in valid_feeds:
            raise ValueError(f"Data feed must be one of: {valid_feeds}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:  # noqa: N805
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    class Config:
        """Pydantic configuration."""

        env_prefix = "ALPACA_"
        case_sensitive = False


@dataclass
class AlpacaEnvironmentConfig:
    """Environment-based configuration for Alpaca Markets."""

    # Environment variable mappings
    ENV_MAPPINGS = {
        "api_key": "ALPACA_API_KEY",
        "secret_key": "ALPACA_SECRET_KEY",
        "base_url": "ALPACA_BASE_URL",
        "data_url": "ALPACA_DATA_URL",
        "use_v2_api": "ALPACA_USE_V2",
        "paper_trading": "ALPACA_PAPER_TRADING",
        "max_retries": "ALPACA_MAX_RETRIES",
        "retry_delay": "ALPACA_RETRY_DELAY",
        "websocket_timeout": "ALPACA_WEBSOCKET_TIMEOUT",
        "order_timeout": "ALPACA_ORDER_TIMEOUT",
        "cache_dir": "ALPACA_CACHE_DIR",
        "cache_ttl": "ALPACA_CACHE_TTL",
        "data_feed": "ALPACA_DATA_FEED",
        "extended_hours": "ALPACA_EXTENDED_HOURS",
        "max_position_size": "ALPACA_MAX_POSITION_SIZE",
        "max_daily_trades": "ALPACA_MAX_DAILY_TRADES",
        "log_level": "ALPACA_LOG_LEVEL",
        "log_trades": "ALPACA_LOG_TRADES",
    }

    @classmethod
    def from_environment(cls) -> AlpacaConfigModel:
        """
        Create configuration from environment variables.

        Returns:
            Validated AlpacaConfigModel instance
        """
        # First try to get from unified configuration system
        try:
            from ..core.unified_config import UnifiedConfig

            config = UnifiedConfig()
            if config.alpaca_api_key and config.alpaca_secret_key:
                logger.info("Loading Alpaca configuration from unified config")
                config_data = {
                    "api_key": config.alpaca_api_key,
                    "secret_key": config.alpaca_secret_key,
                    "base_url": config.alpaca_base_url or "https://paper-api.alpaca.markets",
                    "data_url": config.alpaca_data_url or "https://data.alpaca.markets",
                    "use_v2_api": config.alpaca_use_v2,
                    "paper_trading": config.alpaca_paper_trading,
                    "max_retries": config.alpaca_max_retries,
                    "retry_delay": config.alpaca_retry_delay,
                    "websocket_timeout": config.alpaca_websocket_timeout,
                    "order_timeout": config.alpaca_order_timeout,
                    "cache_dir": config.alpaca_cache_dir or "data/alpaca_cache",
                    "cache_ttl": config.alpaca_cache_ttl,
                    "data_feed": config.alpaca_data_feed,
                    "extended_hours": config.alpaca_extended_hours,
                    "max_position_size": config.alpaca_max_position_size,
                    "max_daily_trades": config.alpaca_max_daily_trades,
                    "log_level": config.alpaca_log_level,
                    "log_trades": config.alpaca_log_trades,
                }
                return AlpacaConfigModel(**config_data)
        except Exception as e:
            logger.debug(f"Could not load from unified config: {e}")

        # Fallback to direct environment variable mapping
        config_data = {}

        # Map environment variables to config fields
        for field_name, env_var in cls.ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if field_name in ["use_v2_api", "paper_trading", "extended_hours", "log_trades"]:
                    config_data[field_name] = env_value.lower() in ["true", "1", "yes", "on"]
                elif field_name in [
                    "max_retries",
                    "websocket_timeout",
                    "order_timeout",
                    "cache_ttl",
                    "max_daily_trades",
                ]:
                    config_data[field_name] = int(env_value)
                elif field_name in ["retry_delay", "max_position_size"]:
                    config_data[field_name] = float(env_value)
                else:
                    config_data[field_name] = env_value

        try:
            return AlpacaConfigModel(**config_data)
        except Exception as e:
            logger.exception(f"Failed to create configuration from environment: {e}")
            raise

    @classmethod
    def validate_environment(cls) -> bool:
        """
        Validate that required environment variables are set.

        Returns:
            True if all required variables are set
        """
        # First try to get from unified configuration system
        try:
            from ..core.unified_config import UnifiedConfig

            config = UnifiedConfig()
            if config.alpaca_api_key and config.alpaca_secret_key:
                logger.info("Found Alpaca credentials in unified configuration")
                return True
        except Exception:
            pass

        # Fallback to direct environment variable check
        required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        return True

    @classmethod
    def get_default_config(cls) -> AlpacaConfigModel:
        """
        Get default configuration for development/testing.

        Returns:
            Default AlpacaConfigModel instance
        """
        return AlpacaConfigModel(api_key="your_api_key_here", secret_key="your_secret_key_here")


class AlpacaConfigManager:
    """Manager for Alpaca configuration with validation and caching."""

    def __init__(self) -> None:
        self._config: AlpacaConfigModel | None = None
        self._config_file: Path | None = None

    def load_config(self, config_file: str | None = None) -> AlpacaConfigModel:
        """
        Load configuration from file or environment.

        Args:
            config_file: Optional path to configuration file

        Returns:
            Validated configuration
        """
        if config_file:
            return self._load_from_file(config_file)
        return self._load_from_environment()

    def _load_from_file(self, config_file: str) -> AlpacaConfigModel:
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            # Validate configuration
            config = AlpacaConfigModel(**config_data)
            self._config = config
            self._config_file = config_path

            logger.info(f"Loaded configuration from {config_file}")
            return config

        except Exception as e:
            logger.exception(f"Failed to load configuration from {config_file}: {e}")
            raise

    def _load_from_environment(self) -> AlpacaConfigModel:
        """Load configuration from environment variables."""
        if not AlpacaEnvironmentConfig.validate_environment():
            raise ValueError("Required environment variables are not set")

        try:
            config = AlpacaEnvironmentConfig.from_environment()
            self._config = config

            logger.info("Loaded configuration from environment variables")
            return config

        except Exception as e:
            logger.exception(f"Failed to load configuration from environment: {e}")
            raise

    def save_config(self, config_file: str) -> None:
        """
        Save current configuration to file.

        Args:
            config_file: Path to save configuration
        """
        if not self._config:
            raise ValueError("No configuration loaded")

        import yaml

        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = self._config.dict()
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Saved configuration to {config_file}")

        except Exception as e:
            logger.exception(f"Failed to save configuration to {config_file}: {e}")
            raise

    def get_config(self) -> AlpacaConfigModel:
        """
        Get current configuration.

        Returns:
            Current configuration
        """
        if not self._config:
            raise ValueError("No configuration loaded. Call load_config() first.")
        return self._config

    def update_config(self, **kwargs: Any) -> None:
        """
        Update configuration with new values.

        Args:
            **kwargs: Configuration updates
        """
        if not self._config:
            raise ValueError("No configuration loaded")

        try:
            # Create new config with updates
            current_dict = self._config.dict()
            current_dict.update(kwargs)

            self._config = AlpacaConfigModel(**current_dict)
            logger.info("Configuration updated")

        except Exception as e:
            logger.exception(f"Failed to update configuration: {e}")
            raise

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid
        """
        if not self._config:
            return False

        try:
            # Pydantic validation is automatic, but we can add custom validation here
            if self._config.paper_trading and not self._config.base_url.endswith("paper-api.alpaca.markets"):
                logger.warning("Paper trading enabled but not using paper trading URL")

            return True

        except Exception as e:
            logger.exception(f"Configuration validation failed: {e}")
            return False

    def create_sample_config(self, output_file: str = "alpaca_config_sample.yaml") -> None:
        """
        Create a sample configuration file.

        Args:
            output_file: Output file path
        """
        sample_config = AlpacaConfigModel.get_default_config()

        import yaml

        try:
            config_dict = sample_config.dict()

            # Add comments for documentation
            config_dict["_comment"] = {
                "api_key": "Your Alpaca API key from https://app.alpaca.markets/",
                "secret_key": "Your Alpaca secret key from https://app.alpaca.markets/",
                "paper_trading": "Set to false for live trading (use with caution)",
                "data_feed": 'Use "iex" for free data or "sip" for paid data',
                "max_position_size": "Maximum position size in dollars",
                "max_daily_trades": "Maximum number of trades per day",
            }

            with open(output_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Created sample configuration file: {output_file}")

        except Exception as e:
            logger.exception(f"Failed to create sample configuration: {e}")
            raise


# Convenience functions
def get_alpaca_config(config_file: str | None = None) -> AlpacaConfigModel:
    """
    Get Alpaca configuration from file or environment.

    Args:
        config_file: Optional path to configuration file

    Returns:
        Validated Alpaca configuration
    """
    manager = AlpacaConfigManager()
    return manager.load_config(config_file)


def create_alpaca_config_from_env() -> AlpacaConfigModel:
    """
    Create Alpaca configuration from environment variables.

    Returns:
        Validated Alpaca configuration
    """
    return AlpacaEnvironmentConfig.from_environment()


def validate_alpaca_environment() -> bool:
    """
    Validate that required Alpaca environment variables are set.

    Returns:
        True if environment is properly configured
    """
    return AlpacaEnvironmentConfig.validate_environment()
