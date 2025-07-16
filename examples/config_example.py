#!/usr/bin/env python3
"""
Example usage of the Trading RL Agent configuration system.

This script demonstrates:
1. Loading default settings
2. Loading from YAML configuration file
3. Loading from .env file
4. Environment variable overrides
5. Accessing configuration values
6. API credential handling
"""

import os
from pathlib import Path

from config import clear_settings_cache, get_settings, load_settings


def example_default_settings() -> None:
    """Example: Load default settings."""
    print("=== Default Settings Example ===")

    # Clear any cached settings
    clear_settings_cache()

    # Load default settings
    settings = load_settings()

    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Data source: {settings.data.primary_source}")
    print(f"Model type: {settings.model.type}")
    print(f"Agent type: {settings.agent.agent_type}")
    print(f"Risk max position: {settings.risk.max_position_size}")
    print(f"Execution broker: {settings.execution.broker}")
    print(f"Monitoring log level: {settings.monitoring.log_level}")
    print()


def example_yaml_config() -> None:
    """Example: Load from YAML configuration file."""
    print("=== YAML Configuration Example ===")

    # Clear cache
    clear_settings_cache()

    # Load from test YAML file
    config_path = Path("tests/test_configs/good_config.yaml")
    if config_path.exists():
        settings = load_settings(config_path=config_path)

        print(f"Environment: {settings.environment}")
        print(f"Debug mode: {settings.debug}")
        print(f"Data source: {settings.data.primary_source}")
        print(f"Data symbols: {settings.data.symbols}")
        print(f"Model batch size: {settings.model.batch_size}")
        print(f"Agent ensemble size: {settings.agent.ensemble_size}")
        print(f"Risk max position: {settings.risk.max_position_size}")
        print(f"Execution paper trading: {settings.execution.paper_trading}")
        print(f"Infrastructure workers: {settings.infrastructure.num_workers}")
        print()
    else:
        print(f"Test config file not found: {config_path}")
        print()


def example_env_file() -> None:
    """Example: Load from .env file."""
    print("=== Environment File Example ===")

    # Clear cache
    clear_settings_cache()

    # Load from test .env file
    env_path = Path("tests/test_configs/test.env")
    if env_path.exists():
        settings = load_settings(env_file=env_path)

        print(f"Environment: {settings.environment}")
        print(f"Debug mode: {settings.debug}")
        print(f"Alpaca API key: {settings.alpaca_api_key[:10]}..." if settings.alpaca_api_key else "None")
        print(f"Alpaca secret key: {settings.alpaca_secret_key[:10]}..." if settings.alpaca_secret_key else "None")
        print(f"News API key: {settings.newsapi_key[:10]}..." if settings.newsapi_key else "None")
        print(f"Data source: {settings.data.primary_source}")
        print(f"Model batch size: {settings.model.batch_size}")
        print(f"Agent type: {settings.agent.agent_type}")
        print(f"Risk max position: {settings.risk.max_position_size}")
        print(f"Monitoring log level: {settings.monitoring.log_level}")
        print()
    else:
        print(f"Test .env file not found: {env_path}")
        print()


def example_environment_variables() -> None:
    """Example: Environment variable overrides."""
    print("=== Environment Variable Overrides Example ===")

    # Set environment variables
    os.environ["TRADING_RL_AGENT_ENVIRONMENT"] = "production"
    os.environ["TRADING_RL_AGENT_DEBUG"] = "false"
    os.environ["TRADING_RL_AGENT_ALPACA_API_KEY"] = "env_api_key_12345"
    os.environ["TRADING_RL_AGENT_DATA__PRIMARY_SOURCE"] = "yfinance"
    os.environ["TRADING_RL_AGENT_MODEL__BATCH_SIZE"] = "256"
    os.environ["TRADING_RL_AGENT_AGENT__AGENT_TYPE"] = "td3"

    # Clear cache and load settings
    clear_settings_cache()
    settings = load_settings()

    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Alpaca API key: {settings.alpaca_api_key}")
    print(f"Data source: {settings.data.primary_source}")
    print(f"Model batch size: {settings.model.batch_size}")
    print(f"Agent type: {settings.agent.agent_type}")
    print()

    # Clean up environment variables
    for key in [
        "TRADING_RL_AGENT_ENVIRONMENT",
        "TRADING_RL_AGENT_DEBUG",
        "TRADING_RL_AGENT_ALPACA_API_KEY",
        "TRADING_RL_AGENT_DATA__PRIMARY_SOURCE",
        "TRADING_RL_AGENT_MODEL__BATCH_SIZE",
        "TRADING_RL_AGENT_AGENT__AGENT_TYPE",
    ]:
        os.environ.pop(key, None)


def example_api_credentials() -> None:
    """Example: API credential handling."""
    print("=== API Credentials Example ===")

    # Set up settings with API keys
    settings = get_settings()
    settings.alpaca_api_key = "test_alpaca_key"
    settings.alpaca_secret_key = "test_alpaca_secret"
    settings.alpaca_base_url = "https://test.alpaca.markets"
    settings.alphavantage_api_key = "test_av_key"

    # Get Alpaca credentials
    alpaca_creds = settings.get_api_credentials("alpaca")
    print(f"Alpaca credentials: {alpaca_creds}")

    # Get Alpha Vantage credentials
    av_creds = settings.get_api_credentials("alphavantage")
    print(f"Alpha Vantage credentials: {av_creds}")

    # Get credentials for unknown exchange
    unknown_creds = settings.get_api_credentials("unknown")
    print(f"Unknown exchange credentials: {unknown_creds}")
    print()


def example_combined_config() -> None:
    """Example: Combined configuration loading."""
    print("=== Combined Configuration Example ===")

    # Clear cache
    clear_settings_cache()

    # Load from both YAML and .env files
    config_path = Path("tests/test_configs/good_config.yaml")
    env_path = Path("tests/test_configs/test.env")

    if config_path.exists() and env_path.exists():
        settings = load_settings(config_path=config_path, env_file=env_path)

        print("Configuration loaded from YAML + .env files:")
        print(f"Environment: {settings.environment}")
        print(f"Debug mode: {settings.debug}")
        print(f"Data source: {settings.data.primary_source}")
        print(f"Model batch size: {settings.model.batch_size}")
        print(f"Agent type: {settings.agent.agent_type}")
        print(f"Alpaca API key: {settings.alpaca_api_key[:10]}..." if settings.alpaca_api_key else "None")
        print()
    else:
        print("Test files not found for combined example")
        print()


def example_settings_caching() -> None:
    """Example: Settings caching behavior."""
    print("=== Settings Caching Example ===")

    # Clear cache
    clear_settings_cache()

    # Load settings first time
    settings1 = load_settings()
    print(f"First load - Environment: {settings1.environment}")

    # Load settings second time (should return cached)
    settings2 = load_settings()
    print(f"Second load - Environment: {settings2.environment}")

    # Check if same object
    print(f"Same object: {settings1 is settings2}")

    # Get settings using get_settings()
    settings3 = get_settings()
    print(f"get_settings() - Environment: {settings3.environment}")
    print(f"Same object: {settings1 is settings3}")
    print()


def main() -> None:
    """Run all examples."""
    print("Trading RL Agent Configuration System Examples")
    print("=" * 50)
    print()

    try:
        example_default_settings()
        example_yaml_config()
        example_env_file()
        example_environment_variables()
        example_api_credentials()
        example_combined_config()
        example_settings_caching()

        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
