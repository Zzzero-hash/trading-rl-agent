"""
Unit tests for the configuration system.

Tests cover:
- Default settings loading
- YAML configuration overlay
- .env file parsing
- Environment variable overrides
- Error handling
- Settings validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import (
    AgentConfig,
    DataConfig,
    ExecutionConfig,
    InfrastructureConfig,
    ModelConfig,
    MonitoringConfig,
    RiskConfig,
    Settings,
    clear_settings_cache,
    get_settings,
    load_settings,
)


class TestDefaultSettings:
    """Test default settings behavior."""

    def test_default_settings_creation(self):
        """Test that default settings are created correctly."""
        settings = Settings()

        # Test environment defaults
        assert settings.environment == "development"
        assert settings.debug is False

        # Test component defaults
        assert settings.data.primary_source == "yfinance"
        assert settings.data.symbols == ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        assert settings.model.type == "cnn_lstm"
        assert settings.agent.agent_type == "sac"
        assert settings.risk.max_position_size == 0.1
        assert settings.execution.broker == "alpaca"
        assert settings.monitoring.log_level == "INFO"
        assert settings.infrastructure.num_workers == 4

    def test_api_keys_default_to_none(self):
        """Test that API keys default to None."""
        settings = Settings()

        assert settings.alpaca_api_key is None
        assert settings.alpaca_secret_key is None
        assert settings.alphavantage_api_key is None
        assert settings.newsapi_key is None
        assert settings.social_api_key is None

    def test_api_key_validation(self):
        """Test API key validation."""
        # Empty string should become None
        settings = Settings(alpaca_api_key="")
        assert settings.alpaca_api_key is None

        # Valid key should remain
        settings = Settings(alpaca_api_key="test_key")
        assert settings.alpaca_api_key == "test_key"


class TestYAMLConfigLoading:
    """Test YAML configuration file loading."""

    def test_load_settings_with_yaml_config(self):
        """Test loading settings from YAML file."""
        config_data = {
            "environment": "production",
            "debug": True,
            "data": {"primary_source": "alpaca", "symbols": ["AAPL", "GOOGL"], "feature_window": 30},
            "model": {"type": "rl", "batch_size": 64, "learning_rate": 0.0001},
            "agent": {"agent_type": "ppo", "ensemble_size": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            settings = load_settings(config_path=config_path)

            # Test overridden values
            assert settings.environment == "production"
            assert settings.debug is True
            assert settings.data.primary_source == "alpaca"
            assert settings.data.symbols == ["AAPL", "GOOGL"]
            assert settings.data.feature_window == 30
            assert settings.model.type == "rl"
            assert settings.model.batch_size == 64
            assert settings.model.learning_rate == 0.0001
            assert settings.agent.agent_type == "ppo"
            assert settings.agent.ensemble_size == 3

            # Test default values are preserved
            assert settings.data.backup_source == "yfinance"
            assert settings.model.algorithm == "sac"
            assert settings.risk.max_position_size == 0.1

        finally:
            config_path.unlink()

    def test_load_settings_with_nonexistent_yaml(self):
        """Test error handling for nonexistent YAML file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_settings(config_path=Path("nonexistent.yaml"))

    def test_load_settings_with_invalid_yaml(self):
        """Test error handling for invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid YAML configuration"):
                load_settings(config_path=config_path)
        finally:
            config_path.unlink()

    def test_load_settings_with_empty_yaml(self):
        """Test loading settings with empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = Path(f.name)

        try:
            settings = load_settings(config_path=config_path)
            # Should use defaults
            assert settings.environment == "development"
            assert settings.data.primary_source == "yfinance"
        finally:
            config_path.unlink()


class TestEnvFileLoading:
    """Test .env file loading."""

    def test_load_settings_with_env_file(self):
        """Test loading settings from .env file."""
        env_content = """
TRADING_RL_AGENT_ENVIRONMENT=production
TRADING_RL_AGENT_DEBUG=true
TRADING_RL_AGENT_ALPACA_API_KEY=test_api_key
TRADING_RL_AGENT_ALPACA_SECRET_KEY=test_secret_key
TRADING_RL_AGENT_NEWSAPI_KEY=news_key
        """.strip()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            env_path = Path(f.name)

        try:
            settings = load_settings(env_file=env_path)

            # Test environment variables are loaded
            assert settings.environment == "production"
            assert settings.debug is True
            assert settings.alpaca_api_key == "test_api_key"
            assert settings.alpaca_secret_key == "test_secret_key"
            assert settings.newsapi_key == "news_key"

        finally:
            env_path.unlink()

    def test_load_settings_with_nonexistent_env_file(self):
        """Test error handling for nonexistent .env file."""
        with pytest.raises(FileNotFoundError, match="Environment file not found"):
            load_settings(env_file=Path("nonexistent.env"))

    def test_env_file_overrides_yaml(self):
        """Test that environment variables override YAML config."""
        config_data = {"environment": "development", "debug": False, "alpaca_api_key": "yaml_key"}

        env_content = """
TRADING_RL_AGENT_ENVIRONMENT=production
TRADING_RL_AGENT_DEBUG=true
TRADING_RL_AGENT_ALPACA_API_KEY=env_key
        """.strip()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as config_f:
            yaml.dump(config_data, config_f)
            config_path = Path(config_f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as env_f:
            env_f.write(env_content)
            env_path = Path(env_f.name)

        try:
            settings = load_settings(config_path=config_path, env_file=env_path)

            # Environment variables should override YAML
            assert settings.environment == "production"
            assert settings.debug is True
            assert settings.alpaca_api_key == "env_key"

        finally:
            config_path.unlink()
            env_path.unlink()


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_environment_variable_overrides(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "TRADING_RL_AGENT_ENVIRONMENT": "staging",
                "TRADING_RL_AGENT_DEBUG": "true",
                "TRADING_RL_AGENT_ALPACA_API_KEY": "env_api_key",
                "TRADING_RL_AGENT_DATA__PRIMARY_SOURCE": "alphavantage",
                "TRADING_RL_AGENT_MODEL__BATCH_SIZE": "128",
            },
        ):
            settings = load_settings()

            assert settings.environment == "staging"
            assert settings.debug is True
            assert settings.alpaca_api_key == "env_api_key"
            assert settings.data.primary_source == "alphavantage"
            assert settings.model.batch_size == 128

    def test_environment_variable_case_insensitive(self):
        """Test that environment variables are case insensitive."""
        with patch.dict(os.environ, {"trading_rl_agent_environment": "production", "TRADING_RL_AGENT_DEBUG": "true"}):
            settings = load_settings()

            assert settings.environment == "production"
            assert settings.debug is True


class TestSettingsCaching:
    """Test settings caching behavior."""

    def test_settings_caching(self):
        """Test that settings are cached globally."""
        # Clear cache first
        clear_settings_cache()

        # Load settings first time
        settings1 = load_settings()

        # Load settings second time (should return cached)
        settings2 = load_settings()

        # Should be the same object
        assert settings1 is settings2

    def test_get_settings_returns_cached(self):
        """Test that get_settings returns cached settings."""
        # Clear cache first
        clear_settings_cache()

        # Load settings
        settings1 = load_settings()

        # Get settings (should return cached)
        settings2 = get_settings()

        # Should be the same object
        assert settings1 is settings2

    def test_clear_settings_cache(self):
        """Test that cache clearing works."""
        # Load settings
        settings1 = load_settings()

        # Clear cache
        clear_settings_cache()

        # Load settings again (should be new object)
        settings2 = load_settings()

        # Should be different objects
        assert settings1 is not settings2


class TestAPIKeyHandling:
    """Test API key handling and validation."""

    def test_get_api_credentials_alpaca(self):
        """Test getting Alpaca API credentials."""
        settings = Settings(
            alpaca_api_key="test_key", alpaca_secret_key="test_secret", alpaca_base_url="https://test.alpaca.markets"
        )

        credentials = settings.get_api_credentials("alpaca")

        assert credentials["api_key"] == "test_key"
        assert credentials["secret_key"] == "test_secret"
        assert credentials["base_url"] == "https://test.alpaca.markets"

    def test_get_api_credentials_alphavantage(self):
        """Test getting Alpha Vantage API credentials."""
        settings = Settings(alphavantage_api_key="av_key")

        credentials = settings.get_api_credentials("alphavantage")

        assert credentials["api_key"] == "av_key"

    def test_get_api_credentials_unknown_exchange(self):
        """Test getting credentials for unknown exchange."""
        settings = Settings()

        credentials = settings.get_api_credentials("unknown")

        assert credentials == {}

    def test_get_api_credentials_missing_keys(self):
        """Test getting credentials when keys are missing."""
        settings = Settings()

        credentials = settings.get_api_credentials("alpaca")

        assert credentials == {}


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_configuration(self):
        """Test that valid configuration loads without errors."""
        config_data = {
            "environment": "production",
            "data": {"symbols": ["AAPL", "GOOGL"], "feature_window": 30},
            "model": {"batch_size": 64, "learning_rate": 0.001},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            settings = load_settings(config_path=config_path)
            assert settings.environment == "production"
            assert settings.data.symbols == ["AAPL", "GOOGL"]
            assert settings.model.batch_size == 64
        finally:
            config_path.unlink()

    def test_invalid_configuration_type(self):
        """Test that invalid configuration types raise errors."""
        config_data = {"data": {"feature_window": "invalid_string"}}  # Should be int

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_settings(config_path=config_path)
        finally:
            config_path.unlink()


class TestComponentConfigs:
    """Test individual component configurations."""

    def test_data_config(self):
        """Test DataConfig validation and defaults."""
        data_config = DataConfig()

        assert data_config.primary_source == "yfinance"
        assert data_config.symbols == ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        assert data_config.feature_window == 50
        assert data_config.technical_indicators is True

    def test_model_config(self):
        """Test ModelConfig validation and defaults."""
        model_config = ModelConfig()

        assert model_config.type == "cnn_lstm"
        assert model_config.cnn_filters == [64, 128, 256]
        assert model_config.lstm_units == 128
        assert model_config.batch_size == 32

    def test_agent_config(self):
        """Test AgentConfig validation and defaults."""
        agent_config = AgentConfig()

        assert agent_config.agent_type == "sac"
        assert agent_config.ensemble_size == 1
        assert agent_config.eval_frequency == 10000

    def test_risk_config(self):
        """Test RiskConfig validation and defaults."""
        risk_config = RiskConfig()

        assert risk_config.max_position_size == 0.1
        assert risk_config.max_leverage == 1.0
        assert risk_config.var_confidence_level == 0.05

    def test_execution_config(self):
        """Test ExecutionConfig validation and defaults."""
        execution_config = ExecutionConfig()

        assert execution_config.broker == "alpaca"
        assert execution_config.paper_trading is True
        assert execution_config.order_timeout == 60

    def test_monitoring_config(self):
        """Test MonitoringConfig validation and defaults."""
        monitoring_config = MonitoringConfig()

        assert monitoring_config.log_level == "INFO"
        assert monitoring_config.mlflow_enabled is True
        assert monitoring_config.metrics_frequency == 300

    def test_infrastructure_config(self):
        """Test InfrastructureConfig validation and defaults."""
        infrastructure_config = InfrastructureConfig()

        assert infrastructure_config.distributed is False
        assert infrastructure_config.num_workers == 4
        assert infrastructure_config.gpu_enabled is True


if __name__ == "__main__":
    pytest.main([__file__])
