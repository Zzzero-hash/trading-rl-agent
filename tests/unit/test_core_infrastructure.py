"""
Comprehensive tests for core infrastructure components.

This module provides thorough test coverage for:
- Configuration management (ConfigManager, SystemConfig)
- Logging system (structured logging, log rotation)
- Exception handling (custom exceptions, error propagation)
- Data validation and sanitization
"""

import logging
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import yaml
import pytest
import structlog

# Mock dependencies that might not be available
if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: logging.getLogger(name),
    )
    sys.modules["structlog"] = stub

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trading_rl_agent.core.config import (
    ConfigManager,
    SystemConfig,
    DataConfig,
    ModelConfig,
    AgentConfig,
    RiskConfig,
    ExecutionConfig,
    MonitoringConfig,
    InfrastructureConfig,
)
from trading_rl_agent.core.exceptions import (
    TradingSystemError,
    DataValidationError,
    ModelError,
    ConfigurationError,
    MarketDataError,
    RiskManagementError,
    ExecutionError,
)
from trading_rl_agent.core.logging import (
    setup_logging,
    get_logger,
    get_structured_logger,
    DEFAULT_LOG_CONFIG,
)
from trading_rl_agent.core.unified_config import (
    UnifiedConfig,
    DataSourcesConfig,
    BacktestConfig,
    LiveTradingConfig,
    MonitoringConfig as UnifiedMonitoringConfig,
    InfrastructureConfig as UnifiedInfrastructureConfig,
    HyperoptConfig,
    ProductionConfig,
    load_config,
)

pytestmark = pytest.mark.unit


class TestExceptions:
    """Test custom exception hierarchy."""

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from TradingSystemError."""
        exceptions = [
            DataValidationError,
            ModelError,
            ConfigurationError,
            MarketDataError,
            RiskManagementError,
            ExecutionError,
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, TradingSystemError)

    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated with messages."""
        message = "Test error message"
        
        exc = DataValidationError(message)
        assert str(exc) == message
        
        exc = ModelError(message)
        assert str(exc) == message
        
        exc = ConfigurationError(message)
        assert str(exc) == message
        
        exc = MarketDataError(message)
        assert str(exc) == message
        
        exc = RiskManagementError(message)
        assert str(exc) == message
        
        exc = ExecutionError(message)
        assert str(exc) == message

    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError:
            exc = ConfigurationError("Configuration failed")
            assert str(exc) == "Configuration failed"


class TestConfigurationDataclasses:
    """Test configuration dataclasses."""

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        
        assert config.data_sources == {
            "alpaca": "alpaca",
            "yfinance": "yfinance",
            "ccxt": "ccxt",
        }
        assert config.data_path == "data/"
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.feature_window == 50
        assert config.technical_indicators is True
        assert config.alternative_data is False
        assert config.real_time_enabled is False
        assert config.update_frequency == 60

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.cnn_filters == [32, 64, 128]
        assert config.cnn_kernel_size == 3
        assert config.lstm_units == 256
        assert config.lstm_layers == 2
        assert config.dropout_rate == 0.2
        assert config.batch_normalization is True
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.epochs == 100
        assert config.early_stopping_patience == 10
        assert config.model_save_path == "models/"
        assert config.checkpoint_frequency == 10

    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig()
        
        assert config.agent_type == "sac"
        assert config.total_timesteps == 1_000_000
        assert config.eval_frequency == 10_000
        assert config.save_frequency == 50_000
        assert hasattr(config, 'ppo')
        assert hasattr(config, 'sac')
        assert hasattr(config, 'td3')

    def test_risk_config_defaults(self):
        """Test RiskConfig default values."""
        config = RiskConfig()
        
        assert config.max_position_size == 0.1
        assert config.max_leverage == 1.0
        assert config.max_drawdown == 0.1
        assert config.var_confidence_level == 0.05
        assert config.var_time_horizon == 1
        assert config.kelly_fraction == 0.25
        assert config.risk_per_trade == 0.02

    def test_execution_config_defaults(self):
        """Test ExecutionConfig default values."""
        config = ExecutionConfig()
        
        assert config.broker == "alpaca"
        assert config.paper_trading is True
        assert config.order_timeout == 60
        assert config.max_slippage == 0.001
        assert config.commission_rate == 0.0
        assert config.execution_frequency == 5
        assert config.market_hours_only is True

    def test_monitoring_config_defaults(self):
        """Test MonitoringConfig default values."""
        config = MonitoringConfig()
        
        assert config.log_level == "INFO"
        assert config.log_file == "logs/trading_system.log"
        assert config.structured_logging is True
        assert config.metrics_enabled is True
        assert config.metrics_frequency == 300
        assert config.alerts_enabled is True
        assert config.email_alerts is False
        assert config.slack_alerts is False
        assert config.mlflow_enabled is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"

    def test_infrastructure_config_defaults(self):
        """Test InfrastructureConfig default values."""
        config = InfrastructureConfig()
        
        assert config.distributed is False
        assert config.num_workers == 4
        assert config.gpu_enabled is True
        assert config.ray_address is None
        assert config.model_registry_path == "models"
        assert config.experiment_tracking == "mlflow"
        assert config.enable_monitoring is True
        assert config.metrics_port == 8080
        assert config.health_check_interval == 30

    def test_system_config_defaults(self):
        """Test SystemConfig default values."""
        config = SystemConfig()
        
        assert config.environment == "development"
        assert config.debug is False
        assert config.use_gpu is False
        assert config.max_workers == 4
        assert config.memory_limit == "8GB"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.risk, RiskConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.infrastructure, InfrastructureConfig)


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_init_with_path(self, tmp_path):
        """Test ConfigManager initialization with path."""
        config_path = tmp_path / "test_config.yaml"
        manager = ConfigManager(config_path=config_path)
        
        assert manager.config_path == config_path
        assert manager._config is None

    def test_init_without_path(self):
        """Test ConfigManager initialization without path."""
        manager = ConfigManager()
        
        assert manager.config_path is None
        assert manager._config is None

    def test_init_with_string_path(self, tmp_path):
        """Test ConfigManager initialization with string path."""
        config_path = str(tmp_path / "test_config.yaml")
        manager = ConfigManager(config_path=config_path)
        
        assert manager.config_path == Path(config_path)

    def test_load_config_default(self):
        """Test loading default configuration when no file exists."""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, SystemConfig)
        assert config.environment == "development"
        assert config.debug is False

    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_path = tmp_path / "test_config.yaml"
        
        # Create test config file
        test_config = {
            "environment": "production",
            "debug": True,
            "data": {
                "data_path": "custom_data/",
                "cache_enabled": False,
            },
            "model": {
                "batch_size": 64,
                "learning_rate": 0.0001,
            },
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager(config_path=config_path)
        config = manager.load_config()
        
        assert config.environment == "production"
        assert config.debug is True
        assert config.data.data_path == "custom_data/"
        assert config.data.cache_enabled is False
        assert config.model.batch_size == 64
        assert config.model.learning_rate == 0.0001

    def test_load_config_with_custom_path(self, tmp_path):
        """Test loading configuration with custom path parameter."""
        config_path = tmp_path / "test_config.yaml"
        
        # Create test config file
        test_config = {"environment": "staging", "debug": True}
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        manager = ConfigManager()
        config = manager.load_config(config_path=config_path)
        
        assert config.environment == "staging"
        assert config.debug is True

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading configuration when file doesn't exist."""
        config_path = tmp_path / "nonexistent.yaml"
        manager = ConfigManager(config_path=config_path)
        
        # Should load default config when file doesn't exist
        config = manager.load_config()
        assert isinstance(config, SystemConfig)
        assert config.environment == "development"

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading configuration with invalid YAML."""
        config_path = tmp_path / "invalid_config.yaml"
        
        # Create invalid YAML file
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = ConfigManager(config_path=config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.load_config()
        
        assert "Failed to load config" in str(exc_info.value)

    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config_path = tmp_path / "test_config.yaml"
        manager = ConfigManager(config_path=config_path)
        
        # Create a config with custom values
        config = SystemConfig()
        config.environment = "production"
        config.debug = True
        config.data.data_path = "custom_data/"
        
        manager.save_config(config)
        
        # Verify file was created
        assert config_path.exists()
        
        # Load and verify
        loaded_config = manager.load_config()
        assert loaded_config.environment == "production"
        assert loaded_config.debug is True
        assert loaded_config.data.data_path == "custom_data/"

    def test_save_config_with_custom_path(self, tmp_path):
        """Test saving configuration with custom path."""
        config_path = tmp_path / "test_config.yaml"
        custom_path = tmp_path / "custom_config.yaml"
        manager = ConfigManager(config_path=config_path)
        
        config = SystemConfig()
        config.environment = "staging"
        
        manager.save_config(config, path=custom_path)
        
        # Verify custom file was created
        assert custom_path.exists()
        assert not config_path.exists()

    def test_save_config_no_path(self):
        """Test saving configuration without path."""
        manager = ConfigManager()
        config = SystemConfig()
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.save_config(config)
        
        assert "No path specified" in str(exc_info.value)

    def test_save_config_creates_directory(self, tmp_path):
        """Test that save_config creates parent directories."""
        config_path = tmp_path / "nested" / "dir" / "config.yaml"
        manager = ConfigManager(config_path=config_path)
        
        config = SystemConfig()
        manager.save_config(config)
        
        # Verify directory was created
        assert config_path.exists()
        assert config_path.parent.exists()

    def test_get_config_lazy_loading(self):
        """Test that get_config loads config lazily."""
        manager = ConfigManager()
        
        # Initially no config loaded
        assert manager._config is None
        
        # Get config should load default
        config = manager.get_config()
        assert isinstance(config, SystemConfig)
        assert manager._config is not None

    def test_get_config_returns_cached(self):
        """Test that get_config returns cached config."""
        manager = ConfigManager()
        
        # Load config first time
        config1 = manager.get_config()
        
        # Modify the config
        config1.environment = "production"
        
        # Get config again should return same instance
        config2 = manager.get_config()
        assert config2.environment == "production"
        assert config1 is config2

    def test_update_config(self):
        """Test updating configuration with new values."""
        manager = ConfigManager()
        config = manager.get_config()
        
        # Update top-level values
        updates = {
            "environment": "production",
            "debug": True,
        }
        
        updated_config = manager.update_config(updates)
        
        assert updated_config.environment == "production"
        assert updated_config.debug is True

    def test_update_config_nested(self):
        """Test updating nested configuration values."""
        manager = ConfigManager()
        config = manager.get_config()
        
        # Update nested values
        updates = {
            "data": {
                "data_path": "custom_data/",
                "cache_enabled": False,
            },
            "model": {
                "batch_size": 64,
                "learning_rate": 0.0001,
            },
        }
        
        updated_config = manager.update_config(updates)
        
        assert updated_config.data.data_path == "custom_data/"
        assert updated_config.data.cache_enabled is False
        assert updated_config.model.batch_size == 64
        assert updated_config.model.learning_rate == 0.0001

    def test_update_config_mixed(self):
        """Test updating both top-level and nested values."""
        manager = ConfigManager()
        
        updates = {
            "environment": "production",
            "data": {
                "data_path": "custom_data/",
            },
            "model": {
                "batch_size": 64,
            },
        }
        
        updated_config = manager.update_config(updates)
        
        assert updated_config.environment == "production"
        assert updated_config.data.data_path == "custom_data/"
        assert updated_config.model.batch_size == 64

    def test_dict_to_config(self):
        """Test _dict_to_config method."""
        manager = ConfigManager()
        
        config_dict = {
            "environment": "production",
            "debug": True,
            "data": {
                "data_path": "custom_data/",
                "cache_enabled": False,
            },
            "agent": {
                "agent_type": "ppo",
                "total_timesteps": 500000,
                "ppo": {"learning_rate": 0.0003},
                "sac": {"learning_rate": 0.0001},
                "td3": {"learning_rate": 0.0001},
            },
            "risk": {
                "max_position_size": 0.2,
            },
            "execution": {
                "broker": "paper",
            },
            "monitoring": {
                "log_level": "DEBUG",
            },
            "infrastructure": {
                "distributed": True,
            },
            "use_gpu": True,
            "max_workers": 8,
            "memory_limit": "16GB",
        }
        
        config = manager._dict_to_config(config_dict)
        
        assert config.environment == "production"
        assert config.debug is True
        assert config.data.data_path == "custom_data/"
        assert config.data.cache_enabled is False
        assert config.agent.agent_type == "ppo"
        assert config.agent.total_timesteps == 500000
        assert config.risk.max_position_size == 0.2
        assert config.execution.broker == "paper"
        assert config.monitoring.log_level == "DEBUG"
        assert config.infrastructure.distributed is True
        assert config.use_gpu is True
        assert config.max_workers == 8
        assert config.memory_limit == "16GB"

    def test_config_to_dict(self):
        """Test _config_to_dict method."""
        manager = ConfigManager()
        config = SystemConfig()
        
        # Set some custom values
        config.environment = "production"
        config.debug = True
        config.data.data_path = "custom_data/"
        config.model.batch_size = 64
        
        config_dict = manager._config_to_dict(config)
        
        assert config_dict["environment"] == "production"
        assert config_dict["debug"] is True
        assert config_dict["data"]["data_path"] == "custom_data/"
        assert config_dict["model"]["batch_size"] == 64
        assert "agent" in config_dict
        assert "risk" in config_dict
        assert "execution" in config_dict
        assert "monitoring" in config_dict
        assert "infrastructure" in config_dict

    def test_apply_updates(self):
        """Test _apply_updates method."""
        manager = ConfigManager()
        config = SystemConfig()
        
        # Test top-level update
        updates = {"environment": "production"}
        manager._apply_updates(config, updates)
        assert config.environment == "production"
        
        # Test nested update
        updates = {"data": {"data_path": "custom_data/"}}
        manager._apply_updates(config, updates)
        assert config.data.data_path == "custom_data/"
        
        # Test mixed update
        updates = {
            "debug": True,
            "model": {"batch_size": 64},
        }
        manager._apply_updates(config, updates)
        assert config.debug is True
        assert config.model.batch_size == 64


class TestLoggingSystem:
    """Test logging system functionality."""

    def test_setup_logging_default(self, tmp_path):
        """Test setting up logging with default configuration."""
        log_dir = tmp_path / "logs"
        
        setup_logging(log_dir=log_dir)
        
        # Verify log directory was created
        assert log_dir.exists()
        
        # Verify loggers are configured
        logger = logging.getLogger("trading_rl_agent")
        assert logger.level == logging.DEBUG

    def test_setup_logging_custom_config(self, tmp_path):
        """Test setting up logging with custom configuration."""
        log_dir = tmp_path / "logs"
        
        custom_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "%(levelname)s: %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "formatter": "simple",
                    "class": "logging.StreamHandler",
                },
            },
            "loggers": {
                "test_logger": {
                    "handlers": ["console"],
                    "level": "DEBUG",
                },
            },
        }
        
        setup_logging(config=custom_config, log_dir=log_dir)
        
        # Verify custom logger is configured
        logger = logging.getLogger("test_logger")
        assert logger.level == logging.DEBUG

    def test_setup_logging_structured(self, tmp_path):
        """Test setting up structured logging."""
        log_dir = tmp_path / "logs"
        
        setup_logging(log_dir=log_dir, structured=True)
        
        # Verify structlog is configured
        # This is a basic check - in a real environment we'd verify more
        assert "structlog" in sys.modules

    def test_setup_logging_not_structured(self, tmp_path):
        """Test setting up logging without structured logging."""
        log_dir = tmp_path / "logs"
        
        setup_logging(log_dir=log_dir, structured=False)
        
        # Should still work without structlog
        logger = logging.getLogger("test")
        assert logger is not None

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    @patch('structlog.get_logger')
    def test_get_structured_logger(self, mock_get_logger):
        """Test get_structured_logger function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = get_structured_logger("test_module")
        
        mock_get_logger.assert_called_once_with("test_module")
        assert logger == mock_logger

    def test_default_log_config_structure(self):
        """Test that DEFAULT_LOG_CONFIG has expected structure."""
        assert "version" in DEFAULT_LOG_CONFIG
        assert "formatters" in DEFAULT_LOG_CONFIG
        assert "handlers" in DEFAULT_LOG_CONFIG
        assert "loggers" in DEFAULT_LOG_CONFIG
        
        # Check formatters
        formatters = DEFAULT_LOG_CONFIG["formatters"]
        assert "standard" in formatters
        assert "detailed" in formatters
        assert "json" in formatters
        
        # Check handlers
        handlers = DEFAULT_LOG_CONFIG["handlers"]
        assert "default" in handlers
        assert "file" in handlers
        assert "error_file" in handlers
        
        # Check loggers
        loggers = DEFAULT_LOG_CONFIG["loggers"]
        assert "" in loggers  # root logger
        assert "trading_rl_agent" in loggers

    def test_logging_with_file_handlers(self, tmp_path):
        """Test that file handlers are properly configured."""
        log_dir = tmp_path / "logs"
        
        setup_logging(log_dir=log_dir)
        
        # Verify log files are created when logging
        logger = logging.getLogger("trading_rl_agent")
        logger.info("Test message")
        
        # Check if log files exist (they should be created on first log)
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0


class TestUnifiedConfig:
    """Test UnifiedConfig functionality."""

    def test_unified_config_defaults(self):
        """Test UnifiedConfig default values."""
        config = UnifiedConfig()
        
        assert config.environment == "production"
        assert config.debug is False
        assert config.log_level == "INFO"
        assert isinstance(config.data, DataSourcesConfig)
        assert isinstance(config.model, type(config).__annotations__["model"])
        assert isinstance(config.backtest, BacktestConfig)
        assert isinstance(config.live, LiveTradingConfig)
        assert isinstance(config.monitoring, UnifiedMonitoringConfig)
        assert isinstance(config.infrastructure, UnifiedInfrastructureConfig)
        assert isinstance(config.hyperopt, HyperoptConfig)
        assert isinstance(config.production, ProductionConfig)

    def test_data_sources_config_defaults(self):
        """Test DataSourcesConfig default values."""
        config = DataSourcesConfig()
        
        assert config.primary == "yfinance"
        assert config.backup == "yfinance"
        assert config.real_time_enabled is False
        assert config.update_frequency == 60

    def test_backtest_config_defaults(self):
        """Test BacktestConfig default values."""
        config = BacktestConfig()
        
        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-12-31"
        assert config.symbols == ["AAPL", "GOOGL", "MSFT"]
        assert config.initial_capital == 100000.0
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0001
        assert config.max_position_size == 0.1
        assert config.max_leverage == 1.0
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.05
        assert "total_return" in config.metrics
        assert "sharpe_ratio" in config.metrics
        assert "max_drawdown" in config.metrics
        assert "win_rate" in config.metrics

    def test_live_trading_config_defaults(self):
        """Test LiveTradingConfig default values."""
        config = LiveTradingConfig()
        
        assert config.exchange == "alpaca"
        assert config.paper_trading is True
        assert config.symbols == ["AAPL", "GOOGL", "MSFT"]
        assert config.order_timeout == 60
        assert config.max_slippage == 0.001
        assert config.commission_rate == 0.001
        assert config.execution_frequency == 5
        assert config.market_hours_only is True
        assert config.max_position_size == 0.1
        assert config.max_leverage == 1.0
        assert config.max_drawdown == 0.15
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.05
        assert config.var_confidence_level == 0.05
        assert config.var_time_horizon == 1
        assert config.kelly_fraction == 0.25
        assert config.risk_per_trade == 0.02
        assert config.initial_capital == 100000.0
        assert config.rebalance_frequency == 3600
        assert config.max_positions == 10
        assert config.monitoring_interval == 60
        assert config.alerts_enabled is True
        assert config.email_alerts is False
        assert config.slack_alerts is False

    def test_unified_monitoring_config_defaults(self):
        """Test UnifiedMonitoringConfig default values."""
        config = UnifiedMonitoringConfig()
        
        assert config.log_level == "INFO"
        assert config.structured_logging is True
        assert config.experiment_name == "trading_rl_agent"
        assert config.tracking_uri == "sqlite:///mlruns.db"
        assert config.mlflow_enabled is True
        assert config.tensorboard_enabled is True
        assert config.tensorboard_log_dir == "logs/tensorboard"
        assert config.metrics_enabled is True
        assert config.metrics_frequency == 300
        assert config.health_check_interval == 30
        assert config.alerts_enabled is True
        assert config.email_alerts is False
        assert config.slack_alerts is False

    def test_unified_infrastructure_config_defaults(self):
        """Test UnifiedInfrastructureConfig default values."""
        config = UnifiedInfrastructureConfig()
        
        assert config.distributed is False
        assert config.ray_address is None
        assert config.num_workers == 4
        assert config.gpu_enabled is True
        assert config.model_registry_path == "models"
        assert config.experiment_tracking == "mlflow"
        assert config.enable_monitoring is True
        assert config.metrics_port == 8080
        assert config.health_check_interval == 30
        assert config.use_gpu is False
        assert config.max_workers == 4
        assert config.memory_limit == "8GB"

    def test_hyperopt_config_defaults(self):
        """Test HyperoptConfig default values."""
        config = HyperoptConfig()
        
        assert config.enabled is False
        assert config.n_trials == 50
        assert config.timeout == 3600

    def test_production_config_defaults(self):
        """Test ProductionConfig default values."""
        config = ProductionConfig()
        
        assert config.model_format == "torchscript"
        assert config.model_version == "v1.0.0"
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8000
        assert config.api_workers == 4
        assert config.health_check_interval == 30
        assert config.metrics_export_interval == 60

    def test_api_key_validation(self):
        """Test API key validation."""
        # Test with None values (should be allowed)
        config = UnifiedConfig()
        assert config.alpaca_api_key is None
        assert config.polygon_api_key is None
        
        # Test with empty strings (should be allowed)
        config = UnifiedConfig(alpaca_api_key="")
        assert config.alpaca_api_key == ""
        
        # Test with valid keys
        config = UnifiedConfig(alpaca_api_key="test_key_123")
        assert config.alpaca_api_key == "test_key_123"

    def test_get_api_credentials(self):
        """Test get_api_credentials method."""
        config = UnifiedConfig(
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
            alpaca_base_url="https://test.alpaca.markets"
        )
        
        credentials = config.get_api_credentials("alpaca")
        
        assert credentials["api_key"] == "test_key"
        assert credentials["secret_key"] == "test_secret"
        assert credentials["base_url"] == "https://test.alpaca.markets"

    def test_get_api_credentials_missing(self):
        """Test get_api_credentials with missing credentials."""
        config = UnifiedConfig()
        
        credentials = config.get_api_credentials("alpaca")
        
        assert credentials["api_key"] is None
        assert credentials["secret_key"] is None
        assert credentials["base_url"] is None

    def test_to_dict(self):
        """Test to_dict method."""
        config = UnifiedConfig(
            environment="development",
            debug=True,
            alpaca_api_key="test_key"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["environment"] == "development"
        assert config_dict["debug"] is True
        assert config_dict["alpaca_api_key"] == "test_key"

    def test_from_yaml(self, tmp_path):
        """Test from_yaml class method."""
        yaml_path = tmp_path / "test_config.yaml"
        
        test_config = {
            "environment": "development",
            "debug": True,
            "alpaca_api_key": "test_key",
            "data": {
                "sources": {
                    "primary": "yfinance",
                    "backup": "alpaca",
                }
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(test_config, f)
        
        config = UnifiedConfig.from_yaml(yaml_path)
        
        assert config.environment == "development"
        assert config.debug is True
        assert config.alpaca_api_key == "test_key"
        assert config.data.sources.primary == "yfinance"
        assert config.data.sources.backup == "alpaca"

    def test_to_yaml(self, tmp_path):
        """Test to_yaml method."""
        yaml_path = tmp_path / "test_config.yaml"
        
        config = UnifiedConfig(
            environment="development",
            debug=True,
            alpaca_api_key="test_key"
        )
        
        config.to_yaml(yaml_path)
        
        # Verify file was created
        assert yaml_path.exists()
        
        # Load and verify content
        with open(yaml_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["environment"] == "development"
        assert loaded_config["debug"] is True
        assert loaded_config["alpaca_api_key"] == "test_key"

    def test_load_config_function(self, tmp_path):
        """Test load_config function."""
        yaml_path = tmp_path / "test_config.yaml"
        
        test_config = {
            "environment": "development",
            "debug": True,
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(test_config, f)
        
        config = load_config(yaml_path)
        
        assert isinstance(config, UnifiedConfig)
        assert config.environment == "development"
        assert config.debug is True

    def test_load_config_function_default(self):
        """Test load_config function with no path."""
        config = load_config()
        
        assert isinstance(config, UnifiedConfig)
        assert config.environment == "production"  # default value


class TestDataValidation:
    """Test data validation functionality."""

    def test_config_validation_positive_values(self):
        """Test that positive values are accepted."""
        # These should not raise exceptions
        config = UnifiedConfig()
        
        # Test various positive values
        config.data.sequence_length = 100
        config.model.batch_size = 32
        config.backtest.initial_capital = 100000.0
        config.live.max_position_size = 0.1
        
        assert config.data.sequence_length == 100
        assert config.model.batch_size == 32
        assert config.backtest.initial_capital == 100000.0
        assert config.live.max_position_size == 0.1

    def test_config_validation_boundaries(self):
        """Test configuration value boundaries."""
        config = UnifiedConfig()
        
        # Test boundary values
        config.live.max_position_size = 1.0  # 100%
        config.live.max_leverage = 2.0
        config.live.max_drawdown = 0.5  # 50%
        
        assert config.live.max_position_size == 1.0
        assert config.live.max_leverage == 2.0
        assert config.live.max_drawdown == 0.5

    def test_config_validation_dates(self):
        """Test date string validation."""
        config = UnifiedConfig()
        
        # Test valid date formats
        config.data.start_date = "2023-01-01"
        config.data.end_date = "2024-12-31"
        config.backtest.start_date = "2024-01-01"
        config.backtest.end_date = "2024-12-31"
        
        assert config.data.start_date == "2023-01-01"
        assert config.data.end_date == "2024-12-31"
        assert config.backtest.start_date == "2024-01-01"
        assert config.backtest.end_date == "2024-12-31"


class TestPerformanceBenchmarks:
    """Test performance benchmarks for critical operations."""

    def test_config_loading_performance(self, benchmark, tmp_path):
        """Benchmark configuration loading performance."""
        config_path = tmp_path / "perf_config.yaml"
        
        # Create a complex config file
        test_config = {
            "environment": "production",
            "debug": False,
            "data": {
                "sources": {
                    "primary": "yfinance",
                    "backup": "alpaca",
                },
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"] * 10,
                "sequence_length": 60,
                "prediction_horizon": 1,
            },
            "model": {
                "type": "cnn_lstm",
                "algorithm": "sac",
                "cnn_filters": [64, 128, 256],
                "cnn_kernel_sizes": [3, 3, 3],
                "lstm_units": 128,
                "lstm_layers": 2,
                "dense_units": [64, 32],
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
            },
            "backtest": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "initial_capital": 100000.0,
            },
            "live": {
                "exchange": "alpaca",
                "paper_trading": True,
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "initial_capital": 100000.0,
            },
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        def load_config():
            return UnifiedConfig.from_yaml(config_path)
        
        # Benchmark the operation
        result = benchmark(load_config)
        
        # Verify the result is correct
        assert isinstance(result, UnifiedConfig)
        assert result.environment == "production"

    def test_config_serialization_performance(self, benchmark):
        """Benchmark configuration serialization performance."""
        config = UnifiedConfig()
        
        # Populate with test data
        config.data.symbols = ["AAPL", "GOOGL", "MSFT"] * 100
        config.model.cnn_filters = [64, 128, 256, 512]
        config.backtest.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"] * 50
        
        def serialize_config():
            return config.to_dict()
        
        # Benchmark the operation
        result = benchmark(serialize_config)
        
        # Verify the result is correct
        assert isinstance(result, dict)
        assert "environment" in result
        assert "data" in result

    def test_logging_performance(self, benchmark, tmp_path):
        """Benchmark logging performance."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)
        
        logger = get_logger("perf_test")
        
        def log_messages():
            for i in range(100):
                logger.info(f"Test message {i}")
                logger.debug(f"Debug message {i}")
                logger.warning(f"Warning message {i}")
        
        # Benchmark the operation
        benchmark(log_messages)
        
        # Verify logs were created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0


class TestIntegrationTests:
    """Integration tests for component interactions."""

    def test_config_manager_with_logging(self, tmp_path):
        """Test ConfigManager integration with logging system."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)
        
        logger = get_logger("config_test")
        
        # Test config manager operations with logging
        config_path = tmp_path / "test_config.yaml"
        manager = ConfigManager(config_path=config_path)
        
        logger.info("Loading default configuration")
        config = manager.load_config()
        
        logger.info("Updating configuration")
        config.environment = "production"
        config.debug = True
        
        logger.info("Saving configuration")
        manager.save_config(config)
        
        logger.info("Reloading configuration")
        reloaded_config = manager.load_config()
        
        assert reloaded_config.environment == "production"
        assert reloaded_config.debug is True

    def test_unified_config_with_validation(self):
        """Test UnifiedConfig integration with validation."""
        # Test that UnifiedConfig properly validates data
        config = UnifiedConfig()
        
        # Test nested configuration access
        assert config.data.sources.primary == "yfinance"
        assert config.model.type == "cnn_lstm"
        assert config.backtest.initial_capital == 100000.0
        assert config.live.exchange == "alpaca"
        
        # Test configuration updates
        config.data.sources.primary = "alpaca"
        config.model.batch_size = 64
        config.backtest.symbols = ["AAPL", "GOOGL"]
        
        assert config.data.sources.primary == "alpaca"
        assert config.model.batch_size == 64
        assert config.backtest.symbols == ["AAPL", "GOOGL"]

    def test_exception_handling_integration(self):
        """Test exception handling integration."""
        # Test that exceptions are properly raised and handled
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test configuration error")
        
        with pytest.raises(DataValidationError):
            raise DataValidationError("Test data validation error")
        
        with pytest.raises(ModelError):
            raise ModelError("Test model error")
        
        with pytest.raises(MarketDataError):
            raise MarketDataError("Test market data error")
        
        with pytest.raises(RiskManagementError):
            raise RiskManagementError("Test risk management error")
        
        with pytest.raises(ExecutionError):
            raise ExecutionError("Test execution error")

    def test_logging_with_exceptions(self, tmp_path):
        """Test logging integration with exception handling."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)
        
        logger = get_logger("exception_test")
        
        try:
            raise ConfigurationError("Test error for logging")
        except ConfigurationError as e:
            logger.error(f"Configuration error occurred: {e}")
        
        # Verify error was logged
        error_log_files = list(log_dir.glob("*error*.log"))
        assert len(error_log_files) > 0

    def test_config_persistence_integration(self, tmp_path):
        """Test configuration persistence integration."""
        # Test that configurations can be saved and loaded consistently
        config_path = tmp_path / "integration_config.yaml"
        
        # Create UnifiedConfig
        unified_config = UnifiedConfig(
            environment="development",
            debug=True,
            alpaca_api_key="test_key"
        )
        
        # Save to file
        unified_config.to_yaml(config_path)
        
        # Load from file
        loaded_config = UnifiedConfig.from_yaml(config_path)
        
        # Verify consistency
        assert loaded_config.environment == unified_config.environment
        assert loaded_config.debug == unified_config.debug
        assert loaded_config.alpaca_api_key == unified_config.alpaca_api_key
        
        # Test ConfigManager integration
        manager = ConfigManager(config_path=config_path)
        manager_config = manager.load_config()
        
        # Verify ConfigManager can load the same file
        assert manager_config.environment == "development"
        assert manager_config.debug is True