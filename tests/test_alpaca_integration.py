"""
Tests for Alpaca Markets Integration

Tests the AlpacaIntegration class and related components.
"""

import os

# Add src to path for imports
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.configs.alpaca_config import (
    AlpacaConfigManager,
    AlpacaConfigModel,
    AlpacaEnvironmentConfig,
    validate_alpaca_environment,
)
from trade_agent.data.alpaca_integration import (
    AlpacaConfig,
    AlpacaIntegration,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioPosition,
    create_alpaca_config_from_env,
)


class TestAlpacaConfig:
    """Test Alpaca configuration classes."""

    def test_alpaca_config_creation(self):
        """Test AlpacaConfig creation."""
        config = AlpacaConfig(api_key="test_key", secret_key="test_secret", paper_trading=True)

        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.paper_trading is True
        assert config.base_url == "https://paper-api.alpaca.markets"

    def test_alpaca_config_validation(self):
        """Test AlpacaConfig validation."""
        with pytest.raises(ValueError):
            AlpacaConfig(api_key="", secret_key="test_secret")

        with pytest.raises(ValueError):
            AlpacaConfig(api_key="test_key", secret_key="")

    def test_create_alpaca_config_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "env_key",
                "ALPACA_SECRET_KEY": "env_secret",
                "ALPACA_PAPER_TRADING": "true",
            },
        ):
            config = create_alpaca_config_from_env()
            assert config.api_key == "env_key"
            assert config.secret_key == "env_secret"
            assert config.paper_trading is True


class TestAlpacaConfigModel:
    """Test Pydantic configuration model."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = AlpacaConfigModel(api_key="test_key", secret_key="test_secret")
        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.paper_trading is True  # default

    def test_invalid_credentials(self):
        """Test invalid credentials validation."""
        with pytest.raises(ValueError):
            AlpacaConfigModel(api_key="", secret_key="test_secret")

        with pytest.raises(ValueError):
            AlpacaConfigModel(api_key="test_key", secret_key="")

    def test_invalid_urls(self):
        """Test URL validation."""
        with pytest.raises(ValueError):
            AlpacaConfigModel(api_key="test_key", secret_key="test_secret", base_url="invalid_url")

    def test_invalid_data_feed(self):
        """Test data feed validation."""
        with pytest.raises(ValueError):
            AlpacaConfigModel(api_key="test_key", secret_key="test_secret", data_feed="invalid_feed")

    def test_invalid_log_level(self):
        """Test log level validation."""
        with pytest.raises(ValueError):
            AlpacaConfigModel(api_key="test_key", secret_key="test_secret", log_level="invalid_level")


class TestAlpacaEnvironmentConfig:
    """Test environment-based configuration."""

    def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch.dict(
            os.environ,
            {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"},
        ):
            assert validate_alpaca_environment() is True

    def test_validate_environment_failure(self):
        """Test failed environment validation."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("trading_rl_agent.core.unified_config.UnifiedConfig") as mock_unified,
        ):
            # Mock unified config to return None for credentials
            mock_config = Mock()
            mock_config.alpaca_api_key = None
            mock_config.alpaca_secret_key = None
            mock_unified.return_value = mock_config

            assert validate_alpaca_environment() is False

    def test_from_environment(self):
        """Test creating config from environment."""
        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "env_key",
                "ALPACA_SECRET_KEY": "env_secret",
                "ALPACA_PAPER_TRADING": "true",
                "ALPACA_MAX_RETRIES": "5",
            },
        ):
            config = AlpacaEnvironmentConfig.from_environment()
            assert config.api_key == "env_key"
            assert config.secret_key == "env_secret"
            assert config.paper_trading is True
            assert config.max_retries == 5


class TestAlpacaConfigManager:
    """Test configuration manager."""

    def test_load_from_environment(self):
        """Test loading config from environment."""
        with patch.dict(os.environ, {"ALPACA_API_KEY": "env_key", "ALPACA_SECRET_KEY": "env_secret"}):
            manager = AlpacaConfigManager()
            config = manager.load_config()
            assert config.api_key == "env_key"
            assert config.secret_key == "env_secret"

    def test_load_from_file(self):
        """Test loading config from file."""
        import yaml

        config_data = {
            "api_key": "file_key",
            "secret_key": "file_secret",
            "paper_trading": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            manager = AlpacaConfigManager()
            config = manager.load_config(config_file)
            assert config.api_key == "file_key"
            assert config.secret_key == "file_secret"
            assert config.paper_trading is True
        finally:
            Path(config_file).unlink()

    def test_save_config(self):
        """Test saving config to file."""
        import yaml

        config = AlpacaConfigModel(api_key="save_key", secret_key="save_secret")

        manager = AlpacaConfigManager()
        manager._config = config

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_file = f.name

        try:
            manager.save_config(config_file)

            # Verify saved config
            with open(config_file) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data["api_key"] == "save_key"
            assert saved_data["secret_key"] == "save_secret"
        finally:
            Path(config_file).unlink()


class TestAlpacaIntegration:
    """Test AlpacaIntegration class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return AlpacaConfig(api_key="test_key", secret_key="test_secret", paper_trading=True)

    @pytest.fixture
    def mock_alpaca_api(self):
        """Create mock Alpaca API."""
        mock_api = Mock()

        # Mock account
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_account.cash = "10000.00"
        mock_account.portfolio_value = "15000.00"
        mock_account.buying_power = "20000.00"
        mock_account.equity = "15000.00"
        mock_account.status = "ACTIVE"
        mock_account.currency = "USD"
        mock_account.pattern_day_trader = False
        mock_account.daytrade_count = 0
        mock_api.get_account.return_value = mock_account

        # Mock positions
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "10"
        mock_position.avg_entry_price = "150.00"
        mock_position.current_price = "155.00"
        mock_position.market_value = "1550.00"
        mock_position.unrealized_pl = "50.00"
        mock_position.unrealized_plpc = "0.0333"
        mock_position.side = "long"
        mock_api.list_positions.return_value = [mock_position]

        # Mock quotes
        mock_quote = Mock()
        mock_quote.bid_price = "154.50"
        mock_quote.ask_price = "155.50"
        mock_quote.bid_size = "100"
        mock_quote.ask_size = "100"
        mock_quote.timestamp = datetime.now()
        mock_api.get_latest_quote.return_value = mock_quote

        return mock_api

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_initialization(self, mock_config):
        """Test AlpacaIntegration initialization."""
        # Create a mock REST class
        mock_rest_class = Mock()
        mock_rest_instance = Mock()
        mock_rest_class.return_value = mock_rest_instance

        with patch.dict("sys.modules", {"alpaca_trade_api": Mock(), "alpaca_trade_api.rest": Mock()}):
            import sys

            sys.modules["alpaca_trade_api.rest"].REST = mock_rest_class

        alpaca = AlpacaIntegration(mock_config)

        assert alpaca.config == mock_config
        assert alpaca._stream_connected is False
        assert len(alpaca._data_callbacks) == 0

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_validate_connection(self, mock_config, mock_alpaca_api):
        """Test connection validation."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_rest.return_value = mock_alpaca_api

            alpaca = AlpacaIntegration(mock_config)
            result = alpaca.validate_connection()

            assert result is True
            mock_alpaca_api.get_account.assert_called_once()

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_get_account_info(self, mock_config, mock_alpaca_api):
        """Test account information retrieval."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_rest.return_value = mock_alpaca_api

            alpaca = AlpacaIntegration(mock_config)
            account_info = alpaca.get_account_info()

            assert account_info["id"] == "test_account"
            assert account_info["cash"] == 10000.0
            assert account_info["portfolio_value"] == 15000.0
            assert account_info["status"] == "ACTIVE"

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_get_real_time_quotes(self, mock_config, mock_alpaca_api):
        """Test real-time quotes retrieval."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_rest.return_value = mock_alpaca_api

            alpaca = AlpacaIntegration(mock_config)
            quotes = alpaca.get_real_time_quotes(["AAPL"])

            assert "AAPL" in quotes
            quote = quotes["AAPL"]
            assert quote["bid_price"] == 154.50
            assert quote["ask_price"] == 155.50
            assert quote["spread"] == 1.0

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_get_positions(self, mock_config, mock_alpaca_api):
        """Test positions retrieval."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_rest.return_value = mock_alpaca_api

            alpaca = AlpacaIntegration(mock_config)
            positions = alpaca.get_positions()

            assert len(positions) == 1
            position = positions[0]
            assert position.symbol == "AAPL"
            assert position.qty == 10.0
            assert position.avg_entry_price == 150.0
            assert position.current_price == 155.0

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_get_portfolio_value(self, mock_config, mock_alpaca_api):
        """Test portfolio value calculation."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_rest.return_value = mock_alpaca_api

            alpaca = AlpacaIntegration(mock_config)
            portfolio = alpaca.get_portfolio_value()

            assert portfolio["total_equity"] == 15000.0
            assert portfolio["cash"] == 10000.0
            assert portfolio["total_market_value"] == 1550.0
            assert portfolio["total_unrealized_pl"] == 50.0
            assert portfolio["position_count"] == 1

    @patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True)
    def test_place_order(self, mock_config):
        """Test order placement."""
        with patch("alpaca_trade_api.rest.REST") as mock_rest:
            mock_api = Mock()
            mock_rest.return_value = mock_api

            # Mock order
            mock_order = Mock()
            mock_order.id = "test_order"
            mock_order.client_order_id = "client_123"
            mock_order.symbol = "AAPL"
            mock_order.qty = "10"
            mock_order.side = "buy"
            mock_order.type = "market"
            mock_order.status = "filled"
            mock_order.filled_at = datetime.now()
            mock_order.filled_avg_price = "155.00"
            mock_order.filled_qty = "10"
            mock_order.submitted_at = datetime.now()
            mock_order.limit_price = None
            mock_order.stop_price = None

            mock_api.submit_order.return_value = mock_order
            mock_api.get_order.return_value = mock_order

            alpaca = AlpacaIntegration(mock_config)

            order_request = OrderRequest(symbol="AAPL", qty=10.0, side=OrderSide.BUY, order_type=OrderType.MARKET)

            result = alpaca.place_order(order_request)

            assert result["order_id"] == "test_order"
            assert result["symbol"] == "AAPL"
            assert result["qty"] == 10.0
            assert result["side"] == "buy"
            assert result["type"] == "market"
            assert result["status"] == "filled"

    def test_context_manager(self, mock_config):
        """Test context manager functionality."""
        with (
            patch("alpaca_trade_api.rest.REST") as mock_rest,
            patch("trading_rl_agent.data.alpaca_integration.ALPACA_AVAILABLE", True),
        ):
            mock_rest.return_value = Mock()

            with AlpacaIntegration(mock_config) as alpaca:
                assert alpaca.config == mock_config
                # Context manager should call stop_data_stream on exit
                alpaca.stop_data_stream.assert_not_called()  # Not called yet


class TestOrderRequest:
    """Test OrderRequest class."""

    def test_order_request_creation(self):
        """Test OrderRequest creation."""
        order_request = OrderRequest(
            symbol="AAPL",
            qty=10.0,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force="day",
            limit_price=150.00,
        )

        assert order_request.symbol == "AAPL"
        assert order_request.qty == 10.0
        assert order_request.side == OrderSide.BUY
        assert order_request.order_type == OrderType.MARKET
        assert order_request.time_in_force == "day"
        assert order_request.limit_price == 150.00


class TestMarketData:
    """Test MarketData class."""

    def test_market_data_creation(self):
        """Test MarketData creation."""
        timestamp = datetime.now()
        market_data = MarketData(
            symbol="AAPL",
            timestamp=timestamp,
            open=150.00,
            high=155.00,
            low=149.00,
            close=154.00,
            volume=1000000,
            vwap=152.50,
            trade_count=5000,
        )

        assert market_data.symbol == "AAPL"
        assert market_data.timestamp == timestamp
        assert market_data.open == 150.00
        assert market_data.high == 155.00
        assert market_data.low == 149.00
        assert market_data.close == 154.00
        assert market_data.volume == 1000000
        assert market_data.vwap == 152.50
        assert market_data.trade_count == 5000


class TestPortfolioPosition:
    """Test PortfolioPosition class."""

    def test_portfolio_position_creation(self):
        """Test PortfolioPosition creation."""
        timestamp = datetime.now()
        position = PortfolioPosition(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.00,
            current_price=155.00,
            market_value=1550.00,
            unrealized_pl=50.00,
            unrealized_plpc=0.0333,
            side="long",
            timestamp=timestamp,
        )

        assert position.symbol == "AAPL"
        assert position.qty == 10.0
        assert position.avg_entry_price == 150.00
        assert position.current_price == 155.00
        assert position.market_value == 1550.00
        assert position.unrealized_pl == 50.00
        assert position.unrealized_plpc == 0.0333
        assert position.side == "long"
        assert position.timestamp == timestamp


if __name__ == "__main__":
    pytest.main([__file__])
