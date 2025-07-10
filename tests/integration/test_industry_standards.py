"""
Tests for Industry-Grade FinRL Integration

Comprehensive test suite for professional data feeds, FinRL environment,
and industry-standard features.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Test professional data feeds
from trading_rl_agent.data.professional_feeds import ProfessionalDataProvider


class TestProfessionalDataProvider:
    """Test suite for professional data providers."""

    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        data = []

        for symbol in ["AAPL", "GOOGL"]:
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": 100 + np.random.randn(),
                        "high": 102 + np.random.randn(),
                        "low": 98 + np.random.randn(),
                        "close": 101 + np.random.randn(),
                        "volume": 1000000 + np.random.randint(0, 500000),
                    },
                )

        return pd.DataFrame(data)

    def test_yahoo_provider_initialization(self):
        """Test Yahoo Finance provider initialization."""
        provider = ProfessionalDataProvider(provider="yahoo")
        assert provider.provider == "yahoo"
        assert provider.validate_connection()

    def test_alpaca_provider_initialization_without_credentials(self):
        """Test Alpaca provider fails without credentials."""
        # Clear environment variables
        old_key = os.environ.pop("ALPACA_API_KEY", None)
        old_secret = os.environ.pop("ALPACA_SECRET_KEY", None)

        try:
            with pytest.raises(ValueError, match="Alpaca API credentials required"):
                ProfessionalDataProvider(provider="alpaca")
        finally:
            # Restore environment variables
            if old_key:
                os.environ["ALPACA_API_KEY"] = old_key
            if old_secret:
                os.environ["ALPACA_SECRET_KEY"] = old_secret

    @patch("src.data.professional_feeds.yf.download")
    def test_yahoo_data_retrieval(self, mock_download, sample_market_data):
        """Test Yahoo Finance data retrieval."""
        # Mock yfinance response
        mock_data = sample_market_data[sample_market_data["symbol"] == "AAPL"].copy()
        mock_data.set_index("date", inplace=True)
        mock_data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        mock_download.return_value = mock_data

        provider = ProfessionalDataProvider(provider="yahoo")
        data = provider.get_market_data(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            include_features=False,
        )

        assert not data.empty
        assert "symbol" in data.columns
        assert "close" in data.columns
        assert len(data) > 0

    @patch("src.data.professional_feeds.ALPACA_AVAILABLE", True)
    @patch("src.data.professional_feeds.tradeapi")
    def test_alpaca_data_retrieval_direct_api(self, mock_tradeapi):
        """Test Alpaca direct API data retrieval."""
        # Set up environment variables
        os.environ["ALPACA_API_KEY"] = "test_key"
        os.environ["ALPACA_SECRET_KEY"] = "test_secret"

        # Mock Alpaca API response
        mock_bars = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", "2024-01-31"),
                "open": np.random.randn(31) + 100,
                "high": np.random.randn(31) + 102,
                "low": np.random.randn(31) + 98,
                "close": np.random.randn(31) + 101,
                "volume": np.random.randint(100000, 1000000, 31),
            },
        )

        mock_api = Mock()
        mock_api.get_bars.return_value.df = mock_bars
        mock_tradeapi.REST.return_value = mock_api

        try:
            provider = ProfessionalDataProvider(provider="alpaca")
            data = provider.get_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                include_features=False,
            )

            assert not data.empty
            assert "symbol" in data.columns

        finally:
            # Clean up environment variables
            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_SECRET_KEY", None)

    def test_unsupported_provider(self):
        """Test error handling for unsupported providers."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            ProfessionalDataProvider(provider="unknown")

    def test_feature_generation_integration(self, sample_market_data):
        """Test integration with feature generation pipeline."""
        with patch("src.data.professional_feeds.yf.Ticker") as mock_ticker:
            # Mock yfinance ticker
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_market_data[
                sample_market_data["symbol"] == "AAPL"
            ].set_index("date")
            mock_ticker.return_value = mock_ticker_instance

            provider = ProfessionalDataProvider(provider="yahoo")
            data = provider.get_market_data(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-31",
                include_features=True,
            )

            # Should have generated additional technical features
            expected_base_cols = [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            assert all(col in data.columns for col in expected_base_cols)

            # Should have more columns due to feature generation
            assert len(data.columns) > len(expected_base_cols)


# Test FinRL environment integration
@pytest.mark.skipif(
    condition=True,  # Skip until FinRL is installed
    reason="FinRL not available in current environment",
)
class TestFinRLIntegration:
    """Test suite for FinRL environment integration."""

    @pytest.fixture
    def sample_finrl_data(self):
        """Sample data in FinRL format."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        data = []

        for date in dates:
            for symbol in ["AAPL", "GOOGL"]:
                data.append(
                    {
                        "date": date,
                        "tic": symbol,
                        "open": 100 + np.random.randn(),
                        "high": 102 + np.random.randn(),
                        "low": 98 + np.random.randn(),
                        "close": 101 + np.random.randn(),
                        "volume": 1000000,
                        "macd": np.random.randn(),
                        "rsi_30": np.random.rand() * 100,
                        "cci_30": np.random.randn() * 100,
                    },
                )

        return pd.DataFrame(data)

    def test_finrl_environment_creation(self, sample_finrl_data):
        """Test FinRL environment creation."""
        from trading_rl_agent.envs.finrl_trading_env import HybridFinRLEnv

        env = HybridFinRLEnv(df=sample_finrl_data)

        assert env.observation_space is not None
        assert env.action_space is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_cnn_lstm_integration(self, sample_finrl_data):
        """Test CNN+LSTM model integration with FinRL environment."""
        from trading_rl_agent.envs.finrl_trading_env import HybridFinRLEnv

        # Mock CNN+LSTM model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.6, 0.8])  # trend, confidence

        env = HybridFinRLEnv(df=sample_finrl_data, cnn_lstm_model=mock_model)

        obs, info = env.reset()

        # Observation should be enhanced with CNN+LSTM predictions
        assert len(obs) > len(sample_finrl_data.columns)

        # Test step function
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)

        assert isinstance(reward, int | float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)


class TestIndustryStandardMetrics:
    """Test suite for industry-standard evaluation metrics."""

    @pytest.fixture
    def sample_returns(self):
        """Sample return series for testing."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        return pd.Series(returns)

    @pytest.fixture
    def sample_benchmark_returns(self):
        """Sample benchmark return series."""
        np.random.seed(43)
        returns = np.random.normal(0.0008, 0.015, 252)  # Market returns
        return pd.Series(returns)

    def test_sharpe_ratio_calculation(self, sample_returns):
        """Test Sharpe ratio calculation."""
        from trading_rl_agent.utils import metrics

        sharpe = metrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_maximum_drawdown_calculation(self, sample_returns):
        """Test maximum drawdown calculation."""
        from trading_rl_agent.utils import metrics

        max_dd = metrics.calculate_max_drawdown(sample_returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero

    def test_sortino_ratio_calculation(self, sample_returns):
        """Test Sortino ratio calculation."""
        from trading_rl_agent.utils import metrics

        sortino = metrics.calculate_sortino_ratio(sample_returns, target_return=0.02)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_var_calculation(self, sample_returns):
        """Test Value at Risk calculation."""
        from trading_rl_agent.utils import metrics

        var_95 = metrics.calculate_var(sample_returns, confidence=0.95)

        assert isinstance(var_95, float)
        assert var_95 <= 0  # VaR should be negative

    def test_information_ratio_calculation(
        self,
        sample_returns,
        sample_benchmark_returns,
    ):
        """Test Information ratio calculation."""
        from trading_rl_agent.utils import metrics

        info_ratio = metrics.calculate_information_ratio(
            sample_returns,
            sample_benchmark_returns,
        )

        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)

    def test_comprehensive_metrics_calculation(
        self,
        sample_returns,
        sample_benchmark_returns,
    ):
        """Test comprehensive metrics calculation."""
        from trading_rl_agent.utils import metrics

        metrics_dict = metrics.calculate_comprehensive_metrics(
            sample_returns,
            sample_benchmark_returns,
        )

        # Check that all expected metrics are present
        expected_metrics = [
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "var_95",
            "expected_shortfall",
            "information_ratio",
            "tracking_error",
            "beta",
            "profit_factor",
            "win_rate",
            "average_win_loss_ratio",
        ]

        for metric in expected_metrics:
            assert metric in metrics_dict
            assert isinstance(metrics_dict[metric], int | float)


class TestRiskManagement:
    """Test suite for industry-grade risk management."""

    def test_position_size_calculation(self):
        """Test position sizing algorithms."""
        from trading_rl_agent.risk.position_sizing import IndustryGradeRiskManager

        risk_manager = IndustryGradeRiskManager(max_position_size=0.1)

        # Test Kelly criterion position sizing
        signal_strength = 0.6  # 60% confidence
        volatility = 0.02  # 2% daily volatility

        position_size = risk_manager.calculate_position_size(
            signal_strength,
            volatility,
        )

        assert isinstance(position_size, float)
        assert 0 <= position_size <= risk_manager.max_position_size

    def test_position_limits_enforcement(self):
        """Test position limit enforcement."""
        from trading_rl_agent.risk.position_sizing import IndustryGradeRiskManager

        risk_manager = IndustryGradeRiskManager(max_position_size=0.1)

        # Test oversized action
        oversized_action = np.array([0.5])  # 50% position (exceeds 10% limit)
        mock_portfolio = {"value": 1000000, "positions": {}}

        constrained_action = risk_manager.check_position_limits(
            oversized_action,
            mock_portfolio,
        )

        assert abs(constrained_action[0]) <= risk_manager.max_position_size

    def test_drawdown_monitoring(self):
        """Test maximum drawdown monitoring."""
        from trading_rl_agent.risk.position_sizing import IndustryGradeRiskManager

        risk_manager = IndustryGradeRiskManager(max_drawdown=0.02)

        # Simulate portfolio value history with drawdown
        portfolio_values = [1000000, 990000, 980000, 970000]  # 3% drawdown
        risk_manager.portfolio_value_history = portfolio_values

        # Should trigger risk controls
        current_drawdown = risk_manager.calculate_current_drawdown()

        assert current_drawdown >= risk_manager.max_drawdown


class TestModelServing:
    """Test suite for production model serving."""

    @pytest.mark.asyncio
    async def test_trading_model_service_prediction(self):
        """Test model serving prediction endpoint."""
        from trading_rl_agent.deployment.model_serving import TradingModelService

        # Mock models
        mock_cnn_lstm = Mock()
        mock_cnn_lstm.predict.return_value = {"prediction": 0.7, "confidence": 0.8}

        mock_rl_agent = Mock()
        mock_rl_agent.select_action.return_value = np.array([0.3])

        # Create service with mocked models
        with (
            patch(
                "src.deployment.model_serving.TradingModelService.load_cnn_lstm_model",
                return_value=mock_cnn_lstm,
            ),
            patch(
                "src.deployment.model_serving.TradingModelService.load_rl_agent",
                return_value=mock_rl_agent,
            ),
        ):
            service = TradingModelService("mock_model_path")

            # Test prediction
            market_data = {
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1000000,
            }

            result = await service.predict(market_data)

            assert "action" in result
            assert "confidence" in result
            assert "risk_metrics" in result
            assert isinstance(result["action"], np.ndarray)

    def test_model_monitoring_metrics(self):
        """Test model monitoring and metrics collection."""
        from trading_rl_agent.monitoring.model_monitoring import ModelMonitor

        monitor = ModelMonitor()

        # Test prediction logging
        prediction = {"confidence": 0.8, "prediction": 0.7}
        latency = 0.05  # 50ms

        monitor.log_prediction(prediction, latency=latency)

        # Verify metrics were recorded
        assert monitor.prediction_counter._value._value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
