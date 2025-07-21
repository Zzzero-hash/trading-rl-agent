"""
Data quality tests for trading system data.

These tests validate data integrity, quality, and consistency using
Great Expectations and Pandera frameworks.
"""

from datetime import datetime, timedelta

import great_expectations as ge
import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandera.typing import DataFrame, Series

from trading_rl_agent.data.data_validator import DataValidator


class TestMarketDataQuality:
    """Test quality of market data."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        data = []

        for date in dates:
            data.append(
                {
                    "date": date,
                    "symbol": "AAPL",
                    "open": np.random.uniform(140, 160),
                    "high": np.random.uniform(145, 165),
                    "low": np.random.uniform(135, 155),
                    "close": np.random.uniform(140, 160),
                    "volume": np.random.randint(1000000, 5000000),
                    "adj_close": np.random.uniform(140, 160),
                }
            )

        return pd.DataFrame(data)

    @pytest.mark.data_quality
    def test_market_data_schema_validation(self, sample_market_data):
        """Test market data schema using Pandera."""

        # Define schema for market data
        class MarketDataSchema(pa.SchemaModel):
            date: Series[datetime] = pa.Field(ge=datetime(2020, 1, 1))
            symbol: Series[str] = pa.Field(str_length=pa.Field(ge=1, le=10))
            open: Series[float] = pa.Field(ge=0)
            high: Series[float] = pa.Field(ge=0)
            low: Series[float] = pa.Field(ge=0)
            close: Series[float] = pa.Field(ge=0)
            volume: Series[int] = pa.Field(ge=0)
            adj_close: Series[float] = pa.Field(ge=0)

            @pa.check("high >= open")
            def high_greater_than_open(self, df: DataFrame) -> Series[bool]:
                return df["high"] >= df["open"]

            @pa.check("high >= close")
            def high_greater_than_close(self, df: DataFrame) -> Series[bool]:
                return df["high"] >= df["close"]

            @pa.check("low <= open")
            def low_less_than_open(self, df: DataFrame) -> Series[bool]:
                return df["low"] <= df["open"]

            @pa.check("low <= close")
            def low_less_than_close(self, df: DataFrame) -> Series[bool]:
                return df["low"] <= df["close"]

        # Validate schema
        try:
            MarketDataSchema.validate(sample_market_data)
            assert True
        except pa.errors.SchemaError as e:
            pytest.fail(f"Schema validation failed: {e}")

    @pytest.mark.data_quality
    def test_market_data_great_expectations(self, sample_market_data):
        """Test market data using Great Expectations."""

        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(sample_market_data)

        # Test expectations
        results = []

        # Test for no null values in critical columns
        results.append(ge_df.expect_column_values_to_not_be_null("symbol"))
        results.append(ge_df.expect_column_values_to_not_be_null("close"))
        results.append(ge_df.expect_column_values_to_not_be_null("volume"))

        # Test for positive values
        results.append(ge_df.expect_column_values_to_be_between("open", 0, None))
        results.append(ge_df.expect_column_values_to_be_between("high", 0, None))
        results.append(ge_df.expect_column_values_to_be_between("low", 0, None))
        results.append(ge_df.expect_column_values_to_be_between("close", 0, None))
        results.append(ge_df.expect_column_values_to_be_between("volume", 0, None))

        # Test for reasonable price ranges (assuming USD)
        results.append(ge_df.expect_column_values_to_be_between("close", 1, 10000))

        # Test for reasonable volume ranges
        results.append(ge_df.expect_column_values_to_be_between("volume", 100, 100000000))

        # Test for date consistency
        results.append(ge_df.expect_column_values_to_be_between("date", datetime(2020, 1, 1), datetime(2030, 12, 31)))

        # Check all expectations passed
        failed_expectations = [r for r in results if not r.success]
        if failed_expectations:
            pytest.fail(f"Great Expectations validation failed: {failed_expectations}")

    @pytest.mark.data_quality
    def test_market_data_consistency(self, sample_market_data):
        """Test market data consistency rules."""

        # Test OHLC consistency
        assert all(sample_market_data["high"] >= sample_market_data["low"]), "High must be >= Low"
        assert all(sample_market_data["high"] >= sample_market_data["open"]), "High must be >= Open"
        assert all(sample_market_data["high"] >= sample_market_data["close"]), "High must be >= Close"
        assert all(sample_market_data["low"] <= sample_market_data["open"]), "Low must be <= Open"
        assert all(sample_market_data["low"] <= sample_market_data["close"]), "Low must be <= Close"

        # Test for reasonable price movements (no extreme jumps)
        price_changes = sample_market_data["close"].pct_change().abs()
        assert all(price_changes < 0.5), "Price changes should be less than 50%"

        # Test for reasonable volume
        assert all(sample_market_data["volume"] > 0), "Volume must be positive"
        assert all(sample_market_data["volume"] < 1e9), "Volume should be reasonable"


class TestPortfolioDataQuality:
    """Test quality of portfolio data."""

    @pytest.fixture
    def sample_portfolio_data(self):
        """Generate sample portfolio data for testing."""
        data = []

        for i in range(100):
            data.append(
                {
                    "timestamp": datetime.now() - timedelta(days=i),
                    "symbol": f"ASSET_{i % 10}",
                    "position": np.random.uniform(-1000, 1000),
                    "price": np.random.uniform(10, 500),
                    "market_value": np.random.uniform(1000, 100000),
                    "unrealized_pnl": np.random.uniform(-10000, 10000),
                    "realized_pnl": np.random.uniform(-5000, 5000),
                }
            )

        return pd.DataFrame(data)

    @pytest.mark.data_quality
    def test_portfolio_data_schema(self, sample_portfolio_data):
        """Test portfolio data schema."""

        class PortfolioDataSchema(pa.SchemaModel):
            timestamp: Series[datetime]
            symbol: Series[str] = pa.Field(str_length=pa.Field(ge=1, le=20))
            position: Series[float]
            price: Series[float] = pa.Field(ge=0)
            market_value: Series[float] = pa.Field(ge=0)
            unrealized_pnl: Series[float]
            realized_pnl: Series[float]

            @pa.check("market_value == position * price")
            def market_value_consistency(self, df: DataFrame) -> Series[bool]:
                return abs(df["market_value"] - df["position"] * df["price"]) < 1e-6

        try:
            PortfolioDataSchema.validate(sample_portfolio_data)
            assert True
        except pa.errors.SchemaError as e:
            pytest.fail(f"Portfolio schema validation failed: {e}")

    @pytest.mark.data_quality
    def test_portfolio_data_consistency(self, sample_portfolio_data):
        """Test portfolio data consistency."""

        # Test market value calculation
        calculated_mv = sample_portfolio_data["position"] * sample_portfolio_data["price"]
        assert np.allclose(sample_portfolio_data["market_value"], calculated_mv, rtol=1e-6)

        # Test for reasonable position sizes
        assert all(abs(sample_portfolio_data["position"]) < 1e6), "Position sizes should be reasonable"

        # Test for reasonable PnL values
        assert all(abs(sample_portfolio_data["unrealized_pnl"]) < 1e7), "Unrealized PnL should be reasonable"
        assert all(abs(sample_portfolio_data["realized_pnl"]) < 1e7), "Realized PnL should be reasonable"


class TestRiskDataQuality:
    """Test quality of risk data."""

    @pytest.fixture
    def sample_risk_data(self):
        """Generate sample risk data for testing."""
        data = []

        for i in range(100):
            data.append(
                {
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "var_95": np.random.uniform(-0.1, -0.01),
                    "cvar_95": np.random.uniform(-0.15, -0.02),
                    "volatility": np.random.uniform(0.01, 0.5),
                    "sharpe_ratio": np.random.uniform(-2, 3),
                    "max_drawdown": np.random.uniform(-0.3, 0),
                    "beta": np.random.uniform(0.5, 1.5),
                }
            )

        return pd.DataFrame(data)

    @pytest.mark.data_quality
    def test_risk_data_schema(self, sample_risk_data):
        """Test risk data schema."""

        class RiskDataSchema(pa.SchemaModel):
            timestamp: Series[datetime]
            var_95: Series[float] = pa.Field(le=0)  # VaR should be negative
            cvar_95: Series[float] = pa.Field(le=0)  # CVaR should be negative
            volatility: Series[float] = pa.Field(ge=0)
            sharpe_ratio: Series[float]
            max_drawdown: Series[float] = pa.Field(le=0)  # Drawdown should be negative
            beta: Series[float] = pa.Field(ge=0)

            @pa.check("cvar_95 <= var_95")
            def cvar_less_than_var(self, df: DataFrame) -> Series[bool]:
                return df["cvar_95"] <= df["var_95"]

            @pa.check("volatility >= 0")
            def volatility_positive(self, df: DataFrame) -> Series[bool]:
                return df["volatility"] >= 0

        try:
            RiskDataSchema.validate(sample_risk_data)
            assert True
        except pa.errors.SchemaError as e:
            pytest.fail(f"Risk data schema validation failed: {e}")

    @pytest.mark.data_quality
    def test_risk_data_consistency(self, sample_risk_data):
        """Test risk data consistency."""

        # Test VaR and CVaR relationship
        assert all(sample_risk_data["cvar_95"] <= sample_risk_data["var_95"]), "CVaR should be <= VaR"

        # Test volatility is positive
        assert all(sample_risk_data["volatility"] >= 0), "Volatility should be non-negative"

        # Test drawdown is negative
        assert all(sample_risk_data["max_drawdown"] <= 0), "Max drawdown should be non-positive"

        # Test beta is reasonable
        assert all(sample_risk_data["beta"] >= 0), "Beta should be non-negative"
        assert all(sample_risk_data["beta"] < 5), "Beta should be reasonable"


class TestDataValidatorIntegration:
    """Test integration with DataValidator class."""

    @pytest.mark.data_quality
    def test_data_validator_market_data(self):
        """Test DataValidator with market data."""
        validator = DataValidator()

        # Create test data
        test_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "MSFT"],
                "price": [150.0, 2800.0, 300.0],
                "volume": [1000000, 500000, 2000000],
            }
        )

        # Validate data
        validation_result = validator.validate_market_data(test_data)
        assert validation_result.is_valid, f"Validation failed: {validation_result.errors}"

    @pytest.mark.data_quality
    def test_data_validator_portfolio_data(self):
        """Test DataValidator with portfolio data."""
        validator = DataValidator()

        # Create test data
        test_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "position": [100, -50],
                "price": [150.0, 2800.0],
                "market_value": [15000.0, -140000.0],
            }
        )

        # Validate data
        validation_result = validator.validate_portfolio_data(test_data)
        assert validation_result.is_valid, f"Validation failed: {validation_result.errors}"


class TestDataQualityMonitoring:
    """Test data quality monitoring and alerting."""

    @pytest.mark.data_quality
    def test_data_quality_metrics(self):
        """Test calculation of data quality metrics."""

        # Create data with some quality issues
        data = pd.DataFrame(
            {
                "price": [100, np.nan, 102, -50, 105],  # Missing and invalid values
                "volume": [1000, 1100, np.nan, 1200, 1300],  # Missing values
                "symbol": ["AAPL", "GOOGL", "MSFT", "TSLA", "AAPL"],
            }
        )

        # Calculate quality metrics
        completeness = data.notna().mean()
        validity = (data["price"] > 0).mean()  # Price should be positive

        assert completeness["price"] == 0.8, "Price completeness should be 80%"
        assert completeness["volume"] == 0.8, "Volume completeness should be 80%"
        assert validity == 0.8, "Price validity should be 80%"

    @pytest.mark.data_quality
    def test_data_quality_alerting(self):
        """Test data quality alerting system."""

        # Simulate data quality issues
        data_issues = {
            "missing_data": 0.15,  # 15% missing data
            "invalid_prices": 0.05,  # 5% invalid prices
            "duplicate_records": 0.02,  # 2% duplicates
        }

        # Define thresholds
        thresholds = {
            "missing_data": 0.10,  # Alert if > 10% missing
            "invalid_prices": 0.05,  # Alert if > 5% invalid
            "duplicate_records": 0.01,  # Alert if > 1% duplicates
        }

        # Check for alerts
        alerts = []
        for issue, value in data_issues.items():
            if value > thresholds[issue]:
                alerts.append(f"{issue}: {value:.1%} exceeds threshold {thresholds[issue]:.1%}")

        # Should have alerts for missing_data and duplicate_records
        assert len(alerts) == 2, f"Expected 2 alerts, got {len(alerts)}: {alerts}"


class TestRealTimeDataQuality:
    """Test real-time data quality monitoring."""

    @pytest.mark.data_quality
    def test_realtime_data_freshness(self):
        """Test real-time data freshness."""

        # Simulate real-time data timestamps
        current_time = datetime.now()
        data_timestamps = [
            current_time - timedelta(seconds=1),  # Fresh
            current_time - timedelta(seconds=5),  # Fresh
            current_time - timedelta(seconds=30),  # Stale
            current_time - timedelta(seconds=60),  # Very stale
        ]

        # Check freshness (data should be less than 10 seconds old)
        fresh_threshold = timedelta(seconds=10)
        fresh_data = [ts for ts in data_timestamps if current_time - ts < fresh_threshold]

        assert len(fresh_data) == 2, f"Expected 2 fresh data points, got {len(fresh_data)}"

    @pytest.mark.data_quality
    def test_realtime_data_consistency(self):
        """Test real-time data consistency."""

        # Simulate real-time price updates
        price_updates = [
            {"symbol": "AAPL", "price": 150.0, "timestamp": datetime.now()},
            {"symbol": "AAPL", "price": 151.0, "timestamp": datetime.now()},
            {"symbol": "AAPL", "price": 149.0, "timestamp": datetime.now()},
            {"symbol": "AAPL", "price": 152.0, "timestamp": datetime.now()},
        ]

        # Check for reasonable price movements
        prices = [update["price"] for update in price_updates]
        price_changes = [abs(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

        # Price changes should be reasonable (< 5%)
        assert all(change < 0.05 for change in price_changes), "Price changes should be reasonable"
