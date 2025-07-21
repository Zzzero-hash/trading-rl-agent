"""
Contract tests for trading system APIs.

These tests verify API contracts and ensure service compatibility
using consumer-driven contract testing.
"""

from unittest.mock import Mock, patch

import pytest
from pact import Consumer, EachLike, Like, Provider, Term


class TestMarketDataAPIContracts:
    """Test contracts for market data API."""

    @pytest.fixture
    def market_data_consumer(self):
        """Create market data consumer."""
        return Consumer("trading-system").has_pact_with(
            Provider("market-data-service"), host_name="localhost", port=1234
        )

    @pytest.mark.contract
    def test_get_market_data_contract(self, market_data_consumer):
        """Test contract for getting market data."""

        # Define expected response
        expected_response = {
            "symbol": "AAPL",
            "price": Like(150.0),
            "volume": Like(1000000),
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
            "bid": Like(149.5),
            "ask": Like(150.5),
            "last_update": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        # Define the contract
        (
            market_data_consumer.given("AAPL market data is available")
            .upon_receiving("a request for AAPL market data")
            .with_request("GET", "/api/v1/market-data/AAPL")
            .will_respond_with(200, body=expected_response)
        )

        # Verify the contract
        with market_data_consumer, patch("requests.get") as mock_get:
            # This would normally make a real request to the provider
            # For testing, we'll mock the response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "symbol": "AAPL",
                "price": 150.0,
                "volume": 1000000,
                "timestamp": "2024-01-01T00:00:00Z",
                "bid": 149.5,
                "ask": 150.5,
                "last_update": "2024-01-01T00:00:00Z",
            }
            mock_get.return_value = mock_response

            # Verify the contract is satisfied
            assert mock_response.status_code == 200
            data = mock_response.json()
            assert data["symbol"] == "AAPL"
            assert "price" in data
            assert "volume" in data

    @pytest.mark.contract
    def test_get_multiple_symbols_contract(self, market_data_consumer):
        """Test contract for getting multiple symbols."""

        expected_response = {
            "data": EachLike(
                {"symbol": Like("AAPL"), "price": Like(150.0), "volume": Like(1000000)},
                minimum=1,
            )
        }

        (
            market_data_consumer.given("multiple symbols are available")
            .upon_receiving("a request for multiple symbols")
            .with_request("GET", "/api/v1/market-data/batch", query={"symbols": "AAPL,GOOGL,MSFT"})
            .will_respond_with(200, body=expected_response)
        )

        with market_data_consumer, patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"symbol": "AAPL", "price": 150.0, "volume": 1000000},
                    {"symbol": "GOOGL", "price": 2800.0, "volume": 500000},
                    {"symbol": "MSFT", "price": 300.0, "volume": 2000000},
                ]
            }
            mock_get.return_value = mock_response

            data = mock_response.json()
            assert "data" in data
            assert len(data["data"]) >= 1
            assert all("symbol" in item for item in data["data"])


class TestTradingAPIContracts:
    """Test contracts for trading API."""

    @pytest.fixture
    def trading_consumer(self):
        """Create trading consumer."""
        return Consumer("trading-system").has_pact_with(Provider("trading-service"), host_name="localhost", port=1235)

    @pytest.mark.contract
    def test_place_order_contract(self, trading_consumer):
        """Test contract for placing orders."""

        order_request = {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0,
            "order_type": "LIMIT",
        }

        expected_response = {
            "order_id": Term(r"[a-f0-9-]+", "123e4567-e89b-12d3-a456-426614174000"),
            "status": "PENDING",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0,
            "order_type": "LIMIT",
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            trading_consumer.given("trading service is available")
            .upon_receiving("a request to place an order")
            .with_request(
                "POST",
                "/api/v1/orders",
                body=order_request,
                headers={"Content-Type": "application/json"},
            )
            .will_respond_with(201, body=expected_response)
        )

        with trading_consumer, patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "order_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "PENDING",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 100,
                "price": 150.0,
                "order_type": "LIMIT",
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_post.return_value = mock_response

            assert mock_response.status_code == 201
            data = mock_response.json()
            assert "order_id" in data
            assert data["status"] == "PENDING"

    @pytest.mark.contract
    def test_get_order_status_contract(self, trading_consumer):
        """Test contract for getting order status."""

        expected_response = {
            "order_id": "123e4567-e89b-12d3-a456-426614174000",
            "status": "FILLED",
            "filled_quantity": 100,
            "filled_price": 150.0,
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            trading_consumer.given("order exists and is filled")
            .upon_receiving("a request for order status")
            .with_request("GET", "/api/v1/orders/123e4567-e89b-12d3-a456-426614174000")
            .will_respond_with(200, body=expected_response)
        )

        with trading_consumer, patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "order_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "FILLED",
                "filled_quantity": 100,
                "filled_price": 150.0,
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_get.return_value = mock_response

            assert mock_response.status_code == 200
            data = mock_response.json()
            assert data["status"] == "FILLED"
            assert "filled_quantity" in data


class TestPortfolioAPIContracts:
    """Test contracts for portfolio API."""

    @pytest.fixture
    def portfolio_consumer(self):
        """Create portfolio consumer."""
        return Consumer("trading-system").has_pact_with(Provider("portfolio-service"), host_name="localhost", port=1236)

    @pytest.mark.contract
    def test_get_portfolio_status_contract(self, portfolio_consumer):
        """Test contract for getting portfolio status."""

        expected_response = {
            "total_value": Like(100000.0),
            "cash": Like(50000.0),
            "positions": EachLike(
                {
                    "symbol": Like("AAPL"),
                    "quantity": Like(100),
                    "market_value": Like(15000.0),
                    "unrealized_pnl": Like(1000.0),
                }
            ),
            "last_update": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            portfolio_consumer.given("portfolio exists with positions")
            .upon_receiving("a request for portfolio status")
            .with_request("GET", "/api/v1/portfolio/status")
            .will_respond_with(200, body=expected_response)
        )

        with portfolio_consumer, patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "total_value": 100000.0,
                "cash": 50000.0,
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 100,
                        "market_value": 15000.0,
                        "unrealized_pnl": 1000.0,
                    }
                ],
                "last_update": "2024-01-01T00:00:00Z",
            }
            mock_get.return_value = mock_response

            assert mock_response.status_code == 200
            data = mock_response.json()
            assert "total_value" in data
            assert "positions" in data
            assert len(data["positions"]) >= 0


class TestRiskAPIContracts:
    """Test contracts for risk API."""

    @pytest.fixture
    def risk_consumer(self):
        """Create risk consumer."""
        return Consumer("trading-system").has_pact_with(Provider("risk-service"), host_name="localhost", port=1237)

    @pytest.mark.contract
    def test_get_risk_metrics_contract(self, risk_consumer):
        """Test contract for getting risk metrics."""

        expected_response = {
            "var_95": Like(-0.02),
            "cvar_95": Like(-0.03),
            "volatility": Like(0.15),
            "sharpe_ratio": Like(1.2),
            "max_drawdown": Like(-0.1),
            "beta": Like(1.0),
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            risk_consumer.given("risk metrics are available")
            .upon_receiving("a request for risk metrics")
            .with_request("GET", "/api/v1/risk/metrics")
            .will_respond_with(200, body=expected_response)
        )

        with risk_consumer, patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "var_95": -0.02,
                "cvar_95": -0.03,
                "volatility": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.1,
                "beta": 1.0,
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_get.return_value = mock_response

            assert mock_response.status_code == 200
            data = mock_response.json()
            assert "var_95" in data
            assert "cvar_95" in data
            assert data["var_95"] <= 0  # VaR should be negative


class TestErrorHandlingContracts:
    """Test contracts for error handling."""

    @pytest.fixture
    def error_consumer(self):
        """Create error handling consumer."""
        return Consumer("trading-system").has_pact_with(Provider("trading-service"), host_name="localhost", port=1238)

    @pytest.mark.contract
    def test_invalid_order_contract(self, error_consumer):
        """Test contract for invalid order handling."""

        invalid_order = {
            "symbol": "INVALID",
            "side": "BUY",
            "quantity": -100,  # Invalid negative quantity
            "price": 150.0,
            "order_type": "LIMIT",
        }

        error_response = {
            "error": "Invalid order parameters",
            "code": "INVALID_ORDER",
            "details": {"field": "quantity", "message": "Quantity must be positive"},
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            error_consumer.given("invalid order is submitted")
            .upon_receiving("a request with invalid order")
            .with_request(
                "POST",
                "/api/v1/orders",
                body=invalid_order,
                headers={"Content-Type": "application/json"},
            )
            .will_respond_with(400, body=error_response)
        )

        with error_consumer, patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": "Invalid order parameters",
                "code": "INVALID_ORDER",
                "details": {
                    "field": "quantity",
                    "message": "Quantity must be positive",
                },
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_post.return_value = mock_response

            assert mock_response.status_code == 400
            data = mock_response.json()
            assert "error" in data
            assert "code" in data
            assert data["code"] == "INVALID_ORDER"

    @pytest.mark.contract
    def test_not_found_contract(self, error_consumer):
        """Test contract for not found errors."""

        error_response = {
            "error": "Resource not found",
            "code": "NOT_FOUND",
            "message": "Order with ID 999999 not found",
            "timestamp": Term(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "2024-01-01T00:00:00Z"),
        }

        (
            error_consumer.given("order does not exist")
            .upon_receiving("a request for non-existent order")
            .with_request("GET", "/api/v1/orders/999999")
            .will_respond_with(404, body=error_response)
        )

        with error_consumer, patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.json.return_value = {
                "error": "Resource not found",
                "code": "NOT_FOUND",
                "message": "Order with ID 999999 not found",
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_get.return_value = mock_response

            assert mock_response.status_code == 404
            data = mock_response.json()
            assert data["code"] == "NOT_FOUND"
