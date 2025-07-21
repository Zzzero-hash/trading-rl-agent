"""
Locust load testing for trading system.

This file defines load tests to verify system performance under various
concurrent user scenarios and load conditions.
"""

import random

from locust import HttpUser, between, events, task


class TradingSystemUser(HttpUser):
    """Simulates a user interacting with the trading system."""

    # Set default host for testing
    host = "http://localhost:8000"
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Initialize user session."""
        # Login or authenticate
        self.client.headers.update({"Content-Type": "application/json", "Authorization": "Bearer test-token"})

    @task(3)
    def get_market_data(self):
        """Get market data - high frequency task."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        symbol = random.choice(symbols)

        with self.client.get(f"/api/v1/market-data/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response structure
                if "price" in data and "volume" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_portfolio_status(self):
        """Get portfolio status - medium frequency task."""
        with self.client.get("/api/v1/portfolio/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate portfolio data
                if "total_value" in data and "positions" in data:
                    response.success()
                else:
                    response.failure("Invalid portfolio structure")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def place_order(self):
        """Place trading order - low frequency but high impact task."""
        order_data = {
            "symbol": random.choice(["AAPL", "GOOGL", "MSFT"]),
            "side": random.choice(["BUY", "SELL"]),
            "quantity": random.randint(1, 100),
            "price": round(random.uniform(100, 500), 2),
            "order_type": "LIMIT",
        }

        with self.client.post("/api/v1/orders", json=order_data, catch_response=True) as response:
            if response.status_code in [200, 201]:
                data = response.json()
                if "order_id" in data:
                    response.success()
                else:
                    response.failure("Missing order ID in response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def get_risk_metrics(self):
        """Get risk metrics - medium frequency task."""
        with self.client.get("/api/v1/risk/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate risk metrics
                if "var" in data and "cvar" in data:
                    response.success()
                else:
                    response.failure("Invalid risk metrics structure")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_performance_metrics(self):
        """Get performance metrics - low frequency task."""
        with self.client.get("/api/v1/performance/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate performance metrics
                if "sharpe_ratio" in data and "returns" in data:
                    response.success()
                else:
                    response.failure("Invalid performance metrics structure")
            else:
                response.failure(f"HTTP {response.status_code}")


class HighFrequencyTrader(HttpUser):
    """Simulates a high-frequency trading user."""

    # Set default host for testing
    host = "http://localhost:8000"
    wait_time = between(0.1, 0.5)  # Very fast requests

    def on_start(self):
        """Initialize HFT user session."""
        self.client.headers.update({"Content-Type": "application/json", "Authorization": "Bearer hft-token"})

    @task(5)
    def get_realtime_data(self):
        """Get real-time market data - very high frequency."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        symbol = random.choice(symbols)

        with self.client.get(f"/api/v1/realtime/{symbol}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def place_market_order(self):
        """Place market order - high frequency."""
        order_data = {
            "symbol": random.choice(["AAPL", "GOOGL", "MSFT"]),
            "side": random.choice(["BUY", "SELL"]),
            "quantity": random.randint(1, 50),
            "order_type": "MARKET",
        }

        with self.client.post("/api/v1/orders/market", json=order_data, catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def cancel_order(self):
        """Cancel order - medium frequency."""
        order_id = f"order_{random.randint(1000, 9999)}"

        with self.client.delete(f"/api/v1/orders/{order_id}", catch_response=True) as response:
            if response.status_code in [
                200,
                404,
            ]:  # 404 is acceptable for non-existent orders
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class RiskManagerUser(HttpUser):
    """Simulates a risk manager user."""

    # Set default host for testing
    host = "http://localhost:8000"
    wait_time = between(5, 15)  # Slower, more thoughtful requests

    def on_start(self):
        """Initialize risk manager session."""
        self.client.headers.update({"Content-Type": "application/json", "Authorization": "Bearer risk-token"})

    @task(3)
    def get_risk_report(self):
        """Get comprehensive risk report."""
        with self.client.get("/api/v1/risk/report", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate comprehensive risk report
                required_fields = ["var", "cvar", "position_limits", "exposure"]
                if all(field in data for field in required_fields):
                    response.success()
                else:
                    response.failure("Missing required risk report fields")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def update_risk_limits(self):
        """Update risk limits."""
        limit_data = {
            "max_position_size": random.randint(1000, 10000),
            "max_daily_loss": random.randint(10000, 100000),
            "var_limit": random.uniform(0.01, 0.05),
        }

        with self.client.put("/api/v1/risk/limits", json=limit_data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_audit_log(self):
        """Get audit log for compliance."""
        with self.client.get("/api/v1/audit/log", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "entries" in data:
                    response.success()
                else:
                    response.failure("Invalid audit log structure")
            else:
                response.failure(f"HTTP {response.status_code}")


class DataAnalystUser(HttpUser):
    """Simulates a data analyst user."""

    # Set default host for testing
    host = "http://localhost:8000"
    wait_time = between(10, 30)  # Very slow, analytical requests

    def on_start(self):
        """Initialize data analyst session."""
        self.client.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer analyst-token",
            }
        )

    @task(2)
    def get_historical_data(self):
        """Get historical data for analysis."""
        symbol = random.choice(["AAPL", "GOOGL", "MSFT", "TSLA"])
        start_date = "2024-01-01"
        end_date = "2024-12-31"

        with self.client.get(
            f"/api/v1/historical/{symbol}?start={start_date}&end={end_date}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    response.success()
                else:
                    response.failure("No historical data returned")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def run_backtest(self):
        """Run backtest analysis."""
        backtest_config = {
            "strategy": "momentum",
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
        }

        with self.client.post("/api/v1/backtest", json=backtest_config, catch_response=True) as response:
            if response.status_code in [200, 202]:  # 202 for async processing
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# Custom event handlers for monitoring
@events.request.add_listener
def my_request_handler(
    _request_type,
    name,
    _response_time,
    _response_length,
    _response,
    _context,
    exception,
    _start_time,
    _url,
    **_kwargs,
):
    """Custom request handler for detailed monitoring."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response.status_code >= 400:
        print(f"Request error: {name} - HTTP {response.status_code}")


@events.test_start.add_listener
def on_test_start(_environment, **_kwargs):
    """Called when test starts."""
    print("Load test starting...")


@events.test_stop.add_listener
def on_test_stop(_environment, **_kwargs):
    """Called when test stops."""
    print("Load test completed.")


# Configuration for different load scenarios
class LoadTestConfig:
    """Configuration for different load test scenarios."""

    @staticmethod
    def get_normal_load_config():
        """Normal trading hours load."""
        return {"users": 50, "spawn_rate": 5, "run_time": "10m"}

    @staticmethod
    def get_high_load_config():
        """High load scenario (market open/close)."""
        return {"users": 200, "spawn_rate": 20, "run_time": "5m"}

    @staticmethod
    def get_stress_test_config():
        """Stress test scenario."""
        return {"users": 500, "spawn_rate": 50, "run_time": "3m"}

    @staticmethod
    def get_spike_test_config():
        """Spike test scenario."""
        return {"users": 1000, "spawn_rate": 100, "run_time": "1m"}
