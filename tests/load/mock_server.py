#!/usr/bin/env python3
"""
Mock server for load testing.

This provides a simple FastAPI server that simulates the trading system
APIs for load testing purposes.
"""

import asyncio
import random
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Trading System Mock API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: int
    price: float
    order_type: str


class OrderResponse(BaseModel):
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: int
    price: float
    order_type: str
    timestamp: str


# Mock data
MOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
MOCK_ORDERS = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Trading System Mock API", "status": "running"}


@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol."""
    if symbol not in MOCK_SYMBOLS:
        raise HTTPException(status_code=404, detail="Symbol not found")

    # Simulate some latency
    await asyncio.sleep(random.uniform(0.01, 0.05))

    return {
        "symbol": symbol,
        "price": round(random.uniform(100, 500), 2),
        "volume": random.randint(100000, 5000000),
        "timestamp": datetime.now().isoformat(),
        "bid": round(random.uniform(99, 499), 2),
        "ask": round(random.uniform(101, 501), 2),
        "last_update": datetime.now().isoformat(),
    }


@app.get("/api/v1/market-data/batch")
async def get_multiple_symbols(symbols: str):
    """Get market data for multiple symbols."""
    symbol_list = symbols.split(",")

    # Simulate some latency
    await asyncio.sleep(random.uniform(0.02, 0.1))

    data = []
    for symbol in symbol_list:
        if symbol in MOCK_SYMBOLS:
            data.append(
                {
                    "symbol": symbol,
                    "price": round(random.uniform(100, 500), 2),
                    "volume": random.randint(100000, 5000000),
                }
            )

    return {"data": data}


@app.get("/api/v1/portfolio/status")
async def get_portfolio_status():
    """Get portfolio status."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.05, 0.15))

    return {
        "total_value": round(random.uniform(50000, 200000), 2),
        "cash": round(random.uniform(10000, 50000), 2),
        "positions": [
            {
                "symbol": "AAPL",
                "quantity": random.randint(10, 100),
                "market_value": round(random.uniform(5000, 25000), 2),
                "unrealized_pnl": round(random.uniform(-5000, 5000), 2),
            }
        ],
        "last_update": datetime.now().isoformat(),
    }


@app.post("/api/v1/orders", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    """Place a trading order."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.1, 0.3))

    order_id = f"order_{int(time.time() * 1000)}"

    response = OrderResponse(
        order_id=order_id,
        status="PENDING",
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=order.price,
        order_type=order.order_type,
        timestamp=datetime.now().isoformat(),
    )

    MOCK_ORDERS[order_id] = response.dict()

    return response


@app.post("/api/v1/orders/market")
async def place_market_order(order: OrderRequest):
    """Place a market order."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.05, 0.15))

    order_id = f"market_order_{int(time.time() * 1000)}"

    return {
        "order_id": order_id,
        "status": "FILLED",
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "filled_price": round(random.uniform(100, 500), 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/orders/{order_id}")
async def get_order_status(order_id: str):
    """Get order status."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.02, 0.08))

    if order_id in MOCK_ORDERS:
        order = MOCK_ORDERS[order_id]
        order["status"] = "FILLED"
        order["filled_quantity"] = order["quantity"]
        order["filled_price"] = order["price"]
        return order

    raise HTTPException(status_code=404, detail="Order not found")


@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.02, 0.08))

    if order_id in MOCK_ORDERS:
        MOCK_ORDERS[order_id]["status"] = "CANCELLED"
        return {"message": "Order cancelled successfully"}

    # Return 404 for non-existent orders (this is acceptable for testing)
    raise HTTPException(status_code=404, detail="Order not found")


@app.get("/api/v1/risk/metrics")
async def get_risk_metrics():
    """Get risk metrics."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.1, 0.2))

    return {
        "var_95": round(random.uniform(-0.05, -0.01), 4),
        "cvar_95": round(random.uniform(-0.08, -0.02), 4),
        "volatility": round(random.uniform(0.1, 0.3), 4),
        "sharpe_ratio": round(random.uniform(-1, 2), 2),
        "max_drawdown": round(random.uniform(-0.2, 0), 4),
        "beta": round(random.uniform(0.8, 1.2), 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.15, 0.25))

    return {
        "sharpe_ratio": round(random.uniform(-1, 2), 2),
        "returns": round(random.uniform(-0.1, 0.2), 4),
        "volatility": round(random.uniform(0.1, 0.3), 4),
        "max_drawdown": round(random.uniform(-0.2, 0), 4),
        "calmar_ratio": round(random.uniform(-2, 3), 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/risk/report")
async def get_risk_report():
    """Get comprehensive risk report."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.2, 0.4))

    return {
        "var": round(random.uniform(-0.05, -0.01), 4),
        "cvar": round(random.uniform(-0.08, -0.02), 4),
        "position_limits": {
            "max_position_size": random.randint(1000, 10000),
            "max_daily_loss": random.randint(10000, 100000),
        },
        "exposure": {
            "total_exposure": round(random.uniform(0.5, 0.9), 2),
            "sector_exposure": {
                "technology": round(random.uniform(0.2, 0.6), 2),
                "finance": round(random.uniform(0.1, 0.4), 2),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.put("/api/v1/risk/limits")
async def update_risk_limits(limits: dict):
    """Update risk limits."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.1, 0.2))

    return {
        "message": "Risk limits updated successfully",
        "limits": limits,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/audit/log")
async def get_audit_log():
    """Get audit log."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.05, 0.15))

    return {
        "entries": [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "ORDER_PLACED",
                "user": "test_user",
                "details": "Order AAPL BUY 100 @ 150.00",
            }
        ]
    }


@app.get("/api/v1/realtime/{symbol}")
async def get_realtime_data(symbol: str):
    """Get real-time market data."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.001, 0.01))

    if symbol not in MOCK_SYMBOLS:
        raise HTTPException(status_code=404, detail="Symbol not found")

    return {
        "symbol": symbol,
        "price": round(random.uniform(100, 500), 2),
        "volume": random.randint(100000, 5000000),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/historical/{symbol}")
async def get_historical_data(symbol: str, _start: str = "2024-01-01", _end: str = "2024-12-31"):
    """Get historical data."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(0.5, 1.0))

    if symbol not in MOCK_SYMBOLS:
        raise HTTPException(status_code=404, detail="Symbol not found")

    # Generate mock historical data
    data = []
    for i in range(100):
        data.append(
            {
                "date": f"2024-{i // 30 + 1:02d}-{i % 30 + 1:02d}",
                "price": round(random.uniform(100, 500), 2),
                "volume": random.randint(100000, 5000000),
            }
        )

    return {"data": data}


@app.post("/api/v1/backtest")
async def run_backtest(_config: dict):
    """Run backtest analysis."""
    # Simulate some latency
    await asyncio.sleep(random.uniform(2.0, 5.0))

    return {
        "backtest_id": f"backtest_{int(time.time() * 1000)}",
        "status": "COMPLETED",
        "results": {
            "total_return": round(random.uniform(-0.1, 0.3), 4),
            "sharpe_ratio": round(random.uniform(-1, 2), 2),
            "max_drawdown": round(random.uniform(-0.2, 0), 4),
            "win_rate": round(random.uniform(0.4, 0.7), 2),
        },
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
