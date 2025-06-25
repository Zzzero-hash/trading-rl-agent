"""
NATS client utilities for trading data streaming.
Provides async message publishing/subscribing for market data.
"""

import asyncio
from datetime import datetime
import json
import logging
from typing import Any, Callable, Dict, Optional

import nats
from nats.errors import ConnectionClosedError, TimeoutError

logger = logging.getLogger(__name__)


class TradingNATSClient:
    """NATS client for trading data streaming."""

    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[nats.jetstream.JetStreamContext] = None

    async def connect(self) -> bool:
        """Connect to NATS server with retry logic."""
        try:
            self.nc = await nats.connect(
                servers=[self.nats_url],
                connect_timeout=10,
                max_reconnect_attempts=5,
                reconnect_time_wait=2,
            )

            # Enable JetStream for persistence
            self.js = self.nc.jetstream()

            logger.info(f"Connected to NATS at {self.nats_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False

    async def disconnect(self):
        """Gracefully disconnect from NATS."""
        if self.nc:
            await self.nc.close()
            logger.info("Disconnected from NATS")

    async def publish_market_data(self, symbol: str, data: dict[str, Any]) -> bool:
        """Publish market data for a symbol."""
        if not self.nc:
            logger.error("Not connected to NATS")
            return False

        try:
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.utcnow().isoformat()

            subject = f"market.data.{symbol}"
            message = json.dumps(data).encode()

            await self.nc.publish(subject, message)
            logger.debug(f"Published to {subject}: {data}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish market data: {e}")
            return False

    async def subscribe_market_data(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], None],
        queue_group: Optional[str] = None,
    ) -> bool:
        """Subscribe to market data for a symbol."""
        if not self.nc:
            logger.error("Not connected to NATS")
            return False

        try:
            subject = f"market.data.{symbol}"

            async def message_handler(msg):
                try:
                    data = json.loads(msg.data.decode())
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

            await self.nc.subscribe(subject, cb=message_handler, queue=queue_group)

            logger.info(f"Subscribed to {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False

    async def publish_signal(self, signal_type: str, data: dict[str, Any]) -> bool:
        """Publish trading signals (buy/sell/hold)."""
        if not self.nc:
            return False

        try:
            subject = f"trading.signals.{signal_type}"
            data["timestamp"] = datetime.utcnow().isoformat()
            message = json.dumps(data).encode()

            await self.nc.publish(subject, message)
            logger.info(f"Published signal to {subject}: {data}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")
            return False

    async def subscribe_signals(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> bool:
        """Subscribe to all trading signals."""
        if not self.nc:
            return False

        try:

            async def signal_handler(msg):
                try:
                    # Extract signal type from subject
                    signal_type = msg.subject.split(".")[-1]
                    data = json.loads(msg.data.decode())
                    await callback(signal_type, data)
                except Exception as e:
                    logger.error(f"Error in signal handler: {e}")

            await self.nc.subscribe("trading.signals.*", cb=signal_handler)
            logger.info("Subscribed to trading signals")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to signals: {e}")
            return False

    async def create_stream(self, name: str, subjects: list) -> bool:
        """Create a JetStream stream for persistence."""
        if not self.js:
            return False

        try:
            await self.js.add_stream(
                name=name,
                subjects=subjects,
                max_age=24 * 60 * 60,  # 24 hours retention
                max_bytes=100 * 1024 * 1024,  # 100MB max
            )
            logger.info(f"Created stream '{name}' for subjects: {subjects}")
            return True

        except Exception as e:
            logger.error(f"Failed to create stream: {e}")
            return False


# Example usage and testing
async def example_usage():
    """Example of how to use the NATS client."""
    client = TradingNATSClient()

    if not await client.connect():
        return

    # Create streams for persistence
    await client.create_stream("MARKET_DATA", ["market.data.*"])
    await client.create_stream("TRADING_SIGNALS", ["trading.signals.*"])

    # Subscribe to market data
    async def handle_market_data(data):
        print(f"Received market data: {data}")

    await client.subscribe_market_data("AAPL", handle_market_data)

    # Subscribe to trading signals
    async def handle_signals(signal_type, data):
        print(f"Received {signal_type} signal: {data}")

    await client.subscribe_signals(handle_signals)

    # Publish some test data
    await client.publish_market_data(
        "AAPL", {"price": 150.25, "volume": 1000, "bid": 150.20, "ask": 150.30}
    )

    await client.publish_signal(
        "buy", {"symbol": "AAPL", "quantity": 100, "price": 150.25, "confidence": 0.85}
    )

    # Keep running to receive messages
    await asyncio.sleep(5)

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
