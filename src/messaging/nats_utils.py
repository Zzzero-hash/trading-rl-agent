"""NATS messaging utilities.

This module provides lightweight helper functions for connecting to a NATS
server and publishing/subscribing to subjects. It uses the asynchronous NATS
client directly without an additional wrapper class.
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
import json
import logging
from typing import Any, Callable, Optional

from nats.aio.client import Client as NATS

logger = logging.getLogger(__name__)


async def connect(nats_url: str = "nats://localhost:4222") -> NATS:
    """Create and connect a NATS client.

    Args:
        nats_url: URL of the NATS server.

    Returns:
        Connected ``NATS`` client instance.
    """

    nc = NATS()
    await nc.connect(
        servers=[nats_url],
        connect_timeout=10,
        max_reconnect_attempts=5,
        reconnect_time_wait=2,
    )
    logger.info(f"Connected to NATS at {nats_url}")
    return nc


async def disconnect(nc: NATS) -> None:
    """Gracefully close the NATS connection."""
    await nc.close()
    logger.info("Disconnected from NATS")


async def publish(nc: NATS, subject: str, data: dict[str, Any]) -> None:
    """Publish a JSON-encoded message to a subject."""
    data = (
        data.copy()
    )  # Create a shallow copy to avoid mutating the caller's dictionary
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat()
    message = json.dumps(data).encode()
    await nc.publish(subject, message)
    logger.debug(f"Published to {subject}: {data}")


async def subscribe(
    nc: NATS,
    subject: str,
    callback: Callable[[dict[str, Any]], Awaitable[None]],
    queue: str | None = None,
) -> None:
    """Subscribe to a subject and process messages with ``callback``."""

    async def handler(msg):
        try:
            payload = json.loads(msg.data.decode())
            await callback(payload)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error(f"Error in NATS handler: {exc}")

    await nc.subscribe(subject, cb=handler, queue=queue)
    logger.info(f"Subscribed to {subject}")
