"""Minimal utilities for working with the NATS client.

The previous iteration of this module exposed ``connect``/``disconnect`` as
custom wrappers along with ``publish`` and ``subscribe`` helpers.  The NATS
client now natively supports an asynchronous context manager via
``nats.connect``.  Call sites should use::

    async with nats.connect(servers=["nats://localhost:4222"]) as nc:
        await nc.publish("topic", b"hello")

``nc.publish`` and ``nc.subscribe`` should be invoked directly without wrapper
functions.  This module only re-exports the ``NATS`` client class for typing
purposes.
"""

from nats.aio.client import Client as NATS

__all__ = ["NATS"]
