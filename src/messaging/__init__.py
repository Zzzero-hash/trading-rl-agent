"""Messaging utilities for NATS."""

from nats import connect
from .nats_utils import NATS

__all__ = ["connect", "NATS"]
