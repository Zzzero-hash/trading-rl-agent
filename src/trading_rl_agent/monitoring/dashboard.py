from __future__ import annotations

"""Minimal dashboard utilities."""

from dataclasses import asdict, is_dataclass
from typing import Any, cast

from ..core.logging import get_logger


class Dashboard:
    """In-memory dashboard for monitoring metrics."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.data: list[dict[str, Any]] = []

    def update(self, metrics: Any) -> None:
        """Update the dashboard with new metrics."""
        metrics_dict = asdict(cast(Any, metrics)) if is_dataclass(metrics) else dict(metrics)

        self.data.append(metrics_dict)
        self.logger.info("Dashboard update", extra={"metrics": metrics_dict})

    def get_latest(self) -> dict[str, Any] | None:
        """Get the most recent dashboard entry."""
        return self.data[-1] if self.data else None
