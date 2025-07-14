from __future__ import annotations

"""Simple metrics collection utilities."""

from dataclasses import asdict, is_dataclass
from typing import Any, cast

from ..core.logging import get_logger


class MetricsCollector:
    """Collects and logs metrics during trading and backtesting."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.history: list[dict[str, Any]] = []

    def log_metrics(self, metrics: Any) -> None:
        """Log metrics and store them in memory."""
        metrics_dict = asdict(cast("Any", metrics)) if is_dataclass(metrics) else dict(metrics)

        self.history.append(metrics_dict)
        self.logger.info("Metrics logged", extra={"metrics": metrics_dict})

    def get_latest(self) -> dict[str, Any] | None:
        """Return the most recently logged metrics."""
        return self.history[-1] if self.history else None
