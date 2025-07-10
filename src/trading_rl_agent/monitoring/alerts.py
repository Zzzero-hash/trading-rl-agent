from __future__ import annotations

"""Simple alert management utilities."""


from ..core.logging import get_logger


class AlertManager:
    """Sends alerts based on monitoring events."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.alerts: list[str] = []

    def send_alert(self, message: str) -> None:
        """Send an alert message."""
        self.alerts.append(message)
        self.logger.warning("ALERT: %s", message)
