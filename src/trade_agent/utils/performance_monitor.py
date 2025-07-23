import time
from contextlib import contextmanager
from typing import Any

from rich.console import Console

console = Console()

class PerformanceMonitor:
    """Monitor pipeline performance metrics."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    @contextmanager
    def time_operation(self, operation_name: str) -> Any:
        """Time an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics[operation_name] = duration
            console.print(f"[cyan]⏱️  {operation_name}: {duration:.2f}s[/cyan]")

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        total_time = sum(self.metrics.values())
        return {
            "total_time": total_time,
            "operations": self.metrics,
            "slowest_operation": max(self.metrics.items(), key=lambda x: x[1]) if self.metrics else None
        }
