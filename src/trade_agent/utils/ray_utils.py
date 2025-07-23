from collections.abc import Callable
from typing import Any

import ray


@ray.remote
def _execute_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute a function as a Ray remote task."""
    return func(*args, **kwargs)

def parallel_execute(
    func: Callable[..., Any],
    items: list[Any],
    *args: Any,
    max_workers: int = 4,
    **kwargs: Any,
) -> list[Any]:
    """Execute a function in parallel using Ray."""
    if not ray.is_initialized():
        ray.init(num_cpus=max_workers)

    tasks = [_execute_task.remote(func, item, *args, **kwargs) for item in items]
    results: list[Any] = ray.get(tasks)
    return results
