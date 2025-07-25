import logging
import time
from collections.abc import Callable
from typing import Any

import ray

from .cluster import (
    check_ray_cluster_status,
    get_optimal_worker_count,
    init_ray,
    print_cluster_info,
    validate_cluster_health,
    wait_for_cluster_ready,
)

logger = logging.getLogger(__name__)


@ray.remote
def _execute_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute a function as a Ray remote task."""
    return func(*args, **kwargs)


@ray.remote
class ProgressTracker:
    """Remote actor to track progress across distributed tasks."""

    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.task_results: dict[str, Any] = {}

    def update_progress(self, task_id: str, success: bool = True, result: Any = None) -> None:
        """Update progress for a completed task."""
        if success:
            self.completed_tasks += 1
            self.task_results[task_id] = result
        else:
            self.failed_tasks += 1

        if (self.completed_tasks + self.failed_tasks) % max(1, self.total_tasks // 10) == 0:
            progress = (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100
            elapsed = time.time() - self.start_time
            logger.info(f"Progress: {progress:.1f}% ({self.completed_tasks} completed, {self.failed_tasks} failed) - {elapsed:.1f}s elapsed")

    def get_status(self) -> dict[str, Any]:
        """Get current status."""
        elapsed = time.time() - self.start_time
        progress = (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "progress_percent": progress,
            "elapsed_time": elapsed,
            "is_complete": (self.completed_tasks + self.failed_tasks) >= self.total_tasks
        }


def robust_ray_init(
    address: str | None = None,
    max_workers: int | None = None,
    local_mode: bool = False,
    show_cluster_info: bool = True,
    **ray_init_kwargs: Any
) -> tuple[bool, dict[str, Any]]:
    """
    Robustly initialize Ray with automatic cluster detection and fallback.

    Args:
        address: Ray cluster address (None for auto-detection)
        max_workers: Maximum number of workers (None for auto-detection based on resources)
        local_mode: Force local mode
        show_cluster_info: Whether to display cluster information after initialization
        **ray_init_kwargs: Additional arguments to pass to ray.init()

    Returns:
        Tuple of (success: bool, info: dict) with initialization details
    """
    logger.info("🚀 Starting robust Ray initialization...")

    # First, check if Ray is already initialized
    if ray.is_initialized():
        logger.info("Ray is already initialized")
        if show_cluster_info:
            print_cluster_info()
        health = validate_cluster_health()
        return True, {
            "status": "already_initialized",
            "mode": "existing",
            "health": health
        }

    # Check cluster status first if no address specified
    if address is None and not local_mode:
        logger.info("🔍 Checking for existing Ray cluster...")
        cluster_status = check_ray_cluster_status()

        if cluster_status["available"] and cluster_status["status"] == "connected":
            logger.info("✅ Found existing Ray cluster")
            address = "auto"  # Let Ray auto-discover
        else:
            logger.info("❌ No Ray cluster found, will start in local mode")
            local_mode = True

    # Use the enhanced init_ray function
    success, init_info = init_ray(
        address=address,
        local_mode=local_mode,
        **ray_init_kwargs
    )

    if not success:
        logger.error("❌ Failed to initialize Ray")
        return False, init_info

    # Wait for cluster to be ready
    if not wait_for_cluster_ready(timeout=30.0):
        logger.warning("⚠️ Cluster may not be fully ready, but continuing...")

    # Auto-detect optimal worker count if not specified
    if max_workers is None:
        worker_info = get_optimal_worker_count()
        max_workers = worker_info["total_workers"]
        logger.info(f"🔧 Auto-detected optimal worker count: {max_workers}")

    # Display cluster information
    if show_cluster_info:
        print_cluster_info()

    success_info = {
        **init_info,
        "max_workers": max_workers,
        "initialization_time": time.time()
    }

    logger.info("✅ Ray initialization completed successfully")
    return True, success_info


def parallel_execute(
    func: Callable[..., Any],
    items: list[Any],
    *args: Any,
    max_workers: int | None = None,
    chunk_size: int | None = None,
    show_progress: bool = True,
    retry_failed: bool = True,
    max_retries: int = 2,
    **kwargs: Any,
) -> list[Any]:
    """
    Execute a function in parallel using Ray with enhanced error handling and monitoring.

    Args:
        func: Function to execute in parallel
        items: List of items to process
        *args: Additional positional arguments for func
        max_workers: Maximum number of workers (auto-detected if None)
        chunk_size: Size of chunks to process (auto-calculated if None)
        show_progress: Whether to show progress updates
        retry_failed: Whether to retry failed tasks
        max_retries: Maximum number of retries for failed tasks
        **kwargs: Additional keyword arguments for func

    Returns:
        List of results in the same order as input items
    """
    if not items:
        logger.warning("No items provided for parallel execution")
        return []

    # Initialize Ray if not already done
    if not ray.is_initialized():
        logger.info("Ray not initialized, initializing now...")
        success, _ = robust_ray_init(max_workers=max_workers, show_cluster_info=False)
        if not success:
            logger.error("Failed to initialize Ray, falling back to sequential execution")
            return [func(item, *args, **kwargs) for item in items]

    # Auto-detect optimal parameters
    if max_workers is None:
        worker_info = get_optimal_worker_count()
        max_workers = min(worker_info["total_workers"], len(items))

    if chunk_size is None:
        # Calculate optimal chunk size based on number of items and workers
        chunk_size = max(1, len(items) // (max_workers * 4))  # 4 chunks per worker

    logger.info(f"🔧 Parallel execution config: {len(items)} items, {max_workers} workers, chunk size: {chunk_size}")

    # Create progress tracker if requested
    progress_tracker = None
    if show_progress:
        progress_tracker = ProgressTracker.remote(len(items))  # type: ignore[attr-defined]

    # Submit tasks in chunks to avoid overwhelming the cluster
    tasks = []
    results = [None] * len(items)  # Pre-allocate results list
    failed_indices = []

    # Submit initial batch of tasks
    for i, item in enumerate(items):
        task = _execute_task.remote(func, item, *args, **kwargs)
        tasks.append((i, task))

    # Collect results
    start_time = time.time()
    completed_tasks = 0

    while tasks:
        # Wait for at least one task to complete
        ready_tasks, remaining_tasks = ray.wait([task for _, task in tasks], num_returns=1, timeout=1.0)

        if ready_tasks:
            # Process completed tasks
            new_tasks = []
            for i, task in tasks:
                if task in ready_tasks:
                    try:
                        result = ray.get(task)
                        results[i] = result
                        completed_tasks += 1

                        if progress_tracker:
                            progress_tracker.update_progress.remote(f"task_{i}", success=True, result=result)

                    except Exception as e:
                        logger.warning(f"Task {i} failed: {e}")
                        failed_indices.append(i)

                        if progress_tracker:
                            progress_tracker.update_progress.remote(f"task_{i}", success=False)
                else:
                    new_tasks.append((i, task))

            tasks = new_tasks

        # Log progress periodically
        if show_progress and completed_tasks > 0 and completed_tasks % max(1, len(items) // 10) == 0:
            elapsed = time.time() - start_time
            progress = completed_tasks / len(items) * 100
            logger.info(f"Progress: {progress:.1f}% ({completed_tasks}/{len(items)}) - {elapsed:.1f}s elapsed")

    # Retry failed tasks if requested
    if retry_failed and failed_indices:
        logger.info(f"🔄 Retrying {len(failed_indices)} failed tasks (max {max_retries} retries)...")

        for retry_attempt in range(max_retries):
            if not failed_indices:
                break

            retry_tasks = []
            for i in failed_indices[:]:  # Copy list to allow modification
                item = items[i]
                task = _execute_task.remote(func, item, *args, **kwargs)
                retry_tasks.append((i, task))

            # Process retry tasks
            while retry_tasks:
                ready_tasks, _ = ray.wait([task for _, task in retry_tasks], num_returns=1, timeout=5.0)

                if ready_tasks:
                    new_retry_tasks = []
                    for i, task in retry_tasks:
                        if task in ready_tasks:
                            try:
                                result = ray.get(task)
                                results[i] = result
                                failed_indices.remove(i)
                                logger.info(f"✅ Retry successful for task {i}")

                                if progress_tracker:
                                    progress_tracker.update_progress.remote(f"retry_task_{i}", success=True, result=result)

                            except Exception as e:
                                logger.warning(f"Retry failed for task {i}: {e}")
                                new_retry_tasks.append((i, task))
                        else:
                            new_retry_tasks.append((i, task))

                    retry_tasks = new_retry_tasks
                else:
                    # Timeout waiting for retry tasks
                    break

    # Handle any remaining failed tasks
    if failed_indices:
        logger.warning(f"⚠️ {len(failed_indices)} tasks failed after all retries")
        for i in failed_indices:
            results[i] = None  # or some error indicator

    # Final progress update
    elapsed = time.time() - start_time
    success_count = len([r for r in results if r is not None])

    if progress_tracker:
        ray.get(progress_tracker.get_status.remote())
        logger.info(f"✅ Parallel execution completed: {success_count}/{len(items)} successful in {elapsed:.1f}s")
    else:
        logger.info(f"✅ Parallel execution completed: {success_count}/{len(items)} successful in {elapsed:.1f}s")

    return results


def parallel_map(
    func: Callable[[Any], Any],
    items: list[Any],
    max_workers: int | None = None,
    **kwargs: Any
) -> list[Any]:
    """
    Simple parallel map function using Ray.

    Args:
        func: Function to map over items
        items: List of items to process
        max_workers: Maximum number of workers
        **kwargs: Additional arguments for parallel_execute

    Returns:
        List of mapped results
    """
    return parallel_execute(func, items, max_workers=max_workers, **kwargs)


def distributed_batch_process(
    func: Callable[..., Any],
    items: list[Any],
    batch_size: int,
    *args: Any,
    max_workers: int | None = None,
    **kwargs: Any
) -> list[Any]:
    """
    Process items in batches across distributed workers.

    Args:
        func: Function that processes a batch of items
        items: List of items to process
        batch_size: Size of each batch
        *args: Additional positional arguments
        max_workers: Maximum number of workers
        **kwargs: Additional keyword arguments

    Returns:
        Flattened list of results from all batches
    """
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    logger.info(f"🔧 Processing {len(items)} items in {len(batches)} batches of size {batch_size}")

    # Process batches in parallel
    batch_results = parallel_execute(
        func, batches, *args,
        max_workers=max_workers,
        show_progress=True,
        **kwargs
    )

    # Flatten results
    flattened_results = []
    for batch_result in batch_results:
        if batch_result is not None:
            if isinstance(batch_result, list):
                flattened_results.extend(batch_result)
            else:
                flattened_results.append(batch_result)

    return flattened_results


def get_ray_cluster_info() -> dict[str, Any]:
    """
    Get comprehensive information about the current Ray cluster.

    Returns:
        Dictionary with cluster information
    """
    if not ray.is_initialized():
        return {
            "initialized": False,
            "error": "Ray not initialized"
        }

    try:
        health = validate_cluster_health()
        worker_rec = get_optimal_worker_count()

        return {
            "initialized": True,
            "health": health,
            "worker_recommendations": worker_rec,
            "cluster_resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
            "nodes": ray.nodes()
        }
    except Exception as e:
        return {
            "initialized": True,
            "error": f"Failed to get cluster info: {e}"
        }


def cleanup_ray(force: bool = False) -> None:
    """
    Safely cleanup Ray resources.

    Args:
        force: Whether to force shutdown even if tasks are running
    """
    if ray.is_initialized():
        try:
            if not force:
                # Check if there are running tasks
                cluster_info = get_ray_cluster_info()
                if cluster_info.get("health", {}).get("healthy", False):
                    logger.info("🧹 Gracefully shutting down Ray...")
                else:
                    logger.warning("⚠️ Ray cluster not healthy, forcing shutdown...")
                    force = True

            ray.shutdown()
            logger.info("✅ Ray shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during Ray shutdown: {e}")
    else:
        logger.info("Ray is not initialized, nothing to cleanup")


# Backward compatibility
def parallel_execute_legacy(
    func: Callable[..., Any],
    items: list[Any],
    *args: Any,
    max_workers: int = 4,
    **kwargs: Any,
) -> list[Any]:
    """Legacy parallel execute function for backward compatibility."""
    return parallel_execute(
        func, items, *args,
        max_workers=max_workers,
        show_progress=False,
        **kwargs
    )
