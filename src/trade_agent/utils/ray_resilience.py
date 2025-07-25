"""
Ray Resilience and Error Handling Module.

This module provides robust error handling, fault tolerance, and resilience patterns
for Ray-based distributed computing applications.
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any

import ray

from .cluster import get_optimal_worker_count, validate_cluster_health
from .ray_utils import robust_ray_init

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in Ray operations."""
    TASK_FAILURE = "task_failure"
    WORKER_FAILURE = "worker_failure"
    CLUSTER_FAILURE = "cluster_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies for handling failures."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


class RayTaskFailure(Exception):
    """Exception raised when Ray task execution fails."""

    def __init__(
        self,
        message: str,
        failure_type: FailureType = FailureType.UNKNOWN,
        task_id: str | None = None,
        worker_id: str | None = None,
        retry_count: int = 0,
        original_exception: Exception | None = None
    ):
        super().__init__(message)
        self.failure_type = failure_type
        self.task_id = task_id
        self.worker_id = worker_id
        self.retry_count = retry_count
        self.original_exception = original_exception
        self.timestamp = time.time()


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for Ray operations.

    Prevents cascading failures by temporarily stopping operations
    when failure rate exceeds threshold.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying to close circuit (seconds)
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._call(func, *args, **kwargs)
        return wrapper

    def _call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        current_time = time.time()

        # Check if circuit should be half-open
        if (self.state == "open" and
            current_time - self.last_failure_time >= self.timeout):
            self.state = "half_open"
            logger.info("Circuit breaker transitioning to half-open state")

        # Reject calls when circuit is open
        if self.state == "open":
            raise RayTaskFailure(
                f"Circuit breaker is open. Last failure: {current_time - self.last_failure_time:.1f}s ago",
                failure_type=FailureType.CLUSTER_FAILURE
            )

        try:
            result = func(*args, **kwargs)

            # Success - reset failure count if half-open or closed
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful operation")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time

            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

            raise e


def with_retry(
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for adding retry logic to Ray operations.

    Args:
        max_retries: Maximum number of retry attempts
        strategy: Retry strategy to use
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed
                        raise RayTaskFailure(
                            f"Function failed after {max_retries} retries: {e!s}",
                            failure_type=_classify_exception(e),
                            retry_count=attempt,
                            original_exception=e
                        ) from e

                    # Calculate delay for next attempt
                    delay = _calculate_retry_delay(
                        attempt, strategy, base_delay, max_delay,
                        backoff_multiplier, jitter
                    )

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e!s}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception
                    raise e

            # Should never reach here, but if we do, raise a generic exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Retry loop completed without success or exception")

        return wrapper
    return decorator


def _calculate_retry_delay(
    attempt: int,
    strategy: RetryStrategy,
    base_delay: float,
    max_delay: float,
    backoff_multiplier: float,
    jitter: bool
) -> float:
    """Calculate delay for retry attempt based on strategy."""
    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = base_delay * (backoff_multiplier ** attempt)
    elif strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = base_delay * (attempt + 1)
    elif strategy == RetryStrategy.FIXED_DELAY:
        delay = base_delay
    elif strategy == RetryStrategy.IMMEDIATE:
        delay = 0.0
    else:
        delay = base_delay

    # Apply jitter to prevent thundering herd
    if jitter and delay > 0:
        delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay

    return min(delay, max_delay)


def _classify_exception(exception: Exception) -> FailureType:
    """Classify exception type for better error handling."""
    exception_str = str(exception).lower()

    if "worker" in exception_str or "actor" in exception_str:
        return FailureType.WORKER_FAILURE
    elif "timeout" in exception_str:
        return FailureType.TIMEOUT
    elif "resource" in exception_str or "memory" in exception_str:
        return FailureType.RESOURCE_EXHAUSTION
    elif "network" in exception_str or "connection" in exception_str:
        return FailureType.NETWORK_FAILURE
    elif "cluster" in exception_str:
        return FailureType.CLUSTER_FAILURE
    else:
        return FailureType.TASK_FAILURE


@contextmanager
def ray_fault_tolerance(
    auto_recover: bool = True,
    recovery_timeout: float = 30.0,
    fallback_to_local: bool = True
) -> Any:
    """
    Context manager for fault-tolerant Ray operations.

    Args:
        auto_recover: Whether to automatically attempt recovery
        recovery_timeout: Timeout for recovery attempts (seconds)
        fallback_to_local: Whether to fallback to local execution on failure

    Example:
        with ray_fault_tolerance():
            results = ray.get([task.remote() for _ in range(100)])
    """
    try:
        # Check initial cluster health
        if ray.is_initialized():
            health = validate_cluster_health()
            if not health["healthy"]:
                logger.warning(f"Starting operation on unhealthy cluster: {health['reason']}")

        yield

    except Exception as e:
        failure_type = _classify_exception(e)
        logger.error(f"Ray operation failed with {failure_type.value}: {e}")

        if auto_recover and failure_type in [
            FailureType.CLUSTER_FAILURE,
            FailureType.WORKER_FAILURE,
            FailureType.NETWORK_FAILURE
        ]:
            logger.info("Attempting automatic recovery...")

            try:
                # Attempt to recover cluster
                if _attempt_cluster_recovery(recovery_timeout):
                    logger.info("✅ Cluster recovery successful")
                    # Re-raise original exception to let caller retry
                    raise e
                else:
                    logger.error("❌ Cluster recovery failed")

            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {recovery_error}")

        if fallback_to_local and failure_type in [
            FailureType.CLUSTER_FAILURE,
            FailureType.WORKER_FAILURE
        ]:
            logger.warning("⚠️ Falling back to local execution mode")
            # The calling code should handle this by checking if Ray is available
            # and switching to sequential processing

        # Re-raise original exception
        raise e


def _attempt_cluster_recovery(_timeout: float) -> bool:
    """
    Attempt to recover Ray cluster.

    Args:
        timeout: Maximum time to spend on recovery (seconds)

    Returns:
        True if recovery successful, False otherwise
    """
    start_time = time.time()

    try:
        # Try to reinitialize Ray
        logger.info("Attempting to reinitialize Ray cluster...")

        # Shutdown current Ray session if it exists
        if ray.is_initialized():
            try:
                ray.shutdown()
                time.sleep(2.0)  # Give it time to clean up
            except Exception:
                pass

        # Attempt to reinitialize with robust init
        success, info = robust_ray_init(show_cluster_info=False)

        if success:
            # Validate cluster health
            health = validate_cluster_health()
            if health["healthy"]:
                elapsed = time.time() - start_time
                logger.info(f"✅ Cluster recovery completed in {elapsed:.1f}s")
                return True
            else:
                logger.warning(f"Cluster reinitialized but not healthy: {health['reason']}")
                return False
        else:
            logger.error(f"Failed to reinitialize Ray: {info.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"Cluster recovery failed: {e}")
        return False


class ResilientRayExecutor:
    """
    Resilient executor for Ray tasks with advanced error handling.

    Provides comprehensive fault tolerance, monitoring, and recovery
    capabilities for Ray-based distributed computing.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        circuit_breaker_threshold: int = 5,
        enable_monitoring: bool = True,
        fallback_to_local: bool = True
    ):
        """
        Initialize resilient executor.

        Args:
            max_retries: Maximum retries per task
            retry_strategy: Strategy for retry delays
            circuit_breaker_threshold: Failures before opening circuit
            enable_monitoring: Whether to enable execution monitoring
            fallback_to_local: Whether to fallback to local execution
        """
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.enable_monitoring = enable_monitoring
        self.fallback_to_local = fallback_to_local

        # Execution statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "local_fallbacks": 0,
            "circuit_breaker_opens": 0
        }

        # Circuit breaker for task execution
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout=60.0,
            expected_exception=RayTaskFailure
        )

    def execute_task(
        self,
        func: Callable,
        *args: Any,
        task_id: str | None = None,
        timeout: float | None = None,
        **kwargs: Any
    ) -> Any:
        """
        Execute a single task with full resilience handling.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            task_id: Optional task identifier for logging
            timeout: Task timeout in seconds
            **kwargs: Keyword arguments for function

        Returns:
            Task result

        Raises:
            RayTaskFailure: If task fails after all retries
        """
        task_id = task_id or f"task_{int(time.time() * 1000)}"
        self.stats["tasks_submitted"] += 1

        try:
            with ray_fault_tolerance(fallback_to_local=self.fallback_to_local):

                @with_retry(
                    max_retries=self.max_retries,
                    strategy=self.retry_strategy,
                    exceptions=(ray.exceptions.RayTaskError, ray.exceptions.WorkerCrashedError)
                )
                @self.circuit_breaker
                def _execute_with_resilience() -> Any:
                    if not ray.is_initialized():
                        if self.fallback_to_local:
                            logger.warning(f"Ray not available for task {task_id}, executing locally")
                            self.stats["local_fallbacks"] += 1
                            return func(*args, **kwargs)
                        else:
                            raise RayTaskFailure(
                                "Ray not initialized and local fallback disabled",
                                failure_type=FailureType.CLUSTER_FAILURE,
                                task_id=task_id
                            )

                    # Execute as Ray remote task
                    remote_func = ray.remote(func)
                    task_ref = remote_func.remote(*args, **kwargs)

                    if timeout:
                        return ray.get(task_ref, timeout=timeout)
                    else:
                        return ray.get(task_ref)

                result = _execute_with_resilience()
                self.stats["tasks_completed"] += 1

                if self.enable_monitoring:
                    logger.debug(f"Task {task_id} completed successfully")

                return result

        except RayTaskFailure as e:
            self.stats["tasks_failed"] += 1
            if e.retry_count > 0:
                self.stats["tasks_retried"] += e.retry_count

            logger.error(f"Task {task_id} failed: {e}")
            raise e

        except Exception as e:
            self.stats["tasks_failed"] += 1
            logger.error(f"Task {task_id} failed with unexpected error: {e}")

            raise RayTaskFailure(
                f"Unexpected error in task {task_id}: {e!s}",
                failure_type=_classify_exception(e),
                task_id=task_id,
                original_exception=e
            ) from e

    def execute_batch(
        self,
        func: Callable,
        items: list[Any],
        max_parallel: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> list[Any]:
        """
        Execute a batch of tasks with resilience handling.

        Args:
            func: Function to execute on each item
            items: List of items to process
            max_parallel: Maximum parallel tasks (auto-detected if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List of results in same order as input items
        """
        if not items:
            return []

        # Auto-detect optimal parallelism
        if max_parallel is None and ray.is_initialized():
            worker_info = get_optimal_worker_count()
            max_parallel = worker_info["total_workers"]

        max_parallel = max_parallel or 4

        logger.info(f"Executing batch of {len(items)} tasks with max {max_parallel} parallel")

        results = [None] * len(items)
        failed_indices = []
        completed = 0

        # Execute in batches to avoid overwhelming cluster
        batch_size = min(max_parallel, len(items))

        for start_idx in range(0, len(items), batch_size):
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]
            list(range(start_idx, end_idx))

            # Submit batch tasks
            for i, item in enumerate(batch_items):
                task_id = f"batch_task_{start_idx + i}"
                try:
                    task_result = self.execute_task(func, item, task_id=task_id)
                    results[start_idx + i] = task_result
                    completed += 1

                except RayTaskFailure:
                    failed_indices.append(start_idx + i)
                    results[start_idx + i] = None

            # Progress callback
            if progress_callback:
                progress_callback(completed, len(items))

        # Log batch results
        success_count = len(items) - len(failed_indices)
        logger.info(f"Batch execution completed: {success_count}/{len(items)} successful")

        if failed_indices:
            logger.warning(f"Failed task indices: {failed_indices}")

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        total_tasks = self.stats["tasks_submitted"]
        success_rate = (self.stats["tasks_completed"] / max(1, total_tasks)) * 100

        return {
            **self.stats,
            "success_rate": success_rate,
            "failure_rate": 100 - success_rate,
            "avg_retries_per_task": self.stats["tasks_retried"] / max(1, total_tasks)
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        for key in self.stats:
            self.stats[key] = 0


# Utility functions for easy integration

def resilient_ray_get(
    object_refs: ray.ObjectRef | list[ray.ObjectRef],
    timeout: float | None = None,
    max_retries: int = 3
) -> Any:
    """
    Resilient version of ray.get() with automatic retry and error handling.

    Args:
        object_refs: Ray object reference(s) to get
        timeout: Timeout for get operation
        max_retries: Maximum retry attempts

    Returns:
        Retrieved object(s)
    """
    @with_retry(
        max_retries=max_retries,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=(ray.exceptions.GetTimeoutError, ray.exceptions.WorkerCrashedError)
    )
    def _get_with_retry() -> Any:
        return ray.get(object_refs, timeout=timeout)

    return _get_with_retry()


def resilient_ray_wait(
    object_refs: list[ray.ObjectRef],
    num_returns: int = 1,
    timeout: float | None = None,
    max_retries: int = 3
) -> Any:
    """
    Resilient version of ray.wait() with automatic retry and error handling.

    Args:
        object_refs: List of Ray object references to wait for
        num_returns: Number of objects to wait for
        timeout: Timeout for wait operation
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (ready, not_ready) object references
    """
    @with_retry(
        max_retries=max_retries,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        exceptions=(ray.exceptions.GetTimeoutError,)
    )
    def _wait_with_retry() -> Any:
        return ray.wait(object_refs, num_returns=num_returns, timeout=timeout)

    return _wait_with_retry()


# Global resilient executor instance
_global_executor: ResilientRayExecutor | None = None


def get_global_executor() -> ResilientRayExecutor:
    """Get or create global resilient executor."""
    global _global_executor

    if _global_executor is None:
        _global_executor = ResilientRayExecutor()

    return _global_executor


def execute_resilient_task(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Execute a task using the global resilient executor."""
    executor = get_global_executor()
    return executor.execute_task(func, *args, **kwargs)
