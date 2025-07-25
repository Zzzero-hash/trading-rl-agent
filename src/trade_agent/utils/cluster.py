import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import ray
import yaml


def check_ray_cluster_status() -> dict[str, Any]:
    """Check Ray cluster status using the Ray CLI.

    Returns:
        Dictionary containing cluster status information or error details.
    """
    try:
        # Try to get cluster status using ray status command
        result = subprocess.run(
            ["ray", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )

        if result.returncode == 0:
            try:
                status_data = json.loads(result.stdout)
                return {
                    "available": True,
                    "status": "connected",
                    "data": status_data,
                    "error": None
                }
            except json.JSONDecodeError as e:
                return {
                    "available": False,
                    "status": "parse_error",
                    "data": None,
                    "error": f"Failed to parse Ray status output: {e}"
                }
        else:
            return {
                "available": False,
                "status": "disconnected",
                "data": None,
                "error": result.stderr.strip() or "No Ray cluster found"
            }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "status": "timeout",
            "data": None,
            "error": "Ray status command timed out"
        }
    except FileNotFoundError:
        return {
            "available": False,
            "status": "no_ray_cli",
            "data": None,
            "error": "Ray CLI not found in PATH"
        }
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "data": None,
            "error": f"Unexpected error checking Ray status: {e}"
        }


def validate_cluster_health() -> dict[str, Any]:
    """Validate Ray cluster health and resource availability.

    Returns:
        Dictionary containing health status and resource information.
    """
    logger = logging.getLogger(__name__)

    # First check if Ray is initialized
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, cannot validate cluster health")
        return {
            "healthy": False,
            "reason": "ray_not_initialized",
            "resources": {},
            "nodes": 0,
            "recommendations": ["Initialize Ray first before validating cluster health"]
        }

    try:
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        # Get node information
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node.get("Alive", False)]

        # Calculate resource utilization
        cpu_total = cluster_resources.get("CPU", 0)
        cpu_available = available_resources.get("CPU", 0)
        cpu_utilized = cpu_total - cpu_available if cpu_total > 0 else 0
        cpu_utilization = (cpu_utilized / cpu_total * 100) if cpu_total > 0 else 0

        gpu_total = cluster_resources.get("GPU", 0)
        gpu_available = available_resources.get("GPU", 0)
        gpu_utilized = gpu_total - gpu_available if gpu_total > 0 else 0
        gpu_utilization = (gpu_utilized / gpu_total * 100) if gpu_total > 0 else 0

        # Determine health status
        healthy = True
        issues = []
        recommendations = []

        # Check for issues
        if len(alive_nodes) == 0:
            healthy = False
            issues.append("No alive nodes found")
            recommendations.append("Check cluster connectivity and node health")

        if cpu_total == 0:
            healthy = False
            issues.append("No CPU resources available")
            recommendations.append("Ensure cluster has CPU resources allocated")

        if cpu_utilization > 95:
            issues.append("High CPU utilization (>95%)")
            recommendations.append("Consider scaling up cluster or reducing workload")

        if gpu_total > 0 and gpu_utilization > 95:
            issues.append("High GPU utilization (>95%)")
            recommendations.append("Consider adding more GPU nodes or optimizing GPU usage")

        # Memory check (if available)
        memory_total = cluster_resources.get("memory", 0)
        memory_available = available_resources.get("memory", 0)
        if memory_total > 0:
            memory_utilization = ((memory_total - memory_available) / memory_total * 100)
            if memory_utilization > 90:
                issues.append(f"High memory utilization ({memory_utilization:.1f}%)")
                recommendations.append("Consider adding more memory or optimizing memory usage")

        return {
            "healthy": healthy,
            "reason": "cluster_operational" if healthy else "; ".join(issues),
            "resources": {
                "cpu": {
                    "total": cpu_total,
                    "available": cpu_available,
                    "utilized": cpu_utilized,
                    "utilization_percent": cpu_utilization
                },
                "gpu": {
                    "total": gpu_total,
                    "available": gpu_available,
                    "utilized": gpu_utilized,
                    "utilization_percent": gpu_utilization
                },
                "memory": {
                    "total": memory_total,
                    "available": memory_available,
                    "utilized": memory_total - memory_available if memory_total > 0 else 0
                }
            },
            "nodes": len(alive_nodes),
            "issues": issues,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error validating cluster health: {e}")
        return {
            "healthy": False,
            "reason": f"validation_error: {e}",
            "resources": {},
            "nodes": 0,
            "recommendations": ["Check Ray cluster connectivity and try again"]
        }


def get_available_devices() -> dict[str, float]:
    """Return available cluster resources (CPUs and GPUs)."""
    if not ray.is_initialized():
        return {"CPU": 0.0, "GPU": 0.0}

    try:
        resources = ray.available_resources()
        return {
            "CPU": resources.get("CPU", 0.0),
            "GPU": resources.get("GPU", 0.0),
        }
    except Exception:
        return {"CPU": 0.0, "GPU": 0.0}


def get_optimal_worker_count() -> dict[str, int]:
    """Get optimal worker count based on available cluster resources.

    Returns:
        Dictionary with recommended worker counts for different workloads.
    """
    if not ray.is_initialized():
        return {"cpu_workers": 1, "gpu_workers": 0, "total_workers": 1}

    try:
        available = ray.available_resources()
        cpu_count = int(available.get("CPU", 1))
        gpu_count = int(available.get("GPU", 0))

        # Conservative recommendations to avoid resource contention
        cpu_workers = max(1, int(cpu_count * 0.8))  # Use 80% of available CPUs
        # For GPU workers, use at least 1 if any GPU is available
        gpu_workers = max(0, min(gpu_count, max(1, int(gpu_count * 0.9)) if gpu_count > 0 else 0))

        return {
            "cpu_workers": cpu_workers,
            "gpu_workers": gpu_workers,
            "total_workers": cpu_workers + gpu_workers,
            "available_cpu": cpu_count,
            "available_gpu": gpu_count
        }
    except Exception:
        return {"cpu_workers": 1, "gpu_workers": 0, "total_workers": 1}


def wait_for_cluster_ready(timeout: float = 60.0, check_interval: float = 2.0) -> bool:
    """Wait for Ray cluster to be ready and healthy.

    Args:
        timeout: Maximum time to wait in seconds.
        check_interval: Time between health checks in seconds.

    Returns:
        True if cluster becomes ready within timeout, False otherwise.
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    while time.time() - start_time < timeout:
        if ray.is_initialized():
            health = validate_cluster_health()
            if health["healthy"]:
                logger.info(f"Ray cluster is ready with {health['nodes']} nodes")
                return True
            else:
                logger.debug(f"Cluster not ready: {health['reason']}")
        else:
            logger.debug("Ray not initialized, waiting...")

        time.sleep(check_interval)

    logger.warning(f"Cluster did not become ready within {timeout} seconds")
    return False


def init_ray(
    address: str | None = None,
    config_path: str | None = None,
    local_mode: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> tuple[bool, dict[str, Any]]:
    """Initialize Ray with robust error handling and cluster detection.

    Parameters
    ----------
    address : str, optional
        Address of the Ray head node. If None, uses the ``RAY_ADDRESS``
        environment variable or the address from ``config_path`` if provided.
    config_path : str, optional
        YAML file containing ``head_address``.
    local_mode : bool, default False
        Whether to run Ray in local mode for easier debugging.
    max_retries : int, default 3
        Maximum number of connection retries.
    retry_delay : float, default 5.0
        Delay between retry attempts in seconds.

    Returns
    -------
    tuple
        (success: bool, info: dict) where info contains connection details.
    """
    logger = logging.getLogger(__name__)

    # Check if Ray is already initialized
    if ray.is_initialized():
        logger.info("Ray is already initialized")
        health = validate_cluster_health()
        return True, {
            "status": "already_initialized",
            "mode": "existing",
            "health": health
        }

    # Determine address
    if address is None:
        address = os.getenv("RAY_ADDRESS")
    if config_path and not address:
        try:
            with Path(config_path).open("r") as f:
                cfg = yaml.safe_load(f)
            address = cfg.get("head_address")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    # Try to connect with retries
    for attempt in range(max_retries):
        try:
            if address:
                logger.info(f"Attempting to connect to Ray cluster at {address} (attempt {attempt + 1}/{max_retries})")
                ray.init(address=address, ignore_reinit_error=True)

                # Validate the connection
                if wait_for_cluster_ready(timeout=10.0):
                    health = validate_cluster_health()
                    logger.info(f"Successfully connected to Ray cluster with {health['nodes']} nodes")
                    return True, {
                        "status": "connected",
                        "mode": "cluster",
                        "address": address,
                        "health": health
                    }
                else:
                    logger.warning("Connected to Ray but cluster not healthy")
                    ray.shutdown()
            else:
                logger.info(f"Starting Ray in local mode (attempt {attempt + 1}/{max_retries})")
                ray.init(local_mode=local_mode, ignore_reinit_error=True)

                # Brief wait for local mode to be ready
                time.sleep(1.0)
                health = validate_cluster_health()
                logger.info("Successfully started Ray in local mode")
                return True, {
                    "status": "local_mode",
                    "mode": "local",
                    "health": health
                }

        except Exception as e:
            logger.warning(f"Ray initialization attempt {attempt + 1} failed: {e}")

            # Cleanup on failure
            try:
                if ray.is_initialized():
                    ray.shutdown()
            except Exception:
                pass

            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    # All attempts failed
    logger.error(f"Failed to initialize Ray after {max_retries} attempts")
    return False, {
        "status": "failed",
        "mode": "none",
        "error": f"All {max_retries} initialization attempts failed"
    }


def print_cluster_info() -> None:
    """Print comprehensive cluster information for debugging."""
    logging.getLogger(__name__)

    print("\n" + "="*60)
    print("RAY CLUSTER INFORMATION")
    print("="*60)

    # Check Ray CLI status
    status = check_ray_cluster_status()
    print(f"\n🔍 Ray CLI Status: {status['status']}")
    if not status["available"]:
        print(f"   Error: {status['error']}")

    # Check Ray initialization
    print(f"\n⚡ Ray Initialized: {ray.is_initialized()}")

    if ray.is_initialized():
        # Health check
        health = validate_cluster_health()
        print(f"\n🏥 Cluster Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
        if not health["healthy"]:
            print(f"   Reason: {health['reason']}")

        # Resource information
        resources = health["resources"]
        print("\n💻 Resources:")
        print(f"   CPUs: {resources['cpu']['available']:.1f}/{resources['cpu']['total']:.1f} available ({resources['cpu']['utilization_percent']:.1f}% used)")
        if resources["gpu"]["total"] > 0:
            print(f"   GPUs: {resources['gpu']['available']:.1f}/{resources['gpu']['total']:.1f} available ({resources['gpu']['utilization_percent']:.1f}% used)")
        else:
            print("   GPUs: None available")

        if resources["memory"]["total"] > 0:
            memory_gb = resources["memory"]["total"] / (1024**3)
            memory_avail_gb = resources["memory"]["available"] / (1024**3)
            print(f"   Memory: {memory_avail_gb:.1f}/{memory_gb:.1f} GB available")

        print(f"\n🌐 Nodes: {health['nodes']} alive")

        # Worker recommendations
        worker_rec = get_optimal_worker_count()
        print("\n👥 Recommended Workers:")
        print(f"   CPU Workers: {worker_rec['cpu_workers']}")
        print(f"   GPU Workers: {worker_rec['gpu_workers']}")
        print(f"   Total Workers: {worker_rec['total_workers']}")

        # Issues and recommendations
        if health["issues"]:
            print("\n⚠️  Issues:")
            for issue in health["issues"]:
                print(f"   • {issue}")

        if health["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in health["recommendations"]:
                print(f"   • {rec}")

    print("\n" + "="*60 + "\n")
