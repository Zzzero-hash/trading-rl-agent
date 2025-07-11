import os
from pathlib import Path

import ray
import yaml


def init_ray(
    address: str | None = None,
    config_path: str | None = None,
    local_mode: bool = False,
) -> None:
    """Initialize Ray with the given address or configuration.

    Parameters
    ----------
    address : str, optional
        Address of the Ray head node. If None, uses the ``RAY_ADDRESS``
        environment variable or the address from ``config_path`` if provided.
    config_path : str, optional
        YAML file containing ``head_address``.
    local_mode : bool, default False
        Whether to run Ray in local mode for easier debugging.
    """
    if address is None:
        address = os.getenv("RAY_ADDRESS")
    if config_path and not address:
        with Path(config_path).open("r") as f:
            cfg = yaml.safe_load(f)
        address = cfg.get("head_address")

    if address:
        ray.init(address=address)
    else:
        ray.init(local_mode=local_mode)


def get_available_devices() -> dict[str, float]:
    """Return available cluster resources (CPUs and GPUs)."""
    resources = ray.cluster_resources() if ray.is_initialized() else ray.available_resources()
    return {
        "CPU": resources.get("CPU", 0.0),
        "GPU": resources.get("GPU", 0.0),
    }
