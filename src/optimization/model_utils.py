"""Utilities for model inspection and performance profiling."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch import nn
from torch.utils.benchmark import Timer
from torchinfo import summary

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_model_summary(
    model: nn.Module,
    input_size: tuple[int, ...],
    **kwargs: Any,
) -> str:
    """Return a summary of the model using ``torchinfo.summary``."""
    return str(summary(model, input_size=input_size, **kwargs))


def profile_model_inference(
    model: nn.Module,
    batch_size: int = 32,
    sequence_length: int = 60,
    num_features: int = 10,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> dict[str, Any]:
    """Profile model inference performance using ``torch.utils.benchmark``."""
    model.eval()
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    x = torch.randn(batch_size, sequence_length, num_features, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)
    timer = Timer(stmt="model(x)", globals={"model": model, "x": x})
    result = timer.timeit(num_runs)
    avg_time = result.mean
    throughput = batch_size / avg_time
    metrics = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_features": num_features,
        "avg_time_ms": avg_time * 1000,
        "throughput_samples_per_sec": throughput,
        "device": str(device),
    }
    if device.type == "cuda":
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.update(
                {
                    "gpu_memory_total_mb": int(info.total) / (1024 * 1024),
                    "gpu_memory_used_mb": int(info.used) / (1024 * 1024),
                    "gpu_memory_free_mb": int(info.free) / (1024 * 1024),
                },
            )
        except Exception as exc:  # pragma: no cover - optional GPU info
            logger.warning("Failed to get GPU memory info: %s", exc)
    return metrics


# The GPU utilities below are copied from the former ``model_summary`` module
# to maintain API compatibility.


def detect_gpus() -> dict[str, Any]:
    """Detect available GPU devices and their capabilities."""
    gpu_info = {"available": torch.cuda.is_available(), "count": 0, "devices": []}
    if not gpu_info["available"]:
        return gpu_info
    gpu_info["count"] = torch.cuda.device_count()
    for i in range(gpu_info["count"]):
        props = torch.cuda.get_device_properties(i)
        dev = {
            "index": i,
            "name": props.name,
            "total_memory": props.total_memory / (1024 * 1024),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            dev.update(
                {
                    "memory_used": int(mem.used) / (1024 * 1024),
                    "memory_free": int(mem.free) / (1024 * 1024),
                },
            )
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                dev.update(
                    {"gpu_utilization": util.gpu, "memory_utilization": util.memory},
                )
            except Exception:
                pass
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
                dev["temperature"] = temp
            except Exception:
                pass
        except Exception:
            pass
        gpu_info["devices"].append(dev)
    return gpu_info


def optimal_gpu_config(
    model_params: int,
    batch_size: int,
    sequence_length: int = 60,
    feature_dim: int = 10,
) -> dict[str, Any]:
    """Determine an approximate optimal GPU configuration."""
    gpu_info = detect_gpus()
    rec = {
        "use_gpu": gpu_info["available"] and gpu_info["count"] > 0,
        "recommended_batch_size": batch_size,
        "mixed_precision": False,
    }
    if not rec["use_gpu"]:
        rec["reason"] = "No GPUs available"
        return rec
    param_memory_mb = model_params * 4 / (1024 * 1024)
    grad_memory_mb = model_params * 4 / (1024 * 1024)
    optimizer_memory_mb = model_params * 8 / (1024 * 1024)
    activation_factor = 2.0
    activation_memory_mb = batch_size * sequence_length * feature_dim * 4 * activation_factor / (1024 * 1024)
    total_memory_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb
    best_gpu = None
    max_free = 0
    for dev in gpu_info["devices"]:
        if dev.get("memory_free", float("inf")) > max_free:
            max_free = dev.get("memory_free", float("inf"))
            best_gpu = dev
    if best_gpu:
        rec.update(
            {
                "gpu_index": best_gpu["index"],
                "gpu_name": best_gpu["name"],
                "gpu_memory_free_mb": best_gpu.get(
                    "memory_free",
                    best_gpu["total_memory"],
                ),
                "estimated_memory_required_mb": total_memory_mb,
            },
        )
        headroom = 0.8
        avail = rec["gpu_memory_free_mb"] * headroom
        if total_memory_mb > avail:
            rec["mixed_precision"] = True
            savings = param_memory_mb * 0.5 + grad_memory_mb * 0.5
            new_total = total_memory_mb - savings
            if new_total > avail:
                max_batch = int(batch_size * (avail / new_total))
                max_batch = 2 ** int(np.log2(max_batch))
                max_batch = max(1, max_batch)
                rec["recommended_batch_size"] = max_batch
                rec["reason"] = f"Using mixed precision and reduced batch size ({max_batch}) due to memory constraints"
            else:
                rec["reason"] = "Using mixed precision due to memory constraints"
        else:
            rec["reason"] = "Using full precision, memory is sufficient"
    return rec


def run_hyperparameter_optimization(
    train_fn: Callable,
    config_space: dict[str, Any],
    num_samples: int = 10,
    max_epochs_per_trial: int = 10,
    resources_per_trial: dict[str, int] | None = None,
    metric: str = "val_loss",
    mode: str = "min",
    output_dir: str = "./optimization_results",
) -> tune.ExperimentAnalysis:
    """Run hyperparameter optimization using Ray Tune."""
    if resources_per_trial is None:
        resources_per_trial = {"cpu": 1, "gpu": 0}
    if any(isinstance(v, dict) and "grid_search" in v for v in config_space.values()):
        search_alg = None
    else:
        search_alg = OptunaSearch(metric=metric, mode=mode)
    scheduler = ASHAScheduler(
        max_t=max_epochs_per_trial,
        grace_period=1,
        reduction_factor=2,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hparam_opt_{timestamp}"
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        storage_uri = output_path.resolve().as_uri()
        run_kwargs = {
            "config": config_space,
            "num_samples": num_samples,
            "scheduler": scheduler,
            "resources_per_trial": resources_per_trial,
            "storage_path": storage_uri,
            "metric": metric,
            "mode": mode,
            "verbose": 1,
            "raise_on_failed_trial": True,
        }
        if search_alg is not None:
            run_kwargs["search_alg"] = search_alg
        analysis = tune.run(train_fn, **run_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Hyperparameter optimization failed: {exc}") from exc
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    logger.info("Best configuration: %s", best_config)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    if best_trial is None:
        raise RuntimeError("No valid trials found for hyperparameter optimization")
    best_result = best_trial.last_result
    summary_data = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "metric": metric,
        "mode": mode,
        "num_samples": num_samples,
        "best_value": best_result[metric],
        "best_config": best_config,
        "best_trial_id": best_trial.trial_id,
    }
    import json

    with (output_path / "summary.json").open("w") as f:
        json.dump(summary_data, f, indent=2)
    try:
        df = analysis.dataframe()
        if not df.empty:
            plt.figure(figsize=(10, 6))
            for trial_id, trial_df in df.groupby("trial_id"):
                plt.plot(
                    trial_df["training_iteration"],
                    trial_df[metric],
                    label=f"Trial {trial_id}",
                )
            plt.xlabel("Iteration")
            plt.ylabel(metric)
            plt.title(f"Hyperparameter Optimization - {metric} vs Iteration")
            plt.legend(loc="best")
            plt.savefig(output_path / f"{metric}_progress.png")
            plt.close()
    except Exception as exc:  # pragma: no cover - optional plotting
        logger.warning("Failed to create visualization: %s", exc)
    return analysis
