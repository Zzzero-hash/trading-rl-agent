"""Model summary and optimization utilities.

This module provides tools for analyzing model architecture, performance, memory usage,
and hyperparameter optimization. It focuses on:

1. Model Architecture Summary - Detailed layer-by-layer analysis
2. Performance Profiling - Training, inference time and memory consumption metrics
3. GPU Optimization - Automatic detection and configuration for optimal GPU usage
4. Hyperparameter Optimization - Bayesian optimization using Ray Tune

Example usage:

>>> from src.optimization.model_summary import ModelSummarizer, profile_model_inference
>>> model = CNNLSTMModel(input_dim=10)
>>> summarizer = ModelSummarizer(model)
>>> print(summarizer.get_summary())
>>> perf_metrics = profile_model_inference(model, batch_size=32, sequence_length=60)
"""

from __future__ import annotations

from datetime import datetime
import inspect
import logging
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Try to import GPU profiling tools if available
try:
    from torch.utils.benchmark import Timer

    TORCH_BENCHMARK_AVAILABLE = True
except ImportError:
    TORCH_BENCHMARK_AVAILABLE = False

try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
except Exception:
    # Handle pynvml.NVMLError or other exceptions
    PYNVML_AVAILABLE = False

# Local imports
try:
    from ..models.cnn_lstm import CNNLSTMConfig, CNNLSTMModel
    from ..models.concat_model import ConcatModel
except ImportError:
    # For direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.models.cnn_lstm import CNNLSTMConfig, CNNLSTMModel
    from src.models.concat_model import ConcatModel

logger = logging.getLogger(__name__)


class ModelSummarizer:
    """Summarizes model architecture and parameters.

    Provides detailed insights into model architecture, parameters count,
    memory usage estimation and layer-by-layer analysis.
    """

    def __init__(self, model: nn.Module):
        """Initialize with a PyTorch model.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to summarize
        """
        self.model = model
        self.num_params = self._count_parameters()
        self.memory_estimate = self._estimate_memory()
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = "cpu"  # Default to CPU if no parameters
        self.layer_info = self._get_layer_info()

    def _count_parameters(self) -> dict[str, int]:
        """Count trainable and non-trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        return {
            "trainable": trainable,
            "non_trainable": non_trainable,
            "total": trainable + non_trainable,
        }

    def _estimate_memory(self) -> dict[str, float]:
        """Estimate memory usage (parameters, gradients, forward pass)."""
        # Memory for parameters (32 bit floats)
        param_memory = self.num_params["total"] * 4 / (1024 * 1024)  # MB

        # Estimate for gradients (32 bit floats)
        grad_memory = self.num_params["trainable"] * 4 / (1024 * 1024)  # MB

        # Rough estimate for forward/backward activations
        # This is very approximate - actual value depends on model architecture
        activation_memory = param_memory * 2  # rough approximation

        return {
            "parameters": param_memory,
            "gradients": grad_memory,
            "activations": activation_memory,
            "total": param_memory + grad_memory + activation_memory,
        }

    def _get_layer_info(self) -> list[dict[str, Any]]:
        """Analyze model layers and extract detailed information."""
        info = []
        for name, module in self.model.named_modules():
            if name == "":  # Skip the root module
                continue

            layer_params = sum(p.numel() for p in module.parameters())
            is_leaf = len(list(module.children())) == 0

            # Only add leaf nodes to avoid duplication
            if is_leaf:
                info.append(
                    {
                        "name": name,
                        "type": module.__class__.__name__,
                        "parameters": layer_params,
                        "requires_grad": (
                            all(p.requires_grad for p in module.parameters())
                            if layer_params > 0
                            else None
                        ),
                    }
                )

        return info

    def get_summary(self, detailed: bool = True) -> str:
        """Generate a comprehensive model summary.

        Parameters
        ----------
        detailed : bool, default True
            Whether to include layer-by-layer details

        Returns
        -------
        str
            Formatted model summary
        """
        lines = []

        # Header with model class name
        model_name = self.model.__class__.__name__
        lines.append(f"{'=' * 80}")
        lines.append(f"Model: {model_name}")
        lines.append(f"{'=' * 80}")

        # General information
        lines.append(f"Device: {self.device}")
        lines.append(f"Parameters: {self.num_params['total']:,} total")
        lines.append(f"  - Trainable: {self.num_params['trainable']:,}")
        lines.append(f"  - Non-trainable: {self.num_params['non_trainable']:,}")
        lines.append(f"Memory Usage (estimated):")
        lines.append(f"  - Parameters: {self.memory_estimate['parameters']:.2f} MB")
        lines.append(f"  - Gradients: {self.memory_estimate['gradients']:.2f} MB")
        lines.append(
            f"  - Activations (est.): {self.memory_estimate['activations']:.2f} MB"
        )
        lines.append(f"  - Total: {self.memory_estimate['total']:.2f} MB")

        # Layer-by-layer information if requested
        if detailed:
            lines.append(f"\nLayer details:")
            lines.append(f"{'-' * 80}")
            lines.append(
                f"{'Layer Name':<40} {'Type':<15} {'Params':>10} {'Trainable':<10}"
            )
            lines.append(f"{'-' * 80}")
            for layer in self.layer_info:
                trainable = (
                    "Yes"
                    if layer["requires_grad"]
                    else "No" if layer["requires_grad"] is not None else "N/A"
                )
                lines.append(
                    f"{layer['name']:<40} {layer['type']:<15} {layer['parameters']:>10,} {trainable:<10}"
                )

        return "\n".join(lines)

    def visualize_model_complexity(self, save_path: str | None = None):
        """Visualize model layer complexity using a bar chart.

        Parameters
        ----------
        save_path : str, optional
            Path to save the visualization. If None, displays the plot.
        """
        # Extract data
        names = [f"{layer['name']} ({layer['type']})" for layer in self.layer_info]
        params = [layer["parameters"] for layer in self.layer_info]

        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, len(names) * 0.3)))

        # Only show the top 20 layers if there are too many
        if len(names) > 20:
            indices = np.argsort(params)[-20:]
            names = [names[i] for i in indices]
            params = [params[i] for i in indices]
            plt.title("Top 20 layers by parameter count")
        else:
            plt.title("Model layers by parameter count")

        plt.barh(names, params)
        plt.xscale("log")
        plt.xlabel("Number of Parameters (log scale)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def profile_model_inference(
    model: nn.Module,
    batch_size: int = 32,
    sequence_length: int = 60,
    num_features: int = 10,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> dict[str, Any]:
    """Profile model inference performance.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to profile
    batch_size : int, default 32
        Batch size for inference
    sequence_length : int, default 60
        Sequence length for time series data
    num_features : int, default 10
        Number of input features
    num_warmup : int, default 5
        Number of warmup iterations
    num_runs : int, default 50
        Number of runs for timing

    Returns
    -------
    dict
        Inference performance metrics
    """
    # Put model in evaluation mode and move to the right device
    model.eval()
    device = next(model.parameters()).device

    # Generate dummy input tensor
    x = torch.randn(batch_size, sequence_length, num_features, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = batch_size / avg_time

    # Use torch benchmark if available for more accurate measurement
    benchmark_time = avg_time
    if TORCH_BENCHMARK_AVAILABLE:
        try:
            timer = Timer(stmt="model(x)", globals={"model": model, "x": x})
            benchmark_time = timer.timeit(num_runs).mean
        except Exception as e:
            logger.warning(f"Failed to use torch benchmark: {str(e)}")

    results = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_features": num_features,
        "avg_time_ms": avg_time * 1000,
        "throughput_samples_per_sec": throughput,
        "benchmark_time_ms": benchmark_time * 1000 if benchmark_time else None,
        "device": str(device),
    }

    # Add GPU memory usage if available
    if PYNVML_AVAILABLE and "cuda" in str(device):
        try:
            import pynvml

            dev_idx = device.index if hasattr(device, "index") else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            results.update(
                {
                    "gpu_memory_total_mb": info.total / (1024 * 1024),
                    "gpu_memory_used_mb": info.used / (1024 * 1024),
                    "gpu_memory_free_mb": info.free / (1024 * 1024),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {str(e)}")

    return results


def detect_gpus() -> dict[str, Any]:
    """Detect available GPU devices and their capabilities.

    Returns
    -------
    dict
        Information about available GPUs
    """
    gpu_info = {"available": torch.cuda.is_available(), "count": 0, "devices": []}

    if not gpu_info["available"]:
        return gpu_info

    gpu_info["count"] = torch.cuda.device_count()

    # Get detailed info for each GPU
    for i in range(gpu_info["count"]):
        device_props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": device_props.name,
            "total_memory": device_props.total_memory / (1024 * 1024),  # MB
            "compute_capability": f"{device_props.major}.{device_props.minor}",
            "multi_processor_count": device_props.multi_processor_count,
        }
        # Add NVML info if available
        if PYNVML_AVAILABLE:
            try:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_info.update(
                    {
                        "memory_used": mem_info.used / (1024 * 1024),  # MB
                        "memory_free": mem_info.free / (1024 * 1024),  # MB
                    }
                )
                # Utilization info
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    device_info.update(
                        {
                            "gpu_utilization": util.gpu,
                            "memory_utilization": util.memory,
                        }
                    )
                except Exception:
                    pass
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    device_info["temperature"] = temp
                except Exception:
                    pass
            except Exception:
                pass
        gpu_info["devices"].append(device_info)

    return gpu_info


def optimal_gpu_config(
    model_params: int,
    batch_size: int,
    sequence_length: int = 60,
    feature_dim: int = 10,
) -> dict[str, Any]:
    """Determine optimal GPU configuration for the given model and batch size.

    Parameters
    ----------
    model_params : int
        Number of model parameters
    batch_size : int
        Batch size for training
    sequence_length : int, default 60
        Sequence length for time series data
    feature_dim : int, default 10
        Feature dimension

    Returns
    -------
    dict
        Recommended GPU configuration
    """
    gpu_info = detect_gpus()
    recommendations = {
        "use_gpu": gpu_info["available"] and gpu_info["count"] > 0,
        "recommended_batch_size": batch_size,
        "mixed_precision": False,
    }

    if not recommendations["use_gpu"]:
        recommendations["reason"] = "No GPUs available"
        return recommendations

    # Estimate memory requirements (very rough approximation)
    # Parameters: 4 bytes per parameter (fp32)
    # Gradients: 4 bytes per parameter (fp32)
    # Optimizer states: 8 bytes per parameter (momentum + adam)
    # Forward activations: depends on model, batch size, sequence length
    param_memory_mb = model_params * 4 / (1024 * 1024)
    grad_memory_mb = model_params * 4 / (1024 * 1024)
    optimizer_memory_mb = model_params * 8 / (1024 * 1024)

    # Rough estimate for activations
    # This is very approximate - actual value depends significantly on model architecture
    activation_factor = 2.0  # multiplier on parameter size
    activation_memory_mb = (
        batch_size
        * sequence_length
        * feature_dim
        * 4
        * activation_factor
        / (1024 * 1024)
    )

    total_memory_mb = (
        param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb
    )

    # Get the GPU with the most free memory
    best_gpu = None
    max_free_memory = 0
    for device in gpu_info["devices"]:
        if device.get("memory_free", float("inf")) > max_free_memory:
            max_free_memory = device.get("memory_free", float("inf"))
            best_gpu = device

    if best_gpu:
        recommendations["gpu_index"] = best_gpu["index"]
        recommendations["gpu_name"] = best_gpu["name"]
        recommendations["gpu_memory_free_mb"] = best_gpu.get(
            "memory_free", best_gpu["total_memory"]
        )
        recommendations["estimated_memory_required_mb"] = total_memory_mb

        # Determine if we need to adjust batch size or use mixed precision
        memory_headroom = 0.8  # Use only 80% of available memory
        available_memory = recommendations["gpu_memory_free_mb"] * memory_headroom

        if total_memory_mb > available_memory:
            # Try mixed precision first
            recommendations["mixed_precision"] = True
            mixed_precision_savings = (
                param_memory_mb * 0.5 + grad_memory_mb * 0.5
            )  # Roughly half for fp16
            new_total_memory = total_memory_mb - mixed_precision_savings

            if new_total_memory > available_memory:
                # Need to reduce batch size too
                max_batch_size = int(batch_size * (available_memory / new_total_memory))
                # Round down to power of 2
                max_batch_size = 2 ** int(np.log2(max_batch_size))
                max_batch_size = max(1, max_batch_size)  # Ensure at least 1

                recommendations["recommended_batch_size"] = max_batch_size
                recommendations["reason"] = (
                    f"Using mixed precision and reduced batch size ({max_batch_size}) due to memory constraints"
                )
            else:
                recommendations["reason"] = (
                    "Using mixed precision due to memory constraints"
                )
        else:
            recommendations["reason"] = "Using full precision, memory is sufficient"

    return recommendations


def run_hyperparameter_optimization(
    train_fn: callable,
    config_space: dict[str, Any],
    num_samples: int = 10,
    max_epochs_per_trial: int = 10,
    resources_per_trial: dict[str, int] = {"cpu": 1, "gpu": 0},
    metric: str = "val_loss",
    mode: str = "min",
    output_dir: str = "./optimization_results",
) -> tune.ExperimentAnalysis:
    """Run hyperparameter optimization using Ray Tune.

    Parameters
    ----------
    train_fn : callable
        Training function that takes a config dict and returns metrics
    config_space : dict
        Configuration space for hyperparameter search
    num_samples : int, default 10
        Number of trials to run
    max_epochs_per_trial : int, default 10
        Maximum epochs per trial
    resources_per_trial : dict, default {"cpu": 1, "gpu": 0}
        Resources to allocate per trial
    metric : str, default "val_loss"
        Metric to optimize
    mode : str, default "min"
        Optimization mode - "min" or "max"
    output_dir : str, default "./optimization_results"
        Directory to save results

    Returns
    -------
    ExperimentAnalysis
        Ray Tune experiment analysis object
    """
    # Initialize search algorithm: use Optuna for non-grid search, else leave None to use default grid search
    if any(isinstance(v, dict) and "grid_search" in v for v in config_space.values()):
        search_alg = None
    else:
        search_alg = OptunaSearch(
            metric=metric,
            mode=mode,
        )

    # Initialize scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=max_epochs_per_trial,
        grace_period=1,
        reduction_factor=2,
    )

    # Add timestamp to directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hparam_opt_{timestamp}"
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Run hyperparameter search
    try:
        # Use a file URI for storage_path to satisfy pyarrow fs requirements
        storage_uri = output_path.resolve().as_uri()
        # Build tune.run kwargs and include search_alg only if provided
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
    except Exception as e:
        # Propagate failure from train_fn or wrap other exceptions
        if "fail" in str(e):
            raise e
        raise RuntimeError(str(e))

    # Log best configuration
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    logger.info(f"Best configuration: {best_config}")

    # Create a summary report
    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    if best_trial is None:
        raise RuntimeError("No valid trials found for hyperparameter optimization")
    best_result = best_trial.last_result

    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "metric": metric,
        "mode": mode,
        "num_samples": num_samples,
        "best_value": best_result[metric],
        "best_config": best_config,
        "best_trial_id": best_trial.trial_id,
    }

    # Save summary as JSON
    import json

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create visualization
    try:
        df = analysis.dataframe()
        if not df.empty:
            # Plot training progression for each trial
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
    except Exception as e:
        logger.warning(f"Failed to create visualization: {str(e)}")

    return analysis


if __name__ == "__main__":
    # Simple example usage
    input_dim = 10
    output_size = 1
    model = CNNLSTMModel(input_dim=input_dim, output_size=output_size)

    summarizer = ModelSummarizer(model)
    print(summarizer.get_summary())

    gpu_info = detect_gpus()
    print(f"\nGPU Information:")
    print(f"Available: {gpu_info['available']}")
    print(f"Count: {gpu_info['count']}")

    if gpu_info["available"]:
        for i, device in enumerate(gpu_info["devices"]):
            print(f"\nGPU {i}: {device['name']}")
            print(f"  Memory: {device['total_memory']:.2f} MB")

    # Profile model inference
    perf = profile_model_inference(model, batch_size=32, sequence_length=60)
    print("\nPerformance Metrics:")
    for key, value in perf.items():
        print(f"  {key}: {value}")

    # Get optimal GPU configuration
    optimal = optimal_gpu_config(summarizer.num_params["total"], batch_size=32)
    print("\nOptimal GPU Configuration:")
    for key, value in optimal.items():
        print(f"  {key}: {value}")
