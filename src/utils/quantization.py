"""
Model Quantization Utilities

This module provides utilities for quantizing ML models to reduce memory
usage and improve inference speed for trading applications.
"""

from typing import Any, Dict, Optional, Union
import warnings

import torch
import torch.nn as nn


def quantize_model(
    model: nn.Module, quantization_type: str = "dynamic", backend: str = "fbgemm"
) -> nn.Module:
    """
    Quantize a PyTorch model for efficient inference.

    Args:
        model: PyTorch model to quantize
        quantization_type: Type of quantization ('dynamic', 'static', 'qat')
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

    Returns:
        Quantized model
    """
    if not torch.backends.quantized.supported_engines:
        warnings.warn("Quantization not supported on this platform")
        return model

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Prepare model for quantization
    model.eval()

    if quantization_type == "dynamic":
        return dynamic_quantization(model)
    elif quantization_type == "static":
        return static_quantization(model)
    elif quantization_type == "qat":
        warnings.warn("QAT requires retraining - returning original model")
        return model
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")


def dynamic_quantization(
    model: nn.Module, dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to model.

    Args:
        model: PyTorch model to quantize
        dtype: Quantization data type

    Returns:
        Dynamically quantized model
    """
    # Define layers to quantize
    layers_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}

    quantized_model = torch.quantization.quantize_dynamic(
        model, qconfig_spec=layers_to_quantize, dtype=dtype
    )

    return quantized_model


def static_quantization(
    model: nn.Module, calibration_data: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Apply static quantization to model (requires calibration data).

    Args:
        model: PyTorch model to quantize
        calibration_data: Sample data for calibration

    Returns:
        Statically quantized model
    """
    if calibration_data is None:
        warnings.warn(
            "Static quantization requires calibration data - using dynamic instead"
        )
        return dynamic_quantization(model)

    # Prepare model for static quantization
    # For now, fallback to dynamic quantization since static requires more setup
    warnings.warn("Static quantization setup complex - using dynamic quantization")
    return dynamic_quantization(model)


def int8_quantization(model: nn.Module) -> nn.Module:
    """
    Apply INT8 quantization to model.

    Args:
        model: PyTorch model to quantize

    Returns:
        INT8 quantized model
    """
    return dynamic_quantization(model, dtype=torch.qint8)


def compare_model_sizes(
    original_model: nn.Module, quantized_model: nn.Module
) -> dict[str, float]:
    """
    Compare sizes of original and quantized models.

    Args:
        original_model: Original model
        quantized_model: Quantized model

    Returns:
        Dictionary with size comparison metrics
    """

    def get_model_size(model):
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size + buffer_size

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    size_reduction = (
        (original_size - quantized_size) / original_size if original_size > 0 else 0
    )

    return {
        "original_size_mb": original_size / (1024 * 1024),
        "quantized_size_mb": quantized_size / (1024 * 1024),
        "compression_ratio": compression_ratio,
        "size_reduction_percent": size_reduction * 100,
    }


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark inference speed of original vs quantized model.

    Args:
        original_model: Original model
        quantized_model: Quantized model
        sample_input: Sample input tensor
        num_runs: Number of inference runs for timing

    Returns:
        Dictionary with timing comparison metrics
    """
    import time

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            original_model(sample_input)
            quantized_model(sample_input)

    # Benchmark original model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            original_model(sample_input)
    original_time = (time.time() - start_time) / num_runs

    # Benchmark quantized model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            quantized_model(sample_input)
    quantized_time = (time.time() - start_time) / num_runs

    speedup = original_time / quantized_time if quantized_time > 0 else 0

    return {
        "original_inference_ms": original_time * 1000,
        "quantized_inference_ms": quantized_time * 1000,
        "speedup_ratio": speedup,
        "speed_improvement_percent": (speedup - 1) * 100 if speedup > 1 else 0,
    }


def quantize_for_deployment(
    model: nn.Module, sample_input: torch.Tensor, target_platform: str = "cpu"
) -> dict[str, Any]:
    """
    Quantize model for deployment with performance analysis.

    Args:
        model: PyTorch model to quantize
        sample_input: Sample input for testing
        target_platform: Target deployment platform ('cpu', 'mobile')

    Returns:
        Dictionary with quantized model and performance metrics
    """
    # Choose quantization settings based on platform
    if target_platform == "mobile":
        backend = "qnnpack"
        quantization_type = "dynamic"
    else:
        backend = "fbgemm"
        quantization_type = "dynamic"

    # Quantize the model
    quantized_model = quantize_model(
        model, quantization_type=quantization_type, backend=backend
    )

    # Analyze performance
    size_metrics = compare_model_sizes(model, quantized_model)
    speed_metrics = benchmark_quantized_model(model, quantized_model, sample_input)

    return {
        "quantized_model": quantized_model,
        "size_metrics": size_metrics,
        "speed_metrics": speed_metrics,
        "quantization_settings": {
            "type": quantization_type,
            "backend": backend,
            "target_platform": target_platform,
        },
    }


# Legacy compatibility functions
def quantize_cnn_lstm(model: nn.Module) -> nn.Module:
    """Legacy function - use quantize_model instead."""
    return quantize_model(model, quantization_type="dynamic")


def quantize_trading_model(model: nn.Module) -> nn.Module:
    """Legacy function - use quantize_model instead."""
    return quantize_model(model, quantization_type="dynamic")
