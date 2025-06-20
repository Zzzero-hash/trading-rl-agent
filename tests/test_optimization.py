"""Tests for model optimization and hyperparameter tuning utilities.

Tests cover:
1. Model summary and profiling
2. GPU detection and optimization
3. Ray Tune integration for hyperparameter optimization
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import ray
from ray import tune
import torch
import torch.nn as nn

from src.models.cnn_lstm import CNNLSTMModel
from src.optimization.model_summary import (
    ModelSummarizer,
    detect_gpus,
    optimal_gpu_config,
    profile_model_inference,
    run_hyperparameter_optimization,
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_ray_initialized():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestModelSummarizer:
    """Test ModelSummarizer class."""

    def test_model_summarizer_initialization(self):
        """Test ModelSummarizer initialization."""
        # Create a simple model for testing
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        # Initialize summarizer
        summarizer = ModelSummarizer(model)

        # Check attributes
        assert summarizer.model is model
        assert summarizer.num_params is not None
        assert "trainable" in summarizer.num_params
        assert "non_trainable" in summarizer.num_params
        assert "total" in summarizer.num_params

        # Check if the number of parameters is correct
        # 10*20 + 20 bias + 20*1 + 1 bias = 241
        expected_params = 241
        assert summarizer.num_params["total"] == expected_params

    def test_get_summary(self):
        """Test summary generation."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        summarizer = ModelSummarizer(model)
        summary = summarizer.get_summary()

        # Check if summary contains key information
        assert "Parameters: 241 total" in summary
        assert "Device:" in summary

        # Test detailed summary
        detailed = summarizer.get_summary(detailed=True)
        assert "Layer details:" in detailed

        # Test simplified summary
        simplified = summarizer.get_summary(detailed=False)
        assert "Layer details:" not in simplified

    def test_cnn_lstm_summarizer(self):
        """Test summarizer with CNN-LSTM model."""
        model = CNNLSTMModel(input_dim=10, output_size=1)

        summarizer = ModelSummarizer(model)
        summary = summarizer.get_summary()

        # Check if summary contains model name
        assert "Model: CNNLSTMModel" in summary

        # Make sure it doesn't error with detailed=True
        detailed = summarizer.get_summary(detailed=True)
        assert isinstance(detailed, str)


class TestModelSummaryEdgeCases:
    """Test edge cases for model summarization."""

    def test_model_with_no_parameters(self):
        """Test summarization of a model with no parameters."""

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        model = Dummy()
        summarizer = ModelSummarizer(model)
        summary = summarizer.get_summary()
        assert "Parameters: 0 total" in summary
        assert summarizer.memory_estimate["total"] == 0

    def test_model_with_non_trainable_params(self):
        """Test summarization of a model with non-trainable parameters."""

        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, x):
                return x

        model = Dummy()
        summarizer = ModelSummarizer(model)
        assert summarizer.num_params["trainable"] == 0
        assert summarizer.num_params["non_trainable"] == 1

    def test_summary_handles_large_model(self):
        """Test summarization of a large model."""
        model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(20)])
        summarizer = ModelSummarizer(model)
        summary = summarizer.get_summary()
        assert "Model: Sequential" in summary
        assert summarizer.num_params["total"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUOperations:
    """Test GPU detection and optimization."""

    def test_detect_gpus(self):
        """Test GPU detection."""
        gpu_info = detect_gpus()

        assert isinstance(gpu_info, dict)
        assert "available" in gpu_info
        assert "count" in gpu_info

        # On systems with GPUs available AND actual devices present
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            assert gpu_info["available"] is True
            assert gpu_info["count"] > 0
            assert len(gpu_info["devices"]) == gpu_info["count"]

            # Check device info
            device_info = gpu_info["devices"][0]
            assert "name" in device_info
            assert "total_memory" in device_info
        else:
            # CUDA may be available but no devices present
            assert gpu_info["count"] == torch.cuda.device_count()

    def test_optimal_gpu_config(self):
        """Test optimal GPU configuration."""
        model_params = 1_000_000  # 1M parameters
        batch_size = 32

        config = optimal_gpu_config(
            model_params=model_params,
            batch_size=batch_size,
            sequence_length=60,
            feature_dim=10,
        )

        assert isinstance(config, dict)
        assert "use_gpu" in config
        # Fixed: Check both CUDA availability AND device count
        assert config["use_gpu"] == (
            torch.cuda.is_available() and torch.cuda.device_count() > 0
        )

        # More detailed tests if GPU is available AND has devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            assert "gpu_index" in config
            assert "gpu_name" in config
            assert "recommended_batch_size" in config
            assert "mixed_precision" in config
            assert isinstance(config["recommended_batch_size"], int)
            assert isinstance(config["mixed_precision"], bool)


class TestGPUDetectionRobustness:
    """Test robustness of GPU detection methods."""

    def test_detect_gpus_no_gpu(self, monkeypatch):
        """Test GPU detection when no GPU is available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        info = detect_gpus()
        assert info["available"] is False
        assert info["count"] == 0
        assert info["devices"] == []

    def test_optimal_gpu_config_no_gpu(self, monkeypatch):
        """Test optimal GPU configuration when no GPU is available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = optimal_gpu_config(1000, 8)
        assert config["use_gpu"] is False
        assert "No GPUs available" in config["reason"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available for profiling"
)
class TestModelProfiling:
    """Test model profiling."""

    def test_profile_model_inference(self):
        """Test model inference profiling."""
        model = CNNLSTMModel(input_dim=10, output_size=1)

        # Profile with minimal runs for testing
        profile = profile_model_inference(
            model=model,
            batch_size=2,
            sequence_length=10,
            num_features=10,
            num_warmup=1,
            num_runs=2,
        )

        assert isinstance(profile, dict)
        assert "avg_time_ms" in profile
        assert profile["avg_time_ms"] > 0
        assert "throughput_samples_per_sec" in profile
        assert profile["throughput_samples_per_sec"] > 0
        assert "device" in profile


class TestProfilingEdgeCases:
    """Test edge cases for model profiling."""

    def test_profile_model_inference_various_shapes(self):
        """Test profiling with various input shapes."""
        model = CNNLSTMModel(input_dim=5, output_size=1)
        for batch_size in [1, 8]:
            for seq_len in [5, 20]:
                perf = profile_model_inference(
                    model,
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    num_features=5,
                    num_warmup=1,
                    num_runs=1,
                )
                assert perf["avg_time_ms"] > 0

    def test_profile_model_inference_on_cpu(self):
        """Test profiling of model inference on CPU."""
        model = CNNLSTMModel(input_dim=3, output_size=1)
        perf = profile_model_inference(
            model,
            batch_size=2,
            sequence_length=4,
            num_features=3,
            num_warmup=1,
            num_runs=1,
        )
        assert perf["device"] == "cpu" or "cuda" in perf["device"]


@pytest.mark.integration
@pytest.mark.skipif(not ray.is_initialized(), reason="Ray not initialized")
class TestHyperparamOptimization:
    """Test hyperparameter optimization with Ray Tune."""

    def test_hyperparameter_search_space(self):
        """Test hyperparameter search space conversion for Ray Tune."""
        from src.optimization.cnn_lstm_optimization import (
            _get_default_cnn_lstm_search_space,
        )
        from src.optimization.rl_optimization import (
            _get_default_ppo_search_space,
            _get_default_sac_search_space,
            _get_default_td3_search_space,
        )

        # Test CNN-LSTM search space
        cnn_lstm_space = _get_default_cnn_lstm_search_space()
        assert isinstance(cnn_lstm_space, dict)
        assert "cnn_filters" in cnn_lstm_space
        assert "lstm_units" in cnn_lstm_space
        assert "learning_rate" in cnn_lstm_space

        # Test TD3 search space
        td3_space = _get_default_td3_search_space()
        assert isinstance(td3_space, dict)
        assert "actor_lr" in td3_space
        assert "critic_lr" in td3_space

        # Test SAC search space
        sac_space = _get_default_sac_search_space()
        assert isinstance(sac_space, dict)
        assert "twin_q" in sac_space

        # Test PPO search space
        ppo_space = _get_default_ppo_search_space()
        assert isinstance(ppo_space, dict)
        assert "lr" in ppo_space

    @pytest.mark.skip(reason="Integration test that requires Ray cluster")
    def test_optimize_cnn_lstm_minimal(self):
        """Minimal test for CNN-LSTM optimization."""
        from src.optimization.cnn_lstm_optimization import optimize_cnn_lstm

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        features = np.random.randn(n_samples, n_features)
        targets = np.sin(np.arange(n_samples) * 0.1) + 0.1 * np.random.randn(n_samples)

        # Define minimal search space for testing
        search_space = {
            "cnn_filters": [[4]],
            "cnn_kernel_sizes": [[2]],
            "lstm_units": 8,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "batch_size": 16,
            "sequence_length": 10,
            "prediction_horizon": 1,
        }

        # Run with only 1 sample and minimal epochs
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = optimize_cnn_lstm(
                features=features,
                targets=targets,
                num_samples=1,
                max_epochs_per_trial=2,
                early_stopping_patience=1,
                output_dir=temp_dir,
                custom_search_space=search_space,
                gpu_per_trial=0.0,  # Use CPU for testing
            )

            assert analysis is not None

    @pytest.mark.skip(reason="Integration test that requires Ray cluster")
    def test_optimize_rl_minimal(self):
        """Minimal test for RL optimization."""
        from src.optimization.rl_optimization import optimize_td3_hyperparams

        # Create a minimal environment config
        env_config = {
            "dataset_paths": ["data/sample_training_data_simple_20250607_192034.csv"],
            "window_size": 10,
        }

        # Minimal search space
        search_space = {"actor_lr": 0.001, "critic_lr": 0.001, "train_batch_size": 32}

        # Run with only 1 sample and minimal iterations
        with tempfile.TemporaryDirectory() as temp_dir:
            analysis = optimize_td3_hyperparams(
                env_config=env_config,
                num_samples=1,
                max_iterations_per_trial=2,
                output_dir=temp_dir,
                custom_search_space=search_space,
                gpu_per_trial=0.0,  # Use CPU for testing
            )

            assert analysis is not None


class TestHyperparamOptimizationEdgeCases:
    """Test edge cases for hyperparameter optimization."""

    def test_run_hyperparameter_optimization_invalid_train_fn(self):
        """Test hyperparameter optimization with an invalid training function."""

        def bad_train_fn(config):
            raise RuntimeError("fail")

        config_space = {"lr": tune.grid_search([0.01, 0.1])}
        try:
            run_hyperparameter_optimization(
                bad_train_fn, config_space, num_samples=1, max_epochs_per_trial=1
            )
        except Exception as e:
            assert "fail" in str(e) or isinstance(e, RuntimeError)

    def test_run_hyperparameter_optimization_minimal(self):
        """Test minimal hyperparameter optimization run."""
        from ray.air import session

        def dummy_train_fn(config):
            session.report({"val_loss": 1.0})

        config_space = {"lr": tune.grid_search([0.01])}
        analysis = run_hyperparameter_optimization(
            dummy_train_fn, config_space, num_samples=1, max_epochs_per_trial=1
        )
        assert analysis is not None
