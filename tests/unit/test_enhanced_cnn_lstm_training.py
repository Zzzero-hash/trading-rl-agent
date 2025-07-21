"""
Comprehensive tests for enhanced CNN+LSTM training module.

This module tests the EnhancedCNNLSTMTrainer class and related functionality
with focus on small incremental fixes and edge cases.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from trade_agent.models.cnn_lstm import CNNLSTMModel
from trade_agent.training.train_cnn_lstm_enhanced import (
    EnhancedCNNLSTMTrainer,
    create_enhanced_model_config,
    create_enhanced_training_config,
)


class TestEnhancedCNNLSTMTrainer:
    """Test suite for EnhancedCNNLSTMTrainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization with default parameters."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        assert trainer.device is not None
        assert trainer.logger is not None
        assert trainer.metrics_history is not None
        assert isinstance(trainer.metrics_history, dict)

    def test_trainer_device_selection(self):
        """Test automatic device selection."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Should automatically select CPU if CUDA not available
        if torch.cuda.is_available():
            assert trainer.device.type in ["cuda", "cpu"]
        else:
            assert trainer.device.type == "cpu"

    def test_trainer_custom_device(self):
        """Test trainer with custom device specification."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config, device="cpu")
        assert trainer.device.type == "cpu"

    def test_create_model(self):
        """Test model creation functionality."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Test with default config
        model = trainer.create_model()
        assert isinstance(model, CNNLSTMModel)

        # Test with custom config
        custom_config = {
            "input_dim": 20,
            "lstm_units": 64,
            "lstm_num_layers": 1,
            "output_dim": 1,
        }
        model = trainer.create_model(model_config=custom_config)
        assert isinstance(model, CNNLSTMModel)
        assert model.lstm_units == 64

    def test_create_optimizer(self):
        """Test optimizer creation functionality."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)
        model = trainer.create_model()

        # Test default optimizer
        optimizer = trainer.create_optimizer(model)
        assert isinstance(optimizer, torch.optim.Adam)

        # Test custom learning rate
        optimizer = trainer.create_optimizer(model, learning_rate=0.01)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_create_scheduler(self):
        """Test learning rate scheduler creation."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)

        # Test scheduler creation
        scheduler = trainer.create_scheduler(optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_calculate_metrics(self):
        """Test metrics calculation functionality."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create dummy predictions and targets
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1])

        metrics = trainer.calculate_metrics(predictions, targets)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_training_workflow(self):
        """Test complete training workflow."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create dummy data
        X = torch.randn(50, 10, 5)  # 50 samples, 10 timesteps, 5 features
        y = torch.randn(50, 1)  # 50 samples, 1 target

        # Create model and train
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)
        trainer.create_scheduler(optimizer)

        # Test training step
        loss = trainer.train_step(model, optimizer, X, y)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_model_checkpointing(self):
        """Test model checkpointing functionality."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)
        model = trainer.create_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_checkpoint.pth"

            # Test saving checkpoint
            trainer.save_checkpoint(model, checkpoint_path, epoch=1, loss=0.5)
            assert checkpoint_path.exists()

            # Test loading checkpoint
            loaded_model, epoch, loss = trainer.load_checkpoint(checkpoint_path)
            assert isinstance(loaded_model, CNNLSTMModel)
            assert epoch == 1
            assert loss == 0.5

    def test_mlflow_setup_success(self):
        """Test MLflow setup with successful connection."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.start_run") as mock_start_run,
        ):
            trainer.setup_mlflow(experiment_name="test_experiment")
            mock_start_run.assert_called_once()

    def test_mlflow_setup_failure(self):
        """Test MLflow setup with connection failure."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        with patch("mlflow.set_tracking_uri", side_effect=Exception("Connection failed")):
            # Should not raise exception, just log warning
            trainer.setup_mlflow(experiment_name="test_experiment")

    def test_tensorboard_setup_success(self):
        """Test TensorBoard setup with successful creation."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"

            with patch("torch.utils.tensorboard.SummaryWriter") as mock_writer:
                trainer.setup_tensorboard(log_dir)
                mock_writer.assert_called_once_with(log_dir)

    def test_tensorboard_setup_failure(self):
        """Test TensorBoard setup with creation failure."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        with patch(
            "torch.utils.tensorboard.SummaryWriter",
            side_effect=Exception("TensorBoard failed"),
        ):
            # Should not raise exception, just log warning
            trainer.setup_tensorboard("invalid_path")

    def test_memory_efficiency(self):
        """Test memory efficiency with large models."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Test with larger model
        large_config = {
            "input_dim": 50,
            "lstm_units": 256,
            "lstm_num_layers": 3,
            "output_dim": 1,
        }

        model = trainer.create_model(model_config=large_config)
        assert isinstance(model, CNNLSTMModel)

        # Test memory usage doesn't explode
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Create some dummy data and run forward pass
        X = torch.randn(10, 20, 50)
        with torch.no_grad():
            _ = model(X)

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            # Memory increase should be reasonable (less than 1GB)
            assert (final_memory - initial_memory) < 1024 * 1024 * 1024


class TestHyperparameterOptimization:
    """Test suite for hyperparameter optimization functionality."""

    def test_optimizer_initialization(self):
        """Test hyperparameter optimizer initialization."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Test optimizer creation
        optimizer = trainer.create_hyperparameter_optimizer()
        assert optimizer is not None

    def test_optimization_objective(self):
        """Test optimization objective function."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create dummy trial
        trial = Mock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_int.return_value = 64

        # Test objective function
        objective_value = trainer.optimization_objective(trial)
        assert isinstance(objective_value, float)

    def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create dummy data
        X = torch.randn(20, 5, 3)  # Small dataset for testing
        y = torch.randn(20, 1)

        # Test optimization with limited trials
        best_params = trainer.optimize_hyperparameters(X, y, n_trials=2)
        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params
        assert "lstm_units" in best_params

    def test_coordinated_cnn_architecture_selection(self):
        """Test coordinated CNN architecture selection."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Test different CNN architectures
        architectures = ["simple", "residual", "dense"]

        for arch in architectures:
            config = create_enhanced_model_config(cnn_architecture=arch)
            assert config["cnn_architecture"] == arch

            # Should be able to create model with this config
            model = trainer.create_model(model_config=config)
            assert isinstance(model, CNNLSTMModel)

    def test_model_config_completeness(self):
        """Test that model config contains all required parameters."""
        config = create_enhanced_model_config()

        required_keys = [
            "input_dim",
            "lstm_units",
            "lstm_num_layers",
            "output_dim",
            "cnn_architecture",
            "dropout_rate",
        ]

        for key in required_keys:
            assert key in config
            assert config[key] is not None


class TestConfigFunctions:
    """Test suite for configuration functions."""

    def test_create_enhanced_model_config(self):
        """Test enhanced model config creation."""
        config = create_enhanced_model_config()

        assert isinstance(config, dict)
        assert "input_dim" in config
        assert "lstm_units" in config
        assert "lstm_num_layers" in config
        assert "output_dim" in config
        assert "cnn_architecture" in config
        assert "dropout_rate" in config

    def test_create_enhanced_training_config(self):
        """Test enhanced training config creation."""
        config = create_enhanced_training_config()

        assert isinstance(config, dict)
        assert "learning_rate" in config
        assert "batch_size" in config
        assert "epochs" in config
        assert "early_stopping_patience" in config

    def test_config_defaults(self):
        """Test config default values are reasonable."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()

        # Test model config defaults
        assert model_config["input_dim"] > 0
        assert model_config["lstm_units"] > 0
        assert model_config["lstm_num_layers"] > 0
        assert 0 <= model_config["dropout_rate"] <= 1

        # Test training config defaults
        assert training_config["learning_rate"] > 0
        assert training_config["batch_size"] > 0
        assert training_config["epochs"] > 0
        assert training_config["early_stopping_patience"] > 0


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_invalid_model_config(self):
        """Test handling of invalid model configuration."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        invalid_config = {
            "input_dim": -1,  # Invalid negative value
            "lstm_units": 0,  # Invalid zero value
        }

        with pytest.raises(ValueError):
            trainer.create_model(model_config=invalid_config)

    def test_invalid_training_data(self):
        """Test handling of invalid training data."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)

        # Test with mismatched dimensions
        X = torch.randn(10, 5, 3)
        y = torch.randn(5, 1)  # Mismatched batch size

        with pytest.raises(ValueError):
            trainer.train_step(model, optimizer, X, y)

    def test_checkpoint_file_not_found(self):
        """Test handling of missing checkpoint file."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = Path(temp_dir) / "non_existent.pth"

            with pytest.raises(FileNotFoundError):
                trainer.load_checkpoint(non_existent_path)

    def test_device_handling(self):
        """Test device handling with invalid device specification."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        with pytest.raises(ValueError):
            EnhancedCNNLSTMTrainer(model_config, training_config, device="invalid_device")


class TestIntegration:
    """Integration tests for the enhanced training module."""

    def test_end_to_end_training(self):
        """Test complete end-to-end training workflow."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create synthetic data
        X = torch.randn(100, 10, 5)
        y = torch.randn(100, 1)

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Create model and train
        model = trainer.create_model()
        optimizer = trainer.create_optimizer(model)
        scheduler = trainer.create_scheduler(optimizer)

        # Train for a few epochs
        for epoch in range(3):
            train_loss = trainer.train_step(model, optimizer, X_train, y_train)

            # Validation
            with torch.no_grad():
                val_predictions = model(X_val)
                val_metrics = trainer.calculate_metrics(val_predictions, y_val)

            scheduler.step(val_metrics["mse"])

            assert isinstance(train_loss, float)
            assert train_loss >= 0
            assert "mse" in val_metrics
            assert "mae" in val_metrics

    def test_hyperparameter_optimization_integration(self):
        """Test integration of hyperparameter optimization."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        trainer = EnhancedCNNLSTMTrainer(model_config, training_config)

        # Create small synthetic dataset
        X = torch.randn(50, 5, 3)
        y = torch.randn(50, 1)

        # Run optimization
        best_params = trainer.optimize_hyperparameters(X, y, n_trials=3)

        # Create model with best parameters
        model = trainer.create_model(model_config=best_params)
        assert isinstance(model, CNNLSTMModel)

        # Test that model can be trained
        optimizer = trainer.create_optimizer(model)
        loss = trainer.train_step(model, optimizer, X, y)
        assert isinstance(loss, float)
        assert loss >= 0
