"""
Tests for enhanced training module.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from trade_agent.training.train_cnn_lstm_enhanced import (
    EnhancedCNNLSTMTrainer,
    HyperparameterOptimizer,
    create_enhanced_model_config,
    create_enhanced_training_config,
)


class TestEnhancedCNNLSTMTrainer:
    """Test suite for EnhancedCNNLSTMTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        seq_length = 20
        n_features = 10

        sequences = np.random.randn(n_samples, seq_length, n_features)
        targets = np.sum(sequences[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1

        return sequences, targets

    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return {
            "cnn_filters": [16, 32],
            "cnn_kernel_sizes": [3, 3],
            "lstm_units": 64,
            "lstm_layers": 1,
            "dropout_rate": 0.1,
            "output_size": 1,
        }

    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 3,
            "weight_decay": 1e-5,
            "val_split": 0.2,
            "early_stopping_patience": 2,
            "lr_patience": 1,
            "max_grad_norm": 1.0,
        }

    def test_trainer_initialization(self, model_config, training_config):
        """Test trainer initialization."""
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )

        assert trainer.model_config == model_config
        assert trainer.training_config == training_config
        assert trainer.model is None
        assert len(trainer.history["train_loss"]) == 0

    def test_trainer_device_selection(self, model_config, training_config):
        """Test automatic device selection."""
        trainer = EnhancedCNNLSTMTrainer(model_config=model_config, training_config=training_config, device=None)

        # Should select a valid device
        assert trainer.device in [torch.device("cpu"), torch.device("cuda")]

    def test_trainer_custom_device(self, model_config, training_config):
        """Test custom device selection."""
        trainer = EnhancedCNNLSTMTrainer(model_config=model_config, training_config=training_config, device="cpu")

        assert trainer.device == torch.device("cpu")

    def test_create_model(self, model_config, training_config):
        """Test model creation."""
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
        )

        input_dim = 10
        model = trainer._create_model(input_dim)

        assert model is not None
        assert hasattr(model, "forward")
        assert model.input_dim == input_dim

    def test_create_optimizer(self, model_config, training_config):
        """Test optimizer creation."""
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
        )

        # Create a model first
        input_dim = 10
        trainer.model = trainer._create_model(input_dim)

        optimizer = trainer._create_optimizer()

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_scheduler(self, model_config, training_config):
        """Test scheduler creation."""
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
        )

        # Create model and optimizer first
        input_dim = 10
        trainer.model = trainer._create_model(input_dim)
        trainer.optimizer = trainer._create_optimizer()

        scheduler = trainer._create_scheduler()

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_calculate_metrics(self, model_config, training_config):
        """Test metrics calculation."""
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
        )

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = trainer._calculate_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mse" in metrics

        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mse"] > 0

    def test_training_workflow(self, sample_data, model_config, training_config):
        """Test complete training workflow."""
        sequences, targets = sample_data

        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.pth"

            result = trainer.train_from_dataset(
                sequences=sequences,
                targets=targets,
                save_path=str(save_path),
            )

        # Verify training results
        assert "best_val_loss" in result
        assert "total_epochs" in result
        assert "final_metrics" in result
        assert "training_time" in result

        # Verify model was trained
        assert trainer.model is not None
        assert len(trainer.history["train_loss"]) > 0
        assert len(trainer.history["val_loss"]) > 0

    def test_model_checkpointing(self, sample_data, model_config, training_config):
        """Test model checkpointing and loading."""
        sequences, targets = sample_data

        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "checkpoint.pth"

            # Train and save
            result = trainer.train_from_dataset(
                sequences=sequences,
                targets=targets,
                save_path=str(save_path),
            )

            # Verify checkpoint was created
            assert save_path.exists()

            # Load checkpoint
            checkpoint = trainer.load_checkpoint(str(save_path))

            assert "epoch" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "val_loss" in checkpoint
            assert "model_config" in checkpoint
            assert "training_config" in checkpoint
            assert "history" in checkpoint

    def test_mlflow_setup_success(self, model_config, training_config):
        """Test successful MLflow setup."""
        # Mock the import inside the method
        with patch("builtins.__import__") as mock_import:
            mock_mlflow = Mock()
            mock_import.return_value = mock_mlflow

            trainer = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                enable_mlflow=True,
                enable_tensorboard=False,
            )

            # Since MLflow is not actually available, it should be disabled
            assert trainer.enable_mlflow is False

    def test_mlflow_setup_failure(self, model_config, training_config):
        """Test MLflow setup failure handling."""
        # Mock the import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("MLflow not available")):
            trainer = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                enable_mlflow=True,
                enable_tensorboard=False,
            )

            assert trainer.enable_mlflow is False

    def test_tensorboard_setup_success(self, model_config, training_config):
        """Test successful TensorBoard setup."""
        # Mock the import inside the method
        with patch("builtins.__import__") as mock_import:
            mock_writer = Mock()
            mock_import.return_value = mock_writer

            trainer = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                enable_mlflow=False,
                enable_tensorboard=True,
            )

            # Since TensorBoard is not actually available, it should be disabled
            assert trainer.enable_tensorboard is False

    def test_tensorboard_setup_failure(self, model_config, training_config):
        """Test TensorBoard setup failure handling."""
        # Mock the import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("TensorBoard not available")):
            trainer = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                enable_mlflow=False,
                enable_tensorboard=True,
            )

            assert trainer.enable_tensorboard is False


class TestHyperparameterOptimizer:
    """Test suite for HyperparameterOptimizer."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for optimization."""
        np.random.seed(42)
        n_samples = 50
        seq_length = 15
        n_features = 8

        sequences = np.random.randn(n_samples, seq_length, n_features)
        targets = np.sum(sequences[:, -3:, :2], axis=(1, 2)) + np.random.randn(n_samples) * 0.1

        return sequences, targets

    def test_optimizer_initialization(self, sample_data):
        """Test optimizer initialization."""
        sequences, targets = sample_data

        optimizer = HyperparameterOptimizer(sequences=sequences, targets=targets, n_trials=5, timeout=30)

        assert optimizer.sequences.shape == sequences.shape
        assert optimizer.targets.shape == targets.shape
        assert optimizer.n_trials == 5
        assert optimizer.timeout == 30

    def test_suggest_model_config(self, sample_data):
        """Test model configuration suggestion."""
        sequences, targets = sample_data
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=1)

        # Mock trial
        trial = Mock()
        trial.suggest_categorical.return_value = [32, 64]
        trial.suggest_int.return_value = 64
        trial.suggest_float.return_value = 0.2

        config = optimizer._suggest_model_config(trial)

        assert "cnn_filters" in config
        assert "cnn_kernel_sizes" in config
        assert "lstm_units" in config
        assert "lstm_layers" in config
        assert "dropout_rate" in config
        assert "use_attention" in config
        assert "use_residual" in config
        assert "output_size" in config

    def test_suggest_training_config(self, sample_data):
        """Test training configuration suggestion."""
        sequences, targets = sample_data
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=1)

        # Mock trial
        trial = Mock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 32
        trial.suggest_int.return_value = 5

        config = optimizer._suggest_training_config(trial)

        assert "learning_rate" in config
        assert "batch_size" in config
        assert "epochs" in config
        assert "weight_decay" in config
        assert "val_split" in config
        assert "early_stopping_patience" in config
        assert "lr_patience" in config
        assert "max_grad_norm" in config

    @patch("trading_rl_agent.training.train_cnn_lstm_enhanced.optuna")
    def test_optimization_workflow(self, mock_optuna, sample_data):
        """Test optimization workflow."""
        sequences, targets = sample_data

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"model_config": {}, "training_config": {}}
        mock_study.best_value = 0.5
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=2)

        result = optimizer.optimize()

        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result

        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()


class TestConfigFunctions:
    """Test suite for configuration functions."""

    def test_create_enhanced_model_config(self):
        """Test enhanced model configuration creation."""
        config = create_enhanced_model_config(
            cnn_filters=[32, 64, 128],
            cnn_kernel_sizes=[3, 5, 7],
            lstm_units=256,
            lstm_layers=3,
            dropout_rate=0.3,
            use_attention=True,
            use_residual=True,
        )

        assert config["cnn_filters"] == [32, 64, 128]
        assert config["cnn_kernel_sizes"] == [3, 5, 7]
        assert config["lstm_units"] == 256
        assert config["lstm_layers"] == 3
        assert config["dropout_rate"] == 0.3
        assert config["use_attention"] is True
        assert config["use_residual"] is True
        assert config["output_size"] == 1

    def test_create_enhanced_training_config(self):
        """Test enhanced training configuration creation."""
        config = create_enhanced_training_config(
            learning_rate=0.0001,
            batch_size=64,
            epochs=200,
            weight_decay=1e-4,
            val_split=0.3,
            early_stopping_patience=15,
            lr_patience=8,
            max_grad_norm=2.0,
        )

        assert config["learning_rate"] == 0.0001
        assert config["batch_size"] == 64
        assert config["epochs"] == 200
        assert config["weight_decay"] == 1e-4
        assert config["val_split"] == 0.3
        assert config["early_stopping_patience"] == 15
        assert config["lr_patience"] == 8
        assert config["max_grad_norm"] == 2.0

    def test_config_defaults(self):
        """Test configuration defaults."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()

        # Check model config defaults
        assert model_config["cnn_filters"] == [64, 128, 256]
        assert model_config["cnn_kernel_sizes"] == [3, 3, 3]
        assert model_config["lstm_units"] == 256
        assert model_config["lstm_layers"] == 2
        assert model_config["dropout_rate"] == 0.2
        assert model_config["use_attention"] is False
        assert model_config["use_residual"] is False

        # Check training config defaults
        assert training_config["learning_rate"] == 0.001
        assert training_config["batch_size"] == 32
        assert training_config["epochs"] == 100
        assert training_config["weight_decay"] == 1e-5
        assert training_config["val_split"] == 0.2
        assert training_config["early_stopping_patience"] == 10
        assert training_config["lr_patience"] == 5
        assert training_config["max_grad_norm"] == 1.0
