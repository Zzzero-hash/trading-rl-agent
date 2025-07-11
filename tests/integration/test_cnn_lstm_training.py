"""
Integration tests for CNN+LSTM training pipeline.

This module tests the complete training workflow including:
- Configuration loading
- Dataset preparation
- Model training
- Evaluation
- Model saving/loading
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from trading_rl_agent.training.cnn_lstm_trainer import CNNLSTMTrainingManager


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        "model": {
            "cnn_filters": [32, 64],
            "cnn_kernel_sizes": [3, 3],
            "cnn_dropout": 0.1,
            "lstm_units": 64,
            "lstm_num_layers": 1,
            "lstm_dropout": 0.1,
            "dense_units": [32],
            "output_dim": 1,
            "activation": "relu",
            "use_attention": False,
        },
        "training": {
            "batch_size": 16,
            "val_split": 0.2,
            "sequence_length": 20,
            "prediction_horizon": 1,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "epochs": 5,  # Small number for testing
            "early_stopping_patience": 3,
            "reduce_lr_patience": 2,
            "reduce_lr_factor": 0.5,
            "loss_function": "mse",
            "device": "cpu",  # Use CPU for testing
            "num_workers": 0,  # No multiprocessing for testing
        },
        "dataset": {
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "timeframe": "1d",
            "real_data_ratio": 0.0,  # Use only synthetic data for testing
            "min_samples_per_symbol": 100,
            "overlap_ratio": 0.8,
            "technical_indicators": True,
            "sentiment_features": False,
            "market_regime_features": False,
            "outlier_threshold": 3.0,
            "missing_value_threshold": 0.05,
            "output_dir": "test_data",
            "save_metadata": True,
        },
        "monitoring": {
            "log_level": "INFO",
            "log_file": "test_logs/cnn_lstm_training.log",
            "experiment_name": "test_cnn_lstm",
            "tracking_uri": "sqlite:///test_mlruns.db",
            "tensorboard_enabled": False,
            "tensorboard_log_dir": "test_logs/tensorboard",
            "checkpoint_dir": "test_models/checkpoints",
            "save_best_only": True,
            "save_frequency": 2,
            "metrics": ["train_loss", "val_loss", "mae", "rmse"],
        },
        "evaluation": {
            "test_split": 0.2,
            "backtest_period": "2023-02-01",
            "metrics": ["mae", "rmse", "r2_score", "correlation"],
            "plot_predictions": False,
            "plot_training_history": False,
            "save_plots": False,
            "plot_dir": "test_plots",
        },
        "hyperopt": {
            "enabled": False,
            "n_trials": 5,
            "timeout": 300,
            "search_space": {},
        },
        "production": {
            "model_format": "pytorch",
            "model_version": "v1.0.0",
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "api_workers": 1,
            "health_check_interval": 30,
            "metrics_export_interval": 60,
        },
    }


@pytest.fixture
def config_file(sample_config, tmp_path) -> Path:
    """Create a temporary configuration file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestCNNLSTMTrainingIntegration:
    """Integration tests for CNN+LSTM training pipeline."""

    def test_config_loading(self, config_file):
        """Test that configuration can be loaded correctly."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        assert trainer.config is not None
        assert "model" in trainer.config
        assert "training" in trainer.config
        assert "dataset" in trainer.config
        assert "monitoring" in trainer.config

        # Check specific values
        assert trainer.config["model"]["cnn_filters"] == [32, 64]
        assert trainer.config["training"]["batch_size"] == 16
        assert trainer.config["dataset"]["symbols"] == ["AAPL"]

    def test_device_setup(self, config_file):
        """Test device setup (should use CPU for testing)."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        assert trainer.device.type == "cpu"

    def test_dataset_preparation(self, config_file):
        """Test dataset preparation with synthetic data."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        sequences, targets, dataset_info = trainer.prepare_dataset(force_rebuild=True)

        # Check dataset shapes
        assert sequences.ndim == 3  # (samples, sequence_length, features)
        assert targets.ndim == 2  # (samples, 1)
        assert sequences.shape[0] == targets.shape[0]
        assert sequences.shape[1] == trainer.config["training"]["sequence_length"]

        # Check dataset info
        assert "total_samples" in dataset_info
        assert "synthetic_samples" in dataset_info
        assert dataset_info["synthetic_samples"] > 0

    def test_model_creation(self, config_file):
        """Test model creation with correct architecture."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset to get input dimension
        sequences, _, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]

        # Create model
        model = trainer.create_model(input_dim)

        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert model.input_dim == input_dim

        # Test forward pass
        batch_size = 4
        seq_len = trainer.config["training"]["sequence_length"]
        test_input = torch.randn(batch_size, seq_len, input_dim)

        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (batch_size, 1)  # (batch_size, output_dim)

    def test_data_loader_creation(self, config_file):
        """Test data loader creation with proper splits."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)

        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(sequences, targets)

        # Check data loaders
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check that we have data in each loader
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0

        # Check batch shapes
        for batch_data, batch_targets in train_loader:
            assert batch_data.shape[0] <= trainer.config["training"]["batch_size"]
            assert batch_data.shape[1] == trainer.config["training"]["sequence_length"]
            assert batch_targets.shape[0] == batch_data.shape[0]
            break

    def test_training_workflow(self, config_file, temp_output_dir):
        """Test complete training workflow."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)

        # Create model
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)

        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(sequences, targets)

        # Setup checkpoint path
        checkpoint_path = temp_output_dir / "test_checkpoint.pth"

        # Train model
        training_results = trainer.train(train_loader, val_loader, save_path=str(checkpoint_path))

        # Check training results
        assert "best_val_loss" in training_results
        assert "final_train_loss" in training_results
        assert "final_val_loss" in training_results
        assert "total_epochs" in training_results
        assert "training_time" in training_results

        # Check that training actually happened
        assert training_results["total_epochs"] > 0
        assert training_results["training_time"] > 0
        assert training_results["best_val_loss"] > 0

        # Check that checkpoint was saved
        assert checkpoint_path.exists()

    def test_model_evaluation(self, config_file, temp_output_dir):
        """Test model evaluation on test set."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)

        # Create model and train
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)

        train_loader, val_loader, test_loader = trainer.create_data_loaders(sequences, targets)

        # Train for a few epochs
        checkpoint_path = temp_output_dir / "test_checkpoint.pth"
        trainer.train(train_loader, val_loader, save_path=str(checkpoint_path))

        # Evaluate on test set
        test_metrics = trainer.evaluate(test_loader)

        # Check metrics
        assert "mae" in test_metrics
        assert "rmse" in test_metrics
        assert "r2_score" in test_metrics
        assert "correlation" in test_metrics

        # Check that metrics are reasonable
        assert test_metrics["mae"] >= 0
        assert test_metrics["rmse"] >= 0
        assert -1 <= test_metrics["correlation"] <= 1

    def test_model_saving_and_loading(self, config_file, temp_output_dir):
        """Test model saving and loading functionality."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset and train model
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)

        train_loader, val_loader, _ = trainer.create_data_loaders(sequences, targets)
        trainer.train(train_loader, val_loader)

        # Save model
        model_path = temp_output_dir / "test_model.pth"
        trainer.save_model(str(model_path))

        # Check that model was saved
        assert model_path.exists()

        # Create new trainer and load model
        new_trainer = CNNLSTMTrainingManager(str(config_file))
        new_trainer.load_model(str(model_path), input_dim)

        # Check that model was loaded correctly
        assert new_trainer.model is not None
        assert new_trainer.model.input_dim == input_dim

        # Test that loaded model can make predictions
        test_data = torch.randn(2, trainer.config["training"]["sequence_length"], input_dim)
        with torch.no_grad():
            predictions = new_trainer.model(test_data)

        assert predictions.shape == (2, 1)

    def test_prediction_functionality(self, config_file, temp_output_dir):
        """Test prediction functionality."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset and train model
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)

        train_loader, val_loader, _ = trainer.create_data_loaders(sequences, targets)
        trainer.train(train_loader, val_loader)

        # Test prediction
        test_data = sequences[:5]  # Use first 5 samples
        predictions = trainer.predict(test_data)

        # Check predictions
        assert predictions.shape == (5, 1)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()

    def test_training_history_tracking(self, config_file):
        """Test that training history is properly tracked."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Prepare dataset and train
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)

        train_loader, val_loader, _ = trainer.create_data_loaders(sequences, targets)
        trainer.train(train_loader, val_loader)

        # Check training history
        assert "train_loss" in trainer.history
        assert "val_loss" in trainer.history
        assert "metrics" in trainer.history
        assert "learning_rates" in trainer.history

        # Check that history has data
        assert len(trainer.history["train_loss"]) > 0
        assert len(trainer.history["val_loss"]) > 0
        assert len(trainer.history["metrics"]) > 0
        assert len(trainer.history["learning_rates"]) > 0

        # Check that losses are decreasing (generally)
        assert trainer.history["train_loss"][-1] <= trainer.history["train_loss"][0]

    def test_error_handling(self, config_file):
        """Test error handling for invalid inputs."""
        trainer = CNNLSTMTrainingManager(str(config_file))

        # Test with invalid model path
        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_model.pth", 10)

        # Test prediction without trained model
        with pytest.raises(ValueError, match="Model must be loaded or trained"):
            trainer.predict(np.random.randn(5, 20, 10))

    def test_config_validation(self, tmp_path):
        """Test configuration validation."""
        # Test missing required sections
        invalid_config = {
            "model": {"cnn_filters": [32]},
            # Missing 'training', 'dataset', 'monitoring'
        }

        config_path = tmp_path / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError, match="Missing required config section"):
            CNNLSTMTrainingManager(str(config_path))


class TestCNNLSTMTrainingCLI:
    """Integration tests for CLI interface."""

    def test_cli_help(self):
        """Test that CLI help works."""
        from trading_rl_agent.training.cli import create_parser

        parser = create_parser()
        help_text = parser.format_help()

        assert "Train CNN+LSTM models for trading" in help_text
        assert "train" in help_text
        assert "evaluate" in help_text
        assert "predict" in help_text

    def test_cli_train_command(self, config_file, temp_output_dir, capsys):
        """Test CLI train command."""
        import sys

        from trading_rl_agent.training.cli import main

        # Mock command line arguments
        sys.argv = [
            "cli.py",
            "train",
            "--config",
            str(config_file),
            "--output-dir",
            str(temp_output_dir),
            "--log-level",
            "INFO",
        ]

        # Run training
        main()

        # Check output
        captured = capsys.readouterr()
        assert "Starting CNN+LSTM model training" in captured.out
        assert "Training completed successfully" in captured.out

        # Check that files were created
        assert (temp_output_dir / "final_model.pth").exists()
        assert (temp_output_dir / "training_results.json").exists()
        assert (temp_output_dir / "best_model_checkpoint.pth").exists()

    def test_cli_evaluate_command(self, config_file, temp_output_dir, capsys):
        """Test CLI evaluate command."""
        import sys

        from trading_rl_agent.training.cli import main

        # First train a model
        trainer = CNNLSTMTrainingManager(str(config_file))
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)
        train_loader, val_loader, _ = trainer.create_data_loaders(sequences, targets)
        trainer.train(train_loader, val_loader)

        model_path = temp_output_dir / "test_model.pth"
        trainer.save_model(str(model_path))

        # Now evaluate
        sys.argv = [
            "cli.py",
            "evaluate",
            "--model-path",
            str(model_path),
            "--config",
            str(config_file),
            "--output-file",
            str(temp_output_dir / "eval_results.json"),
            "--log-level",
            "INFO",
        ]

        main()

        # Check output
        captured = capsys.readouterr()
        assert "Evaluating CNN+LSTM model" in captured.out
        assert "Evaluation completed" in captured.out
        assert "Evaluation Results:" in captured.out

        # Check that results file was created
        assert (temp_output_dir / "eval_results.json").exists()

    def test_cli_predict_command(self, config_file, temp_output_dir, capsys):
        """Test CLI predict command."""
        import sys

        from trading_rl_agent.training.cli import main

        # First train a model
        trainer = CNNLSTMTrainingManager(str(config_file))
        sequences, targets, _ = trainer.prepare_dataset(force_rebuild=True)
        input_dim = sequences.shape[-1]
        trainer.model = trainer.create_model(input_dim)
        train_loader, val_loader, _ = trainer.create_data_loaders(sequences, targets)
        trainer.train(train_loader, val_loader)

        model_path = temp_output_dir / "test_model.pth"
        trainer.save_model(str(model_path))

        # Create test data
        test_data = sequences[:10]
        test_data_path = temp_output_dir / "test_data.npy"
        np.save(test_data_path, test_data)

        # Now predict
        sys.argv = [
            "cli.py",
            "predict",
            "--model-path",
            str(model_path),
            "--data-path",
            str(test_data_path),
            "--config",
            str(config_file),
            "--output-path",
            str(temp_output_dir / "predictions.npy"),
            "--log-level",
            "INFO",
        ]

        main()

        # Check output
        captured = capsys.readouterr()
        assert "Making predictions with CNN+LSTM model" in captured.out
        assert "Predictions completed" in captured.out
        assert "Prediction Summary:" in captured.out

        # Check that predictions file was created
        predictions_path = temp_output_dir / "predictions.npy"
        assert predictions_path.exists()

        # Check predictions
        predictions = np.load(predictions_path)
        assert predictions.shape == (10, 1)
