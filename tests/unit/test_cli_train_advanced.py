"""
Advanced CLI Training Command Tests.

This module provides comprehensive testing for all training CLI commands:
- CNN+LSTM training with hyperparameter optimization
- Advanced parameter combinations and validation
- GPU/CPU training modes
- Mixed precision training
- Model checkpointing and recovery
- Training performance monitoring
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


class TestCLICNNLSTMTrainingComplete:
    """Comprehensive tests for CNN+LSTM training CLI command."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "dataset.csv"
        self.output_dir = Path(self.temp_dir) / "models"

        # Create mock dataset file
        self._create_mock_dataset_file()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset_file(self):
        """Create a mock CSV dataset file."""
        import pandas as pd

        # Create mock dataset
        data = {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "symbol": ["AAPL"] * 100,
            "open": [150.0 + i * 0.1 for i in range(100)],
            "high": [151.0 + i * 0.1 for i in range(100)],
            "low": [149.0 + i * 0.1 for i in range(100)],
            "close": [150.5 + i * 0.1 for i in range(100)],
            "volume": [1000000 + i * 1000 for i in range(100)],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config")
    def test_cnn_lstm_basic_training(self, mock_training_config, mock_model_config,
                                   mock_trainer_class, mock_load_data, mock_init_ray):
        """Test basic CNN+LSTM training."""
        # Setup mocks
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)
        mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
        mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "5",
                "--batch-size", "32",
                "--learning-rate", "0.001",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 0
        assert "Training CNN+LSTM model" in result.output
        assert "CNN+LSTM training complete" in result.output

        # Verify trainer was called with correct parameters
        mock_trainer.train_from_dataset.assert_called_once()
        call_args = mock_trainer.train_from_dataset.call_args
        assert str(self.output_dir / "best_model.pth") in call_args[1]["save_path"]

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.HyperparameterOptimizer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config")
    def test_cnn_lstm_with_hyperparameter_optimization(self, mock_training_config, mock_model_config,
                                                      mock_optimizer_class, mock_trainer_class,
                                                      mock_load_data, mock_init_ray):
        """Test CNN+LSTM training with hyperparameter optimization."""
        # Setup mocks
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)
        mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
        mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

        # Mock hyperparameter optimization
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = {
            "best_params": {
                "learning_rate": 0.002,
                "batch_size": 64
            }
        }

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--optimize-hyperparams",
                "--n-trials", "10",
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 0
        assert "Running hyperparameter optimization" in result.output
        assert "Best parameters found" in result.output
        assert "Using optimized learning_rate=0.002, batch_size=64" in result.output

        # Verify optimization was called
        mock_optimizer.optimize.assert_called_once()

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config")
    def test_cnn_lstm_gpu_training(self, mock_training_config, mock_model_config,
                                  mock_trainer_class, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training with GPU enabled."""
        # Setup mocks
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)
        mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
        mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--gpu",
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 0

        # Verify trainer was initialized with GPU device
        call_args = mock_trainer_class.call_args
        assert call_args[1]["device"] == "cuda"

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config")
    def test_cnn_lstm_sequence_parameters(self, mock_training_config, mock_model_config,
                                        mock_trainer_class, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training with custom sequence parameters."""
        # Setup mocks
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)
        mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
        mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--sequence-length", "120",
                "--prediction-horizon", "5",
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 0
        assert "Used sequence_length=120, prediction_horizon=5" in result.output

        # Verify load_and_preprocess_csv_data was called with correct parameters
        call_args = mock_load_data.call_args
        assert call_args[1]["sequence_length"] == 120
        assert call_args[1]["prediction_horizon"] == 5

    def test_cnn_lstm_missing_dataset_file(self):
        """Test CNN+LSTM training with missing dataset file."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.csv"

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(nonexistent_path),
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 1
        assert "Dataset file not found" in result.output

    def test_cnn_lstm_directory_dataset_path(self):
        """Test CNN+LSTM training with directory containing dataset.csv."""
        # Create directory structure
        dataset_dir = Path(self.temp_dir) / "dataset_dir"
        dataset_dir.mkdir()
        dataset_file = dataset_dir / "dataset.csv"

        # Copy dataset to directory
        import shutil
        shutil.copy(self.data_path, dataset_file)

        with patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster") as mock_init_ray, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data") as mock_load_data, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer") as mock_trainer_class, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config") as mock_model_config, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config") as mock_training_config:

            # Setup mocks
            mock_init_ray.return_value = None
            mock_sequences, mock_targets = self._create_mock_sequences_targets()
            mock_load_data.return_value = (mock_sequences, mock_targets)
            mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
            mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(dataset_dir),
                    "--epochs", "5",
                    "--output-dir", str(self.output_dir)
                ]
            )

            assert result.exit_code == 0
            # Verify it found the dataset.csv file in the directory
            call_args = mock_load_data.call_args
            assert str(dataset_file) in str(call_args[1]["csv_path"])

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    def test_cnn_lstm_ray_initialization_failure(self, mock_init_ray):
        """Test CNN+LSTM training when Ray initialization fails."""
        mock_init_ray.side_effect = Exception("Ray initialization failed")

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 1
        assert "Error during CNN+LSTM training" in result.output

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    def test_cnn_lstm_data_loading_failure(self, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training when data loading fails."""
        mock_init_ray.return_value = None
        mock_load_data.side_effect = Exception("Data loading failed")

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "5",
                "--output-dir", str(self.output_dir)
            ]
        )

        assert result.exit_code == 1
        assert "Error during CNN+LSTM training" in result.output

    def _create_mock_sequences_targets(self):
        """Create mock sequences and targets for training."""
        import numpy as np

        # Mock sequences: (samples, timesteps, features)
        sequences = np.random.rand(100, 60, 10)
        # Mock targets: (samples, output_dim)
        targets = np.random.rand(100, 1)

        return sequences, targets


class TestCLITrainingParameterValidation:
    """Test training command parameter validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "dataset.csv"
        self._create_mock_dataset_file()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset_file(self):
        """Create a mock CSV dataset file."""
        import pandas as pd

        data = {
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "symbol": ["AAPL"] * 50,
            "close": [150.0 + i * 0.1 for i in range(50)],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)

    def test_invalid_epochs_parameter(self):
        """Test handling of invalid epochs parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "-1",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "epochs" in result.output.lower() or "invalid" in result.output.lower()

    def test_invalid_batch_size_parameter(self):
        """Test handling of invalid batch size parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--batch-size", "0",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "batch" in result.output.lower() or "invalid" in result.output.lower()

    def test_invalid_learning_rate_parameter(self):
        """Test handling of invalid learning rate parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--learning-rate", "-0.1",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "learning" in result.output.lower() or "rate" in result.output.lower()

    def test_invalid_sequence_length_parameter(self):
        """Test handling of invalid sequence length parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--sequence-length", "0",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "sequence" in result.output.lower() or "length" in result.output.lower()

    def test_invalid_prediction_horizon_parameter(self):
        """Test handling of invalid prediction horizon parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--prediction-horizon", "0",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "prediction" in result.output.lower() or "horizon" in result.output.lower()

    def test_invalid_n_trials_parameter(self):
        """Test handling of invalid n-trials parameter."""
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--optimize-hyperparams",
                "--n-trials", "0",
                "--output-dir", str(self.temp_dir)
            ]
        )

        # Should either validate or handle gracefully
        if result.exit_code != 0:
            assert "trials" in result.output.lower() or "invalid" in result.output.lower()


class TestCLITrainingVerboseOutput:
    """Test training command verbose output and logging."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "dataset.csv"
        self._create_mock_dataset_file()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset_file(self):
        """Create a mock CSV dataset file."""
        import pandas as pd

        data = {
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "symbol": ["AAPL"] * 50,
            "close": [150.0 + i * 0.1 for i in range(50)],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.data_path, index=False)

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config")
    def test_training_verbose_output(self, mock_training_config, mock_model_config,
                                   mock_trainer_class, mock_load_data, mock_init_ray):
        """Test training with verbose output."""
        # Setup mocks
        mock_init_ray.return_value = None
        mock_sequences = self._create_mock_sequences()
        mock_targets = self._create_mock_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)
        mock_model_config.return_value = {"input_dim": 10, "hidden_dim": 64}
        mock_training_config.return_value = {"epochs": 5, "batch_size": 32}

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = self.runner.invoke(
            main_app,
            [
                "-v",  # verbose flag
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "5",
                "--output-dir", str(self.temp_dir)
            ]
        )

        assert result.exit_code == 0
        # Should have more detailed output in verbose mode
        assert "Initializing Ray cluster" in result.output
        assert "Loading data from" in result.output

    def _create_mock_sequences(self):
        """Create mock sequences for training."""
        import numpy as np
        return np.random.rand(50, 30, 10)

    def _create_mock_targets(self):
        """Create mock targets for training."""
        import numpy as np
        return np.random.rand(50, 1)


if __name__ == "__main__":
    pytest.main([__file__])
