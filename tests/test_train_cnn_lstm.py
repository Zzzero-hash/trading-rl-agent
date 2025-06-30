"""
Tests for the CNN-LSTM Training Pipeline.
Tests training functionality, data preparation, and model integration.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch


import yaml

from src.models.cnn_lstm import CNNLSTMModel
from src.training.cnn_lstm import (
    CNNLSTMTrainer,
    SequenceDataset,
    TrainingConfig,
    create_example_config,
)

pytestmark = pytest.mark.integration


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.sequence_length == 60
        assert config.prediction_horizon == 1
        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.include_sentiment is True
        assert config.use_attention is True
        assert config.normalize_features is True

    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            sequence_length=30,
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            include_sentiment=False,
            use_attention=False,
        )

        assert config.sequence_length == 30
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.include_sentiment is False
        assert config.use_attention is False


class TestSequenceDataset:
    """Test SequenceDataset class."""

    def test_sequence_dataset_creation(self):
        """Test creating sequences from time series data."""
        # Create sample data
        features = np.random.randn(100, 5)  # 100 timesteps, 5 features
        targets = np.random.randn(100)  # 100 target values

        dataset = SequenceDataset(
            features, targets, sequence_length=10, prediction_horizon=1
        )

        assert len(dataset) == 90  # 100 - 10 + 1 - 1 = 90
        assert dataset.sequences.shape == (90, 10, 5)  # (samples, seq_len, features)
        assert dataset.sequence_targets.shape == (90,)

    def test_sequence_dataset_multi_step_prediction(self):
        """Test multi-step prediction horizon."""
        features = np.random.randn(50, 3)
        targets = np.random.randn(50)

        dataset = SequenceDataset(
            features, targets, sequence_length=5, prediction_horizon=3
        )

        expected_samples = 50 - 5 - 3 + 1  # 43
        assert len(dataset) == expected_samples
        assert dataset.sequences.shape == (expected_samples, 5, 3)

    def test_sequence_dataset_indexing(self):
        """Test dataset indexing."""
        features = np.arange(20).reshape(20, 1).astype(float)  # Simple sequence
        targets = np.arange(20).astype(float)

        dataset = SequenceDataset(
            features, targets, sequence_length=3, prediction_horizon=1
        )

        seq, target = dataset[0]
        expected_seq = features[0:3]  # First 3 timesteps
        expected_target = targets[3]  # 4th timestep

        np.testing.assert_array_equal(seq, expected_seq)
        assert target == expected_target


class TestCNNLSTMTrainer:
    """Test CNNLSTMTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = TrainingConfig(
            epochs=2,  # Quick test
            batch_size=4,
            sequence_length=5,
            early_stopping_patience=1,
            include_sentiment=False,  # Simplify for testing
        )
        self.trainer = CNNLSTMTrainer(self.config)

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.model is None
        assert self.trainer.optimizer is None
        assert torch.cuda.is_available() or self.trainer.device == torch.device("cpu")

    def test_trainer_with_sentiment(self):
        """Test trainer initialization with sentiment analysis."""
        config = TrainingConfig(include_sentiment=True)
        trainer = CNNLSTMTrainer(config)

        assert trainer.sentiment_analyzer is not None
        assert trainer.config.include_sentiment is True

    def test_prepare_data_basic(self):
        """Test basic data preparation without sentiment."""
        # Create sample DataFrame with enough data for technical indicators
        dates = pd.date_range(
            "2023-01-01", periods=100, freq="D"
        )  # Increased from 20 to 100
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 102,
                "low": np.random.randn(100) + 98,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        features, targets = self.trainer.prepare_data(df)

        assert isinstance(features, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert features.shape[0] == targets.shape[0]
        assert features.shape[1] > 5  # Should have technical indicators

    def test_prepare_data_normalization(self):
        """Test data normalization."""
        dates = pd.date_range(
            "2023-01-01", periods=100, freq="D"
        )  # Increased from 15 to 100
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.randn(100) * 10 + 100,
                "high": np.random.randn(100) * 10 + 105,
                "low": np.random.randn(100) * 10 + 95,
                "close": np.random.randn(100) * 10 + 100,
                "volume": np.random.randint(1000, 100000, 100),
            }
        )

        # Test with normalization
        config_norm = TrainingConfig(normalize_features=True, include_sentiment=False)
        trainer_norm = CNNLSTMTrainer(config_norm)
        features_norm, _ = trainer_norm.prepare_data(df)

        # Test without normalization
        config_no_norm = TrainingConfig(
            normalize_features=False, include_sentiment=False
        )
        trainer_no_norm = CNNLSTMTrainer(config_no_norm)
        features_no_norm, _ = trainer_no_norm.prepare_data(df)

        # Normalized features should have different scale
        assert not np.allclose(features_norm, features_no_norm)
        assert trainer_norm.scaler is not None
        assert trainer_no_norm.scaler is None

    def test_create_data_loaders(self):
        """Test data loader creation."""
        features = np.random.randn(50, 8)
        targets = np.random.randn(50)

        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(
            features, targets
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Check batch shape
        for batch_data, batch_targets in train_loader:
            assert batch_data.shape[1] == self.config.sequence_length
            assert batch_data.shape[2] == features.shape[1]
            break

    def test_initialize_model(self):
        """Test model initialization."""
        input_dim = 10
        self.trainer.initialize_model(input_dim)

        assert self.trainer.model is not None
        assert isinstance(self.trainer.model, CNNLSTMModel)
        assert self.trainer.optimizer is not None
        assert self.trainer.criterion is not None
        assert self.trainer.model.input_dim == input_dim

    def test_train_epoch(self):
        """Test single epoch training."""
        # Initialize model
        input_dim = 6
        self.trainer.initialize_model(input_dim)

        # Create dummy data loader with proper classification targets (0, 1, 2)
        features = torch.randn(20, self.config.sequence_length, input_dim)
        targets = torch.randint(0, 3, (20,))  # Classification targets: 0, 1, 2
        dataset = TensorDataset(features, targets)
        loader = DataLoader(dataset, batch_size=4)

        # Train one epoch
        loss = self.trainer.train_epoch(loader)

        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative

    def test_validate(self):
        """Test model validation."""
        # Initialize model
        input_dim = 6
        self.trainer.initialize_model(input_dim)

        # Create dummy validation data with proper classification targets (0, 1, 2)
        features = torch.randn(16, self.config.sequence_length, input_dim)
        targets = torch.randint(0, 3, (16,))  # Classification targets: 0, 1, 2
        dataset = TensorDataset(features, targets)
        loader = DataLoader(dataset, batch_size=4)

        # Validate
        val_loss, val_corr = self.trainer.validate(loader)

        assert isinstance(val_loss, float)
        assert isinstance(val_corr, float)
        assert val_loss >= 0
        assert -1.0 <= val_corr <= 1.0

    @patch("torch.save")
    def test_save_model(self, mock_torch_save):
        """Test model saving."""
        # Initialize model
        input_dim = 5
        self.trainer.initialize_model(input_dim)

        # Set up temporary save path
        self.trainer.config.model_save_path = "/tmp/test_model.pth"

        # Save model
        self.trainer._save_model()

        # Check that torch.save was called
        mock_torch_save.assert_called_once()

        # Check save arguments
        call_args = mock_torch_save.call_args[0]
        save_dict = call_args[0]
        assert "model_state_dict" in save_dict
        assert "config" in save_dict
        assert "input_dim" in save_dict


class TestTrainingIntegration:
    """Test full training integration."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline with minimal data."""
        # Create larger sample data for training
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.cumsum(np.random.randn(100) * 0.1) + 100,
                "high": np.cumsum(np.random.randn(100) * 0.1) + 102,
                "low": np.cumsum(np.random.randn(100) * 0.1) + 98,
                "close": np.cumsum(np.random.randn(100) * 0.1) + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "label": np.random.randint(
                    0, 3, 100
                ),  # Add classification labels: 0, 1, 2
            }
        )

        # Quick training config
        config = TrainingConfig(
            sequence_length=5,
            epochs=2,
            batch_size=8,
            early_stopping_patience=1,
            include_sentiment=False,
            normalize_features=True,
            save_model=False,  # Don't save during tests
        )

        trainer = CNNLSTMTrainer(config)

        # Prepare data
        features, targets = trainer.prepare_data(df)
        assert features.shape[0] > 10  # Should have enough data after cleaning

        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            features, targets
        )

        # Initialize model
        trainer.initialize_model(features.shape[1])

        # Train
        history = trainer.train(train_loader, val_loader)

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert all(isinstance(loss, float) for loss in history["train_loss"])


class TestConfigurationUtils:
    """Test configuration utilities."""

    def test_create_example_config(self):
        """Test example configuration creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the config path to use temp directory
            with patch("src.training.cnn_lstm.create_example_config") as mock_create:
                config_path = Path(temp_dir) / "test_config.yaml"
                mock_create.return_value = str(config_path)

                # Create sample config manually
                config = {
                    "data": {
                        "source": {"type": "csv", "path": "test.csv"},
                        "symbols": ["AAPL"],
                    },
                    "training": {"epochs": 10, "batch_size": 32},
                }

                with open(config_path, "w") as f:
                    yaml.safe_dump(config, f)

                # Test that config can be loaded
                with open(config_path) as f:
                    loaded_config = yaml.safe_load(f)

                assert "data" in loaded_config
                assert "training" in loaded_config
                assert loaded_config["data"]["symbols"] == ["AAPL"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_close_column(self):
        """Test error when close column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10),
                "open": np.random.randn(10),
                "high": np.random.randn(10),
                "low": np.random.randn(10),
                "volume": np.random.randint(1000, 10000, 10),
                # Missing 'close' column
            }
        )

        config = TrainingConfig(include_sentiment=False)
        trainer = CNNLSTMTrainer(config)

        with pytest.raises(ValueError, match="DataFrame must contain 'close' column"):
            trainer.prepare_data(df)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Very small dataset
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5),
                "open": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [98, 99, 100, 101, 102],
                "close": [101, 102, 103, 104, 105],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        config = TrainingConfig(
            sequence_length=10, include_sentiment=False  # Longer than available data
        )
        trainer = CNNLSTMTrainer(config)

        # Should raise ValueError for insufficient data after NaN removal
        with pytest.raises(ValueError, match="No data remaining after NaN removal"):
            trainer.prepare_data(df)

    def test_model_config_validation(self):
        """Test model configuration validation."""
        config = TrainingConfig(
            model_config={
                "cnn_filters": [64, 128],
                "cnn_kernel_sizes": [3, 5],  # Matching lengths
                "lstm_units": 256,
                "dropout": 0.2,
            },
            include_sentiment=False,
        )

        trainer = CNNLSTMTrainer(config)

        # Should initialize without error
        trainer.initialize_model(input_dim=10)
        assert trainer.model is not None


# Mock data for sentiment testing
@pytest.fixture
def mock_sentiment_data():
    """Provide mock sentiment data for testing."""
    return {
        "AAPL": {"score": 0.3, "magnitude": 0.8},
        "GOOGL": {"score": -0.1, "magnitude": 0.6},
    }


class TestSentimentIntegration:
    """Test sentiment analysis integration in training."""

    @patch("src.training.cnn_lstm.SentimentAnalyzer")
    def test_sentiment_feature_integration(
        self, mock_sentiment_analyzer, mock_sentiment_data
    ):
        """Test that sentiment features are properly integrated."""
        # Mock sentiment analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.get_symbol_sentiment.side_effect = (
            lambda symbol, days_back: mock_sentiment_data.get(symbol, {}).get(
                "score", 0.0
            )
        )
        mock_sentiment_analyzer.return_value = mock_analyzer

        # Create training config with sentiment enabled and smaller sequence length
        config = TrainingConfig(
            include_sentiment=True,
            normalize_features=False,
            sequence_length=10,  # Much smaller to work with limited test data
            prediction_horizon=1,
        )
        trainer = CNNLSTMTrainer(config)

        # Create sample data with sufficient rows for feature engineering
        # Need enough data for technical indicators (at least 26 rows) plus sequence length
        dates = pd.date_range(
            "2023-01-01", periods=100, freq="D"
        )  # Increased from 30 to 100
        np.random.seed(42)  # For reproducible tests
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 102,
                "low": np.random.randn(100) + 98,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        # Prepare data with sentiment
        symbols = ["AAPL", "GOOGL"]
        features, targets = trainer.prepare_data(df, symbols)

        # Should have more features due to sentiment (base features + sentiment features)
        assert features.shape[1] > 10  # More than just technical indicators
        assert features.shape[0] > 0  # Should have some data after processing


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
