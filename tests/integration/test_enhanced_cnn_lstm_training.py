"""
Integration tests for Enhanced CNN+LSTM Training Pipeline

This module tests the complete training workflow including:
- Dataset building/loading
- Model initialization
- Training with monitoring
- Hyperparameter optimization
- Model saving/loading
- Metrics calculation
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add src to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from train_cnn_lstm_enhanced import (
    EnhancedCNNLSTMTrainer,
    HyperparameterOptimizer,
    create_enhanced_model_config,
    create_enhanced_training_config,
)
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel


class TestEnhancedCNNLSTMTraining:
    """Test suite for enhanced CNN+LSTM training pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 1000
        seq_length = 60
        n_features = 50
        
        # Generate synthetic sequences
        sequences = np.random.randn(n_samples, seq_length, n_features)
        
        # Generate synthetic targets (simple linear combination with noise)
        targets = np.sum(sequences[:, -10:, :5], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
        
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
            "batch_size": 16,
            "epochs": 5,  # Small number for fast testing
            "weight_decay": 1e-5,
            "val_split": 0.2,
            "early_stopping_patience": 3,
            "lr_patience": 2,
            "max_grad_norm": 1.0,
        }
    
    def test_trainer_initialization(self, model_config, training_config):
        """Test trainer initialization with different configurations."""
        
        # Test with MLflow and TensorBoard enabled
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,  # Disable for testing
            enable_tensorboard=False,  # Disable for testing
        )
        
        assert trainer.model_config == model_config
        assert trainer.training_config == training_config
        assert trainer.model is None  # Model not initialized yet
        assert len(trainer.history["train_loss"]) == 0
    
    def test_model_creation(self, sample_data, model_config):
        """Test CNN+LSTM model creation and forward pass."""
        sequences, _ = sample_data
        input_dim = sequences.shape[-1]
        
        model = CNNLSTMModel(input_dim=input_dim, config=model_config)
        
        # Test forward pass
        batch_size = 4
        seq_length = sequences.shape[1]
        x = torch.randn(batch_size, seq_length, input_dim)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_training_workflow(self, sample_data, model_config, training_config):
        """Test complete training workflow."""
        sequences, targets = sample_data
        
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )
        
        # Train the model
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
        
        # Verify metrics are reasonable
        assert result["best_val_loss"] > 0
        assert result["total_epochs"] <= training_config["epochs"]
        
        final_metrics = result["final_metrics"]
        assert "mae" in final_metrics
        assert "rmse" in final_metrics
        assert "r2" in final_metrics
        assert final_metrics["mae"] > 0
        assert final_metrics["rmse"] > 0
    
    def test_hyperparameter_optimization(self, sample_data):
        """Test hyperparameter optimization workflow."""
        sequences, targets = sample_data
        
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=3)  # Small number for testing
        
        result = optimizer.optimize()
        
        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result
        
        best_params = result["best_params"]
        assert "model_config" in best_params
        assert "training_config" in best_params
        
        # Verify optimization found some parameters
        assert result["best_score"] < float("inf")
    
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
            checkpoint = torch.load(save_path)
            assert "epoch" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "val_loss" in checkpoint
            assert "model_config" in checkpoint
            assert "training_config" in checkpoint
            assert "history" in checkpoint
    
    def test_metrics_calculation(self, sample_data, model_config):
        """Test comprehensive metrics calculation."""
        sequences, targets = sample_data
        
        # Create a simple model for testing
        input_dim = sequences.shape[-1]
        model = CNNLSTMModel(input_dim=input_dim, config=model_config)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(sequences[:100])  # Use subset for testing
            predictions = model(X_tensor).cpu().numpy().flatten()
        
        targets_subset = targets[:100]
        
        # Calculate metrics manually
        mae = mean_absolute_error(targets_subset, predictions)
        mse = mean_squared_error(targets_subset, predictions)
        rmse = np.sqrt(mse)
        
        # Verify metrics are reasonable
        assert mae > 0
        assert mse > 0
        assert rmse > 0
        assert not np.isnan(mae)
        assert not np.isnan(mse)
        assert not np.isnan(rmse)
    
    def test_training_visualization(self, sample_data, model_config, training_config):
        """Test training history visualization."""
        sequences, targets = sample_data
        
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )
        
        # Train the model
        trainer.train_from_dataset(sequences=sequences, targets=targets)
        
        # Test plotting
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "training_history.png"
            
            # This should not raise an exception
            trainer.plot_training_history(save_path=str(plot_path))
            
            # Verify plot was created
            assert plot_path.exists()
    
    def test_error_handling(self, sample_data, model_config, training_config):
        """Test error handling in training pipeline."""
        sequences, targets = sample_data
        
        # Test with invalid model config
        invalid_config = model_config.copy()
        invalid_config["cnn_filters"] = [16]  # Mismatch with kernel sizes
        
        with pytest.raises(ValueError):
            trainer = EnhancedCNNLSTMTrainer(
                model_config=invalid_config,
                training_config=training_config,
                enable_mlflow=False,
                enable_tensorboard=False,
            )
            trainer.train_from_dataset(sequences=sequences, targets=targets)
    
    def test_device_handling(self, sample_data, model_config, training_config):
        """Test training on different devices."""
        sequences, targets = sample_data
        
        # Test CPU training
        trainer_cpu = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            device="cpu",
            enable_mlflow=False,
            enable_tensorboard=False,
        )
        
        result_cpu = trainer_cpu.train_from_dataset(sequences=sequences, targets=targets)
        assert result_cpu["best_val_loss"] > 0
        
        # Test GPU training if available
        if torch.cuda.is_available():
            trainer_gpu = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                device="cuda",
                enable_mlflow=False,
                enable_tensorboard=False,
            )
            
            result_gpu = trainer_gpu.train_from_dataset(sequences=sequences, targets=targets)
            assert result_gpu["best_val_loss"] > 0
    
    def test_config_creation(self):
        """Test configuration creation functions."""
        model_config = create_enhanced_model_config()
        training_config = create_enhanced_training_config()
        
        # Verify model config structure
        assert "cnn_filters" in model_config
        assert "cnn_kernel_sizes" in model_config
        assert "lstm_units" in model_config
        assert "lstm_layers" in model_config
        assert "dropout_rate" in model_config
        assert "output_size" in model_config
        
        # Verify training config structure
        assert "learning_rate" in training_config
        assert "batch_size" in training_config
        assert "epochs" in training_config
        assert "weight_decay" in training_config
        assert "val_split" in training_config
        assert "early_stopping_patience" in training_config
    
    def test_memory_efficiency(self, sample_data, model_config, training_config):
        """Test memory efficiency of training."""
        sequences, targets = sample_data
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )
        
        trainer.train_from_dataset(sequences=sequences, targets=targets)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 1GB for this small test)
        assert memory_increase < 1024, f"Memory increase too large: {memory_increase:.1f} MB"


class TestHyperparameterOptimization:
    """Test suite for hyperparameter optimization."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for optimization testing."""
        np.random.seed(42)
        n_samples = 500  # Smaller for faster testing
        seq_length = 30
        n_features = 20
        
        sequences = np.random.randn(n_samples, seq_length, n_features)
        targets = np.sum(sequences[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
        
        return sequences, targets
    
    def test_optimizer_initialization(self, sample_data):
        """Test hyperparameter optimizer initialization."""
        sequences, targets = sample_data
        
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=5)
        
        assert optimizer.sequences.shape == sequences.shape
        assert optimizer.targets.shape == targets.shape
        assert optimizer.n_trials == 5
        assert optimizer.best_params is None
        assert optimizer.best_score == float("inf")
    
    def test_optimization_objective(self, sample_data):
        """Test optimization objective function."""
        sequences, targets = sample_data
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=1)
        
        # Mock optuna trial
        class MockTrial:
            def suggest_categorical(self, name, choices):
                return choices[0]
            
            def suggest_int(self, name, low, high):
                return low
            
            def suggest_float(self, name, low, high, log=False):
                return low
        
        trial = MockTrial()
        result = optimizer.objective(trial)
        
        # Should return a float (validation loss)
        assert isinstance(result, float)
        assert result > 0 or result == float("inf")
    
    def test_optimization_workflow(self, sample_data):
        """Test complete optimization workflow."""
        sequences, targets = sample_data
        
        optimizer = HyperparameterOptimizer(sequences, targets, n_trials=2)
        result = optimizer.optimize()
        
        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result
        
        # Verify optimization completed
        assert result["best_score"] < float("inf")
        assert result["best_params"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])