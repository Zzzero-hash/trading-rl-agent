"""
Comprehensive tests for hybrid agent module.

This module tests the HybridAgent class and related functionality
with focus on small incremental fixes and edge cases.
"""

import numpy as np
import pytest
import torch

from trade_agent.agents.hybrid import HybridAgent
from trade_agent.models.cnn_lstm import CNNLSTMModel


class TestHybridAgent:
    """Test suite for HybridAgent class."""

    def test_initialization_default(self):
        """Test HybridAgent initialization with default parameters."""
        agent = HybridAgent()

        assert agent.state_dim == 50
        assert agent.action_dim == 3
        assert agent.hidden_dim == 128
        assert agent.learning_rate == 3e-4
        assert agent.device is not None
        assert agent.cnn_lstm_model is not None
        assert isinstance(agent.cnn_lstm_model, CNNLSTMModel)

    def test_initialization_custom_parameters(self):
        """Test HybridAgent initialization with custom parameters."""
        agent = HybridAgent(
            state_dim=100,
            action_dim=5,
            hidden_dim=256,
            learning_rate=0.001,
            device="cpu",
        )

        assert agent.state_dim == 100
        assert agent.action_dim == 5
        assert agent.hidden_dim == 256
        assert agent.learning_rate == 0.001
        assert agent.device.type == "cpu"

    def test_initialization_with_existing_model(self):
        """Test HybridAgent initialization with existing CNN+LSTM model."""
        cnn_lstm_model = CNNLSTMModel(input_dim=10, lstm_units=64, lstm_num_layers=2, output_dim=32)

        agent = HybridAgent(cnn_lstm_model=cnn_lstm_model)

        assert agent.cnn_lstm_model is cnn_lstm_model
        assert agent.cnn_lstm_model.lstm_units == 64

    def test_device_auto_selection(self):
        """Test automatic device selection."""
        agent = HybridAgent(device="auto")

        if torch.cuda.is_available():
            assert agent.device.type in ["cuda", "cpu"]
        else:
            assert agent.device.type == "cpu"

    def test_forward_pass(self):
        """Test forward pass through the hybrid agent."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Create dummy input
        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Test forward pass
        output = agent.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, agent.action_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_with_attention(self):
        """Test forward pass with attention mechanism."""
        agent = HybridAgent(
            state_dim=10,
            action_dim=3,
            cnn_lstm_model=CNNLSTMModel(
                input_dim=5,
                lstm_units=64,
                lstm_num_layers=2,
                output_dim=32,
                use_attention=True,
            ),
        )

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)
        output = agent.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, agent.action_dim)

    def test_forward_pass_without_attention(self):
        """Test forward pass without attention mechanism."""
        agent = HybridAgent(
            state_dim=10,
            action_dim=3,
            cnn_lstm_model=CNNLSTMModel(
                input_dim=5,
                lstm_units=64,
                lstm_num_layers=2,
                output_dim=32,
                use_attention=False,
            ),
        )

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)
        output = agent.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, agent.action_dim)

    def test_get_action(self):
        """Test action selection functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Test deterministic action
        action = agent.get_action(x, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (batch_size, agent.action_dim)

        # Test stochastic action
        action = agent.get_action(x, deterministic=False)
        assert isinstance(action, np.ndarray)
        assert action.shape == (batch_size, agent.action_dim)

    def test_get_action_single_sample(self):
        """Test action selection for single sample."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        sequence_length = 20
        input_features = 5

        x = torch.randn(sequence_length, input_features)

        action = agent.get_action(x, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (agent.action_dim,)

    def test_predict(self):
        """Test prediction functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        prediction = agent.predict(x)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (batch_size, agent.action_dim)

    def test_predict_with_numpy_input(self):
        """Test prediction with numpy input."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = np.random.randn(batch_size, sequence_length, input_features)

        prediction = agent.predict(x)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (batch_size, agent.action_dim)

    def test_predict_with_pandas_input(self):
        """Test prediction with pandas input."""
        import pandas as pd

        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = pd.DataFrame(np.random.randn(batch_size * sequence_length, input_features))

        prediction = agent.predict(x)
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (batch_size, agent.action_dim)

    def test_feature_extraction(self):
        """Test feature extraction from CNN+LSTM model."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        features = agent.extract_features(x)
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == batch_size
        assert features.shape[1] > 0  # Should have extracted features

    def test_ensemble_prediction(self):
        """Test ensemble prediction functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Test ensemble prediction
        ensemble_pred = agent.ensemble_predict(x, num_samples=5)
        assert isinstance(ensemble_pred, np.ndarray)
        assert ensemble_pred.shape == (batch_size, agent.action_dim)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        uncertainty = agent.estimate_uncertainty(x, num_samples=10)
        assert isinstance(uncertainty, np.ndarray)
        assert uncertainty.shape == (batch_size, agent.action_dim)
        assert (uncertainty >= 0).all()  # Uncertainty should be non-negative

    def test_model_parameters(self):
        """Test model parameter access."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Test total parameters
        total_params = agent.get_total_parameters()
        assert isinstance(total_params, int)
        assert total_params > 0

        # Test trainable parameters
        trainable_params = agent.get_trainable_parameters()
        assert isinstance(trainable_params, int)
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_model_summary(self):
        """Test model summary generation."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        summary = agent.get_model_summary()
        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "model_architecture" in summary

    def test_save_and_load(self):
        """Test model saving and loading."""
        import tempfile
        from pathlib import Path

        agent = HybridAgent(state_dim=10, action_dim=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hybrid_agent.pth"

            # Save model
            agent.save_model(model_path)
            assert model_path.exists()

            # Load model
            loaded_agent = HybridAgent.load_model(model_path)
            assert isinstance(loaded_agent, HybridAgent)
            assert loaded_agent.state_dim == agent.state_dim
            assert loaded_agent.action_dim == agent.action_dim

    def test_training_mode(self):
        """Test training mode functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Test training mode
        agent.train()
        assert agent.training is True
        assert agent.cnn_lstm_model.training is True

        # Test evaluation mode
        agent.eval()
        assert agent.training is False
        assert agent.cnn_lstm_model.training is False

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        agent = HybridAgent(state_dim=10, action_dim=3)
        agent.train()

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features, requires_grad=True)

        # Forward pass
        output = agent.forward(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_deterministic_output(self):
        """Test deterministic output for same input."""
        agent = HybridAgent(state_dim=10, action_dim=3)
        agent.eval()

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Get two outputs for same input
        output1 = agent.forward(x)
        output2 = agent.forward(x)

        # Should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_different_output_dims(self):
        """Test agent with different output dimensions."""
        for action_dim in [1, 2, 5, 10]:
            agent = HybridAgent(state_dim=10, action_dim=action_dim)

            batch_size = 4
            sequence_length = 20
            input_features = 5

            x = torch.randn(batch_size, sequence_length, input_features)
            output = agent.forward(x)

            assert output.shape == (batch_size, action_dim)

    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Test wrong number of dimensions
        x = torch.randn(4, 5)  # Missing sequence dimension
        with pytest.raises(ValueError):
            agent.forward(x)

        # Test wrong feature dimension
        x = torch.randn(4, 20, 3)  # Wrong number of features
        with pytest.raises(ValueError):
            agent.forward(x)

    def test_config_loading(self):
        """Test configuration loading functionality."""
        config = {
            "state_dim": 100,
            "action_dim": 5,
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "cnn_lstm_config": {
                "input_dim": 20,
                "lstm_units": 128,
                "lstm_num_layers": 3,
                "output_dim": 64,
                "use_attention": True,
            },
        }

        agent = HybridAgent.from_config(config)

        assert agent.state_dim == 100
        assert agent.action_dim == 5
        assert agent.hidden_dim == 256
        assert agent.learning_rate == 0.001
        assert agent.cnn_lstm_model.lstm_units == 128
        assert agent.cnn_lstm_model.use_attention is True

    def test_config_to_dict(self):
        """Test configuration export functionality."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        config = agent.get_config()
        assert isinstance(config, dict)
        assert "state_dim" in config
        assert "action_dim" in config
        assert "hidden_dim" in config
        assert "learning_rate" in config
        assert "cnn_lstm_config" in config

    def test_batch_size_consistency(self):
        """Test consistency across different batch sizes."""
        agent = HybridAgent(state_dim=10, action_dim=3)
        agent.eval()

        sequence_length = 20
        input_features = 5

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, sequence_length, input_features)
            output = agent.forward(x)
            assert output.shape[0] == batch_size

    def test_sequence_length_consistency(self):
        """Test consistency across different sequence lengths."""
        agent = HybridAgent(state_dim=10, action_dim=3)
        agent.eval()

        batch_size = 4
        input_features = 5

        for sequence_length in [10, 20, 50, 100]:
            x = torch.randn(batch_size, sequence_length, input_features)
            output = agent.forward(x)
            assert output.shape[0] == batch_size
            assert output.shape[1] == agent.action_dim


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_invalid_state_dim(self):
        """Test handling of invalid state dimension."""
        with pytest.raises(ValueError):
            HybridAgent(state_dim=-1, action_dim=3)

    def test_invalid_action_dim(self):
        """Test handling of invalid action dimension."""
        with pytest.raises(ValueError):
            HybridAgent(state_dim=10, action_dim=0)

    def test_invalid_hidden_dim(self):
        """Test handling of invalid hidden dimension."""
        with pytest.raises(ValueError):
            HybridAgent(state_dim=10, action_dim=3, hidden_dim=-1)

    def test_invalid_learning_rate(self):
        """Test handling of invalid learning rate."""
        with pytest.raises(ValueError):
            HybridAgent(state_dim=10, action_dim=3, learning_rate=-0.001)

    def test_invalid_device(self):
        """Test handling of invalid device specification."""
        with pytest.raises(ValueError):
            HybridAgent(state_dim=10, action_dim=3, device="invalid_device")

    def test_feature_mismatch(self):
        """Test handling of feature dimension mismatch."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Create input with wrong feature dimension
        x = torch.randn(4, 20, 10)  # 10 features instead of 5

        with pytest.raises(ValueError):
            agent.forward(x)

    def test_empty_input(self):
        """Test handling of empty input."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        x = torch.empty(0, 20, 5)
        with pytest.raises(ValueError):
            agent.forward(x)

    def test_nan_input(self):
        """Test handling of NaN input."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        x = torch.randn(4, 20, 5)
        x[0, 0, 0] = float("nan")

        with pytest.raises(ValueError):
            agent.forward(x)

    def test_inf_input(self):
        """Test handling of infinite input."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        x = torch.randn(4, 20, 5)
        x[0, 0, 0] = float("inf")

        with pytest.raises(ValueError):
            agent.forward(x)


class TestIntegration:
    """Integration tests for hybrid agent."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        # Create synthetic data
        batch_size = 8
        sequence_length = 30
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Test training mode
        agent.train()
        output_train = agent.forward(x)
        assert output_train.shape == (batch_size, 3)

        # Test evaluation mode
        agent.eval()
        output_eval = agent.forward(x)
        assert output_eval.shape == (batch_size, 3)

        # Test action selection
        action = agent.get_action(x, deterministic=True)
        assert action.shape == (batch_size, 3)

        # Test prediction
        prediction = agent.predict(x)
        assert prediction.shape == (batch_size, 3)

        # Test feature extraction
        features = agent.extract_features(x)
        assert features.shape[0] == batch_size

    def test_ensemble_workflow(self):
        """Test ensemble prediction workflow."""
        agent = HybridAgent(state_dim=10, action_dim=3)

        batch_size = 4
        sequence_length = 20
        input_features = 5

        x = torch.randn(batch_size, sequence_length, input_features)

        # Test ensemble prediction
        ensemble_pred = agent.ensemble_predict(x, num_samples=5)
        assert ensemble_pred.shape == (batch_size, 3)

        # Test uncertainty estimation
        uncertainty = agent.estimate_uncertainty(x, num_samples=10)
        assert uncertainty.shape == (batch_size, 3)
        assert (uncertainty >= 0).all()

    def test_model_persistence(self):
        """Test model saving and loading workflow."""
        import tempfile
        from pathlib import Path

        agent = HybridAgent(state_dim=10, action_dim=3)

        # Train the model a bit
        agent.train()
        x = torch.randn(4, 20, 5)
        output_before = agent.forward(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hybrid_agent.pth"

            # Save model
            agent.save_model(model_path)

            # Load model
            loaded_agent = HybridAgent.load_model(model_path)
            loaded_agent.eval()

            # Test that outputs are similar
            output_after = loaded_agent.forward(x)
            assert torch.allclose(output_before, output_after, atol=1e-6)

    def test_config_workflow(self):
        """Test configuration workflow."""
        # Create agent with custom config
        config = {
            "state_dim": 50,
            "action_dim": 4,
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "cnn_lstm_config": {
                "input_dim": 10,
                "lstm_units": 64,
                "lstm_num_layers": 2,
                "output_dim": 32,
                "use_attention": True,
            },
        }

        agent = HybridAgent.from_config(config)

        # Export config
        exported_config = agent.get_config()

        # Should match original config
        assert exported_config["state_dim"] == config["state_dim"]
        assert exported_config["action_dim"] == config["action_dim"]
        assert exported_config["hidden_dim"] == config["hidden_dim"]
        assert exported_config["learning_rate"] == config["learning_rate"]
