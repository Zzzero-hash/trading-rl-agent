"""
Extended tests for CNNLSTMModel to increase coverage.
"""

import pytest
import torch

from trade_agent.models.cnn_lstm import (
    CNNLSTMConfig,
    CNNLSTMModel,
    _load_config,
    create_model,
)


class TestCNNLSTMModelExtended:
    """Extended test suite for CNNLSTMModel."""

    def test_cnn_lstm_model_initialization(self):
        """Test model initialization with different configurations."""
        model = CNNLSTMModel(
            input_dim=10,
            output_size=3,
            use_attention=True,
        )

        assert model.input_dim == 10
        assert model.use_attention is True
        assert hasattr(model, "attention")

    def test_cnn_lstm_forward_pass(self):
        """Test forward pass with different input sizes."""
        model = CNNLSTMModel(input_dim=5, output_size=1)

        batch_sizes = [1, 4, 8]
        seq_lengths = [10, 20, 50]

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                x = torch.randn(batch_size, seq_len, 5)
                output = model(x)

                assert output.shape == (batch_size, 1)
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()

    def test_cnn_lstm_with_attention(self):
        """Test CNN-LSTM model with attention mechanism."""
        model = CNNLSTMModel(input_dim=8, output_size=1, use_attention=True)

        x = torch.randn(4, 15, 8)
        output = model(x)

        assert output.shape == (4, 1)
        assert model.use_attention is True
        assert hasattr(model, "attention")

    def test_cnn_lstm_without_attention(self):
        """Test CNN-LSTM model without attention mechanism."""
        model = CNNLSTMModel(input_dim=6, output_size=1, use_attention=False)

        x = torch.randn(3, 12, 6)
        output = model(x)

        assert output.shape == (3, 1)
        assert model.use_attention is False
        assert not hasattr(model, "attention")

    def test_cnn_lstm_parameter_count(self):
        """Test parameter counting."""
        model = CNNLSTMModel(input_dim=5, output_size=1)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_cnn_lstm_factory_function(self):
        """Test the factory function for creating models."""
        config = CNNLSTMConfig(
            input_dim=6,
            output_size=1,
            cnn_filters=[16, 32],
            cnn_kernel_sizes=[3, 3],
            lstm_units=64,
            use_attention=False,
        )

        model = create_model(config)

        assert isinstance(model, CNNLSTMModel)
        assert model.input_dim == 6

    def test_cnn_lstm_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = CNNLSTMModel(input_dim=3, output_size=1)

        x = torch.randn(2, 8, 3, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_cnn_lstm_deterministic_output(self):
        """Test that model produces deterministic output."""
        model = CNNLSTMModel(
            input_dim=4,
            output_size=1,
            use_attention=False,
        )

        x = torch.randn(3, 10, 4)
        model.eval()

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_cnn_lstm_different_output_dims(self):
        """Test model with different output dimensions."""
        output_dims = [1, 2, 5]

        for output_dim in output_dims:
            model = CNNLSTMModel(input_dim=5, output_size=output_dim)

            x = torch.randn(4, 12, 5)
            output = model(x)

            assert output.shape == (4, output_dim)

    def test_cnn_lstm_invalid_input_dimensions(self):
        """Test that model raises error for invalid input dimensions."""
        model = CNNLSTMModel(input_dim=5, output_size=1)

        # Test with wrong number of features
        x = torch.randn(2, 10, 3)  # 3 features instead of 5
        with pytest.raises(ValueError, match="Expected input features 5, got 3"):
            model(x)

    def test_cnn_lstm_config_loading(self):
        """Test configuration loading from dictionary."""
        config_dict = {
            "cnn_filters": [64, 128],
            "cnn_kernel_sizes": [3, 5],
            "lstm_units": 256,
            "dropout": 0.2,
        }

        model = CNNLSTMModel(input_dim=10, output_size=1, config=config_dict)

        # Verify the model was created with the custom config
        assert model.input_dim == 10
        # Test that the model can process input with the expected output size
        x = torch.randn(2, 10, 10)
        output = model(x)
        assert output.shape == (2, 1)


class TestCNNLSTMConfig:
    """Test suite for CNNLSTMConfig."""

    def test_config_initialization(self):
        """Test config initialization with default values."""
        config = CNNLSTMConfig(input_dim=10)

        assert config.input_dim == 10
        assert config.output_size == 1
        assert config.cnn_filters == (32, 64)
        assert config.cnn_kernel_sizes == (3, 3)
        assert config.lstm_units == 128
        assert config.dropout == 0.5
        assert config.use_attention is False

    def test_config_custom_values(self):
        """Test config initialization with custom values."""
        config = CNNLSTMConfig(
            input_dim=20,
            output_size=3,
            cnn_filters=[64, 128, 256],
            cnn_kernel_sizes=[3, 5, 7],
            lstm_units=512,
            dropout=0.3,
            use_attention=True,
        )

        assert config.input_dim == 20
        assert config.output_size == 3
        assert config.cnn_filters == [64, 128, 256]
        assert config.cnn_kernel_sizes == [3, 5, 7]
        assert config.lstm_units == 512
        assert config.dropout == 0.3
        assert config.use_attention is True

    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = CNNLSTMConfig(
            input_dim=15,
            output_size=2,
            cnn_filters=[32, 64],
            lstm_units=128,
            use_attention=True,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["input_dim"] == 15
        assert config_dict["output_size"] == 2
        assert config_dict["cnn_filters"] == [32, 64]
        assert config_dict["lstm_units"] == 128
        assert config_dict["use_attention"] is True


class TestConfigLoading:
    """Test suite for configuration loading utilities."""

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {"cnn_filters": [64, 128], "lstm_units": 256}
        result = _load_config(config_dict)

        assert result == config_dict

    def test_load_config_from_string(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
        cnn_filters: [32, 64]
        lstm_units: 128
        dropout: 0.2
        """
        config_file.write_text(config_content)

        result = _load_config(str(config_file))

        assert result["cnn_filters"] == [32, 64]
        assert result["lstm_units"] == 128
        assert result["dropout"] == 0.2

    def test_load_config_invalid_path(self):
        """Test loading config from invalid path."""
        with pytest.raises(FileNotFoundError):
            _load_config("nonexistent_file.yaml")

    def test_load_config_invalid_type(self):
        """Test loading config with invalid type."""
        with pytest.raises(ValueError, match="Config must be a dictionary after loading."):
            _load_config(123)  # Invalid type
