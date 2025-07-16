"""
Extended tests for CNNLSTMModel to increase coverage.
"""

import pytest
import torch

from trading_rl_agent.models.cnn_lstm import (
    CNNLSTMModel,
    CNNLSTMModelWithAttention,
    CNNLSTMModelWithResidual,
    MultiHeadAttention,
    ResidualBlock,
    _load_config,
    create_cnn_lstm_model,
)


class TestCNNLSTMModelExtended:
    """Extended test suite for CNNLSTMModel."""

    def test_cnn_lstm_model_initialization(self):
        """Test model initialization with different configurations."""
        model = CNNLSTMModel(
            input_dim=10,
            cnn_filters=[32, 64],
            cnn_kernel_sizes=[3, 5],
            lstm_units=128,
            lstm_num_layers=2,
            lstm_dropout=0.3,
            cnn_dropout=0.2,
            output_dim=3,
            use_attention=True,
            use_residual=True,
            attention_heads=4,
            layer_norm=True,
            batch_norm=True,
        )

        assert model.input_dim == 10
        assert model.cnn_filters == [32, 64]
        assert model.cnn_kernel_sizes == [3, 5]
        assert model.lstm_units == 128
        assert model.lstm_num_layers == 2
        assert model.use_attention is True
        assert model.use_residual is True
        assert model.layer_norm is True
        assert model.batch_norm is True

    def test_cnn_lstm_forward_pass(self):
        """Test forward pass with different input sizes."""
        model = CNNLSTMModel(input_dim=5, cnn_filters=[16, 32], cnn_kernel_sizes=[3, 3], lstm_units=64, output_dim=1)

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
        model = CNNLSTMModelWithAttention(
            input_dim=8, cnn_filters=[16], cnn_kernel_sizes=[3], lstm_units=32, output_dim=1
        )

        x = torch.randn(4, 15, 8)
        output = model(x)

        assert output.shape == (4, 1)
        assert model.use_attention is True
        assert hasattr(model, "attention")

    def test_cnn_lstm_with_residual(self):
        """Test CNN-LSTM model with residual connections."""
        model = CNNLSTMModelWithResidual(
            input_dim=6,
            cnn_filters=[16, 16],  # Same size for residual
            cnn_kernel_sizes=[3, 3],
            lstm_units=32,
            output_dim=1,
        )

        x = torch.randn(3, 12, 6)
        output = model(x)

        assert output.shape == (3, 1)
        assert model.use_residual is True

    def test_cnn_lstm_feature_importance(self):
        """Test feature importance calculation."""
        model = CNNLSTMModel(input_dim=4, cnn_filters=[8], cnn_kernel_sizes=[3], lstm_units=16, output_dim=1)

        x = torch.randn(2, 10, 4, requires_grad=True)
        importance = model.get_feature_importance(x)

        assert importance.shape == (4,)
        assert torch.all(importance >= 0)
        assert torch.all(torch.isfinite(importance))

    def test_cnn_lstm_parameter_count(self):
        """Test parameter counting."""
        model = CNNLSTMModel(input_dim=5, cnn_filters=[16, 32], cnn_kernel_sizes=[3, 3], lstm_units=64, output_dim=1)

        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_cnn_lstm_factory_function(self):
        """Test the factory function for creating models."""
        model = create_cnn_lstm_model(
            input_dim=6, architecture="16_32_3_3", lstm_units=64, lstm_num_layers=2, output_dim=1
        )

        assert isinstance(model, CNNLSTMModel)
        assert model.input_dim == 6
        assert model.cnn_filters == [16, 32]
        assert model.cnn_kernel_sizes == [3, 3]
        assert model.lstm_units == 64
        assert model.lstm_num_layers == 2
        assert model.output_dim == 1

    def test_cnn_lstm_factory_with_keywords(self):
        """Test factory function with attention and residual keywords."""
        model = create_cnn_lstm_model(
            input_dim=4, architecture="attention_residual_8_16_3_3", lstm_units=32, output_dim=1
        )

        assert model.use_attention is True
        assert model.use_residual is True
        assert model.cnn_filters == [8, 16]
        assert model.cnn_kernel_sizes == [3, 3]

    def test_cnn_lstm_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = CNNLSTMModel(input_dim=3, cnn_filters=[8], cnn_kernel_sizes=[3], lstm_units=16, output_dim=1)

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
            cnn_filters=[8],
            cnn_kernel_sizes=[3],
            lstm_units=16,
            output_dim=1,
            lstm_dropout=0.0,
            cnn_dropout=0.0,
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
            model = CNNLSTMModel(
                input_dim=5, cnn_filters=[8], cnn_kernel_sizes=[3], lstm_units=16, output_dim=output_dim
            )

            x = torch.randn(4, 12, 5)
            output = model(x)

            assert output.shape == (4, output_dim)


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention."""

    def test_attention_initialization(self):
        """Test attention mechanism initialization."""
        attention = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        assert attention.d_model == 64
        assert attention.num_heads == 8
        assert attention.d_k == 8  # 64 // 8

    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.1)

        x = torch.randn(2, 10, 32)  # batch_size, seq_len, d_model
        output = attention(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_incompatible_heads(self):
        """Test attention with incompatible number of heads."""
        # d_model=7, num_heads=8 (7 not divisible by 8)
        attention = MultiHeadAttention(d_model=7, num_heads=8, dropout=0.1)

        # Should adjust to compatible number of heads
        assert attention.num_heads == 1  # Fallback to 1 head

        x = torch.randn(2, 5, 7)
        output = attention(x)

        assert output.shape == x.shape

    def test_attention_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        attention = MultiHeadAttention(d_model=16, num_heads=2, dropout=0.1)

        seq_lengths = [5, 10, 20]

        for seq_len in seq_lengths:
            x = torch.randn(3, seq_len, 16)
            output = attention(x)

            assert output.shape == (3, seq_len, 16)


class TestResidualBlock:
    """Test suite for ResidualBlock."""

    def test_residual_block_initialization(self):
        """Test residual block initialization."""
        block = ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, dropout=0.1)

        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "shortcut")

    def test_residual_block_forward_pass(self):
        """Test residual block forward pass."""
        block = ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, dropout=0.1)

        x = torch.randn(2, 8, 20)  # batch_size, channels, seq_len
        output = block(x)

        assert output.shape == (2, 16, 20)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_residual_block_same_channels(self):
        """Test residual block with same input and output channels."""
        block = ResidualBlock(in_channels=16, out_channels=16, kernel_size=5, dropout=0.1)

        x = torch.randn(3, 16, 15)
        output = block(x)

        assert output.shape == (3, 16, 15)

    def test_residual_block_different_kernel_sizes(self):
        """Test residual block with different kernel sizes."""
        kernel_sizes = [3, 5, 7]

        for kernel_size in kernel_sizes:
            block = ResidualBlock(in_channels=8, out_channels=16, kernel_size=kernel_size, dropout=0.1)

            x = torch.randn(2, 8, 25)
            output = block(x)

            assert output.shape == (2, 16, 25)


class TestConfigLoading:
    """Test suite for configuration loading."""

    def test_load_config_from_string(self, tmp_path):
        """Test loading config from string path."""
        config = {"cnn_filters": [32, 64], "cnn_kernel_sizes": [3, 5], "lstm_units": 128, "dropout": 0.2}

        config_path = tmp_path / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = _load_config(str(config_path))
        assert loaded_config == config

    def test_load_config_invalid_path(self):
        """Test loading config from invalid path."""
        with pytest.raises(FileNotFoundError):
            _load_config("nonexistent_file.yaml")
