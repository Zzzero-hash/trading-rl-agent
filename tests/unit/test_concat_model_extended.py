"""
Extended tests for ConcatModel to increase coverage.
"""

import pytest
import torch

from trading_rl_agent.models.concat_model import ConcatModel


class TestConcatModelExtended:
    """Extended test suite for ConcatModel."""

    def test_concat_model_initialization(self):
        """Test model initialization with different parameters."""
        model = ConcatModel(
            dim1=10,
            dim2=5,
            hidden_dim=32,
            output_dim=1,
            num_layers=3,
            dropout=0.3,
            activation="relu",
        )

        assert model.dim1 == 10
        assert model.dim2 == 5
        assert model.hidden_dim == 32
        assert model.output_dim == 1

        # Test that the network has the expected number of layers
        # 3 hidden layers * 3 components each (linear, activation, dropout) + 1 output layer
        expected_layers = 3 * 3 + 1
        assert len(model.network) == expected_layers

    def test_concat_model_different_activations(self):
        """Test model with different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]

        for activation in activations:
            model = ConcatModel(dim1=5, dim2=3, hidden_dim=16, output_dim=1, activation=activation)

            x1 = torch.randn(4, 5)
            x2 = torch.randn(4, 3)
            output = model(x1, x2)

            assert output.shape == (4, 1)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_concat_model_invalid_activation(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            ConcatModel(dim1=5, dim2=3, hidden_dim=16, output_dim=1, activation="invalid")

    def test_concat_model_feature_importance(self):
        """Test feature importance calculation."""
        model = ConcatModel(
            dim1=4,
            dim2=3,
            hidden_dim=16,
            output_dim=1,
            dropout=0.0,  # No dropout for deterministic results
        )

        x1 = torch.randn(8, 4, requires_grad=True)
        x2 = torch.randn(8, 3, requires_grad=True)

        importance1, importance2 = model.get_feature_importance(x1, x2)

        # Check shapes
        assert importance1.shape == (4,)
        assert importance2.shape == (3,)

        # Check that importance values are non-negative
        assert torch.all(importance1 >= 0)
        assert torch.all(importance2 >= 0)

        # Check that importance values are finite
        assert torch.all(torch.isfinite(importance1))
        assert torch.all(torch.isfinite(importance2))

    def test_concat_model_different_output_dims(self):
        """Test model with different output dimensions."""
        output_dims = [1, 2, 5, 10]

        for output_dim in output_dims:
            model = ConcatModel(dim1=6, dim2=4, hidden_dim=20, output_dim=output_dim)

            x1 = torch.randn(10, 6)
            x2 = torch.randn(10, 4)
            output = model(x1, x2)

            assert output.shape == (10, output_dim)

    def test_concat_model_single_layer(self):
        """Test model with single hidden layer."""
        model = ConcatModel(dim1=5, dim2=3, hidden_dim=16, output_dim=1, num_layers=1)

        x1 = torch.randn(6, 5)
        x2 = torch.randn(6, 3)
        output = model(x1, x2)

        assert output.shape == (6, 1)

    def test_concat_model_large_dimensions(self):
        """Test model with large input dimensions."""
        model = ConcatModel(dim1=100, dim2=50, hidden_dim=64, output_dim=10)

        x1 = torch.randn(32, 100)
        x2 = torch.randn(32, 50)
        output = model(x1, x2)

        assert output.shape == (32, 10)

    def test_concat_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = ConcatModel(dim1=4, dim2=3, hidden_dim=16, output_dim=1)

        x1 = torch.randn(5, 4, requires_grad=True)
        x2 = torch.randn(5, 3, requires_grad=True)

        output = model(x1, x2)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x1.grad is not None
        assert x2.grad is not None

        # Check gradient shapes
        assert x1.grad.shape == x1.shape
        assert x2.grad.shape == x2.shape

    def test_concat_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = ConcatModel(dim1=10, dim2=5, hidden_dim=32, output_dim=1, num_layers=2)

        total_params = sum(p.numel() for p in model.parameters())

        # Should have parameters (not zero)
        assert total_params > 0

        # Should be reasonable size (not too large)
        assert total_params < 10000

    def test_concat_model_deterministic_output(self):
        """Test that model produces deterministic output with same input."""
        model = ConcatModel(
            dim1=4,
            dim2=3,
            hidden_dim=16,
            output_dim=1,
            dropout=0.0,  # No dropout for deterministic results
        )

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 3)

        # Set model to eval mode
        model.eval()

        with torch.no_grad():
            output1 = model(x1, x2)
            output2 = model(x1, x2)

        # Outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_concat_model_input_validation(self):
        """Test that model handles invalid input shapes properly."""
        model = ConcatModel(dim1=4, dim2=3, hidden_dim=16, output_dim=1)

        # Correct shapes
        x1 = torch.randn(5, 4)
        x2 = torch.randn(5, 3)
        output = model(x1, x2)
        assert output.shape == (5, 1)

        # Wrong dimensions should raise error
        x1_wrong = torch.randn(5, 3)  # Wrong dim1
        x2_wrong = torch.randn(5, 4)  # Wrong dim2

        with pytest.raises(RuntimeError):
            model(x1_wrong, x2)

        with pytest.raises(RuntimeError):
            model(x1, x2_wrong)

    def test_concat_model_batch_size_consistency(self):
        """Test that model works with different batch sizes."""
        model = ConcatModel(dim1=4, dim2=3, hidden_dim=16, output_dim=1)

        batch_sizes = [1, 5, 10, 32]

        for batch_size in batch_sizes:
            x1 = torch.randn(batch_size, 4)
            x2 = torch.randn(batch_size, 3)
            output = model(x1, x2)

            assert output.shape == (batch_size, 1)
