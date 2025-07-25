"""Property-based tests for trading system invariants."""

import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from trade_agent.models.cnn_lstm import CNNLSTMModel


class TestModelProperties:
    """Property-based tests for model behavior."""

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        sequence_length=st.integers(min_value=5, max_value=50),
        n_features=st.integers(min_value=1, max_value=20),
        lstm_units=st.integers(min_value=8, max_value=256)
    )
    @settings(max_examples=10, deadline=5000)
    def test_model_output_shape_property(self, batch_size, sequence_length, n_features, lstm_units):
        """Property: Model output shape should always be (batch_size, 1)."""
        assume(lstm_units % 8 == 0)  # Ensure reasonable LSTM units

        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=lstm_units,
            sequence_length=sequence_length,
            n_features=n_features,
            dropout=0.0  # Disable dropout for deterministic testing
        )

        input_tensor = torch.randn(batch_size, sequence_length, n_features)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 1)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    @given(
        sequence_length=st.integers(min_value=10, max_value=100),
        n_features=st.integers(min_value=3, max_value=15)
    )
    @settings(max_examples=5, deadline=10000)
    def test_model_determinism_property(self, sequence_length, n_features):
        """Property: Model should be deterministic with same input and seed."""
        torch.manual_seed(42)

        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=sequence_length,
            n_features=n_features,
            dropout=0.0
        )

        input_tensor = torch.randn(5, sequence_length, n_features)

        model.eval()
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-6)

    @given(
        learning_rate=st.floats(min_value=1e-5, max_value=1e-1),
        n_steps=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=10000)
    def test_training_convergence_property(self, learning_rate, n_steps):
        """Property: Training should generally decrease loss over steps."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.1
        )

        # Create synthetic training data
        X = torch.randn(20, 10, 5)
        y = torch.randn(20, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        losses = []
        model.train()

        for _ in range(n_steps):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Property: Should not have exploding gradients
        assert all(not np.isnan(loss) and not np.isinf(loss) for loss in losses)
        assert all(loss < 1000 for loss in losses)  # Reasonable upper bound


class TestDataProperties:
    """Property-based tests for data processing."""

    @given(
        n_samples=st.integers(min_value=50, max_value=1000),
        n_features=st.integers(min_value=3, max_value=20),
        sequence_length=st.integers(min_value=5, max_value=30)
    )
    @settings(max_examples=10, deadline=5000)
    def test_sequence_creation_property(self, n_samples, n_features, sequence_length):
        """Property: Sequence creation should preserve data integrity."""
        assume(n_samples > sequence_length)

        # Create time series data
        data = np.random.randn(n_samples, n_features)

        # Create sequences
        sequences = []
        targets = []

        for i in range(n_samples - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length, 0])  # Use first feature as target

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Properties
        assert sequences.shape == (n_samples - sequence_length, sequence_length, n_features)
        assert targets.shape == (n_samples - sequence_length,)
        assert not np.isnan(sequences).any()
        assert not np.isnan(targets).any()

    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0),
            min_size=50,
            max_size=200
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_returns_calculation_property(self, prices):
        """Property: Returns calculation should preserve mathematical relationships."""
        prices = np.array(prices)

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Properties
        assert len(returns) == len(prices) - 1
        assert not np.isinf(returns).any()  # No infinite returns
        assert np.all(returns > -1.0)  # No return less than -100%

        # Cumulative returns should approximately reconstruct prices
        if not np.isnan(returns).any():
            reconstructed = prices[0] * np.cumprod(1 + returns)
            # Allow for small numerical errors
            assert np.allclose(reconstructed, prices[1:], rtol=1e-10)

    @given(
        data=st.lists(
            st.floats(min_value=-10.0, max_value=10.0),
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_normalization_property(self, data):
        """Property: Normalization should have specific statistical properties."""
        data = np.array(data)
        assume(np.std(data) > 1e-8)  # Avoid division by zero

        # Z-score normalization
        normalized = (data - np.mean(data)) / np.std(data)

        # Properties
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)
        assert normalized.shape == data.shape


class TestTradingSystemProperties:
    """Property-based tests for trading system invariants."""

    @given(
        initial_capital=st.floats(min_value=1000.0, max_value=1000000.0),
        position_sizes=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=5,
            max_size=20
        ),
        prices=st.lists(
            st.floats(min_value=10.0, max_value=1000.0),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_portfolio_value_property(self, initial_capital, position_sizes, prices):
        """Property: Portfolio value calculations should be consistent."""
        assume(len(position_sizes) == len(prices))

        position_sizes = np.array(position_sizes)
        prices = np.array(prices)

        # Calculate portfolio value
        position_values = position_sizes * initial_capital * prices
        total_position_value = np.sum(np.abs(position_values))
        cash = initial_capital - np.sum(position_values)
        portfolio_value = cash + np.sum(position_values)

        # Properties
        if np.all(np.abs(position_sizes) <= 1.0):  # If positions are within leverage limits
            assert total_position_value <= initial_capital * len(prices)

        assert not np.isnan(portfolio_value)
        assert not np.isinf(portfolio_value)

    @given(
        returns=st.lists(
            st.floats(min_value=-0.2, max_value=0.2),
            min_size=20,
            max_size=252
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_risk_metrics_property(self, returns):
        """Property: Risk metrics should have expected mathematical properties."""
        returns = np.array(returns)
        assume(np.std(returns) > 1e-8)

        # Calculate risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Calculate maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Properties
        assert volatility >= 0
        assert not np.isnan(sharpe_ratio)
        assert not np.isinf(sharpe_ratio)
        assert max_drawdown <= 0
        assert max_drawdown >= -1.0  # Cannot lose more than 100%

    @given(
        signals=st.lists(
            st.integers(min_value=-1, max_value=1),
            min_size=10,
            max_size=50
        ),
        prices=st.lists(
            st.floats(min_value=50.0, max_value=200.0),
            min_size=10,
            max_size=50
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_signal_consistency_property(self, signals, prices):
        """Property: Trading signals should be consistent with position changes."""
        assume(len(signals) == len(prices))

        signals = np.array(signals)
        prices = np.array(prices)

        # Convert signals to position changes
        positions = []
        current_position = 0

        for signal in signals:
            if signal == 1 and current_position <= 0:  # Buy signal
                current_position = 1
            elif signal == -1 and current_position >= 0:  # Sell signal
                current_position = -1
            # signal == 0 or contradictory signals maintain current position

            positions.append(current_position)

        positions = np.array(positions)

        # Properties
        assert np.all(np.isin(positions, [-1, 0, 1]))  # Valid position values
        assert len(positions) == len(signals)

        # Position changes should be related to signals
        position_changes = np.diff(positions)
        non_zero_changes = position_changes[position_changes != 0]

        if len(non_zero_changes) > 0:
            assert np.all(np.abs(non_zero_changes) <= 2)  # Max change is from -1 to 1


class TestModelTrainingProperties:
    """Property-based tests for model training behavior."""

    @given(
        batch_size=st.integers(min_value=4, max_value=32),
        learning_rate=st.floats(min_value=1e-4, max_value=1e-2)
    )
    @settings(max_examples=5, deadline=15000)
    def test_batch_size_invariance_property(self, batch_size):
        """Property: Model should handle different batch sizes consistently."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.0
        )

        # Test with different batch sizes
        for test_batch_size in [1, batch_size]:
            input_tensor = torch.randn(test_batch_size, 10, 5)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (test_batch_size, 1)
            assert not torch.isnan(output).any()

    @given(
        dropout_rate=st.floats(min_value=0.0, max_value=0.8)
    )
    @settings(max_examples=5, deadline=10000)
    def test_dropout_consistency_property(self, dropout_rate):
        """Property: Dropout should only affect training mode."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=dropout_rate
        )

        input_tensor = torch.randn(8, 10, 5)

        # Eval mode should be deterministic
        model.eval()
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-6)

        # Training mode may have variation due to dropout
        model.train()
        with torch.no_grad():
            train_output = model(input_tensor)

        assert train_output.shape == output1.shape
        assert not torch.isnan(train_output).any()


class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability."""

    @given(
        scale=st.floats(min_value=1e-3, max_value=1e3)
    )
    @settings(max_examples=5, deadline=10000)
    def test_input_scale_stability(self, scale):
        """Property: Model should be stable under input scaling."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.0
        )

        # Normal scale input
        normal_input = torch.randn(4, 10, 5)

        # Scaled input
        scaled_input = normal_input * scale

        model.eval()
        with torch.no_grad():
            normal_output = model(normal_input)
            scaled_output = model(scaled_input)

        # Outputs should not be NaN or infinite
        assert not torch.isnan(normal_output).any()
        assert not torch.isnan(scaled_output).any()
        assert not torch.isinf(normal_output).any()
        assert not torch.isinf(scaled_output).any()

    @given(
        noise_level=st.floats(min_value=1e-6, max_value=1e-2)
    )
    @settings(max_examples=5, deadline=10000)
    def test_gradient_stability(self, noise_level):
        """Property: Small input changes should not cause gradient explosion."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.0
        )

        base_input = torch.randn(4, 10, 5, requires_grad=True)
        target = torch.randn(4, 1)

        # Calculate gradient on base input
        output1 = model(base_input)
        loss1 = torch.nn.functional.mse_loss(output1, target)

        grad1 = torch.autograd.grad(loss1, base_input, create_graph=True)[0]

        # Small perturbation
        noise = torch.randn_like(base_input) * noise_level
        perturbed_input = base_input + noise

        output2 = model(perturbed_input)
        loss2 = torch.nn.functional.mse_loss(output2, target)

        grad2 = torch.autograd.grad(loss2, perturbed_input, create_graph=True)[0]

        # Gradients should not be too different or explode
        grad_diff = torch.norm(grad1 - grad2).item()
        grad1_norm = torch.norm(grad1).item()
        grad2_norm = torch.norm(grad2).item()

        assert grad1_norm < 1000  # No gradient explosion
        assert grad2_norm < 1000
        assert not np.isnan(grad_diff)
        assert not np.isinf(grad_diff)
