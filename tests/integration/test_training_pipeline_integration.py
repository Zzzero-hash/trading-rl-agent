"""Integration tests for the complete training pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from trade_agent.models.cnn_lstm import CNNLSTMModel


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipelineIntegration:
    """Test complete training pipeline integration."""

    @pytest.fixture
    def training_dataset(self):
        """Create a realistic training dataset."""
        np.random.seed(42)
        n_samples = 1000

        # Generate realistic time series data
        dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        symbols = ["AAPL", "MSFT", "GOOGL"]

        data = []
        for symbol in symbols:
            base_price = np.random.uniform(50, 200)
            prices = [base_price]

            for i in range(1, n_samples):
                # Random walk with mean reversion
                change = np.random.normal(0, 0.02) * prices[-1]
                new_price = max(1, prices[-1] + change)
                prices.append(new_price)

            for i, (date, price) in enumerate(zip(dates, prices, strict=False)):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_px = price + np.random.normal(0, 0.005) * price
                volume = np.random.randint(100000, 1000000)

                # Technical indicators
                sma_5 = np.mean(prices[max(0, i-4):i+1])
                sma_20 = np.mean(prices[max(0, i-19):i+1])

                data.append({
                    "date": date,
                    "symbol": symbol,
                    "open": open_px,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                    "sma_5": sma_5,
                    "sma_20": sma_20,
                    "rsi": np.random.uniform(20, 80),
                    "macd": np.random.normal(0, 1)
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_end_to_end_training_pipeline(self, training_dataset, temp_model_dir):
        """Test complete end-to-end training pipeline."""
        # 1. Data preparation
        features = ["open", "high", "low", "close", "volume", "sma_5", "sma_20", "rsi", "macd"]
        target = "close"

        # Create sequences
        sequence_length = 30
        X, y = self._create_sequences(training_dataset, features, target, sequence_length)

        # 2. Model creation
        model = CNNLSTMModel(
            cnn_layers=[32, 64],
            lstm_units=128,
            sequence_length=sequence_length,
            n_features=len(features),
            dropout=0.2
        )

        # 3. Training setup
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        # 4. Training loop (minimal for integration test)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        model.train()
        for epoch in range(5):  # Just a few epochs for testing
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # 5. Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)

        # 6. Model saving
        model_path = temp_model_dir / "trained_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "cnn_layers": [32, 64],
                "lstm_units": 128,
                "sequence_length": sequence_length,
                "n_features": len(features)
            },
            "training_loss": loss.item(),
            "test_loss": test_loss.item()
        }, model_path)

        # Assertions
        assert model_path.exists()
        assert test_loss.item() > 0
        assert outputs.shape == y_train_tensor.shape
        assert test_outputs.shape == y_test_tensor.shape

    def test_model_persistence_and_loading(self, training_dataset, temp_model_dir):  # noqa: ARG002
        """Test model saving and loading functionality."""
        # Create and train a simple model
        sequence_length = 10
        n_features = 5

        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=sequence_length,
            n_features=n_features,
            dropout=0.1
        )

        # Save model
        model_path = temp_model_dir / "test_model.pth"
        model_config = {
            "cnn_layers": [16, 32],
            "lstm_units": 64,
            "sequence_length": sequence_length,
            "n_features": n_features
        }

        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config
        }, model_path)

        # Load model
        checkpoint = torch.load(model_path, map_location="cpu")
        loaded_model = CNNLSTMModel(**checkpoint["model_config"])
        loaded_model.load_state_dict(checkpoint["model_state_dict"])

        # Test that models produce same output
        test_input = torch.randn(5, sequence_length, n_features)

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    @patch("trade_agent.training.train_cnn_lstm_enhanced.ray")
    def test_distributed_training_simulation(self, mock_ray, training_dataset):  # noqa: ARG002
        """Test distributed training simulation (mocked)."""
        try:
            from trade_agent.training.train_cnn_lstm_enhanced import init_ray_cluster

            # Mock Ray initialization
            mock_ray.is_initialized.return_value = False
            mock_ray.init.return_value = None

            # Test cluster initialization
            result = init_ray_cluster(show_info=False)
            assert isinstance(result, bool)

        except ImportError:
            pytest.skip("Enhanced training module not available")

    def test_hyperparameter_optimization_integration(self, training_dataset):  # noqa: ARG002
        """Test hyperparameter optimization integration."""
        try:
            import optuna

            # Create a simple objective function
            def objective(trial):
                lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
                trial.suggest_categorical("batch_size", [16, 32, 64])
                dropout = trial.suggest_float("dropout", 0.1, 0.5)

                # Simple model training simulation
                model = CNNLSTMModel(
                    cnn_layers=[16, 32],
                    lstm_units=64,
                    sequence_length=10,
                    n_features=5,
                    dropout=dropout
                )

                # Simulate training with random performance
                return np.random.random()

            # Create study and optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=3, timeout=10)

            assert len(study.trials) == 3
            assert study.best_params is not None

        except ImportError:
            pytest.skip("Optuna not available")

    def test_training_with_validation_split(self, training_dataset):
        """Test training with proper validation split."""
        features = ["open", "high", "low", "close", "volume"]
        target = "close"
        sequence_length = 20

        X, y = self._create_sequences(training_dataset, features, target, sequence_length)

        # Split data
        n_samples = len(X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        # Verify splits
        assert len(X_train) + len(X_val) + len(X_test) == n_samples
        assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0

        # Test model with all splits
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=sequence_length,
            n_features=len(features),
            dropout=0.2
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)

        # Test forward pass on all splits
        model.eval()
        with torch.no_grad():
            train_out = model(X_train_tensor)
            val_out = model(X_val_tensor)
            test_out = model(X_test_tensor)

        assert train_out.shape[0] == len(X_train)
        assert val_out.shape[0] == len(X_val)
        assert test_out.shape[0] == len(X_test)

    def test_multi_symbol_training(self, training_dataset):
        """Test training with multiple symbols."""
        symbols = training_dataset["symbol"].unique()
        assert len(symbols) >= 2

        features = ["open", "high", "low", "close", "volume"]
        results = {}

        for symbol in symbols[:2]:  # Test with first 2 symbols
            symbol_data = training_dataset[training_dataset["symbol"] == symbol]

            if len(symbol_data) < 50:  # Skip if insufficient data
                continue

            X, y = self._create_sequences(symbol_data, features, "close", 20)

            if len(X) < 10:  # Skip if insufficient sequences
                continue

            model = CNNLSTMModel(
                cnn_layers=[16, 32],
                lstm_units=64,
                sequence_length=20,
                n_features=len(features),
                dropout=0.2
            )

            # Quick training
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            results[symbol] = loss.item()

        # Should have trained on at least one symbol
        assert len(results) >= 1

    def test_training_early_stopping_simulation(self, training_dataset):
        """Test early stopping simulation."""
        features = ["open", "high", "low", "close", "volume"]
        sequence_length = 15

        X, y = self._create_sequences(training_dataset, features, "close", sequence_length)

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=sequence_length,
            n_features=len(features),
            dropout=0.2
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Simulate early stopping
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        for epoch in range(10):
            # Training
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            # Early stopping logic
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        assert best_val_loss < float("inf")

    def _create_sequences(self, data, features, target, sequence_length):
        """Helper method to create sequences from time series data."""
        # Group by symbol and create sequences
        X, y = [], []

        for symbol in data["symbol"].unique():
            symbol_data = data[data["symbol"] == symbol].sort_values("date")

            if len(symbol_data) < sequence_length + 1:
                continue

            feature_data = symbol_data[features].values
            target_data = symbol_data[target].values

            for i in range(len(feature_data) - sequence_length):
                X.append(feature_data[i:i + sequence_length])
                y.append([target_data[i + sequence_length]])

        return np.array(X), np.array(y)


@pytest.mark.integration
@pytest.mark.model
class TestModelIntegration:
    """Test model integration scenarios."""

    def test_model_architecture_compatibility(self):
        """Test different model architectures are compatible."""
        configs = [
            {
                "cnn_layers": [16, 32],
                "lstm_units": 64,
                "sequence_length": 10,
                "n_features": 5
            },
            {
                "cnn_layers": [32, 64, 128],
                "lstm_units": 256,
                "sequence_length": 30,
                "n_features": 10
            },
            {
                "cnn_layers": [8],
                "lstm_units": 32,
                "sequence_length": 5,
                "n_features": 3
            }
        ]

        for config in configs:
            model = CNNLSTMModel(**config, dropout=0.1)

            # Test forward pass
            batch_size = 4
            input_tensor = torch.randn(
                batch_size,
                config["sequence_length"],
                config["n_features"]
            )

            output = model(input_tensor)
            assert output.shape == (batch_size, 1)

    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.1
        )

        # Create dummy data
        x = torch.randn(8, 10, 5, requires_grad=True)
        y = torch.randn(8, 1)

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        # Backward pass
        loss.backward()

        # Check that gradients exist and are not zero
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "Model parameters should have gradients"

    def test_model_state_dict_consistency(self):
        """Test model state dict consistency."""
        model1 = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.1
        )

        model2 = CNNLSTMModel(
            cnn_layers=[16, 32],
            lstm_units=64,
            sequence_length=10,
            n_features=5,
            dropout=0.1
        )

        # Copy state dict
        model2.load_state_dict(model1.state_dict())

        # Test that outputs are identical
        test_input = torch.randn(5, 10, 5)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)

        assert torch.allclose(output1, output2, atol=1e-6)
