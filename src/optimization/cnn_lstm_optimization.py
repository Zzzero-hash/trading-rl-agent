"""Streamlined CNN-LSTM Optimization.

This module provides a simple, working hyperparameter optimization
that handles both Ray Tune and fallback scenarios correctly.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import after path setup
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Always assume Ray is available
RAY_AVAILABLE = True


def get_default_search_space() -> dict[str, Any]:
    """Get default CNN-LSTM hyperparameter search space.

    Returns
    -------
    Dict[str, Any]
        Default search space for hyperparameter optimization
    """
    return {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "lstm_units": tune.choice([32, 64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        "sequence_length": tune.choice([8, 12, 16]),
    }


def create_simple_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 10,
) -> TensorDataset:
    """Create a simple dataset for optimization."""

    # For optimization, we'll use a simplified approach
    # that doesn't require complex sequence processing
    n_samples = min(len(features), len(targets))

    # Take last sequence_length samples for each feature
    if len(features.shape) == 2:
        # Features already in correct format (n_samples, n_features)
        X = features[:n_samples]
        y = targets[:n_samples]
    else:
        # Handle other shapes
        X = features.reshape(n_samples, -1)
        y = targets[:n_samples]

    # Create sequences by reshaping
    # For now, use a simple approach where each sample is a short sequence
    n_features = X.shape[1]
    seq_len = min(sequence_length, n_features)

    # Pad or truncate features to create uniform sequences
    if n_features < seq_len:
        # Pad with zeros
        padding = np.zeros((n_samples, seq_len - n_features))
        X_seq = np.concatenate([X, padding], axis=1)
    else:
        # Take first seq_len features
        X_seq = X[:, :seq_len]

    # Reshape to (batch, seq_len, 1) for CNN-LSTM
    X_seq = X_seq.reshape(n_samples, seq_len, 1)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    return TensorDataset(X_tensor, y_tensor)


def train_single_trial(
    config: dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    max_epochs: int = 20,
) -> dict[str, Any]:
    """Train a single model configuration."""

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dataset
        sequence_length = config.get("sequence_length", 10)
        dataset = create_simple_dataset(features, targets, sequence_length)

        # Split data
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        batch_size = config.get("batch_size", 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        # The input_dim for CNN-LSTM should be 1 (since we reshape to seq_len x 1)
        input_dim = 1
        model = CNNLSTMModel(
            input_dim=input_dim,
            config={
                "lstm_units": config.get("lstm_units", 64),
                "dropout": config.get("dropout", 0.2),
            },
        ).to(device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
        )
        criterion = nn.MSELoss()

        # Initialize training metrics
        train_loss = float("inf")
        val_loss = float("inf")
        epoch = -1
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_count = 0

            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_count += 1

            # Validation
            model.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)

                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)

                    val_loss += loss.item()
                    val_count += 1

            train_loss = train_loss / max(train_count, 1)
            val_loss = val_loss / max(val_count, 1)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return {
            "val_loss": best_val_loss,
            "train_loss": train_loss,
            "epochs_trained": epoch + 1,
        }

    except Exception as e:
        logger.exception(f"Trial failed: {e}")
        return {
            "val_loss": float("inf"),
            "train_loss": float("inf"),
            "epochs_trained": 0,
            "error": str(e),
        }


def simple_grid_search(
    features: np.ndarray,
    targets: np.ndarray,
    num_samples: int = 5,
    max_epochs_per_trial: int = 20,
) -> dict[str, Any]:
    """Simple grid search optimization."""

    logger.info("Running simple grid search optimization...")

    # Define search space
    param_grid = {
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32],
        "lstm_units": [64, 128],
        "dropout": [0.2, 0.3],
        "sequence_length": [8, 12],
    }

    best_config = None
    best_score = float("inf")
    results = []

    # Sample configurations
    import random

    for trial in range(num_samples):
        config = {
            param: random.choice(values if isinstance(values, list) else [values])
            for param, values in param_grid.items()
        }

        logger.info(f"Trial {trial + 1}/{num_samples}: {config}")

        # Train model
        metrics = train_single_trial(config, features, targets, max_epochs_per_trial)

        # Track results
        result = {**config, **metrics}
        results.append(result)

        # Update best
        score = metrics["val_loss"]
        if score < best_score:
            best_score = score
            best_config = config.copy()
            logger.info(f"New best score: {best_score:.4f}")

    # Create results summary
    return {
        "best_config": best_config,
        "best_score": best_score,
        "all_results": results,
        "method": "simple_grid_search",
    }


def optimize_cnn_lstm_streamlined(
    features: np.ndarray,
    targets: np.ndarray,
    num_samples: int = 5,
    max_epochs_per_trial: int = 20,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Streamlined CNN-LSTM optimization.

    Args:
        features: Input features array (n_samples, n_features)
        targets: Target values array (n_samples,)
        num_samples: Number of optimization trials
        max_epochs_per_trial: Maximum epochs per trial
        output_dir: Directory to save results

    Returns:
        Dictionary with optimization results
    """

    logger.info(f"Starting CNN-LSTM optimization with {num_samples} trials...")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./optimization_results/cnn_lstm_{timestamp}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Always use Ray Tune for optimization
    logger.info("Using Ray Tune for optimization")
    # Ensure Ray cluster is initialized
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray cluster must be initialized before running optimization.",
        )
    results = ray_tune_optimization(
        features_scaled,
        targets,
        num_samples,
        max_epochs_per_trial,
    )

    # Save results
    results_path = Path(output_dir) / "optimization_results.json"
    with results_path.open("w") as f:
        # Convert numpy types to JSON serializable
        def make_serializable(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer | np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj

        json_results = make_serializable(results)
        json.dump(json_results, f, indent=2)

    logger.info(f"Optimization complete. Results saved to {results_path}")
    logger.info(f"Best validation loss: {results['best_score']}")

    return results


def ray_tune_optimization(
    features: np.ndarray,
    targets: np.ndarray,
    num_samples: int,
    max_epochs_per_trial: int,
    custom_search_space: dict[str, Any] | None = None,
    ray_resources_per_trial: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Ray Tune optimization when available.

    Parameters
    ----------
    features : np.ndarray
        Input feature array.
    targets : np.ndarray
        Target array.
    num_samples : int
        Number of trials to run.
    max_epochs_per_trial : int
        Maximum epochs per Ray Tune trial.
    custom_search_space : dict, optional
        Overrides the default search space.
    ray_resources_per_trial : dict, optional
        Resources to allocate per trial (``{"cpu": 1, "gpu": 0}`` by default).

    Returns
    -------
    dict
        Summary of optimization results.
    """
    # Ensure Ray cluster is initialized once
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray cluster must be initialized before running Ray Tune optimization.",
        )

    # Prepare search space
    search_space = get_default_search_space()
    if custom_search_space:
        search_space.update(custom_search_space)

    if ray_resources_per_trial is None:
        ray_resources_per_trial = {"cpu": 1, "gpu": int(torch.cuda.is_available())}

    train_fn = tune.with_parameters(
        train_single_trial,
        features=features,
        targets=targets,
        max_epochs=max_epochs_per_trial,
    )

    scheduler = ASHAScheduler(
        max_t=max_epochs_per_trial,
        grace_period=1,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_fn,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial=ray_resources_per_trial,
        metric="val_loss",
        mode="min",
        name="cnn_lstm_hparam_opt",
        verbose=1,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
    assert best_trial is not None, "No trials found in Ray Tune analysis"
    best_score = float(best_trial.last_result.get("val_loss", float("inf")))

    all_results = analysis.dataframe().to_dict(orient="records")

    return {
        "best_config": best_config,
        "best_score": best_score,
        "all_results": all_results,
        "method": "ray_tune",
    }


# Alias for backward compatibility
optimize_cnn_lstm = optimize_cnn_lstm_streamlined


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    features = np.random.randn(100, 5)
    targets = np.random.randn(100)

    results = optimize_cnn_lstm_streamlined(
        features,
        targets,
        num_samples=3,
        max_epochs_per_trial=5,
    )
    print(f"Best configuration: {results['best_config']}")
    print(f"Best score: {results['best_score']:.4f}")
