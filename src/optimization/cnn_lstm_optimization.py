"""Hyperparameter optimization for CNN-LSTM models.

This module provides utilities for hyperparameter tuning of CNN-LSTM models
using Ray Tune. It includes specialized sampling distributions and
configuration spaces for supervised learning time series models.

Example usage:

>>> from src.optimization.cnn_lstm_optimization import optimize_cnn_lstm
>>> results = optimize_cnn_lstm(
...     features=X,
...     targets=y,
...     num_samples=20,
...     max_epochs_per_trial=50
... )
>>> print(results.best_config)
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

from src.utils.cluster import init_ray, get_available_devices
from src.models.cnn_lstm import CNNLSTMModel, CNNLSTMConfig
from src.train_cnn_lstm import TrainingConfig, SequenceDataset

logger = logging.getLogger(__name__)


def _to_tensor(data) -> torch.Tensor:
    """Convert numpy array or pandas DataFrame to float tensor."""
    if hasattr(data, "values") and not torch.is_tensor(data):
        data = data.values
    return torch.as_tensor(data, dtype=torch.float32)


def _get_default_cnn_lstm_search_space() -> Dict[str, Any]:
    """Get default CNN-LSTM hyperparameter search space."""
    return {
        # Model architecture
        "cnn_filters": tune.choice([
            [16, 32],
            [32, 64],
            [64, 128],
            [32, 64, 128],
        ]),
        "cnn_kernel_sizes": tune.choice([
            [3, 3],
            [5, 5],
            [3, 5],
            [5, 3],
            [3, 3, 3],
        ]),
        "lstm_units": tune.choice([32, 64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        "use_attention": tune.choice([True, False]),
        
        # Training parameters
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "sequence_length": tune.choice([30, 60, 90]),
        "prediction_horizon": tune.choice([1, 3, 5]),
    }


@ray.remote(num_cpus=1, num_gpus=0.25)
def _train_cnn_lstm(
    config: Dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
    val_pct: float = 0.2,
    max_epochs: int = 100,
    early_stopping_patience: int = 10,
) -> Dict[str, Any]:
    """Train CNN-LSTM model with given hyperparameters.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Hyperparameters to use
    features : np.ndarray
        Feature array, shape (n_samples, n_features)
    targets : np.ndarray
        Target array, shape (n_samples,)
    val_pct : float, default 0.2
        Percentage of data to use for validation
    max_epochs : int, default 100
        Maximum number of epochs to train
    early_stopping_patience : int, default 10
        Patience for early stopping
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with training results
    """
    # Extract hyperparameters
    cnn_filters = config.get("cnn_filters", [32, 64])
    cnn_kernel_sizes = config.get("cnn_kernel_sizes", [3, 3])
    lstm_units = config.get("lstm_units", 64)
    dropout = config.get("dropout", 0.2)
    use_attention = config.get("use_attention", True)
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    sequence_length = config.get("sequence_length", 60)
    prediction_horizon = config.get("prediction_horizon", 1)
    
    # Make sure sequence_length and cnn_filters, cnn_kernel_sizes are compatible
    if len(cnn_filters) != len(cnn_kernel_sizes):
        raise ValueError("CNN filters and kernel sizes must have the same length")
    
    # Prepare data
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    dataset = SequenceDataset(
        features_scaled, 
        targets, 
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Split data
    n_samples = len(dataset)
    n_val = int(n_samples * val_pct)
    n_train = n_samples - n_val
    
    # Random split rather than time-based to avoid bias in tune
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    input_dim = features.shape[1]
    model = CNNLSTMModel(
        input_dim=input_dim,
        config={
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "lstm_units": lstm_units,
            "dropout": dropout,
        },
        use_attention=use_attention
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_r2": []}
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)  # Ensuring correct shape
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        
        # Calculate additional metrics
        y_pred = np.concatenate(all_preds).flatten()
        y_true = np.concatenate(all_targets).flatten()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        history["val_rmse"].append(rmse)
        history["val_r2"].append(r2)
          # Report metrics to Ray Tune
        train.report(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_rmse=rmse,
            val_r2=r2
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
    # Return final metrics
    final_metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_rmse": rmse,
        "val_r2": r2,
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
    }
    
    return final_metrics


def optimize_cnn_lstm(
    features: np.ndarray,
    targets: np.ndarray,
    num_samples: int = 20,
    max_epochs_per_trial: int = 50,
    early_stopping_patience: int = 10,
    output_dir: str = "./cnn_lstm_optimization",
    custom_search_space: Optional[Dict[str, Any]] = None,
    cpu_per_trial: float = 1.0,
    gpu_per_trial: float = 0.25,
) -> tune.ExperimentAnalysis:
    """Optimize CNN-LSTM hyperparameters using Ray Tune.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array, shape (n_samples, n_features)
    targets : np.ndarray
        Target array, shape (n_samples,)
    num_samples : int, default 20
        Number of trials to run
    max_epochs_per_trial : int, default 50
        Maximum epochs per trial
    early_stopping_patience : int, default 10
        Patience for early stopping
    output_dir : str, default "./cnn_lstm_optimization"
        Directory to save results
    custom_search_space : dict, optional
        Custom hyperparameter search space (overrides default)
    cpu_per_trial : float, default 1.0
        CPUs to allocate per trial
    gpu_per_trial : float, default 0.25
        GPUs to allocate per trial

    Returns
    -------
    ExperimentAnalysis
        Ray Tune experiment analysis object
    """
    # Initialize Ray if not already done
    if not ray.is_initialized():
        init_ray()
    
    # Prepare search space
    search_space = _get_default_cnn_lstm_search_space()
    if custom_search_space:
        search_space.update(custom_search_space)
    
    # Configure search algorithm
    search_alg = OptunaSearch(
        metric="val_loss",
        mode="min",
    )
    
    # Configure scheduler
    scheduler = ASHAScheduler(
        max_t=max_epochs_per_trial,
        grace_period=5,
        reduction_factor=2,
    )
    
    # Setup output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define the training function
    def train_fn(config):
        return ray.get(_train_cnn_lstm.remote(
            config, 
            features, 
            targets,
            val_pct=0.2,
            max_epochs=max_epochs_per_trial,
            early_stopping_patience=early_stopping_patience,
        ))
    
    # Run optimization
    analysis = tune.run(
        train_fn,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        local_dir=output_dir,
        metric="val_loss",
        mode="min",
        verbose=2,
    )
    
    # Log best config
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    logger.info(f"Best CNN-LSTM config: {best_config}")
    
    # Save best config
    best_config_path = Path(output_dir) / "best_config.pt"
    torch.save(best_config, best_config_path)
    
    return analysis


def train_best_cnn_lstm(
    features: np.ndarray,
    targets: np.ndarray,
    best_config: Dict[str, Any],
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    epochs: int = 100,
    early_stopping_patience: int = 15,
    save_path: Optional[str] = None,
) -> Tuple[CNNLSTMModel, Dict[str, List[float]]]:
    """Train CNN-LSTM model with best hyperparameters.
    
    Parameters
    ----------
    features : np.ndarray
        Feature array, shape (n_samples, n_features)
    targets : np.ndarray
        Target array, shape (n_samples,)
    best_config : Dict[str, Any]
        Best hyperparameters from optimization
    train_pct : float, default 0.7
        Percentage of data to use for training
    val_pct : float, default 0.15
        Percentage of data to use for validation
    epochs : int, default 100
        Maximum number of epochs to train
    early_stopping_patience : int, default 15
        Patience for early stopping
    save_path : str, optional
        Path to save the model. If None, the model won't be saved.

    Returns
    -------
    tuple
        (trained model, training history)
    """
    # Extract hyperparameters
    cnn_filters = best_config.get("cnn_filters", [32, 64])
    cnn_kernel_sizes = best_config.get("cnn_kernel_sizes", [3, 3])
    lstm_units = best_config.get("lstm_units", 64)
    dropout = best_config.get("dropout", 0.2)
    use_attention = best_config.get("use_attention", True)
    learning_rate = best_config.get("learning_rate", 0.001)
    batch_size = best_config.get("batch_size", 32)
    sequence_length = best_config.get("sequence_length", 60)
    prediction_horizon = best_config.get("prediction_horizon", 1)
    
    # Prepare data
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    dataset = SequenceDataset(
        features_scaled, 
        targets, 
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Split data (time-based split for final training)
    n_samples = len(dataset)
    n_train = int(n_samples * train_pct)
    n_val = int(n_samples * val_pct)
    n_test = n_samples - n_train - n_val
    
    # Split datasets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    input_dim = features.shape[1]
    model = CNNLSTMModel(
        input_dim=input_dim,
        config={
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "lstm_units": lstm_units,
            "dropout": dropout,
        },
        use_attention=use_attention
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "test_loss": []}
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        # Test evaluation
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss = test_loss / len(test_loader.dataset)
        history["test_loss"].append(test_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model if requested
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and config together
        save_dict = {
            "model_state_dict": model.state_dict(),
            "config": {
                "cnn_filters": cnn_filters,
                "cnn_kernel_sizes": cnn_kernel_sizes,
                "lstm_units": lstm_units,
                "dropout": dropout,
                "use_attention": use_attention,
                "input_dim": input_dim,
            },
            "scaler": scaler,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
        }
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model, history


def visualize_optimization_results(
    analysis: tune.ExperimentAnalysis,
    output_dir: str = "./cnn_lstm_optimization"
):
    """Visualize hyperparameter optimization results.
    
    Parameters
    ----------
    analysis : tune.ExperimentAnalysis
        Ray Tune experiment analysis object
    output_dir : str, default "./cnn_lstm_optimization"
        Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get results as dataframe
    df = analysis.results_df
    
    if df.empty:
        logger.warning("No results to visualize")
        return
    
    # Scatter plot of key hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot 1: Learning rate vs val_loss
    if 'config/learning_rate' in df.columns and 'val_loss' in df.columns:
        ax = axes[0]
        sc = ax.scatter(df['config/learning_rate'], df['val_loss'], c=df['val_rmse'], cmap='viridis')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Learning Rate vs Validation Loss')
        fig.colorbar(sc, ax=ax, label='RMSE')
    
    # Plot 2: Batch size vs val_loss
    if 'config/batch_size' in df.columns and 'val_loss' in df.columns:
        ax = axes[1]
        sc = ax.scatter(df['config/batch_size'], df['val_loss'], c=df['val_rmse'], cmap='viridis')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Batch Size vs Validation Loss')
        fig.colorbar(sc, ax=ax, label='RMSE')
    
    # Plot 3: LSTM units vs val_loss
    if 'config/lstm_units' in df.columns and 'val_loss' in df.columns:
        ax = axes[2]
        sc = ax.scatter(df['config/lstm_units'], df['val_loss'], c=df['val_rmse'], cmap='viridis')
        ax.set_xlabel('LSTM Units')
        ax.set_ylabel('Validation Loss')
        ax.set_title('LSTM Units vs Validation Loss')
        fig.colorbar(sc, ax=ax, label='RMSE')
    
    # Plot 4: Dropout vs val_loss
    if 'config/dropout' in df.columns and 'val_loss' in df.columns:
        ax = axes[3]
        sc = ax.scatter(df['config/dropout'], df['val_loss'], c=df['val_rmse'], cmap='viridis')
        ax.set_xlabel('Dropout')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Dropout vs Validation Loss')
        fig.colorbar(sc, ax=ax, label='RMSE')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hyperparameter_scatter.png")
    plt.close()
    
    # Create parallel coordinates plot
    try:
        from skopt.plots import plot_parallel_coordinates
        
        # Filter to only include key parameters and metrics
        key_params = [col for col in df.columns if col.startswith('config/')]
        key_metrics = ['val_loss', 'val_rmse', 'val_r2']
        
        plot_cols = key_params + key_metrics
        plot_df = df[plot_cols].copy()
        
        # Rename columns for better readability
        plot_df.columns = [col.replace('config/', '') for col in plot_df.columns]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_parallel_coordinates(plot_df, plot_df.columns, ax=ax)
        ax.set_title('Parallel Coordinates Plot of Hyperparameters and Metrics')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "parallel_coordinates.png")
        plt.close()
    except:
        logger.warning("Could not create parallel coordinates plot")
    
    # Create convergence plot for best trial
    best_trial = analysis.get_best_trial('val_loss', 'min')
    if best_trial and hasattr(best_trial, 'metric_analysis'):
        best_result_df = best_trial.metric_analysis
        
        if not best_result_df.empty and 'val_loss' in best_result_df.columns and 'epoch' in best_result_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(best_result_df['epoch'], best_result_df['train_loss'], label='Training Loss')
            plt.plot(best_result_df['epoch'], best_result_df['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss for Best Trial')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "best_trial_convergence.png")
            plt.close()
    
    # Save summary stats
    try:
        summary = {
            "best_val_loss": analysis.get_best_trial('val_loss', 'min').last_result['val_loss'],
            "best_rmse": analysis.get_best_trial('val_rmse', 'min').last_result['val_rmse'],
            "best_r2": analysis.get_best_trial('val_r2', 'max').last_result['val_r2'],
            "best_config": analysis.get_best_config('val_loss', 'min'),
            "num_trials": len(df),
        }
        
        with open(Path(output_dir) / "optimization_summary.txt", "w") as f:
            f.write("# CNN-LSTM Hyperparameter Optimization Summary\n\n")
            f.write(f"Total trials: {summary['num_trials']}\n")
            f.write(f"Best validation loss: {summary['best_val_loss']:.6f}\n")
            f.write(f"Best RMSE: {summary['best_rmse']:.6f}\n")
            f.write(f"Best R²: {summary['best_r2']:.6f}\n\n")
            
            f.write("## Best Configuration\n\n")
            for param, value in summary['best_config'].items():
                f.write(f"- {param}: {value}\n")
    except:
        logger.warning("Could not create summary stats")


if __name__ == "__main__":
    # Simple demonstration
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    
    # Create time series data with trend and noise
    features = np.random.randn(n_samples, n_features)
    targets = np.sin(np.arange(n_samples) * 0.1) + 0.1 * np.random.randn(n_samples)
    
    # Run optimization with small number of samples for demonstration
    results = optimize_cnn_lstm(
        features=features,
        targets=targets,
        num_samples=4,
        max_epochs_per_trial=10,
        early_stopping_patience=5,
        output_dir="./demo_optimization",
        gpu_per_trial=0.25,
    )
    
    # Print best config
    best_config = results.get_best_config(metric="val_loss", mode="min")
    print(f"Best config: {best_config}")
    
    # Train final model with best config
    model, history = train_best_cnn_lstm(
        features=features,
        targets=targets,
        best_config=best_config,
        save_path="./demo_optimization/best_model.pt",
    )
    
    # Visualize results
    visualize_optimization_results(results, "./demo_optimization")
