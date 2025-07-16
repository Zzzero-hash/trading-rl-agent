"""
Enhanced CNN-LSTM Training Module.

This module provides advanced training capabilities for CNN-LSTM models including
hyperparameter optimization, comprehensive metrics, and experiment tracking.
"""

import time
import warnings
from typing import Any

import numpy as np
import optuna
import torch
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from trading_rl_agent.models.cnn_lstm import CNNLSTMModel


class EnhancedCNNLSTMTrainer:
    """
    Enhanced trainer for CNN-LSTM models with advanced features.

    Features:
    - Comprehensive metrics tracking
    - Early stopping and learning rate scheduling
    - Gradient clipping
    - Model checkpointing
    - Experiment tracking (MLflow/TensorBoard)
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        enable_mlflow: bool = False,
        enable_tensorboard: bool = False,
        device: str | None = None,
    ):
        """
        Initialize the enhanced trainer.

        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            enable_mlflow: Whether to enable MLflow tracking
            enable_tensorboard: Whether to enable TensorBoard logging
            device: Device to use for training ('cpu', 'cuda', or None for auto)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.enable_mlflow = enable_mlflow
        self.enable_tensorboard = enable_tensorboard

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.model: CNNLSTMModel | None = None
        self.optimizer: optim.Optimizer | None = None
        self.scheduler: optim.lr_scheduler._LRScheduler | None = None
        self.criterion = nn.MSELoss()

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": [],
        }

        # Setup experiment tracking
        if enable_mlflow:
            self._setup_mlflow()
        if enable_tensorboard:
            self._setup_tensorboard()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.start_run()
            self.mlflow = mlflow
        except ImportError:
            warnings.warn("MLflow not available, disabling MLflow tracking", stacklevel=2)
            self.enable_mlflow = False

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir="./runs")
        except ImportError:
            warnings.warn("TensorBoard not available, disabling TensorBoard logging", stacklevel=2)
            self.enable_tensorboard = False

    def _create_model(self, input_dim: int) -> CNNLSTMModel:
        """Create CNN-LSTM model from configuration."""
        return CNNLSTMModel(
            input_dim=input_dim,
            cnn_filters=self.model_config.get("cnn_filters", [64, 128, 256]),
            cnn_kernel_sizes=self.model_config.get("cnn_kernel_sizes", [3, 3, 3]),
            lstm_units=self.model_config.get("lstm_units", 256),
            lstm_num_layers=self.model_config.get("lstm_layers", 2),
            lstm_dropout=self.model_config.get("dropout_rate", 0.2),
            cnn_dropout=self.model_config.get("dropout_rate", 0.2),
            output_dim=self.model_config.get("output_size", 1),
            use_attention=self.model_config.get("use_attention", False),
            use_residual=self.model_config.get("use_residual", False),
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        if self.model is None:
            raise ValueError("Model must be created before creating optimizer")
        return optim.Adam(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.0),
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be created before creating scheduler")
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=self.training_config.get("lr_patience", 5),
        )

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate comprehensive metrics."""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

    def train_from_dataset(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Train model from dataset.

        Args:
            sequences: Input sequences of shape (n_samples, seq_length, n_features)
            targets: Target values of shape (n_samples,)
            save_path: Path to save the best model

        Returns:
            Training results dictionary
        """
        start_time = time.time()

        # Split data
        val_split = self.training_config.get("val_split", 0.2)
        X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=val_split, random_state=42)

        # Create model
        input_dim = sequences.shape[-1]
        self.model = self._create_model(input_dim).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=False,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)

        for epoch in range(self.training_config["epochs"]):
            # Training phase
            assert self.model is not None, "Model must be initialized"
            assert self.optimizer is not None, "Optimizer must be initialized"
            assert self.scheduler is not None, "Scheduler must be initialized"

            self.model.train()
            train_losses = []
            train_predictions = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()

                train_losses.append(loss.item())
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_y.detach().cpu().numpy())

            # Validation phase
            self.model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_metrics = self._calculate_metrics(np.array(train_targets), np.array(train_predictions))
            val_metrics = self._calculate_metrics(np.array(val_targets), np.array(val_predictions))

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_path:
                    self._save_checkpoint(save_path, epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Logging
            if self.enable_mlflow:
                self.mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_mae": train_metrics["mae"],
                        "val_mae": val_metrics["mae"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

            if self.enable_tensorboard:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("MAE/Train", train_metrics["mae"], epoch)
                self.writer.add_scalar("MAE/Val", val_metrics["mae"], epoch)
                self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch)

        # Calculate final metrics on validation set
        final_metrics = self._calculate_metrics(np.array(val_targets), np.array(val_predictions))

        training_time = time.time() - start_time

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": len(self.history["train_loss"]),
            "final_metrics": final_metrics,
            "training_time": training_time,
        }

    def _save_checkpoint(self, save_path: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        assert self.model is not None, "Model must be initialized"
        assert self.optimizer is not None, "Optimizer must be initialized"
        assert self.scheduler is not None, "Scheduler must be initialized"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "history": self.history,
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Use the saved model config to recreate the model with correct dimensions
        saved_model_config = checkpoint.get("model_config", self.model_config)
        input_dim = saved_model_config.get("input_dim", 50)  # Default fallback

        # Recreate model and optimizer
        self.model = self._create_model(input_dim).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Load states
        assert self.model is not None, "Model must be initialized"
        assert self.optimizer is not None, "Optimizer must be initialized"
        assert self.scheduler is not None, "Scheduler must be initialized"

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load history
        self.history = checkpoint["history"]


class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        n_trials: int = 100,
        timeout: int | None = None,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            sequences: Input sequences
            targets: Target values
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.sequences = sequences
        self.targets = targets
        self.n_trials = n_trials
        self.timeout = timeout

    def _objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Suggest model configuration
        model_config = self._suggest_model_config(trial)

        # Suggest training configuration
        training_config = self._suggest_training_config(trial)

        # Create trainer
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
        )

        # Train model
        try:
            result = trainer.train_from_dataset(
                sequences=self.sequences,
                targets=self.targets,
            )
            return float(result["best_val_loss"])
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    def _suggest_model_config(self, trial: Trial) -> dict[str, Any]:
        """Suggest model configuration."""
        return {
            "cnn_filters": trial.suggest_categorical(
                "cnn_filters", [[32, 64], [64, 128], [32, 64, 128], [64, 128, 256]]
            ),
            "cnn_kernel_sizes": trial.suggest_categorical("cnn_kernel_sizes", [[3, 3], [5, 5], [3, 5], [3, 3, 3]]),
            "lstm_units": trial.suggest_int("lstm_units", 32, 512),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "use_attention": trial.suggest_categorical("use_attention", [True, False]),
            "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            "output_size": 1,
        }

    def _suggest_training_config(self, trial: Trial) -> dict[str, Any]:
        """Suggest training configuration."""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": 50,  # Fixed for optimization
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "val_split": 0.2,
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 15),
            "lr_patience": trial.suggest_int("lr_patience", 3, 10),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),
        }

    def optimize(self) -> dict[str, Any]:
        """Run hyperparameter optimization."""
        study = optuna.create_study(direction="minimize")
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }


def create_enhanced_model_config(
    cnn_filters: list[int] | None = None,
    cnn_kernel_sizes: list[int] | None = None,
    lstm_units: int = 256,
    lstm_layers: int = 2,
    dropout_rate: float = 0.2,
    use_attention: bool = False,
    use_residual: bool = False,
) -> dict[str, Any]:
    if cnn_filters is None:
        cnn_filters = [64, 128, 256]
    if cnn_kernel_sizes is None:
        cnn_kernel_sizes = [3, 3, 3]
    """Create enhanced model configuration."""
    return {
        "cnn_filters": cnn_filters,
        "cnn_kernel_sizes": cnn_kernel_sizes,
        "lstm_units": lstm_units,
        "lstm_layers": lstm_layers,
        "dropout_rate": dropout_rate,
        "use_attention": use_attention,
        "use_residual": use_residual,
        "output_size": 1,
    }


def create_enhanced_training_config(
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    weight_decay: float = 1e-5,
    val_split: float = 0.2,
    early_stopping_patience: int = 10,
    lr_patience: int = 5,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Create enhanced training configuration."""
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "early_stopping_patience": early_stopping_patience,
        "lr_patience": lr_patience,
        "max_grad_norm": max_grad_norm,
    }
