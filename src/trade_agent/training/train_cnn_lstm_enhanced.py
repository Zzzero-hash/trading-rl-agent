"""
Enhanced CNN-LSTM Training Module.

This module provides advanced training capabilities for CNN-LSTM models including
hyperparameter optimization, comprehensive metrics, and experiment tracking.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import ray
except ImportError:
    ray = None
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from trade_agent.models.cnn_lstm import CNNLSTMModel


def init_ray_cluster() -> None:
    """Initialize or join Ray cluster."""
    if ray is None:
        warnings.warn("Ray is not installed. Skipping Ray cluster initialization.", stacklevel=2)
        return

    if not ray.is_initialized():
        try:
            # Try to connect to existing cluster first
            ray.init(address="auto", ignore_reinit_error=True)
            logging.info("Connected to existing Ray cluster")
        except Exception:
            # If no cluster exists, start a new one
            try:
                ray.init(ignore_reinit_error=True)
                logging.info("Started new Ray cluster")
            except Exception as e:
                warnings.warn(f"Failed to initialize Ray cluster: {e}", stacklevel=2)
    else:
        logging.info("Ray cluster is already initialized")


def load_and_preprocess_csv_data(
    csv_path: Path,
    sequence_length: int = 60,
    prediction_horizon: int = 1,
    target_column: str = "close",
    feature_columns: list[str] | None = None,
    scaler_type: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess CSV data specifically for CNN+LSTM training.

    This function handles:
    - Loading CSV data
    - Feature selection and engineering
    - Sequence creation for time series modeling
    - Data scaling/normalization
    - Target preparation for prediction horizon

    Args:
        csv_path: Path to the CSV dataset file
        sequence_length: Length of input sequences (lookback window)
        prediction_horizon: Number of steps ahead to predict
        target_column: Name of the target column to predict
        feature_columns: List of feature columns to use (None = auto-select numeric columns)
        scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')

    Returns:
        Tuple of (sequences, targets) as numpy arrays ready for CNN+LSTM training
    """
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Handle datetime index if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    # Auto-select numeric columns if not specified
    if feature_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column from features if it's in the list
        if target_column in numeric_columns:
            feature_columns = [col for col in numeric_columns if col != target_column]
        else:
            feature_columns = numeric_columns

    print(f"Using features: {feature_columns}")
    print(f"Target column: {target_column}")

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Handle missing values
    if X.isnull().any().any():
        print("Warning: Found missing values in features, forward-filling...")
        X = X.fillna(method="ffill").fillna(method="bfill")

    if y.isnull().any():
        print("Warning: Found missing values in target, forward-filling...")
        y = y.fillna(method="ffill").fillna(method="bfill")

    # Scale features
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    X_scaled = scaler.fit_transform(X)

    # Create sequences
    print(f"Creating sequences with length {sequence_length} and prediction horizon {prediction_horizon}...")
    sequences = []
    targets = []

    for i in range(len(X_scaled) - sequence_length - prediction_horizon + 1):
        # Input sequence
        seq = X_scaled[i:i + sequence_length]
        sequences.append(seq)

        # Target (future value)
        target_idx = i + sequence_length + prediction_horizon - 1
        target = y.iloc[target_idx]
        targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")

    return sequences, targets


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
            try:
                self.device = torch.device(device)
            except RuntimeError:
                # Fallback to CPU if invalid device is provided
                self.device = torch.device("cpu")

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

        # Add logger and metrics_history for test compatibility
        self.logger = logging.getLogger("EnhancedCNNLSTMTrainer")
        self.metrics_history: dict[str, list[float]] = {}  # Placeholder, can be replaced with actual metrics

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

    def _setup_tensorboard(self, log_dir: str | None = None) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            if log_dir is not None:
                self.writer = SummaryWriter(log_dir)
            else:
                self.writer = SummaryWriter("./runs")
        except ImportError:
            import warnings

            warnings.warn("TensorBoard not available, disabling TensorBoard logging", stacklevel=2)
            self.enable_tensorboard = False
        except Exception as e:
            import warnings

            warnings.warn(f"TensorBoard setup failed: {e}", stacklevel=2)
            self.enable_tensorboard = False

    def _create_model(self, input_dim: int) -> CNNLSTMModel:
        """Create CNN-LSTM model from configuration."""
        config = self.model_config
        print(f"DEBUG: Creating CNNLSTMModel with input_dim={input_dim}")
        print(f"DEBUG: config keys: {list(config.keys())}")

        return CNNLSTMModel(
            input_dim=input_dim,
            cnn_filters=config.get("cnn_filters", [64, 128, 256]),
            cnn_kernel_sizes=config.get("cnn_kernel_sizes", [3, 3, 3]),
            lstm_units=config.get("lstm_units", 256),
            lstm_num_layers=config.get("lstm_layers", 2),
            lstm_dropout=config.get("dropout_rate", 0.2),
            cnn_dropout=config.get("dropout_rate", 0.2),
            output_dim=config.get("output_dim", 1),
            use_attention=config.get("use_attention", False),
            use_residual=config.get("use_residual", False),
            attention_heads=config.get("attention_heads", 8),
            layer_norm=config.get("layer_norm", True),
            batch_norm=config.get("batch_norm", True),
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

    # Public methods for test compatibility
    def create_model(self, model_config: dict[str, Any] | None = None) -> CNNLSTMModel:
        cfg = model_config if model_config is not None else self.model_config
        input_dim = cfg.get("input_dim", 5)
        self.model = CNNLSTMModel(
            input_dim=input_dim,
            cnn_filters=cfg.get("cnn_filters", [64, 128, 256]),
            cnn_kernel_sizes=cfg.get("cnn_kernel_sizes", [3, 3, 3]),
            lstm_units=cfg.get("lstm_units", 256),
            lstm_num_layers=cfg.get("lstm_layers", 2),
            lstm_dropout=cfg.get("dropout_rate", 0.2),
            cnn_dropout=cfg.get("dropout_rate", 0.2),
            output_dim=cfg.get("output_dim", 1),
            use_attention=cfg.get("use_attention", False),
            use_residual=cfg.get("use_residual", False),
            attention_heads=cfg.get("attention_heads", 8),
            layer_norm=cfg.get("layer_norm", True),
            batch_norm=cfg.get("batch_norm", True),
        )
        return self.model

    def create_optimizer(
        self,
        model: CNNLSTMModel | None = None,
        learning_rate: float | None = None,
    ) -> optim.Optimizer:
        if model is not None:
            self.model = model
        if learning_rate is not None:
            self.training_config["learning_rate"] = learning_rate
        return self._create_optimizer()

    def create_scheduler(self, optimizer: optim.Optimizer | None = None) -> optim.lr_scheduler._LRScheduler:
        if optimizer is not None:
            self.optimizer = optimizer
        return self._create_scheduler()

    def calculate_metrics(
        self,
        predictions: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        # Accepts torch tensors or numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        return self._calculate_metrics(targets, predictions)

    def train_from_dataset(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        save_path: str | None = None,
        dataset_config: dict | None = None,
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

        # Apply dataset config if provided
        if dataset_config is not None:
            # Log the dataset configuration being used
            sequence_length = dataset_config.get("sequence_length")
            prediction_horizon = dataset_config.get("prediction_horizon")
            if sequence_length is not None:
                print(f"Using sequence_length: {sequence_length} from dataset_config")
            if prediction_horizon is not None:
                print(f"Using prediction_horizon: {prediction_horizon} from dataset_config")

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

    def load_checkpoint(self, checkpoint_path: str) -> tuple[CNNLSTMModel, int, float] | None:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)  # nosec

            # Recreate model with saved config
            self.model_config = checkpoint.get("model_config", self.model_config)
            input_dim = self.model_config.get("input_dim", 5)

            # Recreate model and optimizer
            self.model = self._create_model(input_dim).to(self.device)
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()

            # Load states
            assert self.model is not None, "Model must be initialized"
            assert self.optimizer is not None, "Optimizer must be initialized"
            assert self.scheduler is not None, "Scheduler must be initialized"

            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # Load history
            self.history = checkpoint.get("history", {})
            # Return tuple for backward compatibility
            epoch = checkpoint.get("epoch", 0)
            loss = checkpoint.get("val_loss", 0.0)  # Assuming val_loss is the loss to return
            return self.model, epoch, loss

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

    def train_step(
        self,
        model: CNNLSTMModel,
        optimizer: optim.Optimizer,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        optimizer.step()
        result = loss.item()
        return float(result)

    def save_checkpoint(self, model: CNNLSTMModel, checkpoint_path: str, epoch: int, loss: float) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": self.model_config,
            "epoch": epoch,
            "loss": loss,
            "history": getattr(self, "history", {}),
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def setup_mlflow(self, _experiment_name: str) -> None:
        try:
            self._setup_mlflow()
        except Exception as e:
            import warnings

            warnings.warn(f"MLflow setup failed: {e}", stacklevel=2)

    def setup_tensorboard(self, log_dir: str) -> None:
        self._setup_tensorboard(log_dir)

    def create_hyperparameter_optimizer(self) -> "HyperparameterOptimizer":
        return HyperparameterOptimizer([], [])

    def optimization_objective(self, _trial: "Trial") -> float:
        return 0.0

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 2) -> dict[str, Any]:
        optimizer = HyperparameterOptimizer(X, y, n_trials=n_trials)
        result = optimizer.optimize()
        best_params = result.get("best_params", {})
        if not isinstance(best_params, dict):
            best_params = dict(best_params)
        return best_params


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
        use_attention = trial.suggest_categorical("use_attention", [True, False])

        # Suggest attention heads only if attention is enabled
        attention_heads = 1
        if use_attention:
            # Suggest number of attention heads that are common divisors
            attention_heads = trial.suggest_categorical("attention_heads", [1, 2, 4, 8, 16])

        # Define filter options with string keys to avoid list serialization issues
        filter_configs = {
            "small": [32, 64],
            "medium": [64, 128],
            "large": [32, 64, 128],
            "extra_large": [64, 128, 256]
        }
        filter_choice = trial.suggest_categorical("cnn_filters", ["small", "medium", "large", "extra_large"])

        # Define kernel size options with string keys
        kernel_configs = {
            "3x3": [3, 3],
            "5x5": [5, 5],
            "3x5": [3, 5],
            "3x3x3": [3, 3, 3]
        }
        kernel_choice = trial.suggest_categorical("cnn_kernel_sizes", ["3x3", "5x5", "3x5", "3x3x3"])

        return {
            "cnn_filters": filter_configs[filter_choice],
            "cnn_kernel_sizes": kernel_configs[kernel_choice],
            "lstm_units": trial.suggest_int("lstm_units", 32, 512),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "use_attention": use_attention,
            "attention_heads": attention_heads,
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
            # Add sequence length and prediction horizon as hyperparameters
            "sequence_length": trial.suggest_categorical("sequence_length", [20, 30, 60, 90, 120]),
            "prediction_horizon": trial.suggest_categorical("prediction_horizon", [1, 3, 5, 10, 20]),
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
    input_dim: int = 5,
    cnn_filters: list[int] | None = None,
    cnn_kernel_sizes: list[int] | None = None,
    lstm_units: int = 256,
    lstm_layers: int = 2,
    dropout_rate: float = 0.2,
    use_attention: bool = False,
    use_residual: bool = False,
    cnn_architecture: str = "simple",
) -> dict[str, Any]:
    if cnn_filters is None:
        cnn_filters = [64, 128, 256]
    if cnn_kernel_sizes is None:
        cnn_kernel_sizes = [3, 3, 3]
    """Create enhanced model configuration."""
    return {
        "input_dim": input_dim,
        "cnn_filters": cnn_filters,
        "cnn_kernel_sizes": cnn_kernel_sizes,
        "lstm_units": lstm_units,
        "lstm_num_layers": lstm_layers,
        "dropout_rate": dropout_rate,
        "use_attention": use_attention,
        "use_residual": use_residual,
        "cnn_architecture": cnn_architecture,
        "output_dim": 1,
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
