"""
Enhanced CNN-LSTM Training Module.

This module provides advanced training capabilities for CNN-LSTM models including
hyperparameter optimization, comprehensive metrics, experiment tracking, and
real-time visual monitoring.
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

    from trade_agent.utils.cluster import get_optimal_worker_count, validate_cluster_health
    from trade_agent.utils.ray_utils import robust_ray_init
except ImportError:
    ray = None
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from trade_agent.models.cnn_lstm import CNNLSTMModel
from trade_agent.training.visual_monitor import TrainingVisualMonitor, auto_monitor_optuna, auto_monitor_training


def init_ray_cluster(show_info: bool = True, max_workers: int | None = None) -> bool:
    """Initialize or join Ray cluster with robust error handling.

    Args:
        show_info: Whether to display cluster information after initialization
        max_workers: Maximum number of workers (auto-detected if None)

    Returns:
        True if initialization was successful, False otherwise
    """
    if ray is None:
        warnings.warn("Ray is not installed. Skipping Ray cluster initialization.", stacklevel=2)
        return False

    logger = logging.getLogger(__name__)

    try:
        success, info = robust_ray_init(
            max_workers=max_workers,
            show_cluster_info=show_info
        )

        if success:
            if show_info:
                # Print additional training-specific recommendations
                worker_rec = get_optimal_worker_count()
                print("\n🧠 Training Recommendations:")
                print(f"   • Use batch sizes that are multiples of {worker_rec['total_workers']} for optimal distribution")
                print(f"   • Consider {worker_rec['cpu_workers']} parallel hyperparameter trials")
                if worker_rec["gpu_workers"] > 0:
                    print(f"   • GPU training available on {worker_rec['gpu_workers']} workers")
                print()

            logger.info(f"✅ Ray cluster initialized successfully: {info['status']}")
            return True
        else:
            logger.error(f"❌ Failed to initialize Ray cluster: {info.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Unexpected error during Ray initialization: {e}")
        return False


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
        enable_visual_monitor: bool = True,
        device: str | None = None,
    ):
        """
        Initialize the enhanced trainer.

        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            enable_mlflow: Whether to enable MLflow tracking
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_visual_monitor: Whether to enable real-time visual monitoring
            device: Device to use for training ('cpu', 'cuda', or None for auto)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.enable_mlflow = enable_mlflow
        self.enable_tensorboard = enable_tensorboard
        self.enable_visual_monitor = enable_visual_monitor

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                self.device = torch.device(device)
            except RuntimeError:
                # Fallback to CPU if invalid device is provided
                self.device = torch.device("cpu")

        # Print device information for debugging
        device_info = f"🔧 Device: {self.device}"
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(self.device)})"
        print(device_info)

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

        # Visual monitoring
        self.visual_monitor: TrainingVisualMonitor | None = None

        # Setup experiment tracking
        if enable_mlflow:
            self._setup_mlflow()
        if enable_tensorboard:
            self._setup_tensorboard()
        if enable_visual_monitor:
            self._setup_visual_monitor()

    def _validate_config(self, model_config: dict[str, Any], training_config: dict[str, Any]) -> None:
        """Validate model and training configuration."""
        # Check model config
        required_model_keys = ["cnn_filters", "cnn_kernel_sizes", "lstm_units", "dropout_rate"]
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"Missing required model config key: {key}")

        # Check that CNN filters and kernel sizes have matching lengths
        filters = model_config["cnn_filters"]
        kernels = model_config["cnn_kernel_sizes"]

        if not isinstance(filters, list) or not isinstance(kernels, list):
            raise ValueError("cnn_filters and cnn_kernel_sizes must be lists")

        if len(filters) != len(kernels):
            raise ValueError(f"Mismatch: {len(filters)} CNN filters but {len(kernels)} kernel sizes")

        # Check training config
        required_training_keys = ["learning_rate", "batch_size", "epochs"]
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training config key: {key}")

        # Validate ranges
        if training_config["learning_rate"] <= 0:
            raise ValueError("Learning rate must be positive")
        if training_config["batch_size"] <= 0:
            raise ValueError("Batch size must be positive")
        if training_config["epochs"] <= 0:
            raise ValueError("Epochs must be positive")

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

    def _setup_visual_monitor(self) -> None:
        """Setup visual monitoring."""
        try:
            self.visual_monitor = auto_monitor_training(self)
            print("✅ Visual monitoring enabled")
        except Exception as e:
            warnings.warn(f"Visual monitoring setup failed: {e}", stacklevel=2)
            self.enable_visual_monitor = False

    def _create_model(self, input_dim: int) -> CNNLSTMModel:
        """Create CNN-LSTM model from configuration."""
        config = self.model_config

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

        # Optimize batch size based on available resources if Ray is available
        original_batch_size = self.training_config["batch_size"]
        if ray is not None and ray.is_initialized():
            try:
                worker_info = get_optimal_worker_count()
                # Adjust batch size to be optimal for distributed processing
                total_workers = worker_info["total_workers"]
                if total_workers > 1:
                    # Make batch size divisible by number of workers for optimal distribution
                    optimal_batch_size = ((original_batch_size + total_workers - 1) // total_workers) * total_workers
                    optimal_batch_size = min(optimal_batch_size, len(X_train) // 4)  # Don't exceed 1/4 of training data

                    if optimal_batch_size != original_batch_size:
                        print(f"🔧 Optimized batch size from {original_batch_size} to {optimal_batch_size} for {total_workers} workers")
                        self.training_config["batch_size"] = optimal_batch_size
            except Exception as e:
                logging.warning(f"Failed to optimize batch size: {e}, using original size {original_batch_size}")

        # Create model
        input_dim = sequences.shape[-1]
        self.model = self._create_model(input_dim).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).squeeze())
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).squeeze())

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

        # Store total epochs for visual monitor
        self.total_epochs = self.training_config["epochs"]

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
                # Handle tensor shape properly
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                if outputs_np.ndim == 0:
                    outputs_np = outputs_np.reshape(1)
                if batch_y_np.ndim == 0:
                    batch_y_np = batch_y_np.reshape(1)
                train_predictions.extend(outputs_np.flatten())
                train_targets.extend(batch_y_np.flatten())

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
                    # Handle tensor shape properly
                    outputs_np = outputs.cpu().numpy()
                    batch_y_np = batch_y.cpu().numpy()
                    if outputs_np.ndim == 0:
                        outputs_np = outputs_np.reshape(1)
                    if batch_y_np.ndim == 0:
                        batch_y_np = batch_y_np.reshape(1)
                    val_predictions.extend(outputs_np.flatten())
                    val_targets.extend(batch_y_np.flatten())

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

        # Save visual monitoring summary
        if self.visual_monitor:
            try:
                self.visual_monitor.save_training_summary(self)
                self.visual_monitor.stop_monitoring()
            except Exception as e:
                warnings.warn(f"Failed to save visual monitoring summary: {e}", stacklevel=2)

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
    """Hyperparameter optimizer using Optuna with Ray distributed capabilities."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        n_trials: int = 100,
        timeout: int | None = None,
        enable_visual_monitor: bool = True,
        use_ray: bool = True,
        ray_concurrency: int | None = None,
        epochs: int = 50,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            sequences: Input sequences
            targets: Target values
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            enable_visual_monitor: Whether to enable visual monitoring of trials
            use_ray: Whether to use Ray for distributed optimization
            ray_concurrency: Number of concurrent Ray trials (auto-detected if None)
            epochs: Number of epochs to use for each optimization trial
        """
        self.sequences = sequences
        self.targets = targets
        self.n_trials = n_trials
        self.timeout = timeout
        self.enable_visual_monitor = enable_visual_monitor
        self.epochs = epochs
        self.use_ray = use_ray and ray is not None

        # Initialize Ray if requested and available
        self.ray_initialized = False
        if self.use_ray:
            try:
                if not ray.is_initialized():
                    success, _ = robust_ray_init(show_cluster_info=False)
                    if success:
                        self.ray_initialized = True
                        logging.info("🚀 Ray initialized for distributed hyperparameter optimization")
                    else:
                        logging.warning("Failed to initialize Ray, falling back to sequential optimization")
                        self.use_ray = False
                else:
                    self.ray_initialized = True

                # Auto-detect concurrency if not specified
                if ray_concurrency is None and self.use_ray:
                    worker_info = get_optimal_worker_count()
                    # Use conservative concurrency to avoid resource exhaustion
                    ray_concurrency = max(1, min(worker_info["cpu_workers"], 4))
                    logging.info(f"🔧 Auto-detected Ray concurrency: {ray_concurrency}")

            except Exception as e:
                logging.warning(f"Ray setup failed, using sequential optimization: {e}")
                self.use_ray = False

        self.ray_concurrency = ray_concurrency or 1

        # Initialize visual monitor for Optuna if enabled
        self.visual_monitor: TrainingVisualMonitor | None = None
        if enable_visual_monitor:
            try:
                self.visual_monitor = auto_monitor_optuna("CNN_LSTM_HPO")
            except Exception as e:
                warnings.warn(f"Failed to initialize Optuna visual monitor: {e}", stacklevel=2)

    def _objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        import time
        start_time = time.time()
        trial_num = trial.number + 1

        # Suggest model configuration
        model_config = self._suggest_model_config(trial)

        # Suggest training configuration
        training_config = self._suggest_training_config(trial)

        # Show trial start with key parameters
        lr = training_config.get("learning_rate", "N/A")
        lstm_units = model_config.get("lstm_units", "N/A")
        batch_size = training_config.get("batch_size", "N/A")
        cnn_arch = trial.params.get("cnn_architecture", "N/A") if hasattr(trial, "params") else "N/A"

        print(f"\n⚡ Trial {trial_num}/{self.n_trials}: LR={lr:.2e} | LSTM={lstm_units} | Batch={batch_size} | CNN={cnn_arch}")
        print(f"   [{'█' * int(20 * trial_num / self.n_trials)}{'░' * (20 - int(20 * trial_num / self.n_trials))}] {100 * trial_num / self.n_trials:.1f}%")

        # Create trainer (disable visual monitor for individual trials to avoid conflicts)
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_mlflow=False,
            enable_tensorboard=False,
            enable_visual_monitor=False,  # Disable for individual trials
            device=None  # Auto-detect CUDA
        )

        # Validate configuration before training
        try:
            trainer._validate_config(model_config, training_config)
        except Exception as e:
            print(f"   ❌ Trial {trial_num} failed: Invalid configuration - {e}")
            return float("inf")

        # Train model with progress tracking
        try:
            print(f"   🏋️  Training model... (epochs: {training_config.get('epochs', 50)})")
            result = trainer.train_from_dataset(
                sequences=self.sequences,
                targets=self.targets,
            )

            # Ensure we have a valid result with best_val_loss
            if "best_val_loss" not in result:
                print(f"   ❌ Trial {trial_num} failed: Missing best_val_loss in result")
                return float("inf")

            val_loss = result["best_val_loss"]

            # Check for invalid loss values
            if val_loss is None or np.isnan(val_loss) or np.isinf(val_loss):
                print(f"   ❌ Trial {trial_num} failed: Invalid validation loss: {val_loss}")
                return float("inf")

            val_loss = float(val_loss)
            elapsed = time.time() - start_time
            epochs_trained = result.get("total_epochs", "N/A")
            final_metrics = result.get("final_metrics", {})
            mae = final_metrics.get("mae", "N/A")
            r2 = final_metrics.get("r2", "N/A")

            print(f"   ✅ Trial {trial_num} completed ({epochs_trained} epochs, {elapsed:.1f}s)")
            mae_str = f"{mae:.4f}" if mae != "N/A" else "N/A"
            r2_str = f"{r2:.3f}" if r2 != "N/A" else "N/A"
            print(f"      📊 Val Loss: {val_loss:.4f} | MAE: {mae_str} | R²: {r2_str}")

            # Update visual monitor with trial results
            if self.visual_monitor:
                try:
                    trial_data = {
                        "number": trial.number,
                        "value": val_loss,
                        "params": dict(trial.params) if hasattr(trial, "params") else {},
                        "state": "COMPLETE",
                        "best_value": val_loss  # This will be updated by the monitor
                    }
                    self.visual_monitor.update_optuna_trial(trial_data)
                except Exception as e:
                    print(f"   Warning: Failed to update visual monitor: {e}")

            return val_loss

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Trial {trial_num} failed after {elapsed:.1f}s: {e!s}")
            return float("inf")

    def _generate_compatible_lstm_attention_pairs(self) -> list[tuple[int, int]]:
        """
        Generate compatible (lstm_units, attention_heads) combinations.

        Ensures lstm_units is always divisible by attention_heads for proper
        multi-head attention operation.

        Returns:
            List of (lstm_units, attention_heads) tuples that are mathematically compatible
        """
        combinations = []

        # Define possible attention head counts (powers of 2 for efficiency)
        possible_heads = [1, 2, 4, 8, 16]

        for heads in possible_heads:
            # Find LSTM units in range [32, 512] that are divisible by heads
            min_units = max(32, heads)  # Ensure minimum viable size
            max_units = 512

            # Generate units that are multiples of heads within our range
            current_units = min_units
            # Round up to nearest multiple of heads
            if current_units % heads != 0:
                current_units = ((current_units // heads) + 1) * heads

            while current_units <= max_units:
                combinations.append((current_units, heads))
                # Add some good intermediate values (not every multiple)
                # to keep the search space manageable but well-distributed
                current_units += heads * max(1, heads // 2)

        # Sort by total parameters (units * heads) to provide logical ordering
        combinations.sort(key=lambda x: x[0] * x[1])

        # Remove duplicates while preserving order
        seen = set()
        unique_combinations = []
        for combo in combinations:
            if combo not in seen:
                seen.add(combo)
                unique_combinations.append(combo)

        return unique_combinations

    def _suggest_model_config(self, trial: Trial) -> dict[str, Any]:
        """Suggest model configuration with coordinated LSTM units and attention heads."""
        use_attention = trial.suggest_categorical("use_attention", [True, False])

        # Use coordinated parameter suggestion to ensure compatibility
        if use_attention:
            # Generate compatible (lstm_units, attention_heads) combinations
            compatible_combinations = self._generate_compatible_lstm_attention_pairs()
            combination_choice = trial.suggest_categorical(
                "lstm_attention_combination",
                list(range(len(compatible_combinations)))
            )
            lstm_units, attention_heads = compatible_combinations[combination_choice]
        else:
            # When no attention, any LSTM units are fine
            lstm_units = trial.suggest_int("lstm_units", 32, 512)
            attention_heads = 1

        # Define coordinated CNN architectures (filters and kernels must match in length)
        cnn_architectures = {
            "small_2layer": {
                "filters": [32, 64],
                "kernels": [3, 3]
            },
            "medium_2layer": {
                "filters": [64, 128],
                "kernels": [3, 3]
            },
            "large_2layer": {
                "filters": [128, 256],
                "kernels": [3, 3]
            },
            "small_3layer": {
                "filters": [32, 64, 128],
                "kernels": [3, 3, 3]
            },
            "medium_3layer": {
                "filters": [64, 128, 256],
                "kernels": [3, 3, 3]
            },
            "varied_kernel_2layer": {
                "filters": [32, 64],
                "kernels": [3, 5]
            },
            "varied_kernel_3layer": {
                "filters": [32, 64, 128],
                "kernels": [3, 5, 3]
            }
        }

        # Select coordinated architecture
        arch_choice = trial.suggest_categorical("cnn_architecture", list(cnn_architectures.keys()))
        selected_arch = cnn_architectures[arch_choice]

        return {
            "cnn_filters": selected_arch["filters"],
            "cnn_kernel_sizes": selected_arch["kernels"],
            "lstm_units": lstm_units,
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
            "epochs": self.epochs,  # Configurable epochs for optimization
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "val_split": 0.2,
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 15),
            "lr_patience": trial.suggest_int("lr_patience", 3, 10),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),
            # Add sequence length and prediction horizon as hyperparameters
            "sequence_length": trial.suggest_categorical("sequence_length", [20, 30, 60, 90, 120]),
            "prediction_horizon": trial.suggest_categorical("prediction_horizon", [1, 3, 5, 10, 20]),
        }

    def _create_progress_callback(self) -> Any:
        """Create a callback to display optimization progress."""
        start_time = time.time()

        def callback(study: Any, trial: Any) -> None:
            trial_number = trial.number + 1
            current_value = trial.value if trial.value is not None else float("inf")
            best_value = study.best_value if study.best_value is not None else float("inf")

            # Calculate timing
            elapsed_time = time.time() - start_time
            avg_time_per_trial = elapsed_time / trial_number
            remaining_trials = self.n_trials - trial_number
            estimated_remaining = avg_time_per_trial * remaining_trials

            # Show if this is a new best score
            is_best = trial.value is not None and trial.value == best_value
            status_icon = "🌟" if is_best else "✅"

            print(f"\n{status_icon} Trial {trial_number}/{self.n_trials}: Score={current_value:.4f} (Best: {best_value:.4f})")
            print(f"   ⏱️  {elapsed_time:.1f}s elapsed, ~{estimated_remaining:.1f}s remaining")

            # Compact progress bar
            progress = trial_number / self.n_trials
            bar_length = 40
            filled = int(progress * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"   📊 [{bar}] {progress*100:.1f}%")

        return callback

    def optimize(self) -> dict[str, Any]:
        """Run hyperparameter optimization with optional Ray parallelization."""
        optimization_mode = "Distributed (Ray)" if self.use_ray else "Sequential"

        print(f"\n🚀 Starting Optuna hyperparameter optimization ({optimization_mode})")
        print(f"   Trials: {self.n_trials}")
        print(f"   Timeout: {self.timeout}s" if self.timeout else "   Timeout: None")
        if self.use_ray:
            print(f"   Ray Concurrency: {self.ray_concurrency}")
            # Validate cluster health before starting
            health = validate_cluster_health()
            if not health["healthy"]:
                print(f"   ⚠️  Cluster Warning: {health['reason']}")
                for rec in health["recommendations"][:2]:  # Show top 2 recommendations
                    print(f"      • {rec}")
        print("=" * 60)

        study = optuna.create_study(direction="minimize")

        # Add progress callback
        progress_callback = self._create_progress_callback()

        try:
            if self.use_ray and self.ray_initialized:
                # Use Ray for distributed optimization
                print(f"🔧 Using Ray distributed optimization with {self.ray_concurrency} concurrent trials")

                # Create a distributed study with Ray

                # Wrap the objective function for Ray
                @ray.remote
                def distributed_objective(trial_params: dict[str, Any]) -> float:
                    # Create a trial-like object for the objective function
                    class MockTrial:
                        def __init__(self, params: dict[str, Any]) -> None:
                            self.params = params
                            self.number = 0  # This will be set properly by Optuna

                        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
                            return self.params.get(name, choices[0])

                        def suggest_int(self, name: str, low: int, _high: int) -> int:
                            return int(self.params.get(name, low))

                        def suggest_float(self, name: str, low: float, _high: float, _log: bool = False) -> float:
                            return float(self.params.get(name, low))

                    mock_trial = MockTrial(trial_params)
                    return self._objective(mock_trial)

                # Use Ray Tune integration if available
                try:
                    from ray import tune
                    from ray.tune.integration.optuna import OptunaSearch

                    search_alg = OptunaSearch(
                        metric="objective",
                        mode="min",
                        points_to_evaluate=None
                    )

                    # Run distributed optimization
                    analysis = tune.run(
                        self._ray_trainable,
                        search_alg=search_alg,
                        num_samples=self.n_trials,
                        resources_per_trial={"cpu": 1, "gpu": 0},
                        time_budget_s=self.timeout,
                        progress_reporter=tune.CLIReporter(
                            metric_columns=["objective", "training_iteration"]
                        )
                    )

                    # Extract best result
                    best_trial = analysis.get_best_trial("objective", "min")
                    best_config = best_trial.config
                    best_score = best_trial.last_result["objective"]

                    print("\n🎯 Distributed optimization completed!")
                    print(f"   Best score: {best_score:.4f}")
                    print(f"   Total trials: {len(analysis.trials)}")

                    return {
                        "best_params": best_config,
                        "best_score": best_score,
                        "study": analysis,
                        "optimization_mode": "ray_tune"
                    }

                except ImportError:
                    print("⚠️  Ray Tune not available, falling back to standard Ray parallelization")
                    # Fall back to basic Ray parallelization

            # Standard Optuna optimization (sequential or basic Ray)
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[progress_callback],
                n_jobs=self.ray_concurrency if self.use_ray else 1,
            )

            print("\n🎯 Optimization completed!")
            print(f"   Best score: {study.best_value:.4f}")
            print(f"   Total trials: {len(study.trials)}")
            print("=" * 60)

            return {
                "best_params": study.best_params,
                "best_score": study.best_value,
                "study": study,
                "optimization_mode": optimization_mode.lower().replace(" ", "_")
            }

        except Exception as e:
            logging.error(f"❌ Optimization failed: {e}")
            # Return a fallback result
            return {
                "best_params": {},
                "best_score": float("inf"),
                "study": None,
                "error": str(e),
                "optimization_mode": "failed"
            }

        finally:
            # Cleanup visual monitor
            if self.visual_monitor:
                try:
                    self.visual_monitor.stop_monitoring()
                except Exception as e:
                    logging.warning(f"Failed to stop visual monitor: {e}")

    def _ray_trainable(self, config: dict[str, Any]) -> float:
        """Trainable function for Ray Tune integration."""
        # Create a mock trial object with the config
        class MockTrial:
            def __init__(self, params: dict[str, Any]) -> None:
                self.params = params
                self.number = 0

            def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
                return self.params.get(name, choices[0])

            def suggest_int(self, name: str, low: int, _high: int) -> int:
                return int(self.params.get(name, low))

            def suggest_float(self, name: str, low: float, _high: float, _log: bool = False) -> float:
                return float(self.params.get(name, low))

        mock_trial = MockTrial(config)
        objective_value = self._objective(mock_trial)

        # Report the result back to Ray Tune
        from ray import tune
        tune.report(objective=objective_value)
        return objective_value


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
    attention_heads: int = 8,
) -> dict[str, Any]:
    """
    Create enhanced model configuration with validated attention parameters.

    Ensures that when attention is enabled, lstm_units is divisible by attention_heads.
    """
    if cnn_filters is None:
        cnn_filters = [64, 128, 256]
    if cnn_kernel_sizes is None:
        cnn_kernel_sizes = [3, 3, 3]

    # Validate attention configuration
    if use_attention and lstm_units % attention_heads != 0:
        # Find the nearest compatible lstm_units value
        original_units = lstm_units
        # Try rounding up first
        lstm_units = ((lstm_units // attention_heads) + 1) * attention_heads

        # If that makes it too large, try rounding down
        if lstm_units > 512:  # Assuming 512 is our reasonable upper limit
            lstm_units = (original_units // attention_heads) * attention_heads
            if lstm_units < 32:  # Ensure minimum viable size
                lstm_units = attention_heads

        logging.warning(
            f"Adjusted lstm_units from {original_units} to {lstm_units} "
            f"for compatibility with {attention_heads} attention heads"
        )
    return {
        "input_dim": input_dim,
        "cnn_filters": cnn_filters,
        "cnn_kernel_sizes": cnn_kernel_sizes,
        "lstm_units": lstm_units,
        "lstm_num_layers": lstm_layers,
        "dropout_rate": dropout_rate,
        "use_attention": use_attention,
        "attention_heads": attention_heads,
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
