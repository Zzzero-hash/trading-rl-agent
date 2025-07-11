"""
Comprehensive CNN+LSTM Training Manager

This module provides a complete training pipeline for CNN+LSTM models with:
- Configuration management
- Dataset integration
- Training monitoring and logging
- Model checkpointing
- Hyperparameter optimization
- Evaluation and metrics tracking
"""

import logging
import time
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from trading_rl_agent.data.robust_dataset_builder import DatasetConfig, RobustDatasetBuilder
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
from trading_rl_agent.utils.metrics import calculate_max_drawdown, calculate_sharpe_ratio

logger = logging.getLogger(__name__)


class CNNLSTMTrainingManager:
    """
    Comprehensive training manager for CNN+LSTM models.

    This class handles the complete training pipeline including:
    - Configuration management
    - Dataset preparation
    - Model training with monitoring
    - Evaluation and metrics tracking
    - Model checkpointing and saving
    """

    def __init__(self, config_path: str):
        """Initialize the training manager with configuration."""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.model: CNNLSTMModel | None = None
        self.history: dict[str, list[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
            "learning_rates": [],
        }

        # Setup logging and monitoring
        self._setup_logging()
        self._setup_monitoring()

        logger.info("🚀 CNN+LSTM Training Manager initialized")
        logger.info(f"📊 Device: {self.device}")
        logger.info(f"📁 Config loaded from: {config_path}")

    def _assert_model_exists(self) -> None:
        """Assert that model exists, raising ValueError if not."""
        if self.model is None:
            raise ValueError("Model must be created before this operation")

    def _get_model(self) -> CNNLSTMModel:
        """Get the model, raising ValueError if not created."""
        if self.model is None:
            raise ValueError("Model must be created before this operation")
        return self.model

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ["model", "training", "dataset", "monitoring"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        return cast(dict[str, Any], config)

    def _setup_device(self) -> torch.device:
        """Setup training device (CPU/GPU)."""
        device_config = self.config["training"].get("device", "auto")

        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)

        return device

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config["monitoring"]
        log_level = getattr(logging, log_config.get("log_level", "INFO"))
        log_file = log_config.get("log_file", "logs/cnn_lstm_training.log")

        # Create log directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and experiment tracking."""
        monitoring_config = self.config["monitoring"]

        # Setup MLflow
        tracking_uri = monitoring_config.get("tracking_uri", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(tracking_uri)

        # Setup TensorBoard
        if monitoring_config.get("tensorboard_enabled", True):
            tensorboard_dir = monitoring_config.get("tensorboard_log_dir", "logs/tensorboard")
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
        else:
            self.tensorboard_writer = None

        # Setup checkpoint directory
        checkpoint_dir = monitoring_config.get("checkpoint_dir", "models/checkpoints")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, force_rebuild: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        """Prepare dataset for training."""
        dataset_config = self.config["dataset"]

        # Create dataset configuration
        config = DatasetConfig(
            symbols=dataset_config["symbols"],
            start_date=dataset_config["start_date"],
            end_date=dataset_config["end_date"],
            timeframe=dataset_config["timeframe"],
            real_data_ratio=dataset_config["real_data_ratio"],
            min_samples_per_symbol=dataset_config["min_samples_per_symbol"],
            sequence_length=self.config["training"]["sequence_length"],
            prediction_horizon=self.config["training"]["prediction_horizon"],
            overlap_ratio=dataset_config["overlap_ratio"],
            technical_indicators=dataset_config["technical_indicators"],
            sentiment_features=dataset_config["sentiment_features"],
            market_regime_features=dataset_config["market_regime_features"],
            outlier_threshold=dataset_config["outlier_threshold"],
            missing_value_threshold=dataset_config["missing_value_threshold"],
            output_dir=dataset_config["output_dir"],
            save_metadata=dataset_config["save_metadata"],
        )

        logger.info("📊 Preparing dataset...")

        # Build or load dataset
        if force_rebuild:
            sequences, targets, dataset_info = RobustDatasetBuilder(config).build_dataset()
        else:
            sequences, targets, dataset_info = RobustDatasetBuilder.load_or_build(config)

        logger.info(f"✅ Dataset prepared: {sequences.shape} sequences, {targets.shape} targets")
        logger.info(f"📈 Dataset info: {dataset_info}")

        return sequences, targets, dataset_info

    def create_model(self, input_dim: int) -> CNNLSTMModel:
        """Create CNN+LSTM model from configuration."""
        model_config = self.config["model"]

        logger.info("🧠 Creating CNN+LSTM model...")

        model = CNNLSTMModel(
            input_dim=input_dim,
            cnn_filters=model_config["cnn_filters"],
            cnn_kernel_sizes=model_config["cnn_kernel_sizes"],
            cnn_dropout=model_config["cnn_dropout"],
            lstm_units=model_config["lstm_units"],
            lstm_num_layers=model_config["lstm_num_layers"],
            lstm_dropout=model_config["lstm_dropout"],
            dense_units=model_config["dense_units"],
            output_dim=model_config["output_dim"],
            activation=model_config["activation"],
            use_attention=model_config["use_attention"],
        ).to(self.device)

        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("📊 Model created:")
        logger.info(f"  🔢 Total parameters: {total_params:,}")
        logger.info(f"  🎯 Trainable parameters: {trainable_params:,}")
        logger.info(f"  📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

        return model

    def create_data_loaders(
        self, sequences: np.ndarray, targets: np.ndarray
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        training_config = self.config["training"]
        evaluation_config = self.config["evaluation"]

        # Split data
        val_split = training_config["val_split"]
        test_split = evaluation_config.get("test_split", 0.2)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, targets, test_size=test_split, random_state=42, shuffle=False
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split / (1 - test_split), random_state=42, shuffle=False
        )

        logger.info("📊 Data splits:")
        logger.info(f"  🏋️ Training: {X_train.shape}")
        logger.info(f"  📊 Validation: {X_val.shape}")
        logger.info(f"  🧪 Test: {X_test.shape}")

        # Create data loaders
        batch_size = training_config["batch_size"]
        num_workers = training_config.get("num_workers", 4)

        train_loader = self._create_dataloader(X_train, y_train, batch_size, num_workers, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, batch_size, num_workers, shuffle=False)
        test_loader = self._create_dataloader(X_test, y_test, batch_size, num_workers, shuffle=False)

        return train_loader, val_loader, test_loader

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, num_workers: int, shuffle: bool
    ) -> DataLoader:
        """Create a PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str | None = None) -> dict[str, Any]:
        """Train the CNN+LSTM model."""
        training_config = self.config["training"]

        logger.info("🚀 Starting CNN+LSTM training...")

        # Setup optimizer
        optimizer = self._create_optimizer()

        # Setup loss function
        criterion = self._create_loss_function()

        # Setup learning rate scheduler
        scheduler = self._create_scheduler(optimizer)

        # Training loop
        epochs = training_config["epochs"]
        early_stopping_patience = training_config["early_stopping_patience"]

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        # Start MLflow run
        with mlflow.start_run(experiment_name=self.config["monitoring"]["experiment_name"]):
            # Log parameters
            mlflow.log_params(self._flatten_config(self.config))

            for epoch in range(epochs):
                epoch_start_time = time.time()

                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)

                # Validation phase
                val_loss, val_metrics = self._validate_epoch(val_loader, criterion)

                # Learning rate scheduling
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]["lr"]

                # Update history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["metrics"].append(val_metrics)
                self.history["learning_rates"].append(new_lr)

                # Log to MLflow and TensorBoard
                self._log_metrics(epoch, train_loss, val_loss, val_metrics, new_lr)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if save_path:
                        self._save_checkpoint(save_path, epoch, val_loss)
                        logger.info("💾 Best model checkpoint saved")
                else:
                    patience_counter += 1

                # Log epoch summary
                epoch_time = time.time() - epoch_start_time
                self._log_epoch_summary(epoch, epochs, train_loss, val_loss, val_metrics, epoch_time)

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    break

            # Log final metrics
            mlflow.log_metrics(
                {
                    "best_val_loss": best_val_loss,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "total_training_time": time.time() - start_time,
                }
            )

            # Save model to MLflow
            if self.model is not None:
                mlflow.pytorch.log_model(self._get_model(), "model")

        logger.info("✅ Training completed!")
        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "total_epochs": epoch + 1,
            "training_time": time.time() - start_time,
        }

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        self._assert_model_exists()
        training_config = self.config["training"]
        optimizer_name = training_config.get("optimizer", "adam")
        learning_rate = training_config["learning_rate"]
        weight_decay = training_config.get("weight_decay", 1e-5)

        if optimizer_name.lower() == "adam":
            return optim.Adam(self._get_model().parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name.lower() == "sgd":
            return optim.SGD(self._get_model().parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name.lower() == "rmsprop":
            return optim.RMSprop(self._get_model().parameters(), lr=learning_rate, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_name = self.config["training"].get("loss_function", "mse")

        if loss_name.lower() == "mse":
            return nn.MSELoss()
        if loss_name.lower() == "mae":
            return nn.L1Loss()
        if loss_name.lower() == "huber":
            return nn.HuberLoss()
        raise ValueError(f"Unsupported loss function: {loss_name}")

    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        training_config = self.config["training"]
        patience = training_config.get("reduce_lr_patience", 5)
        factor = training_config.get("reduce_lr_factor", 0.5)

        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor, verbose=True
        )

    def _train_epoch(
        self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int
    ) -> float:
        """Train for one epoch."""
        self._assert_model_exists()
        self._get_model().train()
        total_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self._get_model()(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.debug(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        return total_loss / num_batches

    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> tuple[float, dict[str, float]]:
        """Validate for one epoch."""
        self._assert_model_exists()
        self._get_model().eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self._get_model()(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        avg_loss = total_loss / len(dataloader)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        return avg_loss, metrics

    def _calculate_metrics(self, targets: list[float], predictions: list[float]) -> dict[str, float]:
        """Calculate evaluation metrics."""
        targets = np.array(targets)
        predictions = np.array(predictions)

        return {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2_score": r2_score(targets, predictions),
            "correlation": np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0,
        }

    def _log_metrics(
        self, epoch: int, train_loss: float, val_loss: float, metrics: dict[str, float], lr: float
    ) -> None:
        """Log metrics to MLflow and TensorBoard."""
        # Log to MLflow
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr, **metrics}, step=epoch)

        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("Loss/Train", train_loss, epoch)
            self.tensorboard_writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.tensorboard_writer.add_scalar("Learning_Rate", lr, epoch)

            for metric_name, metric_value in metrics.items():
                self.tensorboard_writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        metrics: dict[str, float],
        epoch_time: float,
    ) -> None:
        """Log epoch summary."""
        logger.info(f"📋 Epoch {epoch + 1}/{total_epochs} Summary:")
        logger.info(f"  🔥 Train Loss: {train_loss:.6f}")
        logger.info(f"  📊 Val Loss: {val_loss:.6f}")
        logger.info(f"  📏 Val MAE: {metrics['mae']:.6f}")
        logger.info(f"  📈 Val RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  🔗 Val R²: {metrics['r2_score']:.4f}")
        logger.info(f"  ⏱️ Epoch Time: {epoch_time:.1f}s")

    def _save_checkpoint(self, save_path: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        self._assert_model_exists()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._get_model().state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "history": self.history,
        }
        torch.save(checkpoint, save_path)

    def _flatten_config(self, config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested configuration for MLflow logging."""
        flattened = {}
        for key, value in config.items():
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, f"{prefix}{key}."))
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened

    def evaluate(self, test_loader: DataLoader) -> dict[str, Any]:
        """Evaluate the trained model on test data."""
        self._assert_model_exists()
        logger.info("🧪 Evaluating model on test data...")
        self._get_model().eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self._get_model()(data)
                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        metrics = self._calculate_comprehensive_metrics(all_targets, all_predictions)
        logger.info("✅ Evaluation completed!")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  📊 {metric_name}: {metric_value:.6f}")
        return metrics

    def _calculate_comprehensive_metrics(self, targets: list[float], predictions: list[float]) -> dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        targets = np.array(targets)
        predictions = np.array(predictions)

        # Basic metrics
        metrics = {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2_score": r2_score(targets, predictions),
            "correlation": np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0,
        }

        # Trading-specific metrics
        returns = np.diff(targets) / targets[:-1]
        pred_returns = np.diff(predictions) / predictions[:-1]

        if len(returns) > 0:
            metrics.update(
                {
                    "sharpe_ratio": calculate_sharpe_ratio(returns),
                    "max_drawdown": calculate_max_drawdown(returns),
                    "win_rate": np.mean(returns > 0),
                }
            )

        return metrics

    def save_model(self, save_path: str, model_format: str = "pytorch") -> None:
        """Save the trained model."""
        self._assert_model_exists()
        logger.info(f"💾 Saving model to {save_path}")
        if model_format == "pytorch":
            torch.save(self._get_model().state_dict(), save_path)
        elif model_format == "torchscript":
            self._get_model().eval()
            example_input = torch.randn(1, self.config["training"]["sequence_length"], self._get_model().input_dim)
            traced_model = torch.jit.trace(self._get_model(), example_input)
            traced_model.save(save_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        logger.info("✅ Model saved successfully!")

    def load_model(self, model_path: str, input_dim: int) -> None:
        """Load a trained model."""
        logger.info(f"📂 Loading model from {model_path}")
        self.model = self.create_model(input_dim)
        if self.model is None:
            raise ValueError("Model must be created before loading state dict")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info("✅ Model loaded successfully!")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        self._assert_model_exists()
        self._get_model().eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            predictions = self._get_model()(data_tensor)
            return predictions.cpu().numpy()
