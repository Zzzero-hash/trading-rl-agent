"""
Enhanced CNN+LSTM Training Pipeline with MLflow, TensorBoard, and Hyperparameter Optimization

This script provides a production-ready training pipeline for CNN+LSTM models with:
- MLflow experiment tracking
- TensorBoard logging
- Hyperparameter optimization with Optuna
- Comprehensive metrics and visualization
- Model checkpointing and early stopping
- Real-time training monitoring
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.data.robust_dataset_builder import (
    DatasetConfig,
    RobustDatasetBuilder,
)
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

logger = logging.getLogger(__name__)


class EnhancedCNNLSTMTrainer:
    """Enhanced trainer for CNN+LSTM models with comprehensive monitoring and optimization."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        device: str | None = None,
        experiment_name: str = "cnn_lstm_training",
        enable_mlflow: bool = True,
        enable_tensorboard: bool = True,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        self.enable_mlflow = enable_mlflow
        self.enable_tensorboard = enable_tensorboard
        
        self.model: Optional[CNNLSTMModel] = None
        self.writer: Optional[SummaryWriter] = None
        self.mlflow_run: Optional[mlflow.ActiveRun] = None
        
        self.history: Dict[str, list[Any]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_rmse": [],
            "val_rmse": [],
            "learning_rate": [],
            "gradient_norm": [],
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Initialize MLflow
        if self.enable_mlflow:
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run()
            logger.info(f"MLflow experiment started: {self.experiment_name}")
        
        # Initialize TensorBoard
        if self.enable_tensorboard:
            log_dir = Path("runs") / self.experiment_name / time.strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to: {log_dir}")

    def train_from_dataset(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        save_path: str | None = None,
    ) -> Dict[str, Any]:
        """Train the CNN+LSTM model with enhanced monitoring."""
        
        logger.info("🚀 Starting Enhanced CNN+LSTM training...")
        
        # Log parameters
        if self.enable_mlflow:
            mlflow.log_params(self.model_config)
            mlflow.log_params(self.training_config)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences,
            targets,
            test_size=self.training_config.get("val_split", 0.2),
            random_state=42,
            shuffle=False,  # Keep temporal order
        )
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # Initialize model
        input_dim = sequences.shape[-1]
        logger.info(f"🧠 Initializing CNN+LSTM model with {input_dim} input features...")
        
        self.model = CNNLSTMModel(input_dim=input_dim, config=self.model_config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"  🔢 Total parameters: {total_params:,}")
        logger.info(f"  🎯 Trainable parameters: {trainable_params:,}")
        
        # Log model architecture
        if self.writer:
            dummy_input = torch.randn(1, sequences.shape[1], input_dim).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 0.001),
            weight_decay=self.training_config.get("weight_decay", 1e-5),
        )
        
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.training_config.get("lr_patience", 5),
            factor=0.5,
            verbose=True,
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience = self.training_config.get("early_stopping_patience", 15)
        patience_counter = 0
        epochs = self.training_config.get("epochs", 100)
        start_time = time.time()
        
        logger.info(f"🎯 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["train_rmse"].append(train_metrics["rmse"])
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["learning_rate"].append(new_lr)
            
            # Log to MLflow and TensorBoard
            self._log_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics, new_lr)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self._save_checkpoint(save_path, epoch, val_loss)
                    logger.info("💾 Best model checkpoint saved")
            else:
                patience_counter += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_summary(epoch, epochs, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final evaluation
        final_metrics = self._final_evaluation(X_val, y_val)
        
        # Log final results
        if self.enable_mlflow:
            mlflow.log_metrics(final_metrics)
            mlflow.log_artifact(save_path) if save_path else None
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        # End MLflow run
        if self.mlflow_run:
            mlflow.end_run()
        
        training_summary = {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "final_metrics": final_metrics,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "training_time": time.time() - start_time,
        }
        
        logger.info("✅ Enhanced training completed successfully!")
        return training_summary

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.training_config.get("batch_size", 32)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with enhanced monitoring."""
        
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        gradient_norms = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping and monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.training_config.get("max_grad_norm", 1.0)
            )
            gradient_norms.append(grad_norm.item())
            
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            
            predictions.extend(output.cpu().detach().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{batch_loss:.6f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.6f}",
                "grad_norm": f"{grad_norm:.4f}",
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_grad_norm = np.mean(gradient_norms)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2": r2_score(targets, predictions),
            "explained_variance": explained_variance_score(targets, predictions),
        }
        
        self.history["gradient_norm"].append(avg_grad_norm)
        
        return avg_loss, metrics

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, Dict[str, float]]:
        """Validate for one epoch with enhanced monitoring."""
        
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation", leave=False)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                batch_loss = loss.item()
                total_loss += batch_loss
                
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
                
                pbar.set_postfix({
                    "val_loss": f"{batch_loss:.6f}",
                    "avg_val_loss": f"{total_loss / (batch_idx + 1):.6f}"
                })
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate comprehensive metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2": r2_score(targets, predictions),
            "explained_variance": explained_variance_score(targets, predictions),
            "correlation": np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0,
        }
        
        return avg_loss, metrics

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """Log metrics to MLflow and TensorBoard."""
        
        # Log to MLflow
        if self.enable_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", learning_rate, step=epoch)
            
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
            
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value, step=epoch)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Learning_Rate", learning_rate, epoch)
            
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f"Metrics/Train_{metric_name}", value, epoch)
            
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f"Metrics/Val_{metric_name}", value, epoch)

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
    ) -> None:
        """Log comprehensive epoch summary."""
        
        logger.info(f"\n📋 Epoch {epoch + 1}/{total_epochs} Summary:")
        logger.info(f"  🔥 Train Loss: {train_loss:.6f}")
        logger.info(f"  📊 Val Loss: {val_loss:.6f}")
        logger.info(f"  📏 Train MAE: {train_metrics['mae']:.6f}")
        logger.info(f"  📏 Val MAE: {val_metrics['mae']:.6f}")
        logger.info(f"  📈 Train RMSE: {train_metrics['rmse']:.6f}")
        logger.info(f"  📈 Val RMSE: {val_metrics['rmse']:.6f}")
        logger.info(f"  🔗 Val R²: {val_metrics['r2']:.4f}")
        logger.info(f"  ⏱️ Epoch Time: {epoch_time:.1f}s")

    def _final_evaluation(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Perform comprehensive final evaluation."""
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        targets = y_val.flatten()
        
        return {
            "mse": mean_squared_error(targets, predictions),
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "r2": r2_score(targets, predictions),
            "explained_variance": explained_variance_score(targets, predictions),
            "correlation": np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0,
            "std_predictions": np.std(predictions),
            "std_targets": np.std(targets),
            "mean_predictions": np.mean(predictions),
            "mean_targets": np.mean(targets),
        }

    def _save_checkpoint(self, save_path: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint with enhanced metadata."""
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "history": self.history,
        }
        
        torch.save(checkpoint, save_path)

    def plot_training_history(self, save_path: str | None = None) -> None:
        """Create comprehensive training visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("CNN+LSTM Training History", fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss", color="blue")
        axes[0, 0].plot(self.history["val_loss"], label="Validation Loss", color="red")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE curves
        axes[0, 1].plot(self.history["train_mae"], label="Train MAE", color="blue")
        axes[0, 1].plot(self.history["val_mae"], label="Validation MAE", color="red")
        axes[0, 1].set_title("Mean Absolute Error")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE curves
        axes[0, 2].plot(self.history["train_rmse"], label="Train RMSE", color="blue")
        axes[0, 2].plot(self.history["val_rmse"], label="Validation RMSE", color="red")
        axes[0, 2].set_title("Root Mean Square Error")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("RMSE")
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.history["learning_rate"], color="green")
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)
        
        # Gradient norm
        if self.history["gradient_norm"]:
            axes[1, 1].plot(self.history["gradient_norm"], color="purple")
            axes[1, 1].set_title("Gradient Norm")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Gradient Norm")
            axes[1, 1].grid(True)
        
        # Loss ratio (train/val)
        loss_ratio = [t/v if v > 0 else 0 for t, v in zip(self.history["train_loss"], self.history["val_loss"])]
        axes[1, 2].plot(loss_ratio, color="orange")
        axes[1, 2].set_title("Train/Val Loss Ratio")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Ratio")
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")
        
        if interactive:
            plt.show()


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, n_trials: int = 50):
        self.sequences = sequences
        self.targets = targets
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float("inf")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        
        # Define coordinated CNN architecture choices to ensure matching lengths
        cnn_architectures = [
            # (filters, kernel_sizes) - each pair has matching length
            ([16, 32], [3, 3]),
            ([32, 64], [3, 3]),
            ([64, 128], [3, 3]),
            ([32, 64, 128], [3, 3, 3]),
            ([16, 32, 64], [3, 3, 3]),
            ([32, 64, 128, 256], [3, 3, 3, 3]),
            ([16, 32, 64, 128], [5, 5, 5, 5]),
            ([32, 64], [5, 5]),
            ([64, 128], [5, 5]),
            ([16, 32, 64], [3, 5, 3]),  # Mixed kernel sizes
            ([32, 64, 128], [5, 3, 5]),  # Mixed kernel sizes
        ]
        
        # Select a coordinated CNN architecture
        selected_architecture = trial.suggest_categorical("cnn_architecture", cnn_architectures)
        cnn_filters, cnn_kernel_sizes = selected_architecture
        
        # Define hyperparameter search space
        model_config = {
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "lstm_units": trial.suggest_categorical("lstm_units", [64, 128, 256]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        }
        
        training_config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": 20,  # Reduced for faster optimization
            "val_split": 0.2,
            "early_stopping_patience": 5,
        }
        
        try:
            # Train model with current hyperparameters
            trainer = EnhancedCNNLSTMTrainer(
                model_config=model_config,
                training_config=training_config,
                enable_mlflow=False,
                enable_tensorboard=False,
            )
            
            result = trainer.train_from_dataset(
                sequences=self.sequences,
                targets=self.targets,
            )
            
            # Return validation loss as objective
            val_loss = result["best_val_loss"]
            
            # Update best parameters
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_params = {
                    "model_config": model_config,
                    "training_config": training_config,
                }
            
            return val_loss
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        logger.info(f"🔍 Starting hyperparameter optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        logger.info("✅ Optimization completed!")
        logger.info(f"🎯 Best validation loss: {study.best_value:.6f}")
        logger.info(f"🔧 Best parameters: {study.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": study.best_value,
            "study": study,
        }


def create_enhanced_model_config() -> Dict[str, Any]:
    """Create enhanced model configuration."""
    return {
        "cnn_filters": [32, 64, 128],
        "cnn_kernel_sizes": [3, 3, 3],
        "lstm_units": 128,
        "lstm_layers": 2,
        "dropout_rate": 0.2,
        "output_size": 1,
    }


def create_enhanced_training_config() -> Dict[str, Any]:
    """Create enhanced training configuration."""
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "weight_decay": 1e-5,
        "val_split": 0.2,
        "early_stopping_patience": 15,
        "lr_patience": 5,
        "max_grad_norm": 1.0,
    }


def main() -> Dict[str, Any]:
    """Main enhanced training pipeline."""
    
    parser = argparse.ArgumentParser(description="Enhanced CNN+LSTM Training with MLflow and Optimization")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        help="Stock symbols to include",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date for data",
    )
    parser.add_argument("--end-date", default="2024-12-31", help="End date for data")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for LSTM",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/enhanced_cnn_lstm_training",
        help="Output directory",
    )
    parser.add_argument("--load-dataset", help="Path to existing dataset directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting Enhanced CNN+LSTM training pipeline...")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎯 Symbols: {args.symbols}")
    logger.info(f"📅 Date range: {args.start_date} to {args.end_date}")
    logger.info(f"📏 Sequence length: {args.sequence_length}")
    logger.info(f"🔄 Epochs: {args.epochs}")
    logger.info(f"🖥️ GPU enabled: {args.gpu}")
    logger.info(f"🔍 Hyperparameter optimization: {args.optimize_hyperparams}")
    
    try:
        # Step 1: Build or load dataset
        if args.load_dataset:
            logger.info(f"Loading existing dataset from {args.load_dataset}")
            dataset_dir = Path(args.load_dataset)
            sequences = np.load(dataset_dir / "sequences.npy")
            targets = np.load(dataset_dir / "targets.npy")
        else:
            dataset_config = DatasetConfig(
                symbols=args.symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                timeframe="1d",
                real_data_ratio=0.8,
                min_samples_per_symbol=1000,
                sequence_length=args.sequence_length,
                prediction_horizon=1,
                overlap_ratio=0.8,
                technical_indicators=True,
                sentiment_features=False,
                market_regime_features=True,
                output_dir=str(output_dir / "dataset"),
            )
            
            sequences, targets, dataset_info = RobustDatasetBuilder.load_or_build(dataset_config)
            load_msg = "loaded" if dataset_info.get("loaded") else "built"
            logger.info(f"Dataset {load_msg}: {dataset_info}")
        
        # Step 2: Hyperparameter optimization (optional)
        if args.optimize_hyperparams:
            logger.info("🔍 Running hyperparameter optimization...")
            optimizer = HyperparameterOptimizer(sequences, targets, args.n_trials)
            opt_result = optimizer.optimize()
            
            # Use optimized parameters
            model_config = opt_result["best_params"]["model_config"]
            training_config = opt_result["best_params"]["training_config"]
            training_config["epochs"] = args.epochs  # Override with command line argument
            
            # Save optimization results
            with (output_dir / "optimization_results.json").open("w") as f:
                json.dump({
                    "best_params": opt_result["best_params"],
                    "best_score": opt_result["best_score"],
                }, f, indent=2)
        else:
            # Use default configurations
            model_config = create_enhanced_model_config()
            training_config = create_enhanced_training_config()
            training_config["epochs"] = args.epochs
        
        # Save configurations
        with (output_dir / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)
        
        with (output_dir / "training_config.json").open("w") as f:
            json.dump(training_config, f, indent=2)
        
        # Step 3: Train the model
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            device=device,
            experiment_name="enhanced_cnn_lstm_training",
            enable_mlflow=not args.no_mlflow,
            enable_tensorboard=not args.no_tensorboard,
        )
        
        model_save_path = output_dir / "best_model.pth"
        
        training_summary = trainer.train_from_dataset(
            sequences=sequences,
            targets=targets,
            save_path=str(model_save_path),
        )
        
        # Step 4: Save training summary and plots
        with (output_dir / "training_summary.json").open("w") as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        trainer.plot_training_history(
            save_path=str(output_dir / "training_history.png"),
        )
        
        # Final summary
        final_metrics = training_summary["final_metrics"]
        total_epochs = training_summary["total_epochs"]
        training_time = training_summary["training_time"]
        
        logger.info("\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Training Summary:")
        logger.info(f"  🔄 Total epochs: {total_epochs}")
        logger.info(f"  🎯 Best validation loss: {training_summary['best_val_loss']:.6f}")
        logger.info(f"  📏 Final MAE: {final_metrics['mae']:.6f}")
        logger.info(f"  📈 Final RMSE: {final_metrics['rmse']:.6f}")
        logger.info(f"  🔗 Final R²: {final_metrics['r2']:.4f}")
        logger.info(f"  ⏱️ Total training time: {training_time / 60:.1f} minutes")
        logger.info("\n💾 Output Files:")
        logger.info(f"  🤖 Model: {model_save_path}")
        logger.info(f"  📋 Summary: {output_dir / 'training_summary.json'}")
        logger.info(f"  📊 Config: {output_dir / 'model_config.json'}")
        logger.info(f"  📈 Plot: {output_dir / 'training_history.png'}")
        if args.optimize_hyperparams:
            logger.info(f"  🔍 Optimization: {output_dir / 'optimization_results.json'}")
        logger.info("=" * 60)
        
        return training_summary
        
    except Exception as e:
        logger.exception(f"❌ Enhanced training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()