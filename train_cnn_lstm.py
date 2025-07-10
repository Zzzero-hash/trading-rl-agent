"""
CNN+LSTM Training Pipeline with Robust Dataset Integration

This script provides end-to-end training for CNN+LSTM models using the robust
dataset builder, with comprehensive monitoring and evaluation capabilities.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.data.robust_dataset_builder import (
    DatasetConfig,
    RobustDatasetBuilder,
)
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

logger = logging.getLogger(__name__)


class CNNLSTMTrainer:
    """Comprehensive trainer for CNN+LSTM models with robust dataset integration."""

    def __init__(
        self,
        model_config: dict,
        training_config: dict,
        device: str | None = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def train_from_dataset(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        save_path: str | None = None,
    ) -> dict:
        """Train the CNN+LSTM model from prepared sequences."""

        logger.info("🚀 Starting CNN+LSTM training...")

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
        input_dim = sequences.shape[-1]  # Number of features
        self.model = CNNLSTMModel(input_dim=input_dim, config=self.model_config)
        self.model.to(self.device)

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
            patience=5,
            factor=0.5,
            verbose=True,
        )

        # Training loop
        best_val_loss = float("inf")
        patience = self.training_config.get("early_stopping_patience", 10)
        patience_counter = 0

        epochs = self.training_config.get("epochs", 100)

        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["metrics"].append(val_metrics)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self._save_checkpoint(save_path, epoch, val_loss)
            else:
                patience_counter += 1

            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, Val MAE: {val_metrics['mae']:.6f}",
                )

            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final evaluation
        final_metrics = self._final_evaluation(X_val, y_val)

        training_summary = {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "final_metrics": final_metrics,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }

        logger.info("✅ Training completed successfully!")
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
    ) -> float:
        """Train for one epoch."""

        self.model.train()
        total_loss = 0.0

        for _batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        """Validate for one epoch."""

        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader)

        # Calculate additional metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        metrics = {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "correlation": (np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0),
        }

        return avg_loss, metrics

    def _final_evaluation(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Perform final comprehensive evaluation."""

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        targets = y_val.flatten()

        # Comprehensive metrics
        return {
            "mse": mean_squared_error(targets, predictions),
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "correlation": (np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0),
            "std_predictions": np.std(predictions),
            "std_targets": np.std(targets),
            "mean_predictions": np.mean(predictions),
            "mean_targets": np.mean(targets),
        }

    def _save_checkpoint(self, save_path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "history": self.history,
        }

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str, input_dim: int):
        """Load model from checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = CNNLSTMModel(
            input_dim=input_dim,
            config=checkpoint["model_config"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model_config = checkpoint["model_config"]
        self.training_config = checkpoint["training_config"]
        self.history = checkpoint["history"]

        logger.info(f"Model loaded from {checkpoint_path}")

    def plot_training_history(self, save_path: str | None = None):
        """Plot training history."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE over time
        mae_values = [m["mae"] for m in self.history["metrics"]]
        axes[0, 1].plot(mae_values)
        axes[0, 1].set_title("Validation MAE")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].grid(True)

        # Correlation over time
        corr_values = [m["correlation"] for m in self.history["metrics"]]
        axes[1, 0].plot(corr_values)
        axes[1, 0].set_title("Validation Correlation")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Correlation")
        axes[1, 0].grid(True)

        # RMSE over time
        rmse_values = [m["rmse"] for m in self.history["metrics"]]
        axes[1, 1].plot(rmse_values)
        axes[1, 1].set_title("Validation RMSE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()


def create_model_config() -> dict:
    """Create CNN+LSTM model configuration."""
    return {
        "cnn_filters": [64, 128, 256],
        "cnn_kernel_sizes": [3, 3, 3],
        "lstm_units": 128,
        "dropout": 0.2,
    }


def create_training_config() -> dict:
    """Create training configuration."""
    return {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "val_split": 0.2,
        "early_stopping_patience": 15,
    }


def main():
    """Main training pipeline."""

    parser = argparse.ArgumentParser(description="Train CNN+LSTM with Robust Dataset")
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
        default="outputs/cnn_lstm_training",
        help="Output directory",
    )
    parser.add_argument("--load-dataset", help="Path to existing dataset directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting CNN+LSTM training pipeline...")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Step 1: Build or load dataset
        if args.load_dataset:
            logger.info(f"Loading existing dataset from {args.load_dataset}")
            dataset_dir = Path(args.load_dataset)
            sequences = np.load(dataset_dir / "sequences.npy")
            targets = np.load(dataset_dir / "targets.npy")

            with Path(dataset_dir / "metadata.json").open(dataset_dir / "metadata.json") as f:
                _ = json.load(f)  # Load for validation but don't use
        else:
            logger.info("Building new dataset...")

            # Create dataset configuration
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
                sentiment_features=False,  # Disable for faster processing
                market_regime_features=True,
                output_dir=str(output_dir / "dataset"),
            )

            # Build dataset
            builder = RobustDatasetBuilder(dataset_config)
            sequences, targets, dataset_info = builder.build_dataset()

            logger.info(f"Dataset built: {dataset_info}")

        # Step 2: Setup model and training configurations
        model_config = create_model_config()
        training_config = create_training_config()

        # Save configurations
        with Path(output_dir / "model_config.json").open(output_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        with Path(output_dir / "training_config.json").open(output_dir / "training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)

        # Step 3: Train the model
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        trainer = CNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            device=device,
        )

        model_save_path = output_dir / "best_model.pth"

        training_summary = trainer.train_from_dataset(
            sequences=sequences,
            targets=targets,
            save_path=str(model_save_path),
        )

        # Step 4: Save training summary and plots
        with Path(output_dir / "training_summary.json").open(output_dir / "training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2, default=str)

        trainer.plot_training_history(
            save_path=str(output_dir / "training_history.png"),
        )

        # Step 5: Create real-time inference example
        if not args.load_dataset:
            # Save dataset path for real-time inference
            rt_config = {
                "dataset_version_dir": str(builder.output_dir),
                "model_checkpoint": str(model_save_path),
                "model_config": model_config,
                "usage_example": {
                    "load_model": f"trainer.load_checkpoint('{model_save_path}', input_dim={sequences.shape[-1]})",
                    "load_realtime_processor": f"rt_loader = RealTimeDatasetLoader('{builder.output_dir}')",
                    "process_new_data": "processed_seq = rt_loader.process_realtime_data(new_market_data)",
                },
            }

            with Path(output_dir / "realtime_inference_config.json").open(
                output_dir / "realtime_inference_config.json", "w"
            ) as f:
                json.dump(rt_config, f, indent=2)

        logger.info("✅ Training pipeline completed successfully!")
        logger.info(f"📊 Best validation loss: {training_summary['best_val_loss']:.6f}")
        logger.info(
            f"📈 Final correlation: {training_summary['final_metrics']['correlation']:.4f}",
        )
        logger.info(f"💾 Model saved to: {model_save_path}")
        logger.info(f"📋 Summary saved to: {output_dir}")

        return training_summary

    except Exception as e:
        logger.exception(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
