"""
CNN+LSTM Training Pipeline with Robust Dataset Integration

This script provides end-to-end training for CNN+LSTM models using the robust
dataset builder, with comprehensive monitoring and evaluation capabilities.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        logger.info("\n🧠 Initializing CNN+LSTM model...")
        logger.info(f"  📊 Input dimensions: {input_dim} features")

        self.model = CNNLSTMModel(input_dim=input_dim, config=self.model_config)
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"  🔢 Total parameters: {total_params:,}")
        logger.info(f"  🎯 Trainable parameters: {trainable_params:,}")
        logger.info(f"  📊 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

        # Model architecture summary
        logger.info(
            f"  🏗️ Architecture: CNN({self.model_config['cnn_filters']}) + LSTM({self.model_config['lstm_units']})"
        )

        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.get("learning_rate", 0.001),
            weight_decay=self.training_config.get("weight_decay", 1e-5),
        )

        criterion = nn.MSELoss()
        # Some older PyTorch versions (<1.2) do not accept the ``verbose`` kwarg. Build defensively.
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=5,
                factor=0.5,
                verbose=True,
            )
        except TypeError:
            # Fallback for torch versions that lack the argument
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=5,
                factor=0.5,
            )

        # Training loop
        best_val_loss = float("inf")
        patience = self.training_config.get("early_stopping_patience", 10)
        patience_counter = 0

        epochs = self.training_config.get("epochs", 100)
        start_time = time.time()

        logger.info(f"🎯 Starting training for {epochs} epochs...")
        logger.info(f"📊 Training set: {len(train_loader.dataset):,} samples")
        logger.info(f"📊 Validation set: {len(val_loader.dataset):,} samples")
        logger.info(f"📦 Batch size: {self.training_config.get('batch_size', 32)}")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training phase with monitoring
            logger.info(f"\n📈 Epoch {epoch + 1}/{epochs} - Training...")
            train_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)

            # Validation phase with monitoring
            logger.info(f"📊 Epoch {epoch + 1}/{epochs} - Validation...")
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if old_lr != new_lr:
                logger.info(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["metrics"].append(val_metrics)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            eta = (total_time / (epoch + 1)) * (epochs - epoch - 1)

            # Early stopping check
            improvement = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                improvement = "✅ BEST"
                if save_path:
                    self._save_checkpoint(save_path, epoch, val_loss)
                    logger.info("💾 Model checkpoint saved")
            else:
                patience_counter += 1
                improvement = f"❌ No improvement ({patience_counter}/{patience})"

            # Comprehensive epoch summary
            logger.info(f"\n📋 Epoch {epoch + 1}/{epochs} Summary:")
            logger.info(f"  🔥 Train Loss: {train_loss:.6f}")
            logger.info(f"  📊 Val Loss: {val_loss:.6f} {improvement}")
            logger.info(f"  📏 Val MAE: {val_metrics['mae']:.6f}")
            logger.info(f"  📈 Val RMSE: {val_metrics['rmse']:.6f}")
            logger.info(f"  🔗 Val Correlation: {val_metrics['correlation']:.4f}")
            logger.info(f"  ⏱️ Epoch Time: {epoch_time:.1f}s")
            logger.info(f"  🕐 Total Time: {total_time/60:.1f}m")
            logger.info(f"  ⏳ ETA: {eta/60:.1f}m")
            logger.info(f"  🎯 Learning Rate: {new_lr:.2e}")

            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"⏱️ Total training time: {total_time/60:.1f} minutes")
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
        epoch: int = 0,
    ) -> float:
        """Train for one epoch with detailed monitoring."""

        self.model.train()
        total_loss = 0.0
        batch_losses = []

        # Progress bar for batches
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            loss.backward()

            # Gradient clipping with monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.6f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.6f}",
                    "grad_norm": f"{grad_norm:.4f}",
                }
            )

            # Log detailed stats every 50 batches for debugging
            if batch_idx % 50 == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(
                    f"  Batch {batch_idx:3d}: Loss={batch_loss:.6f}, Avg={avg_loss:.6f}, GradNorm={grad_norm:.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        loss_std = np.std(batch_losses)

        logger.info(f"  🔥 Training completed: avg_loss={avg_loss:.6f}, std={loss_std:.6f}")
        return avg_loss

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        """Validate for one epoch with monitoring."""

        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        batch_losses = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation", leave=False)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)

                batch_loss = loss.item()
                total_loss += batch_loss
                batch_losses.append(batch_loss)

                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())

                # Update progress bar
                pbar.set_postfix({"val_loss": f"{batch_loss:.6f}", "avg_val_loss": f"{total_loss/(batch_idx+1):.6f}"})

        avg_loss = total_loss / len(dataloader)

        # Calculate additional metrics
        predictions = np.array(predictions)
        targets = np.array(targets)

        metrics = {
            "mae": mean_absolute_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "correlation": (np.corrcoef(targets, predictions)[0, 1] if len(targets) > 1 else 0.0),
        }

        logger.info(f"  📊 Validation completed: avg_loss={avg_loss:.6f}, samples={len(predictions)}")

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
        # Handle PyTorch 2.6+ weights_only behavior for sklearn scalers
        import torch.serialization
        from sklearn.preprocessing._data import RobustScaler

        torch.serialization.add_safe_globals([RobustScaler])

        try:
            # First try with weights_only=True (PyTorch 2.6+ default)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting CNN+LSTM training pipeline...")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎯 Symbols: {args.symbols}")
    logger.info(f"📅 Date range: {args.start_date} to {args.end_date}")
    logger.info(f"📏 Sequence length: {args.sequence_length}")
    logger.info(f"🔄 Epochs: {args.epochs}")
    logger.info(f"🖥️ GPU enabled: {args.gpu}")

    try:
        # Step 1: Build or load dataset
        if args.load_dataset:
            logger.info(f"Loading existing dataset from {args.load_dataset}")
            dataset_dir = Path(args.load_dataset)
            sequences = np.load(dataset_dir / "sequences.npy")
            targets = np.load(dataset_dir / "targets.npy")

            # Optional: validate metadata exists
            meta_path = dataset_dir / "metadata.json"
            if meta_path.exists():
                with meta_path.open() as f:
                    _ = json.load(f)
        else:
            # Prepare configuration for potential build
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

            # Attempt to load existing dataset or build a new one
            sequences, targets, dataset_info = RobustDatasetBuilder.load_or_build(dataset_config)

            load_msg = "loaded" if dataset_info.get("loaded") else "built"
            logger.info(f"Dataset {load_msg}: {dataset_info}")

        # Step 2: Setup model and training configurations
        model_config = create_model_config()
        training_config = create_training_config()
        training_config["epochs"] = args.epochs  # Override with command line argument

        # Save configurations
        with (output_dir / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)

        with (output_dir / "training_config.json").open("w") as f:
            json.dump(training_config, f, indent=2)

        # Step 3: Train the model
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        logger.info("\n🖥️ Device Configuration:")
        logger.info(f"  Device: {device}")
        if torch.cuda.is_available():
            logger.info("  CUDA available: Yes")
            logger.info(f"  GPU count: {torch.cuda.device_count()}")
            if device == "cuda":
                logger.info(f"  GPU name: {torch.cuda.get_device_name()}")
                logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
        else:
            logger.info("  CUDA available: No")

        logger.info("\n🧠 Model Configuration:")
        for key, value in model_config.items():
            logger.info(f"  {key}: {value}")

        logger.info("\n⚙️ Training Configuration:")
        for key, value in training_config.items():
            logger.info(f"  {key}: {value}")

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
        with (output_dir / "training_summary.json").open("w") as f:
            json.dump(training_summary, f, indent=2, default=str)

        trainer.plot_training_history(
            save_path=str(output_dir / "training_history.png"),
        )

        # Step 5: Create real-time inference example
        if not args.load_dataset:
            # Determine path to the dataset version we just used/built
            dataset_version_dir = dataset_info.get("source_directory", dataset_info.get("output_directory", ""))

            # Save dataset path for real-time inference
            rt_config = {
                "dataset_version_dir": dataset_version_dir,
                "model_checkpoint": str(model_save_path),
                "model_config": model_config,
                "usage_example": {
                    "load_model": f"trainer.load_checkpoint('{model_save_path}', input_dim={sequences.shape[-1]})",
                    "load_realtime_processor": f"rt_loader = RealTimeDatasetLoader('{dataset_version_dir}')",
                    "process_new_data": "processed_seq = rt_loader.process_realtime_data(new_market_data)",
                },
            }

            with (output_dir / "realtime_inference_config.json").open("w") as f:
                json.dump(rt_config, f, indent=2)

        # Final summary
        final_metrics = training_summary["final_metrics"]
        total_epochs = training_summary["total_epochs"]

        logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Training Summary:")
        logger.info(f"  🔄 Total epochs: {total_epochs}")
        logger.info(f"  🎯 Best validation loss: {training_summary['best_val_loss']:.6f}")
        logger.info(f"  📏 Final MAE: {final_metrics['mae']:.6f}")
        logger.info(f"  📈 Final RMSE: {final_metrics['rmse']:.6f}")
        logger.info(f"  🔗 Final correlation: {final_metrics['correlation']:.4f}")
        logger.info(f"  📊 Prediction std: {final_metrics['std_predictions']:.6f}")
        logger.info(f"  📊 Target std: {final_metrics['std_targets']:.6f}")
        logger.info("\n💾 Output Files:")
        logger.info(f"  🤖 Model: {model_save_path}")
        logger.info(f"  📋 Summary: {output_dir / 'training_summary.json'}")
        logger.info(f"  📊 Config: {output_dir / 'model_config.json'}")
        logger.info(f"  📈 Plot: {output_dir / 'training_history.png'}")
        logger.info("=" * 60)

        return training_summary

    except Exception as e:
        logger.exception(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
