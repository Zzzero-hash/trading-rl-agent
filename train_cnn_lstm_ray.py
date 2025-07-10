"""
Ray-Enabled CNN+LSTM Training Pipeline with Distributed Optimization

This script leverages Ray for distributed training, hyperparameter optimization,
and resource management to maximize training efficiency and performance.
"""

import argparse
import json
import logging
import os

# Add src to Python path
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.data.robust_dataset_builder import (
    DatasetConfig,
    RobustDatasetBuilder,
)
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

logger = logging.getLogger(__name__)


class RayTrainer:
    """Ray-enabled trainer for distributed CNN+LSTM training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def train_model(self, trial_config: dict[str, Any], checkpoint_dir: str | None = None):
        """Train a single model configuration with Ray."""

        # Load data (should be shared across all trials)
        data_path = self.config["data_path"]
        sequences = np.load(f"{data_path}/sequences.npy")
        targets = np.load(f"{data_path}/targets.npy")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42, shuffle=False
        )

        # Create data loaders
        batch_size = trial_config.get("batch_size", 64)
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # Initialize model with trial config
        input_dim = sequences.shape[-1]
        model_config = {
            "cnn_filters": trial_config["cnn_filters"],
            "cnn_kernel_sizes": [3] * len(trial_config["cnn_filters"]),
            "lstm_units": trial_config["lstm_units"],
            "dropout": trial_config["dropout"],
        }

        model = CNNLSTMModel(input_dim=input_dim, config=model_config)
        model.to(self.device)

        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=trial_config["learning_rate"], weight_decay=trial_config["weight_decay"]
        )

        criterion = nn.MSELoss()
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        # Load checkpoint if resuming
        start_epoch = 0
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 10

        for epoch in range(start_epoch, trial_config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            predictions, targets_list = [], []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

                    predictions.extend(output.cpu().numpy().flatten())
                    targets_list.extend(target.cpu().numpy().flatten())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # Calculate metrics
            predictions = np.array(predictions)
            targets_array = np.array(targets_list)

            mae = mean_absolute_error(targets_array, predictions)
            rmse = np.sqrt(mean_squared_error(targets_array, predictions))
            correlation = np.corrcoef(targets_array, predictions)[0, 1] if len(targets_array) > 1 else 0.0

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Report metrics to Ray Tune
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation,
                "epoch": epoch,
            }

            # Save checkpoint
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
            }

            with Checkpoint.from_dict(checkpoint_data) as checkpoint:
                tune.report(metrics, checkpoint=checkpoint)

            # Early stopping
            if patience_counter >= max_patience:
                break

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_ray_optimization(
    data_path: str,
    output_dir: str,
    num_samples: int = 20,
    max_epochs: int = 100,
    gpus_per_trial: float = 0.5,
    cpus_per_trial: float = 2.0,
):
    """Run distributed hyperparameter optimization with Ray Tune."""

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            # Configure based on available resources
            num_cpus=os.cpu_count(),
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            include_dashboard=True,
            dashboard_host="localhost",  # More secure than 0.0.0.0
            dashboard_port=8265,
            log_to_driver=True,
        )

    logger.info(f"🚀 Ray initialized with {ray.cluster_resources()}")

    # Define search space
    search_space = {
        # Model architecture
        "cnn_filters": tune.choice(
            [
                [32, 64],
                [64, 128],
                [64, 128, 256],
                [32, 64, 128, 256],
            ]
        ),
        "lstm_units": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        # Training parameters
        "batch_size": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "epochs": max_epochs,
    }

    # Setup optimization algorithm
    optuna_search = OptunaSearch(
        metric="val_loss",
        mode="min",
        points_to_evaluate=[
            # Provide good starting points
            {
                "cnn_filters": [64, 128],
                "lstm_units": 128,
                "dropout": 0.2,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
            }
        ],
    )

    # Setup scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=max_epochs,
        grace_period=10,
        reduction_factor=2,
    )

    # Create trainer
    trainer = RayTrainer({"data_path": data_path})

    # Setup Ray Tune
    trainable = tune.with_resources(trainer.train_model, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial})

    # Run optimization
    logger.info(f"🎯 Starting Ray Tune optimization with {num_samples} trials...")

    analysis = tune.run(
        trainable,
        config=search_space,
        search_alg=optuna_search,
        scheduler=scheduler,
        num_samples=num_samples,
        storage_path=str(Path(output_dir) / "ray_results"),
        name="cnn_lstm_optimization",
        resume="AUTO",
        checkpoint_freq=5,
        keep_checkpoints_num=3,
        verbose=1,
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "val_loss", "mae", "correlation", "epoch"],
            sort_by_metric=True,
        ),
    )

    # Get best results
    best_trial = analysis.get_best_trial("val_loss", "min")
    best_config = best_trial.config
    best_metrics = best_trial.last_result

    logger.info("🏆 Optimization completed!")
    logger.info(f"📊 Best trial: {best_trial.trial_id}")
    logger.info(f"🎯 Best val_loss: {best_metrics['val_loss']:.6f}")
    logger.info(f"📈 Best correlation: {best_metrics['correlation']:.4f}")
    logger.info(f"📋 Best config: {best_config}")

    # Save results
    results = {
        "best_trial_id": best_trial.trial_id,
        "best_config": best_config,
        "best_metrics": best_metrics,
        "optimization_summary": {
            "total_trials": len(analysis.trials),
            "optimization_time": analysis.total_time,
            "search_space": search_space,
        },
    }

    results_path = Path(output_dir) / "ray_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create visualization
    create_optimization_plots(analysis, output_dir)

    return analysis, best_config


def create_optimization_plots(analysis, output_dir: str):
    """Create comprehensive optimization visualization."""

    # Get results dataframe
    df = analysis.results_df

    if df.empty:
        logger.warning("No results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training progress
    axes[0, 0].plot(df["train_loss"], label="Train Loss", alpha=0.7)
    axes[0, 0].plot(df["val_loss"], label="Val Loss", alpha=0.7)
    axes[0, 0].set_title("Training Progress")
    axes[0, 0].set_xlabel("Trial")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Correlation vs Val Loss
    axes[0, 1].scatter(df["val_loss"], df["correlation"], alpha=0.7)
    axes[0, 1].set_title("Correlation vs Validation Loss")
    axes[0, 1].set_xlabel("Validation Loss")
    axes[0, 1].set_ylabel("Correlation")
    axes[0, 1].grid(True)

    # Learning rate impact
    if "config/learning_rate" in df.columns:
        axes[1, 0].scatter(df["config/learning_rate"], df["val_loss"], alpha=0.7)
        axes[1, 0].set_title("Learning Rate Impact")
        axes[1, 0].set_xlabel("Learning Rate")
        axes[1, 0].set_ylabel("Validation Loss")
        axes[1, 0].set_xscale("log")
        axes[1, 0].grid(True)

    # LSTM units impact
    if "config/lstm_units" in df.columns:
        axes[1, 1].scatter(df["config/lstm_units"], df["val_loss"], alpha=0.7)
        axes[1, 1].set_title("LSTM Units Impact")
        axes[1, 1].set_xlabel("LSTM Units")
        axes[1, 1].set_ylabel("Validation Loss")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ray_optimization_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"📊 Optimization plots saved to {output_dir}")


def train_best_model(best_config: dict[str, Any], data_path: str, output_dir: str):
    """Train the final model with best configuration."""

    logger.info("🎯 Training final model with best configuration...")

    # Load data
    sequences = np.load(f"{data_path}/sequences.npy")
    targets = np.load(f"{data_path}/targets.npy")

    # Use the original trainer with best config
    from train_cnn_lstm import CNNLSTMTrainer, create_model_config, create_training_config

    # Override configs with best parameters
    model_config = create_model_config()
    model_config.update(
        {
            "cnn_filters": best_config["cnn_filters"],
            "lstm_units": best_config["lstm_units"],
            "dropout": best_config["dropout"],
        }
    )

    training_config = create_training_config()
    training_config.update(
        {
            "batch_size": best_config["batch_size"],
            "learning_rate": best_config["learning_rate"],
            "weight_decay": best_config["weight_decay"],
            "epochs": 200,  # Full training
        }
    )

    # Train final model
    trainer = CNNLSTMTrainer(
        model_config=model_config,
        training_config=training_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_save_path = Path(output_dir) / "best_ray_model.pth"
    training_summary = trainer.train_from_dataset(
        sequences=sequences,
        targets=targets,
        save_path=str(model_save_path),
    )

    # Save final results
    final_results = {
        "best_config": best_config,
        "model_config": model_config,
        "training_config": training_config,
        "training_summary": training_summary,
        "model_path": str(model_save_path),
    }

    with open(Path(output_dir) / "final_ray_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    return training_summary


def main():
    """Main Ray-enabled training pipeline."""

    parser = argparse.ArgumentParser(description="Ray-Enabled CNN+LSTM Training")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT"])
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--output-dir", default="outputs/ray_cnn_lstm")
    parser.add_argument("--load-dataset", help="Path to existing dataset")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--gpus-per-trial", type=float, default=0.5, help="GPUs per trial")
    parser.add_argument("--cpus-per-trial", type=float, default=2.0, help="CPUs per trial")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization, use default config")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting Ray-enabled CNN+LSTM pipeline...")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🎯 Symbols: {args.symbols}")
    logger.info(f"🔬 Optimization trials: {args.num_trials}")

    try:
        # Step 1: Build or load dataset
        if args.load_dataset:
            logger.info(f"📂 Loading existing dataset from {args.load_dataset}")
            data_path = args.load_dataset
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

            # Attempt to reuse existing dataset if present
            _, _, dataset_info = RobustDatasetBuilder.load_or_build(dataset_config)
            data_path = dataset_info.get("source_directory", dataset_info.get("output_directory"))

            load_msg = "loaded" if dataset_info.get("loaded") else "built"
            logger.info(f"✅ Dataset {load_msg}: {dataset_info}")

        # Step 2: Ray optimization or use default config
        if not args.skip_optimization:
            logger.info("🎯 Starting Ray Tune optimization...")
            analysis, best_config = run_ray_optimization(
                data_path=str(data_path),
                output_dir=str(output_dir),
                num_samples=args.num_trials,
                max_epochs=args.max_epochs,
                gpus_per_trial=args.gpus_per_trial,
                cpus_per_trial=args.cpus_per_trial,
            )
        else:
            logger.info("⚡ Skipping optimization, using default config...")
            best_config = {
                "cnn_filters": [64, 128],
                "lstm_units": 128,
                "dropout": 0.2,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
            }

        # Step 3: Train final model with best configuration
        logger.info("🏆 Training final model with best configuration...")
        final_results = train_best_model(best_config, str(data_path), str(output_dir))

        logger.info("🎉 Ray-enabled training pipeline completed successfully!")
        logger.info(f"📊 Best validation loss: {final_results['best_val_loss']:.6f}")
        logger.info(f"📈 Final correlation: {final_results['final_metrics']['correlation']:.4f}")
        logger.info(f"💾 Results saved to: {output_dir}")

        return final_results

    except Exception as e:
        logger.exception(f"❌ Ray training pipeline failed: {e}")
        raise
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("🔄 Ray shutdown completed")


if __name__ == "__main__":
    main()
