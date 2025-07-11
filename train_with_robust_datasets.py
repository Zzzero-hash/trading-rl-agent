#!/usr/bin/env python3
"""
Comprehensive CNN+LSTM Training with Robust Datasets

This script demonstrates how to train CNN+LSTM models on diverse,
robust datasets with comprehensive monitoring and evaluation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from train_cnn_lstm import CNNLSTMTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_advanced_model_configs():
    """Create different model configurations for experimentation."""

    return {
        "standard": {
            "cnn_filters": [64, 128, 256],
            "cnn_kernel_sizes": [3, 3, 3],
            "lstm_units": 128,
            "dropout": 0.2,
            "bidirectional": False,
        },
        "deep": {
            "cnn_filters": [32, 64, 128, 256, 512],
            "cnn_kernel_sizes": [3, 3, 3, 3, 3],
            "lstm_units": 256,
            "dropout": 0.3,
            "bidirectional": True,
        },
        "lightweight": {
            "cnn_filters": [32, 64, 128],
            "cnn_kernel_sizes": [3, 3, 3],
            "lstm_units": 64,
            "dropout": 0.1,
            "bidirectional": False,
        },
        "wide": {
            "cnn_filters": [128, 256, 512],
            "cnn_kernel_sizes": [5, 5, 5],
            "lstm_units": 512,
            "dropout": 0.2,
            "bidirectional": True,
        },
    }


def create_advanced_training_configs():
    """Create different training configurations."""

    return {
        "standard": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "val_split": 0.2,
            "early_stopping_patience": 15,
        },
        "aggressive": {
            "epochs": 200,
            "batch_size": 32,
            "learning_rate": 0.002,
            "weight_decay": 1e-4,
            "val_split": 0.15,
            "early_stopping_patience": 25,
        },
        "conservative": {
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 0.0005,
            "weight_decay": 1e-6,
            "val_split": 0.25,
            "early_stopping_patience": 10,
        },
    }


def train_on_dataset(dataset_path, model_config, training_config, output_dir, device="auto"):
    """Train a model on a specific dataset."""

    print(f"\n🚀 Training on dataset: {dataset_path}")
    print(f"📁 Output directory: {output_dir}")

    # Load dataset
    dataset_dir = Path(dataset_path)
    sequences = np.load(dataset_dir / "sequences.npy")
    targets = np.load(dataset_dir / "targets.npy")

    # Load metadata
    with (dataset_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    print(f"📊 Dataset shape: {sequences.shape}")
    print(f"🎯 Target shape: {targets.shape}")
    print(f"📋 Features: {sequences.shape[-1]}")

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create trainer
    trainer = CNNLSTMTrainer(model_config=model_config, training_config=training_config, device=device)

    # Train model
    training_summary = trainer.train_from_dataset(
        sequences=sequences, targets=targets, save_path=str(output_dir / "best_model.pth")
    )

    # Save training results
    results = {
        "dataset_path": str(dataset_path),
        "dataset_metadata": metadata,
        "model_config": model_config,
        "training_config": training_config,
        "training_summary": training_summary,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    with (output_dir / "training_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save training plots
    trainer.plot_training_history(save_path=str(output_dir / "training_history.png"))

    return results


def compare_models(results_dict):
    """Compare performance across different model configurations."""

    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    comparison_data = []

    for model_name, result in results_dict.items():
        if result.get("status") == "success":
            summary = result["training_summary"]
            final_metrics = summary["final_metrics"]

            comparison_data.append(
                {
                    "model": model_name,
                    "best_val_loss": summary["best_val_loss"],
                    "final_mae": final_metrics["mae"],
                    "final_rmse": final_metrics["rmse"],
                    "final_correlation": final_metrics["correlation"],
                    "total_epochs": summary["total_epochs"],
                    "training_time": summary.get("training_time", 0),
                }
            )

    if not comparison_data:
        print("❌ No successful training results to compare")
        return None

    # Create comparison DataFrame
    import pandas as pd

    df = pd.DataFrame(comparison_data)

    print("\n📈 Performance Comparison:")
    print(df.to_string(index=False, float_format="%.6f"))

    # Find best models
    best_loss = df.loc[df["best_val_loss"].idxmin()]
    best_mae = df.loc[df["final_mae"].idxmin()]
    best_correlation = df.loc[df["final_correlation"].idxmax()]

    print("\n🏆 Best Models:")
    print(f"  Lowest Validation Loss: {best_loss['model']} ({best_loss['best_val_loss']:.6f})")
    print(f"  Lowest MAE: {best_mae['model']} ({best_mae['final_mae']:.6f})")
    print(f"  Highest Correlation: {best_correlation['model']} ({best_correlation['final_correlation']:.4f})")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Validation loss comparison
    axes[0, 0].bar(df["model"], df["best_val_loss"])
    axes[0, 0].set_title("Best Validation Loss (Lower is Better)")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # MAE comparison
    axes[0, 1].bar(df["model"], df["final_mae"])
    axes[0, 1].set_title("Final MAE (Lower is Better)")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Correlation comparison
    axes[1, 0].bar(df["model"], df["final_correlation"])
    axes[1, 0].set_title("Final Correlation (Higher is Better)")
    axes[1, 0].set_ylabel("Correlation")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Training epochs
    axes[1, 1].bar(df["model"], df["total_epochs"])
    axes[1, 1].set_title("Training Epochs")
    axes[1, 1].set_ylabel("Epochs")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    print("\n📊 Comparison plot saved to: model_comparison.png")

    return df


def main():
    """Main training pipeline with multiple configurations."""

    parser = argparse.ArgumentParser(description="Train CNN+LSTM with Robust Datasets")
    parser.add_argument(
        "--dataset-path", default="outputs/demo_training/dataset/20250711_003545", help="Path to dataset directory"
    )
    parser.add_argument("--output-dir", default="outputs/robust_training", help="Output directory for training results")
    parser.add_argument(
        "--model-config",
        choices=["standard", "deep", "lightweight", "wide"],
        default="standard",
        help="Model configuration to use",
    )
    parser.add_argument(
        "--training-config",
        choices=["standard", "aggressive", "conservative"],
        default="standard",
        help="Training configuration to use",
    )
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--compare-models", action="store_true", help="Train multiple model configurations and compare")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 ROBUST CNN+LSTM TRAINING PIPELINE")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model config: {args.model_config}")
    print(f"⚙️ Training config: {args.training_config}")
    print(f"🖥️ Device: {args.device}")

    if args.compare_models:
        # Train multiple model configurations
        print("\n🔄 Training multiple model configurations...")

        model_configs = create_advanced_model_configs()
        training_configs = create_advanced_training_configs()

        results = {}

        for model_name, model_config in model_configs.items():
            for training_name, training_config in training_configs.items():
                config_name = f"{model_name}_{training_name}"

                print(f"\n📊 Training {config_name}...")

                try:
                    config_output_dir = output_dir / config_name
                    config_output_dir.mkdir(parents=True, exist_ok=True)

                    result = train_on_dataset(
                        dataset_path=args.dataset_path,
                        model_config=model_config,
                        training_config=training_config,
                        output_dir=config_output_dir,
                        device=args.device,
                    )

                    results[config_name] = {"status": "success", "training_summary": result["training_summary"]}

                    print(f"✅ {config_name} training completed successfully!")

                except Exception as e:
                    print(f"❌ {config_name} training failed: {e}")
                    results[config_name] = {"status": "failed", "error": str(e)}

        # Compare models
        comparison_df = compare_models(results)

        # Save comparison results
        if comparison_df is not None:
            comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
            print(f"📊 Comparison results saved to: {output_dir / 'model_comparison.csv'}")

        # Save all results
        with (output_dir / "all_training_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    else:
        # Train single configuration
        model_configs = create_advanced_model_configs()
        training_configs = create_advanced_training_configs()

        model_config = model_configs[args.model_config]
        training_config = training_configs[args.training_config]

        result = train_on_dataset(
            dataset_path=args.dataset_path,
            model_config=model_config,
            training_config=training_config,
            output_dir=output_dir,
            device=args.device,
        )

        print("\n🎉 Training completed successfully!")
        print(f"📊 Final correlation: {result['training_summary']['final_metrics']['correlation']:.4f}")
        print(f"📏 Final MAE: {result['training_summary']['final_metrics']['mae']:.6f}")

    print(f"\n💾 All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
