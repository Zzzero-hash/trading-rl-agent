#!/usr/bin/env python3
"""
Demonstration script for automatic visual monitoring during CNN-LSTM training.

This script shows how the visual monitor automatically starts when training begins
and provides real-time feedback on training progress and Optuna optimization.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from train_cnn_lstm_enhanced import (
    EnhancedCNNLSTMTrainer,
    HyperparameterOptimizer,
    create_enhanced_model_config,
    create_enhanced_training_config,
    load_and_preprocess_csv_data,
)


def create_demo_data(n_samples: int = 1000, n_features: int = 5, sequence_length: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic time series data for demonstration.

    Args:
        n_samples: Number of data samples
        n_features: Number of input features
        sequence_length: Length of input sequences

    Returns:
        Tuple of (sequences, targets)
    """
    print("🔧 Creating synthetic demo data...")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Sequence length: {sequence_length}")

    # Generate synthetic time series with trends and patterns
    np.random.seed(42)

    # Create base time series
    time_steps = n_samples + sequence_length
    time = np.arange(time_steps)

    # Generate features with different patterns
    features = np.zeros((time_steps, n_features))

    for i in range(n_features):
        # Different patterns for each feature
        if i == 0:  # Trend + noise
            features[:, i] = 0.01 * time + 0.1 * np.sin(0.1 * time) + 0.05 * np.random.randn(time_steps)
        elif i == 1:  # Seasonal pattern
            features[:, i] = np.sin(0.05 * time) + 0.5 * np.cos(0.02 * time) + 0.03 * np.random.randn(time_steps)
        elif i == 2:  # Random walk
            features[:, i] = np.cumsum(0.02 * np.random.randn(time_steps))
        elif i == 3:  # Exponential decay with noise
            features[:, i] = np.exp(-0.001 * time) + 0.1 * np.random.randn(time_steps)
        else:  # Mixed pattern
            features[:, i] = (0.5 * np.sin(0.03 * time) +
                            0.3 * np.cos(0.07 * time) +
                            0.1 * np.random.randn(time_steps))

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create target as a function of features with some lag
    target = (0.3 * features_scaled[:, 0] +
              0.2 * features_scaled[:, 1] +
              0.15 * features_scaled[:, 2] +
              0.1 * np.roll(features_scaled[:, 0], 1) +  # Lag effect
              0.05 * np.random.randn(time_steps))

    # Create sequences
    sequences = []
    targets = []

    for i in range(n_samples):
        seq = features_scaled[i:i + sequence_length]
        tgt = target[i + sequence_length]
        sequences.append(seq)
        targets.append(tgt)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print("✅ Demo data created successfully")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Targets shape: {targets.shape}")

    return sequences, targets


def demo_basic_training_with_visual_monitor() -> None:
    """Demonstrate basic training with automatic visual monitoring."""
    print("\n" + "="*80)
    print("🎬 DEMO: Basic CNN-LSTM Training with Visual Monitor")
    print("="*80)

    # Create demo data
    sequences, targets = create_demo_data(n_samples=800, n_features=5, sequence_length=30)

    # Create model and training configurations
    model_config = create_enhanced_model_config(
        input_dim=5,
        cnn_filters=[32, 64],
        cnn_kernel_sizes=[3, 3],
        lstm_units=64,
        lstm_layers=2,
        dropout_rate=0.3,
        use_attention=True
    )

    training_config = create_enhanced_training_config(
        learning_rate=0.001,
        batch_size=32,
        epochs=20,  # Reduced for demo
        early_stopping_patience=5
    )

    print("\n📋 Configuration:")
    print(f"   Model: CNN({model_config['cnn_filters']}) -> LSTM({model_config['lstm_units']})")
    print(f"   Training: {training_config['epochs']} epochs, LR={training_config['learning_rate']}")
    print("   Visual Monitor: ✅ Enabled (auto-start)")

    # Create trainer with visual monitoring enabled (default)
    trainer = EnhancedCNNLSTMTrainer(
        model_config=model_config,
        training_config=training_config,
        enable_visual_monitor=True,  # This is the key feature!
        enable_mlflow=False,
        enable_tensorboard=False
    )

    print("\n🚀 Starting training with automatic visual monitoring...")
    print("   📊 Visual monitor will start automatically!")
    print("   📈 Check the generated plots and dashboard files")

    # Train the model - visual monitor starts automatically!
    results = trainer.train_from_dataset(sequences, targets)

    print("\n✅ Training completed!")
    print(f"   Best validation loss: {results['best_val_loss']:.4f}")
    print(f"   Total epochs: {results['total_epochs']}")
    print(f"   Training time: {results['training_time']:.2f}s")
    print("   📁 Visual assets saved to: ./training_visualizations/")

    # Results displayed above, no return needed for demo


def demo_optuna_optimization_with_visual_monitor() -> None:
    """Demonstrate Optuna optimization with visual trial monitoring."""
    print("\n" + "="*80)
    print("🔬 DEMO: Optuna Hyperparameter Optimization with Visual Monitor")
    print("="*80)

    # Create demo data
    sequences, targets = create_demo_data(n_samples=600, n_features=5, sequence_length=40)

    print("\n📋 Optimization Setup:")
    print("   Trials: 8 (reduced for demo)")
    print("   Visual Monitor: ✅ Enabled (auto-start)")
    print("   Search space: CNN architecture, LSTM units, learning rate, batch size")

    # Create hyperparameter optimizer with visual monitoring
    optimizer = HyperparameterOptimizer(
        sequences=sequences,
        targets=targets,
        n_trials=8,  # Reduced for demo
        enable_visual_monitor=True  # This enables Optuna trial visualization!
    )

    print("\n🔍 Starting Optuna optimization with automatic visual monitoring...")
    print("   📊 Trial progress will be visualized in real-time!")
    print("   🎯 Hyperparameter relationships will be analyzed")

    # Run optimization - visual monitor tracks trials automatically!
    results = optimizer.optimize()

    print("\n🎯 Optimization completed!")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Total trials: {len(results['study'].trials)}")
    print("   📁 Optuna visualizations saved to: ./training_visualizations/")

    # Display best parameters
    print("\n🏆 Best parameters found:")
    for param, value in results["best_params"].items():
        print(f"   {param}: {value}")

    # Results displayed above, no return needed for demo


def demo_csv_data_training() -> None:
    """Demonstrate training with CSV data and visual monitoring."""
    print("\n" + "="*80)
    print("📊 DEMO: CSV Data Training with Visual Monitor")
    print("="*80)

    # Create a sample CSV file for demonstration
    demo_csv_path = Path("./demo_trading_data.csv")

    if not demo_csv_path.exists():
        print("🔧 Creating sample CSV data...")

        # Generate sample trading data
        np.random.seed(42)
        n_days = 1000
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

        # Simulate price movements
        price = 100.0
        prices = [price]

        for _ in range(n_days - 1):
            change = np.random.normal(0, 0.02) * price  # 2% daily volatility
            price += change
            prices.append(price)

        # Create OHLCV data
        data = {
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": [np.random.randint(1000000, 10000000) for _ in prices]
        }

        # Add technical indicators
        close_prices = np.array(prices)
        data["sma_20"] = pd.Series(close_prices).rolling(20).mean().fillna(method="bfill")
        data["rsi"] = np.random.uniform(20, 80, n_days)  # Simplified RSI
        data["volatility"] = pd.Series(close_prices).rolling(20).std().fillna(method="bfill")

        df = pd.DataFrame(data)
        df.to_csv(demo_csv_path, index=False)
        print(f"   ✅ Sample CSV created: {demo_csv_path}")

    print(f"\n📈 Loading and preprocessing CSV data from {demo_csv_path}")

    # Load and preprocess the CSV data
    sequences, targets = load_and_preprocess_csv_data(
        csv_path=demo_csv_path,
        sequence_length=30,
        prediction_horizon=1,
        target_column="close",
        feature_columns=["open", "high", "low", "volume", "sma_20", "rsi", "volatility"]
    )

    # Create configurations
    model_config = create_enhanced_model_config(
        input_dim=sequences.shape[-1],  # Automatically detected from data
        cnn_filters=[64, 128],
        lstm_units=128,
        dropout_rate=0.2
    )

    training_config = create_enhanced_training_config(
        learning_rate=0.0005,
        batch_size=64,
        epochs=15,
        early_stopping_patience=8
    )

    print("\n📋 CSV Training Configuration:")
    print(f"   Data shape: {sequences.shape}")
    print(f"   Features: {sequences.shape[-1]}")
    print(f"   Sequence length: {sequences.shape[1]}")
    print("   Visual Monitor: ✅ Auto-enabled")

    # Create trainer
    trainer = EnhancedCNNLSTMTrainer(
        model_config=model_config,
        training_config=training_config,
        enable_visual_monitor=True
    )

    print("\n🚀 Starting CSV data training with visual monitoring...")

    # Train with visual monitoring
    results = trainer.train_from_dataset(sequences, targets)

    print("\n✅ CSV training completed!")
    print(f"   Best validation loss: {results['best_val_loss']:.4f}")
    print("   📁 All visualizations saved automatically")

    # Clean up demo file
    if demo_csv_path.exists():
        demo_csv_path.unlink()
        print("   🗑️  Cleaned up demo CSV file")

    # Results displayed above, no return needed for demo


def run_all_demos() -> bool:
    """Run all demonstration scenarios."""
    print("🎭 CNN-LSTM Visual Monitoring Demonstration Suite")
    print("=" * 80)
    print("This demo shows how visual monitoring automatically starts")
    print("when you begin any CNN-LSTM training session!")
    print("=" * 80)

    try:
        # Demo 1: Basic training
        demo_basic_training_with_visual_monitor()

        # Demo 2: Optuna optimization
        demo_optuna_optimization_with_visual_monitor()

        # Demo 3: CSV data training
        demo_csv_data_training()

        print("\n🎉 All demonstrations completed successfully!")
        print("\n📁 Check the './training_visualizations/' directory for:")
        print("   • Real-time training dashboards (HTML files)")
        print("   • Optuna trial progress visualizations")
        print("   • Training summary plots and data")
        print("   • Interactive Plotly charts")

        print("\n✨ Key Features Demonstrated:")
        print("   ✅ Automatic visual monitor activation on training start")
        print("   ✅ Real-time training metrics visualization")
        print("   ✅ Optuna trial progress tracking")
        print("   ✅ Interactive Plotly dashboards")
        print("   ✅ Automatic saving of visualization assets")
        print("   ✅ Support for both synthetic and CSV data")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    """Run the demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="CNN-LSTM Visual Monitoring Demo")
    parser.add_argument(
        "--demo",
        choices=["basic", "optuna", "csv", "all"],
        default="all",
        help="Which demo to run"
    )

    args = parser.parse_args()

    if args.demo == "basic":
        demo_basic_training_with_visual_monitor()
    elif args.demo == "optuna":
        demo_optuna_optimization_with_visual_monitor()
    elif args.demo == "csv":
        demo_csv_data_training()
    else:
        run_all_demos()
