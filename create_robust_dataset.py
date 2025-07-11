#!/usr/bin/env python3
"""
Robust Dataset Generation Script for CNN+LSTM Training

This script demonstrates how to create diverse, reproducible datasets
with various configurations for robust model training.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.data.robust_dataset_builder import DatasetConfig, RobustDatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_diverse_dataset_configs():
    """Create multiple dataset configurations for diversity."""

    # Base symbols for different market sectors
    tech_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    finance_symbols = ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK"]
    healthcare_symbols = ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR"]
    energy_symbols = ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC"]

    configs = {
        # 1. Tech-focused dataset (high volatility, growth)
        "tech_focused": DatasetConfig(
            symbols=tech_symbols,
            start_date="2020-01-01",
            end_date="2024-12-31",
            timeframe="1d",
            real_data_ratio=0.9,  # High real data ratio for tech
            min_samples_per_symbol=1500,
            sequence_length=60,
            prediction_horizon=1,
            overlap_ratio=0.8,
            technical_indicators=True,
            sentiment_features=True,  # Important for tech stocks
            market_regime_features=True,
            output_dir="data/robust_multi_asset_dataset/tech_focused",
            version_tag="tech_v1",
        ),
        # 2. Multi-sector balanced dataset
        "multi_sector": DatasetConfig(
            symbols=tech_symbols[:4] + finance_symbols[:2] + healthcare_symbols[:2],
            start_date="2020-01-01",
            end_date="2024-12-31",
            timeframe="1d",
            real_data_ratio=0.8,
            min_samples_per_symbol=1200,
            sequence_length=60,
            prediction_horizon=1,
            overlap_ratio=0.8,
            technical_indicators=True,
            sentiment_features=True,
            market_regime_features=True,
            output_dir="data/robust_multi_asset_dataset/multi_sector",
            version_tag="multisector_v1",
        ),
        # 3. High-frequency dataset (for more granular patterns)
        "high_freq": DatasetConfig(
            symbols=tech_symbols[:6],
            start_date="2023-01-01",
            end_date="2024-12-31",
            timeframe="1h",  # Hourly data
            real_data_ratio=0.7,
            min_samples_per_symbol=2000,
            sequence_length=120,  # Longer sequences for hourly data
            prediction_horizon=1,
            overlap_ratio=0.9,  # More overlap for high-frequency
            technical_indicators=True,
            sentiment_features=False,  # Disable for speed
            market_regime_features=True,
            output_dir="data/robust_multi_asset_dataset/high_freq",
            version_tag="highfreq_v1",
        ),
        # 4. Conservative dataset (stable stocks)
        "conservative": DatasetConfig(
            symbols=finance_symbols + healthcare_symbols[:4],
            start_date="2020-01-01",
            end_date="2024-12-31",
            timeframe="1d",
            real_data_ratio=0.95,  # Very high real data ratio
            min_samples_per_symbol=1000,
            sequence_length=40,  # Shorter sequences for stable stocks
            prediction_horizon=1,
            overlap_ratio=0.7,
            technical_indicators=True,
            sentiment_features=False,
            market_regime_features=True,
            output_dir="data/robust_multi_asset_dataset/conservative",
            version_tag="conservative_v1",
        ),
        # 5. Crisis-aware dataset (includes 2020 crash)
        "crisis_aware": DatasetConfig(
            symbols=tech_symbols[:4] + finance_symbols[:2] + ["SPY", "QQQ"],  # Include ETFs
            start_date="2019-01-01",  # Include pre-crisis data
            end_date="2024-12-31",
            timeframe="1d",
            real_data_ratio=0.9,
            min_samples_per_symbol=1500,
            sequence_length=80,  # Longer sequences to capture crisis patterns
            prediction_horizon=1,
            overlap_ratio=0.8,
            technical_indicators=True,
            sentiment_features=True,
            market_regime_features=True,
            output_dir="data/robust_multi_asset_dataset/crisis_aware",
            version_tag="crisis_v1",
        ),
    }

    return {
        "sequence_length": 60,
        "prediction_horizon": 1,
        "train_split": 0.7,
        "validation_split": 0.15,
        "test_split": 0.15,
        "feature_columns": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "log_returns",
            "volatility",
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
        ],
        "target_column": "future_returns",
        "scaling_method": "standard",
        "augmentation": True,
        "noise_factor": 0.01,
        "time_shift_range": 2,
        "validation_method": "time_series_split",
    }


def analyze_dataset_quality(sequences, targets, metadata):
    """Analyze dataset quality and diversity."""

    print("\n" + "=" * 60)
    print("📊 DATASET QUALITY ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"📈 Dataset Shape: {sequences.shape}")
    print(f"🎯 Target Shape: {targets.shape}")
    print(f"🔢 Total Sequences: {len(sequences):,}")
    print(f"📊 Features per timestep: {sequences.shape[-1]}")

    # Target analysis
    target_stats = {
        "mean": np.mean(targets),
        "std": np.std(targets),
        "min": np.min(targets),
        "max": np.max(targets),
        "positive_ratio": np.mean(targets > 0),
        "negative_ratio": np.mean(targets < 0),
    }

    print("\n🎯 Target Statistics:")
    print(f"  Mean: {target_stats['mean']:.6f}")
    print(f"  Std: {target_stats['std']:.6f}")
    print(f"  Range: [{target_stats['min']:.6f}, {target_stats['max']:.6f}]")
    print(f"  Positive ratio: {target_stats['positive_ratio']:.2%}")
    print(f"  Negative ratio: {target_stats['negative_ratio']:.2%}")

    # Feature analysis
    feature_means = np.mean(sequences, axis=(0, 1))
    feature_stds = np.std(sequences, axis=(0, 1))

    print("\n🔧 Feature Analysis:")
    print(f"  Feature means range: [{np.min(feature_means):.3f}, {np.max(feature_means):.3f}]")
    print(f"  Feature stds range: [{np.min(feature_stds):.3f}, {np.max(feature_stds):.3f}]")

    # Data quality checks
    nan_count = np.isnan(sequences).sum()
    inf_count = np.isinf(sequences).sum()

    print("\n✅ Data Quality:")
    print(f"  NaN values: {nan_count}")
    print(f"  Infinite values: {inf_count}")
    print(f"  Data completeness: {100 * (1 - nan_count/sequences.size):.2f}%")

    # Metadata summary
    if metadata:
        print("\n📋 Metadata Summary:")
        print(f"  Symbols: {metadata.get('raw_data', {}).get('symbols', [])}")
        print(
            f"  Real data ratio: {metadata.get('raw_data', {}).get('real_samples', 0) / metadata.get('raw_data', {}).get('total_samples', 1):.1%}"
        )
        print(f"  Timeframe: {metadata.get('raw_data', {}).get('timeframe', 'unknown')}")
        print(f"  Feature types: {metadata.get('feature_engineering', {}).get('feature_types', {})}")

    print("=" * 60)


def generate_robust_datasets():
    """Generate multiple robust datasets with different characteristics."""

    print("🚀 Starting Robust Dataset Generation Pipeline")
    print("=" * 60)

    configs = create_diverse_dataset_configs()
    results = {}

    for config_name, config in configs.items():
        print(f"\n📊 Generating {config_name.upper()} dataset...")
        print(f"  Symbols: {config.symbols}")
        print(f"  Date range: {config.start_date} to {config.end_date}")
        print(f"  Timeframe: {config.timeframe}")
        print(f"  Sequence length: {config.sequence_length}")

        try:
            # Build dataset
            builder = RobustDatasetBuilder(config)
            sequences, targets, dataset_info = builder.build_dataset()

            # Analyze quality
            analyze_dataset_quality(sequences, targets, dataset_info)

            # Store results
            results[config_name] = {
                "config": config,
                "sequences": sequences,
                "targets": targets,
                "dataset_info": dataset_info,
                "status": "success",
            }

            print(f"✅ {config_name} dataset generated successfully!")

        except Exception as e:
            print(f"❌ Failed to generate {config_name} dataset: {e}")
            results[config_name] = {"status": "failed", "error": str(e)}

    # Save comprehensive results
    results_summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "total_configs": len(configs),
        "successful_generations": sum(1 for r in results.values() if r["status"] == "success"),
        "failed_generations": sum(1 for r in results.values() if r["status"] == "failed"),
        "configs": {
            name: {
                "symbols": config.symbols,
                "timeframe": config.timeframe,
                "sequence_length": config.sequence_length,
                "real_data_ratio": config.real_data_ratio,
            }
            for name, config in configs.items()
        },
        "results": {
            name: {
                "status": result["status"],
                "dataset_info": result.get("dataset_info", {}),
                "error": result.get("error", ""),
            }
            for name, result in results.items()
        },
    }

    # Save summary
    output_dir = Path("data/robust_multi_asset_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "generation_summary.json").open("w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n💾 Generation summary saved to: {output_dir / 'generation_summary.json'}")

    return results


def create_training_ready_datasets():
    """Create datasets specifically optimized for training."""

    print("\n🎯 Creating Training-Optimized Datasets")
    print("=" * 60)

    # Training-specific configurations
    training_configs = {
        "quick_test": DatasetConfig(
            symbols=["AAPL", "GOOGL", "MSFT"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            timeframe="1d",
            real_data_ratio=0.8,
            min_samples_per_symbol=500,
            sequence_length=30,
            prediction_horizon=1,
            overlap_ratio=0.8,
            technical_indicators=True,
            sentiment_features=False,
            market_regime_features=True,
            output_dir="data/training_datasets/quick_test",
            version_tag="quick_v1",
        ),
        "production_ready": DatasetConfig(
            symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
            start_date="2020-01-01",
            end_date="2024-12-31",
            timeframe="1d",
            real_data_ratio=0.9,
            min_samples_per_symbol=1500,
            sequence_length=60,
            prediction_horizon=1,
            overlap_ratio=0.8,
            technical_indicators=True,
            sentiment_features=True,
            market_regime_features=True,
            output_dir="data/training_datasets/production_ready",
            version_tag="production_v1",
        ),
    }

    for config_name, config in training_configs.items():
        print(f"\n📊 Generating {config_name} training dataset...")

        try:
            builder = RobustDatasetBuilder(config)
            sequences, targets, dataset_info = builder.build_dataset()

            print(f"✅ {config_name} training dataset ready!")
            print(f"  Shape: {sequences.shape}")
            print(f"  Target correlation: {np.corrcoef(targets, np.arange(len(targets)))[0,1]:.4f}")

        except Exception as e:
            print(f"❌ Failed to generate {config_name}: {e}")


if __name__ == "__main__":
    print("🎯 ROBUST DATASET GENERATION PIPELINE")
    print("=" * 60)

    # Generate diverse datasets
    results = generate_robust_datasets()

    # Create training-ready datasets
    create_training_ready_datasets()

    print("\n🎉 Dataset generation pipeline completed!")
    print("📁 Check the 'data/robust_multi_asset_dataset' directory for results.")
