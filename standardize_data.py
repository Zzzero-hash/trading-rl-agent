#!/usr/bin/env python3
"""
Data Standardization Demo

This script demonstrates how to use the DataStandardizer to ensure consistent
feature engineering between training and live inference.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.trading_rl_agent.data.csv_utils import save_csv_chunked
from src.trading_rl_agent.data.data_standardizer import (
    DataStandardizer,
    FeatureConfig,
    LiveDataProcessor,
    create_standardized_dataset,
)


def demo_standardization() -> None:
    """Demonstrate the data standardization process."""
    print("🚀 Data Standardization Demo")
    print("=" * 60)

    # Load the main dataset
    data_path = "data/advanced_trading_dataset_20250617_145352.csv"
    if not Path(data_path).exists():
        print(f"❌ Dataset not found: {data_path}")
        return

    print(f"📊 Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Original shape: {df.shape}")
    print(f"   Original columns: {len(df.columns)}")

    # Create standardized dataset
    print("\n🔧 Creating standardized dataset...")
    standardized_df, standardizer = create_standardized_dataset(
        df=df, save_path="outputs/data_standardizer.pkl", feature_config=FeatureConfig()
    )

    print(f"   Standardized shape: {standardized_df.shape}")
    print(f"   Feature count: {standardizer.get_feature_count()}")
    print("   Standardizer saved to: outputs/data_standardizer.pkl")

    # Show feature information
    print("\n📋 Feature Information:")
    print(f"   Price features: {len(standardizer.feature_config.price_features)}")
    print(f"   Technical indicators: {len(standardizer.feature_config.technical_indicators)}")
    print(f"   Candlestick patterns: {len(standardizer.feature_config.candlestick_patterns)}")
    print(f"   Sentiment features: {len(standardizer.feature_config.sentiment_features)}")
    print(f"   Time features: {len(standardizer.feature_config.time_features)}")

    # Show sample features
    print("\n📝 Sample features:")
    all_features = standardizer.get_feature_names()
    for i, feature in enumerate(all_features[:10]):
        print(f"   {i + 1:2d}. {feature}")
    if len(all_features) > 10:
        print(f"   ... and {len(all_features) - 10} more features")

    # Demonstrate live data processing
    print("\n🎯 Live Data Processing Demo:")

    # Create live data processor
    live_processor = LiveDataProcessor(standardizer)

    # Simulate live data (missing some features, different order, etc.)
    live_data = {
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000000,
        "log_return": 0.01,
        "rsi_14": 65.0,
        # Missing some features intentionally
        "timestamp": "2024-01-01",
        "symbol": "AAPL",
    }

    print(f"   Live data input: {list(live_data.keys())}")

    # Process live data
    processed_live_data = live_processor.process_single_row(live_data)

    print(f"   Processed shape: {processed_live_data.shape}")
    print(f"   Processed features: {len(processed_live_data.columns)}")
    print(f"   Feature count matches: {len(processed_live_data.columns) == standardizer.get_feature_count()}")

    # Show that features are in correct order
    print("\n✅ Feature order validation:")
    expected_features = standardizer.get_feature_names()
    actual_features = list(processed_live_data.columns)

    if expected_features == actual_features:
        print("   ✅ Feature order is correct!")
    else:
        print("   ❌ Feature order mismatch!")
        print(f"   Expected first 5: {expected_features[:5]}")
        print(f"   Actual first 5: {actual_features[:5]}")

    # Demonstrate batch processing
    print("\n📦 Batch Processing Demo:")

    batch_data = [
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000000},
        {"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 1100000},
        {"open": 102.0, "high": 104.0, "low": 101.0, "close": 103.0, "volume": 1200000},
    ]

    processed_batch = live_processor.process_batch(batch_data)
    print(f"   Batch input: {len(batch_data)} rows")
    print(f"   Batch output: {processed_batch.shape}")
    print(f"   All features present: {len(processed_batch.columns) == standardizer.get_feature_count()}")

    # Show data quality improvements
    print("\n🔍 Data Quality Improvements:")

    # Check for missing values
    original_missing = df.isnull().sum().sum()
    standardized_missing = standardized_df.isnull().sum().sum()
    print(f"   Missing values - Original: {original_missing}, Standardized: {standardized_missing}")

    # Check for infinite values
    original_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    standardized_inf = np.isinf(standardized_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"   Missing values - Original: {original_inf}, Standardized: {standardized_inf}")

    # Check for negative prices
    price_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in ["price", "close", "open", "high", "low"])
    ]
    original_negative = sum((df[col] < 0).sum() for col in price_cols if col in df.columns)
    standardized_negative = sum(
        (standardized_df[col] < 0).sum() for col in price_cols if col in standardized_df.columns
    )
    print(f"   Negative prices - Original: {original_negative}, Standardized: {standardized_negative}")

    # Save standardized dataset using chunked approach
    output_path = "outputs/standardized_dataset.csv"
    save_csv_chunked(standardized_df, output_path, chunk_size=10000, show_progress=True)
    print(f"\n💾 Standardized dataset saved to: {output_path}")

    # Create model input template
    print("\n📋 Model Input Template:")
    template = standardizer.create_live_data_template()
    print(f"   Template shape: {template.shape}")
    print(f"   Template columns: {list(template.columns)}")

    # Save template using chunked approach
    template_path = "outputs/model_input_template.csv"
    save_csv_chunked(template, template_path, chunk_size=10000, show_progress=True)
    print(f"   Template saved to: {template_path}")

    print("\n✅ Standardization demo complete!")
    print("\n📚 Next Steps:")
    print("   1. Use 'outputs/data_standardizer.pkl' for consistent preprocessing")
    print("   2. Use 'outputs/standardized_dataset.csv' for training")
    print("   3. Use LiveDataProcessor for live inference")
    print(f"   4. Ensure model input dimension matches: {standardizer.get_feature_count()}")


def demo_model_integration() -> None:
    """Demonstrate how to integrate with model training."""
    print("\n🤖 Model Integration Demo")
    print("=" * 60)

    # Load the standardizer
    standardizer_path = "outputs/data_standardizer.pkl"
    if not Path(standardizer_path).exists():
        print(f"❌ Standardizer not found: {standardizer_path}")
        return

    standardizer = DataStandardizer.load(standardizer_path)
    print(f"✅ Loaded standardizer with {standardizer.get_feature_count()} features")

    # Show how to use in model training
    print("\n🎯 Model Training Integration:")
    print(f"   Input dimension: {standardizer.get_feature_count()}")
    print(f"   Feature names: {standardizer.get_feature_names()}")

    # Simulate model creation
    print("\n📊 Model Configuration:")
    print(f"   CNN+LSTM input_dim = {standardizer.get_feature_count()}")
    print(f"   Ensure model expects exactly {standardizer.get_feature_count()} features")

    # Show live inference workflow
    print("\n🚀 Live Inference Workflow:")
    print("   1. Receive live market data")
    print("   2. Process with LiveDataProcessor")
    print(f"   3. Ensure {standardizer.get_feature_count()} features")
    print("   4. Feed to trained model")
    print("   5. Get prediction")

    # Create example live inference code
    print("\n💻 Example Live Inference Code:")
    print(f"""
# Load standardizer
standardizer = DataStandardizer.load('outputs/data_standardizer.pkl')
live_processor = LiveDataProcessor(standardizer)

# Load trained model
model = CNNLSTMModel(input_dim={standardizer.get_feature_count()})
model.load_state_dict(torch.load('outputs/best_model.pth'))

# Process live data
live_data = {{'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0, 'volume': 1000000}}
processed_data = live_processor.process_single_row(live_data)

# Make prediction
with torch.no_grad():
    input_tensor = torch.FloatTensor(processed_data.values)
    prediction = model(input_tensor)
    print(f"Prediction: {{prediction.item()}}")
""")


def main() -> None:
    """Main function to run the standardization demo."""
    try:
        demo_standardization()
        demo_model_integration()

        print("\n🎉 All demos completed successfully!")
        print("\n📁 Generated files:")
        print("   - outputs/data_standardizer.pkl (Standardizer)")
        print("   - outputs/data_standardizer.json (Config)")
        print("   - outputs/standardized_dataset.csv (Training data)")
        print("   - outputs/model_input_template.csv (Template)")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
