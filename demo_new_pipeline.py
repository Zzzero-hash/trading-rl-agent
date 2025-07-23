#!/usr/bin/env python3
"""
Demo script showing the new auto-processing pipeline.

This script demonstrates how the new pipeline eliminates the raw folder
and creates ready-to-use datasets in one seamless flow.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main() -> None:
    """Demonstrate the new auto-processing pipeline."""
    print("🚀 Trading RL Agent - New Auto-Processing Pipeline Demo")
    print("=" * 60)

    try:
        from trade_agent.data.prepare import create_auto_processed_dataset

        # Demo 1: Basic auto-processing
        print("\n📊 Demo 1: Basic Auto-Processing Dataset")
        print("-" * 40)

        symbols = ["AAPL", "GOOGL", "MSFT"]
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        result = create_auto_processed_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            dataset_name="demo_basic",
            processing_method="robust",
            feature_set="technical",
            include_sentiment=False,  # Disable for demo
            max_workers=4,
            export_formats=["csv", "parquet"],
        )

        print(f"✅ Created dataset: {result['dataset_dir']}")
        print(f"📊 Data shape: {result['metadata']['row_count']} rows x {result['metadata']['column_count']} columns")
        print("📁 Files created:")
        for fmt, path in result["file_paths"].items():
            print(f"   - {path}")

        # Demo 2: Show the old vs new approach
        print("\n🔄 Old vs New Approach Comparison")
        print("-" * 40)

        print("❌ OLD APPROACH (deprecated):")
        print("   1. trade-agent data pipeline --download --symbols 'AAPL'")
        print("   2. Files saved to data/raw/")
        print("   3. trade-agent data pipeline --sentiment")
        print("   4. Files saved to data/sentiment/")
        print("   5. trade-agent data pipeline --process")
        print("   6. Files processed from data/raw/ to data/processed/")
        print("   ⚠️  Raw files persist, taking up space")
        print("   ⚠️  Multiple steps, complex workflow")

        print("\n✅ NEW APPROACH (recommended):")
        print("   1. trade-agent data pipeline --run --symbols 'AAPL'")
        print("   2. Auto-processing: download → sentiment → standardize")
        print("   3. Ready-to-use dataset created directly in data/dataset_TIMESTAMP/")
        print("   ✨ No raw folder needed")
        print("   ✨ One command, complete dataset")
        print("   ✨ Unique dataset directories")
        print("   ✨ Multiple export formats")

        # Demo 3: Show directory structure
        print("\n📁 New Directory Structure")
        print("-" * 40)

        data_dir = Path("data")
        if data_dir.exists():
            print("data/")
            for item in sorted(data_dir.iterdir()):
                if item.is_dir():
                    print(f"├── {item.name}/")
                    if item.name.startswith("dataset_") or item.name == "demo_basic":
                        for subitem in sorted(item.iterdir())[:5]:  # Show first 5 files
                            print(f"│   ├── {subitem.name}")
                        if len(list(item.iterdir())) > 5:
                            print(f"│   └── ... ({len(list(item.iterdir())) - 5} more files)")

        print("\n💡 Usage Examples:")
        print("-" * 20)
        print("# Quick demo with auto-selected symbols")
        print("trade-agent data pipeline --run")
        print()
        print("# Custom symbols with sentiment")
        print("trade-agent data pipeline --run --symbols 'AAPL,GOOGL,TSLA' --sentiment-days 14")
        print()
        print("# Technical analysis focused dataset")
        print("trade-agent data pipeline --run --features technical --dataset-name tech_analysis")
        print()
        print("# High-performance batch processing")
        print("trade-agent data pipeline --run --max-symbols 100 --workers 16 --no-sentiment")

        print("\n🎉 Demo completed successfully!")
        print(f"✨ Check your new dataset at: {result['dataset_dir']}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is expected if dependencies are not available in demo mode")

if __name__ == "__main__":
    main()
