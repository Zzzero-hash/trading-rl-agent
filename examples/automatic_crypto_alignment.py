#!/usr/bin/env python3
"""
Automatic Crypto Alignment Example

This example demonstrates how the system automatically detects crypto assets
in datasets and applies the last_known_value alignment strategy by default.

Key features:
1. Automatic crypto detection
2. Default last_known_value strategy
3. Seamless integration in data pipeline
4. No manual configuration required
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trade_agent.data.market_calendar import classify_portfolio_assets, get_trading_calendar
from trade_agent.data.pipeline import DataPipeline
from trade_agent.data.professional_feeds import ProfessionalDataProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_automatic_detection() -> None:
    """Demonstrate automatic crypto detection and alignment."""
    logger.info("=== Automatic Crypto Detection and Alignment ===")

    # Test different portfolio compositions
    portfolios = [
        {
            "name": "Traditional Only",
            "symbols": ["AAPL", "GOOGL", "MSFT", "SPY"],
            "expected_alignment": False
        },
        {
            "name": "Crypto Only",
            "symbols": ["BTC-USD", "ETH-USD", "ADA-USD"],
            "expected_alignment": False
        },
        {
            "name": "Mixed Portfolio",
            "symbols": ["AAPL", "BTC-USD", "SPY", "ETH-USD"],
            "expected_alignment": True
        },
        {
            "name": "Traditional Mixed",
            "symbols": ["AAPL", "GOOGL", "SPY", "EURUSD=X"],
            "expected_alignment": False
        }
    ]

    get_trading_calendar()

    for portfolio in portfolios:
        logger.info(f"\n--- {portfolio['name']} ---")
        symbols = portfolio["symbols"]
        logger.info(f"Symbols: {', '.join(symbols)}")  # type: ignore[arg-type]

        # Classify portfolio
        classification = classify_portfolio_assets(portfolio["symbols"])

        logger.info(f"Crypto assets: {classification['crypto']}")
        logger.info(f"Traditional assets: {classification['traditional']}")
        logger.info(f"Mixed portfolio: {classification['mixed']}")

        # Check if alignment would be applied
        if classification["mixed"]:
            logger.info("✅ Mixed portfolio detected - alignment will be applied automatically")
            logger.info("📊 Default strategy: last_known_value")
        else:
            logger.info("i  No mixed portfolio - no alignment needed")


def demonstrate_automatic_alignment() -> None:
    """Demonstrate automatic alignment in data pipeline."""
    logger.info("\n=== Automatic Alignment in Data Pipeline ===")

    try:
        # Initialize data provider
        provider = ProfessionalDataProvider("yahoo")

        # Test with mixed portfolio
        symbols = ["AAPL", "BTC-USD", "SPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last week

        logger.info(f"Fetching data for mixed portfolio: {', '.join(symbols)}")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Get data with automatic alignment (default behavior)
        data = provider.get_market_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1Day",
            include_features=False,  # Skip features for this demo
            align_mixed_portfolio=True,  # This is the default
            alignment_strategy="last_known_value",  # This is the default
        )

        if not data.empty:
            logger.info(f"✅ Successfully fetched and aligned {len(data)} data points")

            # Show alignment metadata
            if "alignment_method" in data.columns:
                methods = data["alignment_method"].value_counts()
                logger.info(f"Alignment methods used: {dict(methods)}")

            if "data_source" in data.columns:
                sources = data["data_source"].value_counts()
                logger.info(f"Data sources: {dict(sources)}")

            # Show sample of aligned data
            sample = data.head(10)
            logger.info("\nSample aligned data:")
            # Check which columns are available (handle both 'timestamp' and 'date')
            time_col = "timestamp" if "timestamp" in data.columns else "date"
            available_cols = [time_col, "symbol", "close", "volume"]
            if "alignment_method" in data.columns:
                available_cols.append("alignment_method")
            logger.info(sample[available_cols].to_string(index=False))

            # Verify alignment worked correctly
            crypto_data = data[data["symbol"].isin(["BTC-USD"])]
            traditional_data = data[data["symbol"].isin(["AAPL", "SPY"])]

            logger.info("\nAlignment verification:")
            logger.info(f"Crypto data points: {len(crypto_data)}")
            logger.info(f"Traditional data points: {len(traditional_data)}")

            # Check for zero volume during market closure
            zero_volume_count = len(data[data["volume"] == 0])
            logger.info(f"Zero volume entries (market closure): {zero_volume_count}")

    except Exception as e:
        logger.error(f"Error in automatic alignment demo: {e}")


def demonstrate_pipeline_integration() -> None:
    """Demonstrate integration with the main data pipeline."""
    logger.info("\n=== Pipeline Integration Demo ===")

    try:
        # Initialize data pipeline
        pipeline = DataPipeline()

        # Test with mixed portfolio
        symbols = ["AAPL", "BTC-USD", "SPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        logger.info(f"Using DataPipeline with mixed portfolio: {', '.join(symbols)}")

        # Download data (automatic alignment is built-in)
        downloaded_files = pipeline.download_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if downloaded_files:
            logger.info(f"✅ Pipeline successfully downloaded {len(downloaded_files)} files")
            logger.info(f"Files: {', '.join(downloaded_files)}")

            # Load the first file to check alignment
            if downloaded_files:
                import pandas as pd
                sample_data = pd.read_csv(downloaded_files[0])
                logger.info(f"Sample data points: {len(sample_data)}")

                # Check if alignment was applied
                if "alignment_method" in sample_data.columns:
                    logger.info("✅ Automatic alignment was applied")
                    methods = sample_data["alignment_method"].value_counts()
                    logger.info(f"Methods used: {dict(methods)}")
                else:
                    logger.info("i  No alignment needed (no mixed portfolio detected)")
        else:
            logger.warning("No files were downloaded")

    except Exception as e:
        logger.error(f"Error in pipeline integration demo: {e}")


def demonstrate_configuration_options() -> None:
    """Demonstrate configuration options for alignment."""
    logger.info("\n=== Configuration Options ===")

    try:
        provider = ProfessionalDataProvider("yahoo")

        symbols = ["AAPL", "BTC-USD", "SPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Test different alignment strategies
        strategies = ["last_known_value", "forward_fill", "interpolate"]

        for strategy in strategies:
            logger.info(f"\n--- Testing {strategy} strategy ---")

            data = provider.get_market_data(
                symbols=symbols,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe="1Day",
                include_features=False,
                align_mixed_portfolio=True,
                alignment_strategy=strategy,
            )

            if not data.empty:
                logger.info(f"Data points: {len(data)}")

                if "alignment_method" in data.columns:
                    methods = data["alignment_method"].value_counts()
                    logger.info(f"Alignment methods: {dict(methods)}")

                # Check volume handling
                zero_volume_count = len(data[data["volume"] == 0])
                logger.info(f"Zero volume entries: {zero_volume_count}")

        logger.info("\n📋 Configuration Summary:")
        logger.info("• Default alignment_strategy: 'last_known_value'")
        logger.info("• Default align_mixed_portfolio: True")
        logger.info("• Automatic crypto detection: Enabled")
        logger.info("• Zero volume during market closure: True (for last_known_value)")

    except Exception as e:
        logger.error(f"Error in configuration demo: {e}")


def main() -> None:
    """Run the automatic crypto alignment demonstration."""
    logger.info("Starting Automatic Crypto Alignment Demonstration")
    logger.info("=" * 60)

    try:
        # Run demonstrations
        demonstrate_automatic_detection()
        demonstrate_automatic_alignment()
        demonstrate_pipeline_integration()
        demonstrate_configuration_options()

        logger.info("\n" + "=" * 60)
        logger.info("✅ Automatic crypto alignment demonstration completed!")
        logger.info("\n🎯 Key Takeaways:")
        logger.info("• Crypto detection is automatic")
        logger.info("• last_known_value is the default strategy")
        logger.info("• No manual configuration required")
        logger.info("• Seamless integration in data pipeline")

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
