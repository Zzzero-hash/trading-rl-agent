#!/usr/bin/env python3
"""
Alignment Strategies Comparison Example

This example demonstrates and compares different strategies for aligning
crypto data to traditional market hours in mixed portfolios.

Strategies compared:
1. last_known_value (recommended) - Uses last known crypto values until next data point
2. forward_fill - Simple forward-filling of all OHLCV data
3. interpolate - Interpolates between crypto data points

Key considerations:
- OHLC integrity preservation
- Volume data handling
- Price continuity
- Data leakage prevention
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trade_agent.data.market_calendar import get_trading_calendar
from trade_agent.data.professional_feeds import ProfessionalDataProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_data() -> pd.DataFrame:
    """Create sample data to demonstrate alignment strategies."""
    # Create sample crypto data with weekend trading
    crypto_data = []

    # Friday trading
    crypto_data.append({
        "timestamp": "2024-01-05 16:00:00",
        "symbol": "BTC-USD",
        "open": 45000,
        "high": 45200,
        "low": 44900,
        "close": 45100,
        "volume": 1000
    })

    # Saturday trading (crypto continues)
    crypto_data.append({
        "timestamp": "2024-01-06 12:00:00",
        "symbol": "BTC-USD",
        "open": 45100,
        "high": 45500,
        "low": 45000,
        "close": 45300,
        "volume": 800
    })

    # Sunday trading (crypto continues)
    crypto_data.append({
        "timestamp": "2024-01-07 18:00:00",
        "symbol": "BTC-USD",
        "open": 45300,
        "high": 45600,
        "low": 45200,
        "close": 45400,
        "volume": 600
    })

    # Monday trading
    crypto_data.append({
        "timestamp": "2024-01-08 09:30:00",
        "symbol": "BTC-USD",
        "open": 45400,
        "high": 45800,
        "low": 45300,
        "close": 45600,
        "volume": 1200
    })

    # Create sample traditional market data (no weekend data)
    traditional_data = []

    # Friday trading
    traditional_data.append({
        "timestamp": "2024-01-05 16:00:00",
        "symbol": "AAPL",
        "open": 150.00,
        "high": 151.50,
        "low": 149.80,
        "close": 151.20,
        "volume": 50000
    })

    # Monday trading (no weekend data)
    traditional_data.append({
        "timestamp": "2024-01-08 09:30:00",
        "symbol": "AAPL",
        "open": 151.20,
        "high": 153.00,
        "low": 150.50,
        "close": 152.80,
        "volume": 60000
    })

    # Combine data
    all_data = crypto_data + traditional_data
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def compare_alignment_strategies() -> None:
    """Compare different alignment strategies."""
    logger.info("=== Alignment Strategies Comparison ===")

    # Create sample data
    data = create_sample_data()
    symbols = ["BTC-USD", "AAPL"]

    logger.info("Original data:")
    logger.info(data.to_string(index=False))

    # Get trading calendar
    calendar = get_trading_calendar()

    # Test different alignment strategies
    strategies = ["last_known_value", "forward_fill", "interpolate"]

    results = {}

    for strategy in strategies:
        logger.info(f"\n--- {strategy.upper()} Strategy ---")

        # Align data using this strategy
        aligned_data = calendar.align_data_timestamps(data, symbols, strategy)

        # Store results
        results[strategy] = aligned_data

        logger.info(f"Aligned data ({len(aligned_data)} rows):")
        logger.info(aligned_data.to_string(index=False))

        # Show key differences
        if not aligned_data.empty:
            btc_data = aligned_data[aligned_data["symbol"] == "BTC-USD"]
            aapl_data = aligned_data[aligned_data["symbol"] == "AAPL"]

            logger.info(f"BTC-USD data points: {len(btc_data)}")
            logger.info(f"AAPL data points: {len(aapl_data)}")

            # Check for volume handling
            zero_volume_count = len(aligned_data[aligned_data["volume"] == 0])
            logger.info(f"Zero volume entries: {zero_volume_count}")


def analyze_strategy_characteristics() -> None:
    """Analyze the characteristics of each alignment strategy."""
    logger.info("\n=== Strategy Characteristics Analysis ===")

    characteristics = {
        "last_known_value": {
            "description": "Uses last known crypto values until next data point",
            "pros": [
                "Preserves OHLC integrity",
                "Avoids misleading forward-filling",
                "Sets volume to zero during market closure",
                "Maintains price continuity",
                "No data leakage"
            ],
            "cons": [
                "May miss intraday crypto movements",
                "Conservative approach"
            ],
            "best_for": [
                "Model training",
                "Backtesting",
                "Risk management",
                "Portfolio analysis"
            ]
        },
        "forward_fill": {
            "description": "Simple forward-filling of all OHLCV data",
            "pros": [
                "Simple implementation",
                "Preserves all original data points",
                "Fast processing"
            ],
            "cons": [
                "Misleading volume data",
                "May create artificial price continuity",
                "Potential data leakage",
                "Doesn't reflect market reality"
            ],
            "best_for": [
                "Quick prototyping",
                "Simple analysis",
                "When volume accuracy isn't critical"
            ]
        },
        "interpolate": {
            "description": "Interpolates between crypto data points",
            "pros": [
                "Smooth price transitions",
                "May capture intraday movements",
                "Mathematically sound"
            ],
            "cons": [
                "Creates artificial price points",
                "May not reflect actual trading",
                "Complex implementation",
                "Potential for overfitting"
            ],
            "best_for": [
                "High-frequency analysis",
                "Smooth visualizations",
                "When interpolation makes sense"
            ]
        }
    }

    for strategy, info in characteristics.items():
        logger.info(f"\n{strategy.upper()}:")
        logger.info(f"  Description: {info['description']}")
        logger.info(f"  Pros: {', '.join(info['pros'])}")
        logger.info(f"  Cons: {', '.join(info['cons'])}")
        logger.info(f"  Best for: {', '.join(info['best_for'])}")


def demonstrate_real_world_usage() -> None:
    """Demonstrate real-world usage with actual data."""
    logger.info("\n=== Real-World Usage Example ===")

    # Example with real data provider
    try:
        provider = ProfessionalDataProvider("yahoo")

        # Get data for a mixed portfolio
        symbols = ["AAPL", "BTC-USD"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last week

        logger.info(f"Fetching real data for {symbols} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Test with last_known_value strategy (recommended)
        data = provider.get_market_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1Day",
            include_features=False,
            align_mixed_portfolio=True,
            alignment_strategy="last_known_value"
        )

        if not data.empty:
            logger.info(f"Successfully aligned {len(data)} data points")

            # Show sample of aligned data
            sample = data.head(10)
            logger.info("Sample aligned data:")
            logger.info(sample[["timestamp", "symbol", "close", "volume", "alignment_method"]].to_string(index=False))

            # Check alignment metadata
            if "alignment_method" in data.columns:
                methods = data["alignment_method"].value_counts()
                logger.info(f"Alignment methods used: {dict(methods)}")

    except Exception as e:
        logger.error(f"Error fetching real data: {e}")


def provide_recommendations() -> None:
    """Provide recommendations for choosing alignment strategies."""
    logger.info("\n=== Recommendations ===")

    recommendations = {
        "default": {
            "strategy": "last_known_value",
            "reason": "Best balance of accuracy and practicality for most use cases"
        },
        "model_training": {
            "strategy": "last_known_value",
            "reason": "Preserves data integrity and prevents data leakage"
        },
        "backtesting": {
            "strategy": "last_known_value",
            "reason": "Most realistic representation of actual trading conditions"
        },
        "live_trading": {
            "strategy": "last_known_value",
            "reason": "Consistent with training data and risk management"
        },
        "research": {
            "strategy": "interpolate",
            "reason": "May provide smoother data for certain research applications"
        },
        "quick_prototyping": {
            "strategy": "forward_fill",
            "reason": "Fastest to implement for initial testing"
        }
    }

    for use_case, rec in recommendations.items():
        logger.info(f"{use_case.replace('_', ' ').title()}: {rec['strategy']} - {rec['reason']}")

    logger.info("\nKey Considerations:")
    logger.info("1. Always use the same strategy for training and inference")
    logger.info("2. Consider the impact on volume data (zero vs forward-filled)")
    logger.info("3. Test different strategies on your specific use case")
    logger.info("4. Monitor for data leakage and unrealistic price movements")


def main() -> None:
    """Run the alignment strategies comparison."""
    logger.info("Starting Alignment Strategies Comparison")
    logger.info("=" * 60)

    try:
        # Run comparisons
        compare_alignment_strategies()
        analyze_strategy_characteristics()
        demonstrate_real_world_usage()
        provide_recommendations()

        logger.info("\n" + "=" * 60)
        logger.info("Alignment strategies comparison completed!")

    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise


if __name__ == "__main__":
    main()
