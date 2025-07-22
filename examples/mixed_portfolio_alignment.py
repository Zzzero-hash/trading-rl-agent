#!/usr/bin/env python3
"""
Mixed Portfolio Timestamp Alignment Example

This example demonstrates how the trading system handles mixed portfolios
containing both crypto (24/7 trading) and traditional assets (specific hours/holidays).

Key concepts:
1. Crypto trades 24/7 while traditional markets have specific hours
2. Mixed portfolios need timestamp alignment for consistent analysis
3. The market calendar handles this automatically
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trade_agent.data.market_calendar import classify_portfolio_assets, get_trading_calendar
from trade_agent.data.professional_feeds import ProfessionalDataProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_market_calendar() -> None:
    """Demonstrate basic market calendar functionality."""
    logger.info("=== Market Calendar Demonstration ===")

    calendar = get_trading_calendar()

    # Test different asset types
    test_symbols = ["AAPL", "BTC-USD", "SPY", "EURUSD=X"]

    for symbol in test_symbols:
        info = calendar.get_market_hours_info(symbol)
        logger.info(f"{symbol}: {info['market_type']} - 24/7: {info['trades_24_7']}")

    # Test market open/close times
    now = datetime.now()
    logger.info(f"\nCurrent time: {now}")

    for symbol in test_symbols:
        is_open = calendar.is_market_open(symbol, now)
        next_open = calendar.get_next_market_open(symbol, now)
        logger.info(f"{symbol}: Open now: {is_open}, Next open: {next_open}")


def demonstrate_portfolio_classification() -> None:
    """Demonstrate portfolio asset classification."""
    logger.info("\n=== Portfolio Classification Demonstration ===")

    # Test different portfolio compositions
    portfolios = [
        ["AAPL", "GOOGL", "MSFT"],  # Traditional only
        ["BTC-USD", "ETH-USD", "ADA-USD"],  # Crypto only
        ["AAPL", "BTC-USD", "SPY", "ETH-USD"],  # Mixed portfolio
        ["AAPL", "GOOGL", "SPY", "EURUSD=X"],  # Traditional mixed
    ]

    for i, portfolio in enumerate(portfolios, 1):
        classification = classify_portfolio_assets(portfolio)
        logger.info(f"Portfolio {i} ({', '.join(portfolio)}):")
        logger.info(f"  Crypto: {classification['crypto']}")
        logger.info(f"  Traditional: {classification['traditional']}")
        logger.info(f"  Mixed: {classification['mixed']}")


def demonstrate_data_alignment() -> None:
    """Demonstrate timestamp alignment for mixed portfolios."""
    logger.info("\n=== Data Alignment Demonstration ===")

    # Create a mixed portfolio
    symbols = ["AAPL", "BTC-USD", "SPY", "ETH-USD"]

    # Initialize data provider
    provider = ProfessionalDataProvider("yahoo")

    # Get data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    logger.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        # Get data with alignment enabled
        data_with_alignment = provider.get_market_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1Day",
            include_features=False,  # Skip features for this demo
            align_mixed_portfolio=True,
        )

        # Get data without alignment for comparison
        data_without_alignment = provider.get_market_data(
            symbols=symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1Day",
            include_features=False,
            align_mixed_portfolio=False,
        )

        logger.info(f"Data with alignment: {len(data_with_alignment)} rows")
        logger.info(f"Data without alignment: {len(data_without_alignment)} rows")

        # Show timestamp distribution
        if not data_with_alignment.empty:
            logger.info("\nTimestamp distribution with alignment:")
            timestamp_counts = data_with_alignment.groupby("symbol")["timestamp"].count()
            for symbol, count in timestamp_counts.items():
                logger.info(f"  {symbol}: {count} data points")

        if not data_without_alignment.empty:
            logger.info("\nTimestamp distribution without alignment:")
            timestamp_counts = data_without_alignment.groupby("symbol")["timestamp"].count()
            for symbol, count in timestamp_counts.items():
                logger.info(f"  {symbol}: {count} data points")

        # Show sample of aligned data
        if not data_with_alignment.empty:
            logger.info("\nSample of aligned data:")
            sample_data = data_with_alignment.head(10)
            for _, row in sample_data.iterrows():
                logger.info(f"  {row['timestamp']} - {row['symbol']}: ${row['close']:.2f}")

    except Exception as e:
        logger.error(f"Error fetching data: {e}")


def demonstrate_holiday_handling() -> None:
    """Demonstrate how holidays are handled."""
    logger.info("\n=== Holiday Handling Demonstration ===")

    calendar = get_trading_calendar()

    # Test some known holidays
    test_dates = [
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # Martin Luther King Jr. Day
        "2024-07-04",  # Independence Day
        "2024-12-25",  # Christmas Day
        "2024-01-02",  # Regular trading day
    ]

    for date_str in test_dates:
        test_date = datetime.strptime(date_str, "%Y-%m-%d")
        is_open = calendar.is_market_open("AAPL", test_date)
        logger.info(f"{date_str} ({test_date.strftime('%A')}): Market open: {is_open}")


def main() -> None:
    """Run all demonstrations."""
    logger.info("Starting Mixed Portfolio Alignment Demonstrations")
    logger.info("=" * 60)

    try:
        # Run demonstrations
        demonstrate_market_calendar()
        demonstrate_portfolio_classification()
        demonstrate_data_alignment()
        demonstrate_holiday_handling()

        logger.info("\n" + "=" * 60)
        logger.info("All demonstrations completed successfully!")

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
