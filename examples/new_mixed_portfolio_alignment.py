#!/usr/bin/env python3
"""
New Mixed Portfolio Alignment Example

This example demonstrates the NEW behavior where:
1. All crypto candles are retained (24/7 data)
2. Traditional assets are filled with last known values during weekends/holidays
3. The result is a complete dataset with no missing values
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trade_agent.data.market_calendar import get_trading_calendar

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_new_alignment_behavior() -> None:
    """Demonstrate the new alignment behavior."""
    logger.info("=== NEW Mixed Portfolio Alignment Behavior ===")

    # Create sample data
    data = create_sample_mixed_data()

    logger.info("Original data:")
    logger.info(data.to_string())

    # Apply new alignment
    calendar = get_trading_calendar()
    symbols = ["AAPL", "BTC-USD"]

    aligned_data = calendar.align_data_timestamps(data, symbols, "last_known_value")

    logger.info("\nAligned data (NEW BEHAVIOR):")
    logger.info(aligned_data.to_string())

    # Show the difference
    logger.info("\nKey differences:")
    logger.info(f"Original AAPL data points: {len(data[data['symbol'] == 'AAPL'])}")
    logger.info(f"Aligned AAPL data points: {len(aligned_data[aligned_data['symbol'] == 'AAPL'])}")
    logger.info(f"Original BTC-USD data points: {len(data[data['symbol'] == 'BTC-USD'])}")
    logger.info(f"Aligned BTC-USD data points: {len(aligned_data[aligned_data['symbol'] == 'BTC-USD'])}")

    # Show data sources
    logger.info("\nData sources:")
    logger.info(aligned_data[["timestamp", "symbol", "data_source", "alignment_method"]].to_string())


def create_sample_mixed_data() -> pd.DataFrame:
    """Create sample mixed portfolio data."""
    data = []

    # Friday trading (both markets open)
    data.append({
        "timestamp": "2024-01-05 16:00:00",
        "symbol": "AAPL",
        "open": 150.00,
        "high": 151.50,
        "low": 149.80,
        "close": 151.20,
        "volume": 50000
    })

    data.append({
        "timestamp": "2024-01-05 16:00:00",
        "symbol": "BTC-USD",
        "open": 45000,
        "high": 45200,
        "low": 44900,
        "close": 45100,
        "volume": 1000
    })

    # Saturday trading (only crypto)
    data.append({
        "timestamp": "2024-01-06 12:00:00",
        "symbol": "BTC-USD",
        "open": 45100,
        "high": 45500,
        "low": 45000,
        "close": 45300,
        "volume": 800
    })

    # Sunday trading (only crypto)
    data.append({
        "timestamp": "2024-01-07 18:00:00",
        "symbol": "BTC-USD",
        "open": 45300,
        "high": 45600,
        "low": 45200,
        "close": 45400,
        "volume": 600
    })

    # Monday trading (both markets open)
    data.append({
        "timestamp": "2024-01-08 09:30:00",
        "symbol": "AAPL",
        "open": 151.20,
        "high": 153.00,
        "low": 150.50,
        "close": 152.80,
        "volume": 60000
    })

    data.append({
        "timestamp": "2024-01-08 09:30:00",
        "symbol": "BTC-USD",
        "open": 45400,
        "high": 45800,
        "low": 45300,
        "close": 45600,
        "volume": 1200
    })

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    demonstrate_new_alignment_behavior()
