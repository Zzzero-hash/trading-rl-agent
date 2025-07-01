#!/usr/bin/env python3
"""
Sample Data Generator for CNN-LSTM Training Pipeline

This script generates synthetic market data with technical indicators
and sentiment features for testing the complete training pipeline.
"""

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_ta as ta

# Add src to path for imports
# (Assumes package installed in environment)
from src.data.sentiment import SentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_sample_price_data(
    symbol: str = "AAPL",
    days: int = 365,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic price data with realistic patterns."""

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="1H")

    # Generate price movements with trend and volatility
    np.random.seed(42)  # For reproducibility
    n_points = len(dates)

    # Add trend component (slight upward bias)
    trend = np.linspace(0, 0.2, n_points)

    # Add random walk component
    returns = np.random.normal(0, volatility, n_points)

    # Add some periodic patterns (market cycles)
    daily_pattern = 0.01 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    weekly_pattern = 0.005 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))

    # Combine all components
    log_returns = trend + returns + daily_pattern + weekly_pattern

    # Convert to prices
    prices = start_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC data
    # Open: previous close with small gap
    opens = np.roll(prices, 1)
    opens[0] = start_price
    opens = opens * (1 + np.random.normal(0, 0.001, n_points))

    # High/Low: add intraday volatility
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))

    # Ensure OHLC consistency
    for i in range(n_points):
        high_val = max(opens[i], prices[i], highs[i])
        low_val = min(opens[i], prices[i], lows[i])
        highs[i] = high_val
        lows[i] = low_val

    # Volume: inversely correlated with price sometimes
    base_volume = 1000000
    volume_factor = 1 + 0.5 * np.abs(returns)  # Higher volume on big moves
    volumes = base_volume * volume_factor * (1 + np.random.normal(0, 0.3, n_points))
    volumes = np.abs(volumes).astype(int)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the price data."""

    # Operate on DataFrame copy
    df = df.copy()

    # Compute indicators via pandas-ta accessor
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)

    # Rename pandas-ta default columns
    df.rename(
        columns={
            "SMA_10": "sma_10",
            "SMA_20": "sma_20",
            "SMA_50": "sma_50",
            "EMA_12": "ema_12",
            "EMA_26": "ema_26",
            "MACD_12_26_9": "macd",
            "MACDs_12_26_9": "macd_signal",
            "RSI_14": "rsi",
            "BBM_20_2.0": "bb_middle",
            "BBU_20_2.0": "bb_upper",
            "BBL_20_2.0": "bb_lower",
        },
        inplace=True,
    )

    # Compute derived features
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )
    df["volume_sma"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    df["price_change"] = df["close"].pct_change()
    df["price_change_5"] = df["close"].pct_change(5)
    df["volatility"] = df["price_change"].rolling(window=20).std()

    return df


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment analysis features to the data."""

    df = df.copy()

    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()

    # Get unique symbols
    symbols = df["symbol"].unique()

    # Generate sentiment data for each timestamp
    logging.info("Generating sentiment data...")

    sentiment_data = []
    for timestamp in df["timestamp"].unique():
        for symbol in symbols:
            try:
                # Get sentiment data (will use mock data)
                sentiment = sentiment_analyzer.get_symbol_sentiment(symbol)

                # If sentiment is a float, treat it as composite_sentiment and set others to 0.0
                if isinstance(sentiment, float) or isinstance(sentiment, int):
                    sentiment_data.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "news_sentiment": 0.0,
                            "social_sentiment": 0.0,
                            "composite_sentiment": float(sentiment),
                            "sentiment_volume": 1,
                        }
                    )
                else:
                    sentiment_data.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "news_sentiment": getattr(sentiment, "news_sentiment", 0.0),
                            "social_sentiment": getattr(
                                sentiment, "social_sentiment", 0.0
                            ),
                            "composite_sentiment": getattr(
                                sentiment, "composite_sentiment", 0.0
                            ),
                            "sentiment_volume": (
                                len(getattr(sentiment, "raw_data", []))
                                if hasattr(sentiment, "raw_data")
                                and getattr(sentiment, "raw_data")
                                else 1
                            ),
                        }
                    )
            except Exception as e:
                logging.warning(
                    f"Could not get sentiment for {symbol} at {timestamp}: {e}"
                )
                # Use neutral sentiment as fallback
                sentiment_data.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "news_sentiment": 0.0,
                        "social_sentiment": 0.0,
                        "composite_sentiment": 0.0,
                        "sentiment_volume": 1,
                    }
                )

    # Create sentiment DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)

    # Merge with price data
    df = df.merge(sentiment_df, on=["timestamp", "symbol"], how="left")

    # Fill any missing sentiment values
    sentiment_cols = [
        "news_sentiment",
        "social_sentiment",
        "composite_sentiment",
        "sentiment_volume",
    ]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    return df


def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Generate trading labels (Buy/Hold/Sell) based on future price movements."""

    df = df.copy()

    # Calculate future returns
    future_return = df.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)

    # Define thresholds for buy/sell signals
    buy_threshold = 0.02  # 2% increase
    sell_threshold = -0.02  # 2% decrease

    # Generate labels
    conditions = [future_return > buy_threshold, future_return < sell_threshold]
    choices = [2, 0]  # 2=Buy, 0=Sell, 1=Hold (default)

    df["label"] = np.select(conditions, choices, default=1)

    # Remove rows where we can't calculate future returns
    df = df.dropna(subset=["label"])

    return df


def save_sample_data(df: pd.DataFrame, data_dir: str = "data") -> str:
    """Save the generated sample data to CSV file."""

    os.makedirs(data_dir, exist_ok=True)

    # Save with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample_training_data_{timestamp}.csv"
    filepath = os.path.join(data_dir, filename)

    df.to_csv(filepath, index=False)
    logging.info(f"Sample data saved to: {filepath}")
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return filepath


def generate_sample_training_data(filepath: str):
    """Generate sample training data for TD3 integration tests."""
    data = generate_sample_price_data(
        symbol="TEST", days=30, start_price=100.0, volatility=0.01
    )
    data.to_csv(filepath, index=False)


def main():
    """Generate complete sample dataset for CNN-LSTM training."""

    logging.info("Generating sample data for CNN-LSTM training pipeline...")

    # Generate base price data
    logging.info("1. Generating synthetic price data...")
    df = generate_sample_price_data(
        symbol="AAPL", days=180, start_price=150.0, volatility=0.025
    )

    # Add technical indicators
    logging.info("2. Adding technical indicators...")
    df = add_technical_indicators(df)

    # Add sentiment features
    logging.info("3. Adding sentiment features...")
    df = add_sentiment_features(df)

    # Generate labels
    logging.info("4. Generating trading labels...")
    df = generate_labels(df, horizon=1)

    # Clean data (remove NaN values)
    logging.info("5. Cleaning data...")
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    logging.info(f"Removed {initial_rows - final_rows} rows with NaN values")

    # Save data
    logging.info("6. Saving sample data...")
    filepath = save_sample_data(df)

    # Print summary statistics
    logging.info("\n=== DATA SUMMARY ===")
    logging.info(f"Total samples: {len(df)}")
    logging.info(
        f"Feature columns: {len([col for col in df.columns if col not in ['timestamp', 'symbol', 'label']])}"
    )
    logging.info("Label distribution:")
    logging.info(f"{df['label'].value_counts().sort_index()}")
    logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return filepath


if __name__ == "__main__":
    main()
    output_path = "data/sample_training_data_simple_20250607_192034.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_sample_training_data(output_path)
    logging.info(f"Sample training data saved to {output_path}")
