#!/usr/bin/env python3
"""
Simple Sample Data Generator for CNN-LSTM Training Pipeline

This script generates synthetic market data with technical indicators
for testing the complete training pipeline without sentiment analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def generate_sample_price_data(
    symbol: str = "AAPL",
    days: int = 365,
    start_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate synthetic price data with realistic patterns."""
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    
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
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': symbol,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the price data."""
    
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Moving averages
    df['sma_10'] = close.rolling(window=10).mean()
    df['sma_20'] = close.rolling(window=20).mean()
    df['sma_50'] = close.rolling(window=50).mean()
    
    # Exponential moving averages
    df['ema_12'] = close.ewm(span=12).mean()
    df['ema_26'] = close.ewm(span=26).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma']
    
    # Price changes
    df['price_change'] = close.pct_change()
    df['price_change_5'] = close.pct_change(5)
    
    # Volatility
    df['volatility'] = df['price_change'].rolling(window=20).std()
    
    return df


def add_synthetic_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic sentiment analysis features to the data."""
    
    df = df.copy()
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic sentiment features
    n_points = len(df)
    
    # News sentiment - tends to follow price movements with some lag
    price_changes = df['price_change'].fillna(0)
    news_sentiment = np.random.normal(0, 0.3, n_points)
    # Add correlation with lagged price changes
    for i in range(1, n_points):
        news_sentiment[i] += 0.4 * price_changes[i-1] if i > 0 else 0
    
    # Social sentiment - more volatile and less correlated
    social_sentiment = np.random.normal(0, 0.5, n_points)
    
    # Composite sentiment - weighted average
    composite_sentiment = 0.6 * news_sentiment + 0.4 * social_sentiment
    
    # Normalize to [-1, 1] range
    df['news_sentiment'] = np.clip(news_sentiment, -1, 1)
    df['social_sentiment'] = np.clip(social_sentiment, -1, 1)
    df['composite_sentiment'] = np.clip(composite_sentiment, -1, 1)
    
    # Sentiment volume (number of articles/posts)
    base_volume = 10
    sentiment_volume = np.random.poisson(base_volume, n_points)
    df['sentiment_volume'] = sentiment_volume
    
    return df


def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Generate trading labels (Buy/Hold/Sell) based on future price movements."""
    
    df = df.copy()
    
    # Calculate future returns
    future_return = df.groupby('symbol')['close'].pct_change(horizon).shift(-horizon)
    
    # Define thresholds for buy/sell signals
    buy_threshold = 0.02   # 2% increase
    sell_threshold = -0.02  # 2% decrease
    
    # Generate labels
    conditions = [
        future_return > buy_threshold,
        future_return < sell_threshold
    ]
    choices = [2, 0]  # 2=Buy, 0=Sell, 1=Hold (default)
    
    df['label'] = np.select(conditions, choices, default=1)
    
    # Remove rows where we can't calculate future returns
    df = df.dropna(subset=['label'])
    
    return df


def save_sample_data(df: pd.DataFrame, data_dir: str = "data") -> str:
    """Save the generated sample data to CSV file."""
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Save with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample_training_data_simple_{timestamp}.csv"
    filepath = os.path.join(data_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to: {filepath}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return filepath


def main():
    """Generate complete sample dataset for CNN-LSTM training."""
    
    print("Generating sample data for CNN-LSTM training pipeline...")
    
    # Generate base price data
    print("1. Generating synthetic price data...")
    df = generate_sample_price_data(symbol="AAPL", days=180, start_price=150.0, volatility=0.025)
    
    # Add technical indicators
    print("2. Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Add synthetic sentiment features
    print("3. Adding synthetic sentiment features...")
    df = add_synthetic_sentiment_features(df)
    
    # Generate labels
    print("4. Generating trading labels...")
    df = generate_labels(df, horizon=1)
    
    # Clean data (remove NaN values)
    print("5. Cleaning data...")
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    # Save data
    print("6. Saving sample data...")
    filepath = save_sample_data(df)
    
    # Print summary statistics
    print("\n=== DATA SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Feature columns: {len([col for col in df.columns if col not in ['timestamp', 'symbol', 'label']])}")
    print(f"Label distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Show feature correlation with labels
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'label']]
    correlations = df[feature_cols + ['label']].corr()['label'].abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with labels:")
    for i, (feature, corr) in enumerate(correlations.head(10).items()):
        if feature != 'label':
            print(f"{i+1:2d}. {feature:20s}: {corr:.4f}")
    
    return filepath


if __name__ == "__main__":
    main()
