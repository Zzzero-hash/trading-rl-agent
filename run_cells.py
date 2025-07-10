#!/usr/bin/env python3
"""Run individual cells from main.ipynb interactively."""

import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.append("/workspaces/trading-rl-agent/src")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import required libraries
import matplotlib as mpl
import pandas as pd

mpl.use("Agg")  # Use non-interactive backend
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Import project modules

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print("✅ All required libraries imported successfully!")
print(f"📍 Working directory: {Path.cwd()}")
print(f"🐍 Python version: {sys.version}")

# Global variables to maintain state between cells
global_vars = {}


def run_cell_3():
    """Configure Multi-Asset Dataset Generation"""
    print("\n" + "=" * 60)
    print("📊 Running Cell 3: Data Sources and Pipeline Configuration")
    print("=" * 60)

    # Configure Multi-Asset Dataset Generation
    CONFIG = {
        # Date range for historical data
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "timeframe": "1d",  # Daily data
        # Stock Market Symbols (Large Cap + Tech + Finance + Energy)
        "stock_symbols": [
            # Tech Giants
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            # Financial
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            # Consumer Goods
            "JNJ",
            "PG",
            "KO",
            "PEP",
            "WMT",
            # Industrial & Energy
            "XOM",
            "CVX",
            "CAT",
            "BA",
            "GE",
        ],
        # Cryptocurrency Symbols (Major coins + DeFi + Altcoins)
        "crypto_symbols": [
            "BTC-USD",
            "ETH-USD",
            "BNB-USD",
            "ADA-USD",
            "XRP-USD",
            "SOL-USD",
            "DOT-USD",
            "MATIC-USD",
            "AVAX-USD",
            "LINK-USD",
        ],
        # Forex Symbols (Major + Minor + Exotic pairs)
        "forex_symbols": [
            # Major pairs
            "EURUSD=X",
            "GBPUSD=X",
            "USDJPY=X",
            "USDCHF=X",
            # Minor pairs
            "EURGBP=X",
            "EURJPY=X",
            "GBPJPY=X",
            # Commodity currencies
            "AUDUSD=X",
            "NZDUSD=X",
            "USDCAD=X",
        ],
        # Synthetic Data Configuration
        "synthetic_symbols": ["SYNTH_STOCK_1", "SYNTH_STOCK_2", "SYNTH_CRYPTO_1", "SYNTH_FOREX_1", "SYNTH_COMMODITY_1"],
        # Dataset composition
        "real_data_ratio": 0.75,  # 75% real data, 25% synthetic
        "min_samples_per_symbol": 800,  # Minimum samples per symbol
        # Feature engineering
        "technical_indicators": True,
        "sentiment_features": True,
        "market_regime_features": True,
        # Output configuration
        "output_dir": "data/robust_multi_asset_dataset",
        "save_intermediate": True,
        "create_visualizations": True,
    }

    # Combine all symbols for the robust dataset builder
    ALL_SYMBOLS = (
        CONFIG["stock_symbols"] + CONFIG["crypto_symbols"] + CONFIG["forex_symbols"] + CONFIG["synthetic_symbols"]
    )

    print("📊 Dataset Configuration Summary:")
    print(f"  📈 Stock symbols: {len(CONFIG['stock_symbols'])}")
    print(f"  ₿ Crypto symbols: {len(CONFIG['crypto_symbols'])}")
    print(f"  💱 Forex symbols: {len(CONFIG['forex_symbols'])}")
    print(f"  🎲 Synthetic symbols: {len(CONFIG['synthetic_symbols'])}")
    print(f"  🔢 Total symbols: {len(ALL_SYMBOLS)}")
    print(f"  📅 Date range: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print(f"  ⚖️ Real/Synthetic ratio: {CONFIG['real_data_ratio']:.0%}/{1-CONFIG['real_data_ratio']:.0%}")

    # Store in global variables
    global_vars["CONFIG"] = CONFIG
    global_vars["ALL_SYMBOLS"] = ALL_SYMBOLS

    return CONFIG, ALL_SYMBOLS


def run_cell_4():
    """Fetch Stock Market Data"""
    print("\n" + "=" * 60)
    print("📈 Running Cell 4: Fetch Stock Market Data")
    print("=" * 60)

    CONFIG = global_vars.get("CONFIG")
    if not CONFIG:
        print("⚠️ CONFIG not found, running Cell 3 first...")
        run_cell_3()
        CONFIG = global_vars["CONFIG"]

    def fetch_stock_data(symbols: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """Fetch stock market data for multiple symbols with error handling."""

        stock_data = {}
        failed_symbols = []

        print(f"📈 Fetching stock market data for {len(symbols)} symbols...")

        # For demo, only fetch first 5 symbols to save time
        demo_symbols = symbols[:5]
        print(f"  🎯 Demo mode: fetching only {len(demo_symbols)} symbols")

        for i, symbol in enumerate(demo_symbols, 1):
            try:
                print(f"  [{i:2d}/{len(demo_symbols)}] Fetching {symbol}...", end=" ")

                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval="1d")

                if not df.empty:
                    # Standardize column names
                    df = df.rename(
                        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                    )

                    # Reset index to get timestamp as column
                    df = df.reset_index()
                    df = df.rename(columns={"Date": "timestamp"})

                    # Add symbol and source information
                    df["symbol"] = symbol
                    df["data_source"] = "stock_real"
                    df["asset_class"] = "stock"

                    # Keep only required columns
                    required_cols = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "symbol",
                        "data_source",
                        "asset_class",
                    ]
                    df = df[required_cols]

                    stock_data[symbol] = df
                    print(f"✅ {len(df)} samples")
                else:
                    print("❌ No data")
                    failed_symbols.append(symbol)

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                failed_symbols.append(symbol)

        print("\n📊 Stock Data Summary:")
        print(f"  ✅ Successfully fetched: {len(stock_data)} symbols")
        print(f"  ❌ Failed to fetch: {len(failed_symbols)} symbols")
        if failed_symbols:
            print(f"  Failed symbols: {failed_symbols}")

        return stock_data

    # Fetch stock market data
    stock_datasets = fetch_stock_data(CONFIG["stock_symbols"], CONFIG["start_date"], CONFIG["end_date"])

    global_vars["stock_datasets"] = stock_datasets

    # Show sample data
    if stock_datasets:
        first_symbol = next(iter(stock_datasets.keys()))
        print(f"\n📋 Sample data for {first_symbol}:")
        print(stock_datasets[first_symbol].head())

    return stock_datasets


def run_demo():
    """Run a demo of the main notebook cells."""
    print("🚀 Starting Trading RL Agent Demo")
    print("=" * 80)

    # Cell 3: Configuration
    CONFIG, ALL_SYMBOLS = run_cell_3()

    # Cell 4: Fetch Stock Data
    stock_datasets = run_cell_4()

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)

    print("\n📊 Summary:")
    print("  - Configuration loaded: ✅")
    print(f"  - Stock data fetched: {len(stock_datasets)} symbols")
    print("  - Ready for next steps: cryptocurrency, forex, and synthetic data")

    return global_vars


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        print("\n🎯 Available functions:")
        print("  - run_cell_3(): Configure dataset generation")
        print("  - run_cell_4(): Fetch stock market data")
        print("  - run_demo(): Run complete demo")
        print("\nRun: python run_cells.py demo")
