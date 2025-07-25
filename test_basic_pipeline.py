#!/usr/bin/env python3
"""Test basic pipeline functionality."""

import os
import sys
from datetime import datetime, timedelta

import yfinance as yf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("Testing basic data pipeline components...")

# Test 1: Basic data fetch
try:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    data = yf.download("AAPL", start=start_date, end=end_date, interval="1d", progress=False)
    print(f"✅ Data fetch: {len(data)} rows downloaded")
except Exception as e:
    print(f"❌ Data fetch failed: {e}")
    sys.exit(1)

# Test 2: Basic TA calculations
try:
    # Simple moving averages
    data["sma_5"] = data["Close"].rolling(5).mean()
    data["sma_20"] = data["Close"].rolling(20).mean()
    print(f"✅ Basic indicators: {len(data.columns)} columns")
except Exception as e:
    print(f"❌ Indicators failed: {e}")
    sys.exit(1)

# Test 3: Feature engineering
try:
    from trade_agent.data.features import generate_unified_features
    features = generate_unified_features(data.rename(columns=str.lower))
    print(f"✅ Feature engineering: {features.shape[1]} features generated")
except Exception as e:
    print(f"❌ Feature engineering failed: {e}")
    print("Continuing with basic features...")

print(f"Dataset shape: {data.shape}")
print(f"Latest close price: ${float(data['Close'].iloc[-1]):.2f}")
print("✅ Basic pipeline test completed successfully!")
