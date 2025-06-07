#!/usr/bin/env python3
"""Simple test to debug integration issues."""

import sys
from pathlib import Path

print("🧪 Simple Integration Test")
print("="*50)

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")

# Test 1: Basic imports
print("\n1. Testing basic imports...")
try:
    from src.data.sentiment import SentimentAnalyzer
    print("✅ SentimentAnalyzer import successful")
except Exception as e:
    print(f"❌ SentimentAnalyzer import failed: {e}")
    sys.exit(1)

try:
    from src.train_cnn_lstm import CNNLSTMTrainer
    print("✅ CNNLSTMTrainer import successful")
except Exception as e:
    print(f"❌ CNNLSTMTrainer import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Basic initialization
print("\n2. Testing basic initialization...")
try:
    trainer = CNNLSTMTrainer()
    print("✅ CNNLSTMTrainer initialization successful")
except Exception as e:
    print(f"❌ CNNLSTMTrainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check sample data
print("\n3. Testing sample data...")
data_dir = current_dir / "data"
sample_files = list(data_dir.glob("sample_training_data_*.csv"))

if sample_files:
    print(f"✅ Found {len(sample_files)} sample data files")
    sample_file = max(sample_files, key=lambda f: f.stat().st_mtime)
    print(f"   Latest: {sample_file.name}")
else:
    print("❌ No sample data files found")

print("\n🎉 Basic tests completed successfully!")
