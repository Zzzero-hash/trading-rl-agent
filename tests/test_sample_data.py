#!/usr/bin/env python3
"""
Test script to verify that the sample_data.csv file works with our training pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_sample_data():
    """Test that sample_data.csv is properly formatted for training."""
    
    print("🧪 Testing sample_data.csv for Training Pipeline Compatibility")
    print("="*70)
    
    # Check if file exists
    data_path = Path("data/sample_data.csv")
    if not data_path.exists():
        print("❌ sample_data.csv not found!")
        return False
    
    try:
        # Load the data
        print("📥 Loading data...")
        df = pd.read_csv(data_path, low_memory=False)
        
        print(f"✅ Data loaded successfully")
        print(f"   • Shape: {df.shape}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check required columns for training
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        else:
            print(f"✅ All required columns present")
        
        # Check for targets
        if 'target' in df.columns:
            print(f"✅ Target column found")
            target_dist = df['target'].value_counts().sort_index()
            print(f"   • Target distribution: {target_dist.to_dict()}")
        else:
            print(f"⚠️ No target column - will need to generate")
        
        # Check data quality
        total_missing = df.isnull().sum().sum()
        print(f"📊 Data quality:")
        print(f"   • Missing values: {total_missing:,} ({total_missing/df.size*100:.2f}%)")
        print(f"   • Complete rows: {len(df.dropna()):,} ({len(df.dropna())/len(df)*100:.1f}%)")
        
        # Check symbols and date range
        print(f"📈 Data coverage:")
        print(f"   • Symbols: {df['symbol'].nunique()} unique")
        print(f"   • Records per symbol: {len(df) // df['symbol'].nunique():.0f} avg")
        
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                valid_timestamps = df['timestamp'].notna().sum()
                print(f"   • Valid timestamps: {valid_timestamps:,} ({valid_timestamps/len(df)*100:.1f}%)")
                if valid_timestamps > 0:
                    print(f"   • Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
                    print(f"   • Days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")
            except Exception as e:
                print(f"   ⚠️ Timestamp parsing issue: {e}")
                print(f"   • Raw timestamp sample: {df['timestamp'].iloc[0] if len(df) > 0 else 'N/A'}")
        
        # Test basic feature engineering works
        print(f"🔧 Testing feature engineering compatibility...")
        sample_symbol = df['symbol'].iloc[0]
        sample_data = df[df['symbol'] == sample_symbol].head(100).copy()
        
        # Test basic technical indicators
        sample_data['sma_5'] = sample_data['close'].rolling(5).mean()
        sample_data['returns'] = sample_data['close'].pct_change()
        
        if sample_data['sma_5'].notna().sum() > 0:
            print(f"✅ Feature engineering test passed")
        else:
            print(f"⚠️ Feature engineering may have issues")
        
        # Summary
        print(f"\n🎯 TRAINING READINESS SUMMARY")
        print(f"="*40)
        print(f"✅ File format: Compatible")
        print(f"✅ Required columns: Present")
        print(f"✅ Data volume: {len(df):,} records")
        print(f"✅ Feature count: {len(df.columns)} columns")
        print(f"{'✅' if 'target' in df.columns else '⚠️'} Targets: {'Present' if 'target' in df.columns else 'Need generation'}")
        print(f"✅ Quality: {(1-total_missing/df.size)*100:.1f}% complete")
        
        print(f"\n🚀 sample_data.csv is ready for CNN-LSTM training!")
        print(f"📁 Use this file in your training scripts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sample_data.csv: {e}")
        return False

if __name__ == "__main__":
    success = test_sample_data()
    if success:
        print(f"\n✅ All tests passed! Your dataset is ready for training.")
    else:
        print(f"\n❌ Some tests failed. Check the data generation process.")
