#!/usr/bin/env python3
"""
Quick validation script for the advanced trading dataset.
Run this to verify the dataset is ready for training.
"""

import pandas as pd
import sys

def validate_dataset(path="data/sample_data.csv"):
    """Validate the trading dataset."""
    try:
        df = pd.read_csv(path)
        
        # Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            return False
            
        # Check data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        if missing_pct > 10:
            print(f"❌ Too much missing data: {missing_pct:.1f}%")
            return False
            
        print(f"✅ Dataset valid: {len(df)} records, {missing_pct:.1f}% missing")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_data.csv"
    success = validate_dataset(path)
    sys.exit(0 if success else 1)
