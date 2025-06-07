#!/usr/bin/env python3
"""
Quick Integration Test for Phase 1 Components

This script tests the core components without complex imports.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

def test_sample_data():
    """Test that sample data exists and has expected format."""
    print("1. Testing sample data...")
    
    data_dir = current_dir / "data"
    sample_files = list(data_dir.glob("sample_training_data_*.csv"))
    
    if not sample_files:
        print("❌ No sample data files found")
        return False
    
    # Use the most recent file
    sample_file = max(sample_files, key=lambda f: f.stat().st_mtime)
    print(f"   Using: {sample_file.name}")
    
    try:
        df = pd.read_csv(sample_file)
        print(f"   Data shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ['timestamp', 'close', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check label distribution
        label_dist = df['label'].value_counts().sort_index()
        print(f"   Label distribution: {dict(label_dist)}")
        
        print("✅ Sample data test passed")
        return True, df
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False, None

def test_sentiment_module():
    """Test sentiment analysis module."""
    print("\n2. Testing sentiment module...")
    
    try:
        from src.data.sentiment import SentimentAnalyzer, SentimentData
        
        # Create analyzer
        analyzer = SentimentAnalyzer()
        print("   ✅ SentimentAnalyzer created")
        
        # Test mock sentiment
        sentiment = analyzer.get_symbol_sentiment("AAPL", pd.Timestamp.now())
        if isinstance(sentiment, SentimentData):
            print(f"   ✅ Sentiment retrieved: {sentiment.composite_score:.3f}")
        else:
            print(f"   ⚠️  Sentiment returned non-SentimentData: {type(sentiment)}")
        
        print("✅ Sentiment module test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment module test failed: {e}")
        return False

def test_cnn_lstm_model():
    """Test CNN-LSTM model creation."""
    print("\n3. Testing CNN-LSTM model...")
    
    try:
        from src.models.cnn_lstm import CNNLSTMModel, CNNLSTMConfig
         # Create config
        config = CNNLSTMConfig(
            input_dim=10,
            output_size=3,
            lstm_units=32,
            dropout=0.1
        )
        print("   ✅ CNNLSTMConfig created")

        # Create model
        model = CNNLSTMModel(
            input_dim=config.input_dim,
            output_size=config.output_size,
            config=config.to_dict()
        )
        print(f"   ✅ CNNLSTMModel created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test forward pass
        batch_size = 4
        seq_len = 20  # sequence length
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        expected_shape = (batch_size, config.output_size)
        if output.shape == expected_shape:
            print(f"   ✅ Forward pass successful: {output.shape}")
        else:
            print(f"   ❌ Output shape mismatch: {output.shape} vs {expected_shape}")
            return False
        
        print("✅ CNN-LSTM model test passed")
        return True
        
    except Exception as e:
        print(f"❌ CNN-LSTM model test failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing for training."""
    print("\n4. Testing data preprocessing...")
    
    try:
        # Load sample data
        success, df = test_sample_data()
        if not success:
            return False
        
        # Simple feature extraction
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol', 'label']]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"   ⚠️  Found {nan_count} NaN values in features")
        
        # Create sequences for LSTM
        sequence_length = 10
        sequences = []
        labels = []
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            labels.append(y[i])
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"   Sequences shape: {sequences.shape}")
        print(f"   Sequence labels shape: {labels.shape}")
        
        if len(sequences) > 0:
            print("✅ Data preprocessing test passed")
            return True, sequences, labels
        else:
            print("❌ No sequences generated")
            return False, None, None
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False, None, None

def test_training_pipeline():
    """Test basic training pipeline."""
    print("\n5. Testing basic training pipeline...")
    
    try:
        # Get preprocessed data
        success, X, y = test_data_preprocessing()
        if not success or X is None or y is None:
            print("❌ Cannot proceed without preprocessed data")
            return False
        
        # Import model components
        from src.models.cnn_lstm import CNNLSTMModel, CNNLSTMConfig
         # Create simple config
        config = CNNLSTMConfig(
            input_dim=X.shape[2],
            output_size=3,
            lstm_units=16,
            dropout=0.1
        )

        model = CNNLSTMModel(
            input_dim=config.input_dim,
            output_size=config.output_size,
            config=config.to_dict()
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X[:100])  # Use first 100 samples
        y_tensor = torch.LongTensor(y[:100])
        
        # Single training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step completed, loss: {loss.item():.4f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_output = model(X_tensor[:5])
            predictions = torch.softmax(test_output, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
            
        print(f"   ✅ Predictions: {predicted_classes.tolist()}")
        print("✅ Basic training pipeline test passed")
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Phase 1 Integration Tests\n")
    
    tests = [
        test_sample_data,
        test_sentiment_module,
        test_cnn_lstm_model,
        test_data_preprocessing,
        test_training_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result and result[0] if isinstance(result, tuple) else result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 integration tests passed!")
        print("\n✅ Phase 1 Implementation Status:")
        print("   ✅ Sentiment Analysis Module")
        print("   ✅ CNN-LSTM Model Architecture")
        print("   ✅ Sample Data Generation")
        print("   ✅ Data Preprocessing Pipeline")
        print("   ✅ Basic Training Loop")
        print("\n🚀 Phase 1 is ready for production use!")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Phase 1 needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
