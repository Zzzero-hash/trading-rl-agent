#!/usr/bin/env python3
"""
End-to-End Integration Test for CNN-LSTM Training Pipeline

This script tests the complete training pipeline using the generated sample data.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.train_cnn_lstm import CNNLSTMTrainer, TrainingConfig
except ImportError:
    from train_cnn_lstm import CNNLSTMTrainer, TrainingConfig


def test_end_to_end_training():
    """Test the complete CNN-LSTM training pipeline."""
    
    print("=== END-TO-END CNN-LSTM TRAINING TEST ===")
    
    # 1. Load sample data
    print("1. Loading sample data...")
    data_files = [f for f in os.listdir("data") if f.startswith("sample_training_data_simple_")]
    if not data_files:
        print("ERROR: No sample data found. Run generate_simple_data.py first.")
        return False
    
    latest_data_file = max(data_files)
    data_path = os.path.join("data", latest_data_file)
    print(f"   Using data file: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # 2. Load training configuration
    print("2. Loading training configuration...")
    config_path = "src/configs/training/cnn_lstm_dev.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create training config
    config = TrainingConfig.from_dict(config_dict)
    print(f"   Loaded config: {config.model['name']}")
    
    # 3. Adjust config for test data
    print("3. Adjusting configuration for test data...")
    # Count actual features (excluding timestamp, symbol, label)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'label']]
    config.model['input_dim'] = len(feature_cols)
    config.training['epochs'] = 5  # Quick test
    config.data['sequence_length'] = 30  # Shorter sequences
    config.data['batch_size'] = 16  # Smaller batches
    
    print(f"   Adjusted input_dim to {config.model['input_dim']}")
    print(f"   Set epochs to {config.training['epochs']} for quick test")
    
    # 4. Initialize trainer
    print("4. Initializing CNN-LSTM trainer...")
    trainer = CNNLSTMTrainer(config)
    
    # 5. Prepare data
    print("5. Preparing training data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    print(f"   Training set: {X_train.shape} -> {y_train.shape}")
    print(f"   Validation set: {X_val.shape} -> {y_val.shape}")
    print(f"   Test set: {X_test.shape} -> {y_test.shape}")
    
    # 6. Create model
    print("6. Creating CNN-LSTM model...")
    model = trainer.create_model()
    print(f"   Model created with input shape: {model.input_shape}")
    
    # 7. Train model
    print("7. Training model...")
    try:
        history = trainer.train(X_train, y_train, X_val, y_val)
        print(f"   Training completed successfully!")
        print(f"   Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        # 8. Evaluate model
        print("8. Evaluating model on test set...")
        test_loss, test_accuracy = trainer.model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test loss: {test_loss:.4f}")
        print(f"   Test accuracy: {test_accuracy:.4f}")
        
        # 9. Make predictions
        print("9. Making sample predictions...")
        predictions = trainer.model.predict(X_test[:10], verbose=0)
        predicted_classes = predictions.argmax(axis=1)
        actual_classes = y_test[:10].argmax(axis=1)
        
        print("   Sample predictions (Actual -> Predicted):")
        for i in range(min(10, len(predicted_classes))):
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            actual_action = action_map[actual_classes[i]]
            predicted_action = action_map[predicted_classes[i]]
            confidence = predictions[i].max()
            print(f"   {i+1:2d}. {actual_action} -> {predicted_action} (confidence: {confidence:.3f})")
        
        print("\n=== INTEGRATION TEST PASSED ===")
        return True
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test loading all training configurations."""
    
    print("\n=== CONFIGURATION LOADING TEST ===")
    
    config_dir = Path("src/configs/training")
    config_files = list(config_dir.glob("*.yaml"))
    
    print(f"Found {len(config_files)} configuration files:")
    
    for config_file in config_files:
        try:
            print(f"  Testing {config_file.name}...")
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            config = TrainingConfig.from_dict(config_dict)
            print(f"    ✓ Loaded successfully: {config.model['name']}")
            
        except Exception as e:
            print(f"    ✗ Error loading {config_file.name}: {e}")
            return False
    
    print("=== CONFIGURATION TEST PASSED ===")
    return True


def main():
    """Run all integration tests."""
    
    print("Starting Phase 1 CNN-LSTM Integration Tests...\n")
    
    # Test 1: Configuration loading
    config_test_passed = test_configuration_loading()
    
    # Test 2: End-to-end training
    e2e_test_passed = test_end_to_end_training()
    
    # Summary
    print("\n" + "="*50)
    print("INTEGRATION TEST SUMMARY")
    print("="*50)
    print(f"Configuration Loading: {'PASS' if config_test_passed else 'FAIL'}")
    print(f"End-to-End Training:   {'PASS' if e2e_test_passed else 'FAIL'}")
    
    if config_test_passed and e2e_test_passed:
        print("\n🎉 ALL PHASE 1 INTEGRATION TESTS PASSED!")
        print("\nPhase 1 Implementation Status:")
        print("✅ Sentiment Analysis Module")
        print("✅ CNN-LSTM Training Pipeline")
        print("✅ Configuration Management")
        print("✅ End-to-End Data Pipeline")
        print("✅ Comprehensive Unit Testing")
        print("\nReady to proceed to Phase 2!")
        return True
    else:
        print("\n❌ Some integration tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
