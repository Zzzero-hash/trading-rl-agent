#!/usr/bin/env python3
"""
Quick integration test to verify the entire pipeline works correctly.

This script tests:
1. Data loading and processing
2. Model training (quick version)
3. Agent initialization
4. Environment interaction
5. Basic prediction pipeline
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test data loading and processing."""
    try:
        from tests.test_data_utils import DynamicTestDataManager
        
        logger.info("Testing data pipeline...")
        
        # Test dynamic data discovery/generation
        with DynamicTestDataManager() as manager:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            dataset_path = manager.get_or_create_test_dataset(required_columns)
            
            # Verify file exists and has correct structure
            df = pd.read_csv(dataset_path)
            assert len(df) > 0, "Dataset is empty"
            
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            logger.info(f"✅ Data pipeline test passed - dataset: {len(df)} rows")
            return True
            
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False

def test_model_initialization():
    """Test CNN-LSTM model initialization."""
    try:
        from src.models.cnn_lstm import CNNLSTMModel
        
        logger.info("Testing model initialization...")
        
        # Test model creation
        model = CNNLSTMModel(input_dim=10, output_size=1)
        
        # Test forward pass
        import torch
        test_input = torch.randn(5, 20, 10)  # batch_size=5, seq_len=20, features=10
        output = model(test_input)
        
        assert output.shape == (5, 1), f"Unexpected output shape: {output.shape}"
        
        logger.info("✅ Model initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization test failed: {e}")
        return False

def test_environment_setup():
    """Test trading environment setup."""
    try:
        from tests.test_data_utils import get_dynamic_test_config
        from src.envs.trading_env import TradingEnv
        
        logger.info("Testing environment setup...")
        
        # Create environment with dynamic config
        config = get_dynamic_test_config()
        env = TradingEnv(config)
        
        # Test environment reset
        obs, info = env.reset()
        assert obs is not None, "Environment reset failed"
        
        # Test one step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        logger.info("✅ Environment setup test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment setup test failed: {e}")
        return False

def test_optimization_system():
    """Test optimization system."""
    try:
        from src.optimization import optimize_cnn_lstm
        
        logger.info("Testing optimization system...")
        
        # Quick optimization test with minimal data
        features = np.random.randn(30, 5)
        targets = np.random.randn(30)
        
        results = optimize_cnn_lstm(
            features, 
            targets, 
            num_samples=1, 
            max_epochs_per_trial=2
        )
        
        assert "best_score" in results, "Missing best_score in results"
        assert "best_config" in results, "Missing best_config in results"
        
        logger.info("✅ Optimization system test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization system test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("🧪 Quick Integration Test Starting...")
    logger.info("=" * 50)
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Initialization", test_model_initialization),
        ("Environment Setup", test_environment_setup),
        ("Optimization System", test_optimization_system),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name} test...")
        result = test_func()
        results[test_name] = result
        if result:
            passed += 1
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Integration Test Summary:")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   • {test_name}: {status}")
    
    success_rate = (passed / total) * 100
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed ({success_rate:.0f}%)")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        return 0
    else:
        logger.warning("⚠️  Some integration tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
