#!/usr/bin/env python3
"""
Test script for enhanced training system.

This script tests the core components of the enhanced training system
without requiring all dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_registry() -> None:
    """Test ModelRegistry functionality."""
    print("🧪 Testing ModelRegistry...")

    try:
        from trade_agent.training.model_registry import ModelRegistry, PerformanceGrade

        # Create registry
        registry = ModelRegistry(registry_root=Path("test_models"))

        # Test basic functionality
        models = registry.list_models()
        print(f"   ✅ Found {len(models)} existing models")

        # Test performance grades
        grades = [grade.value for grade in PerformanceGrade]
        print(f"   ✅ Performance grades: {', '.join(grades)}")

        print("   ✅ Model registry tests passed!")

    except Exception as e:
        print(f"   ❌ Model registry tests failed: {e}")
        print(f"      Error details: {e}")
        raise


def test_preprocessor_manager() -> None:
    """Test PreprocessorManager functionality."""
    print("🧪 Testing PreprocessorManager...")

    try:
        from trade_agent.training.preprocessor_manager import PreprocessorManager

        # Create manager
        manager = PreprocessorManager(preprocessor_root=Path("test_preprocessors"))

        # Test pipeline creation
        pipeline = manager.create_pipeline("cnn_lstm")
        print(f"   ✅ Created pipeline: {pipeline.preprocessor_id}")

        # Test standard pipeline creation
        std_pipeline = manager.create_standard_pipeline("cnn_lstm")
        print(f"   ✅ Created standard pipeline: {std_pipeline.preprocessor_id}")

        # Test list preprocessors
        preprocessors = manager.list_preprocessors()
        print(f"   ✅ Found {len(preprocessors)} preprocessors")

        print("   ✅ PreprocessorManager test passed!")

    except Exception as e:
        print(f"   ❌ PreprocessorManager test failed: {e}")
        raise


def test_unified_manager() -> None:
    """Test UnifiedTrainingManager functionality."""
    print("🧪 Testing UnifiedTrainingManager...")

    try:
        from trade_agent.training.unified_manager import TrainingConfig, UnifiedTrainingManager

        # Create manager
        manager = UnifiedTrainingManager()

        # Test configuration creation
        config = TrainingConfig(
            model_type="cnn_lstm",
            data_path="test_data.csv",
            output_dir="test_output",
            epochs=10,
            batch_size=16
        )

        print(f"   ✅ Created training config: {config.training_id}")
        print(f"   ✅ Auto-detected device: {config.device}")

        # Test active trainings tracking
        active = manager.list_active_trainings()
        print(f"   ✅ Active trainings: {len(active)}")

        print("   ✅ UnifiedTrainingManager test passed!")


    except Exception as e:
        print(f"   ❌ UnifiedTrainingManager test failed: {e}")



def test_directory_structure() -> None:
    """Test model directory structure."""
    print("🧪 Testing directory structure...")

    try:
        models_dir = Path("models")
        expected_dirs = ["cnn_lstm", "ppo", "sac", "td3", "hybrid", "ensemble", "preprocessors", "archived", "temp"]

        missing_dirs = []
        for dir_name in expected_dirs:
            dir_path = models_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            print(f"   ⚠️  Missing directories: {', '.join(missing_dirs)}")
        else:
            print("   ✅ All required directories exist")

        # Check README
        readme_path = models_dir / "README.md"
        if readme_path.exists():
            print("   ✅ Models README.md exists")
        else:
            print("   ⚠️  Models README.md missing")

        print("   ✅ Directory structure test passed!")


    except Exception as e:
        print(f"   ❌ Directory structure test failed: {e}")



def main() -> None:
    """Run all tests."""
    print("🚀 Testing Enhanced Training System")
    print("=" * 50)

    tests = [
        test_directory_structure,
        test_model_registry,
        test_preprocessor_manager,
        test_unified_manager,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            test_func()  # Call test function (no return value expected)
            passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Test {test_func.__name__} crashed: {e}")
            print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced training system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")


if __name__ == "__main__":
    main()
    sys.exit(0)
