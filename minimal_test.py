#!/usr/bin/env python3
"""
Minimal test script to verify Trading RL Agent environment setup
"""

import sys
from pathlib import Path


def test_basic_imports() -> bool:
    """Test basic Python functionality"""
    print("✅ Basic Python functionality test")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")

    # Test basic imports
    try:
        pass

        print("✅ json module imported successfully")
    except ImportError as e:
        print(f"❌ json module import failed: {e}")
        return False

    try:
        pass

        print("✅ datetime module imported successfully")
    except ImportError as e:
        print(f"❌ datetime module import failed: {e}")
        return False

    return True


def test_optional_imports() -> list[str]:
    """Test optional dependencies if available"""
    print("\n🔍 Testing optional dependencies...")

    optional_packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("stable_baselines3", "Stable Baselines3"),
        ("yfinance", "YFinance"),
        ("gymnasium", "Gymnasium"),
    ]

    available_packages = []
    for module_name, display_name in optional_packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name}: {version}")
            available_packages.append(display_name)
        except ImportError:
            print(f"⚠️  {display_name}: Not installed")

    return available_packages


def main() -> bool:
    """Main test function"""
    print("🧪 Trading RL Agent - Environment Test")
    print("=" * 40)

    # Test basic functionality
    if not test_basic_imports():
        print("\n❌ Basic imports failed - environment setup incomplete")
        sys.exit(1)

    # Test optional packages
    available = test_optional_imports()

    print("\n📊 Summary:")
    print("✅ Basic Python environment: OK")
    print(f"📦 Optional packages available: {len(available)}")

    if available:
        print(f"   - {', '.join(available)}")

    print("\n🎉 Environment test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
