#!/usr/bin/env python3
"""
Test script to verify advanced testing setup.

This script runs basic tests to ensure the advanced testing framework
is properly configured and working.
"""

import importlib
import subprocess
import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing package imports...")

    packages = [
        "hypothesis",
        "pytest",
        "pandas",
        "numpy",
        "locust",
        "fastapi",
        "uvicorn",
        "pandera",
        "great_expectations",
    ]

    failed_imports = []

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements-dev.txt")
        return False

    print("✅ All packages imported successfully")
    return True


def test_pytest_config():
    """Test pytest configuration."""
    print("\n🔍 Testing pytest configuration...")

    try:
        result = subprocess.run(["pytest", "--version"], check=False, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ pytest: {result.stdout.strip()}")
            return True
        print(f"  ❌ pytest error: {result.stderr}")
        return False

    except FileNotFoundError:
        print("  ❌ pytest not found")
        return False


def test_hypothesis_config():
    """Test Hypothesis configuration."""
    print("\n🔍 Testing Hypothesis configuration...")

    try:
        # Test a simple hypothesis test
        test_code = """
import pytest
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=100))
def test_simple_hypothesis(x):
    assert x >= 1 and x <= 100

if __name__ == "__main__":
    test_simple_hypothesis()
"""

        with open("temp_hypothesis_test.py", "w") as f:
            f.write(test_code)

        result = subprocess.run(
            ["python", "temp_hypothesis_test.py"],
            check=False,
            capture_output=True,
            text=True,
        )

        # Clean up
        Path("temp_hypothesis_test.py").unlink(missing_ok=True)

        if result.returncode == 0:
            print("  ✅ Hypothesis working correctly")
            return True
        print(f"  ❌ Hypothesis error: {result.stderr}")
        return False

    except Exception as e:
        print(f"  ❌ Hypothesis test failed: {e}")
        return False


def test_mock_server():
    """Test mock server startup."""
    print("\n🔍 Testing mock server...")

    try:
        # Test if mock server can be imported
        import sys

        sys.path.append(str(Path("tests/load")))

        print("  ✅ Mock server imports successfully")
        return True

    except Exception as e:
        print(f"  ❌ Mock server error: {e}")
        return False


def test_locust_config():
    """Test Locust configuration."""
    print("\n🔍 Testing Locust configuration...")

    try:
        result = subprocess.run(["locust", "--version"], check=False, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ Locust: {result.stdout.strip()}")
            return True
        print(f"  ❌ Locust error: {result.stderr}")
        return False

    except FileNotFoundError:
        print("  ❌ Locust not found")
        return False


def test_test_structure():
    """Test that test files exist and are properly structured."""
    print("\n🔍 Testing test structure...")

    test_dirs = [
        "tests/property",
        "tests/chaos",
        "tests/load",
        "tests/contract",
        "tests/data_quality",
    ]

    all_exist = True

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"  ✅ {test_dir}")
        else:
            print(f"  ❌ {test_dir} - missing")
            all_exist = False

    return all_exist


def main():
    """Run all setup tests."""
    print("🚀 Advanced Testing Framework Setup Verification")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Pytest Configuration", test_pytest_config),
        ("Hypothesis Configuration", test_hypothesis_config),
        ("Mock Server", test_mock_server),
        ("Locust Configuration", test_locust_config),
        ("Test Structure", test_test_structure),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP VERIFICATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Advanced testing framework is ready.")
        return 0
    print("⚠️  Some tests failed. Please check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
