#!/usr/bin/env python3
"""
Simple validation script to check if mypy errors are resolved.
"""

from pathlib import Path
import subprocess
import sys


def run_mypy_check():
    """Run mypy check on the trading_rl_agent package."""
    print("🔍 Running mypy check on trading_rl_agent...")

    try:
        # Run mypy on the src directory
        result = subprocess.run(
            ["python", "-m", "mypy", "src/", "--show-error-codes"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ No mypy errors found!")
            print("🎉 All type checking issues have been resolved!")
            return True
        else:
            print("❌ Mypy errors found:")
            print(result.stdout)
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running mypy: {e}")
        return False


def check_mypy_config():
    """Check if mypy.ini is properly configured."""
    print("🔍 Checking mypy.ini configuration...")

    mypy_ini = Path(__file__).parent / "mypy.ini"
    if not mypy_ini.exists():
        print("❌ mypy.ini not found")
        return False

    content = mypy_ini.read_text()

    # Check for the fixed patterns
    if "[mypy-build_datasets]" in content:
        print("✅ mypy.ini has been updated with proper module patterns")
        return True
    else:
        print("❌ mypy.ini still has old wildcard patterns")
        return False


def main():
    """Main validation function."""
    print("🛠️  Validating Trading RL Agent Restructure")
    print("=" * 50)

    # Check mypy config
    config_ok = check_mypy_config()

    # Run mypy check
    mypy_ok = run_mypy_check()

    if config_ok and mypy_ok:
        print("\n🎉 SUCCESS: All issues have been resolved!")
        print("🚀 The restructured codebase is ready for development!")
        return True
    else:
        print("\n❌ ISSUES FOUND: Some problems still exist")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
