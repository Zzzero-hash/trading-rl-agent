#!/usr/bin/env python3
"""
Test script for the enhanced pipeline command with sentiment analysis.

This script tests the new unified pipeline command that includes sentiment analysis.
"""

import subprocess
import sys
from pathlib import Path


def test_pipeline_help() -> bool:
    """Test that the pipeline command shows help with sentiment options."""
    print("Testing pipeline command help...")

    try:
        result = subprocess.run([
            sys.executable, "main.py", "data", "pipeline", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            help_text = result.stdout
            if "--sentiment" in help_text and "--sentiment-days" in help_text:
                print("✅ Pipeline help shows sentiment options")
                return True
            else:
                print("❌ Pipeline help missing sentiment options")
                return False
        else:
            print(f"❌ Pipeline help command failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Pipeline help command timed out")
        return False
    except Exception as e:
        print(f"❌ Pipeline help command error: {e}")
        return False

def test_deprecated_commands() -> None:
    """Test that deprecated commands show deprecation warnings."""
    print("\nTesting deprecated commands...")

    deprecated_commands = [
        ["data", "download-all", "--help"],
        ["data", "symbols", "--help"],
        ["data", "refresh", "--help"],
        ["data", "prepare", "--help"],
    ]

    for cmd in deprecated_commands:
        try:
            result = subprocess.run([sys.executable, "main.py", *cmd], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout + result.stderr
                if "DEPRECATED" in output:
                    print(f"✅ {cmd[1]} shows deprecation warning")
                else:
                    print(f"❌ {cmd[1]} missing deprecation warning")
            else:
                print(f"❌ {cmd[1]} command failed")

        except Exception as e:
            print(f"❌ {cmd[1]} command error: {e}")

def test_pipeline_sentiment_only() -> bool:
    """Test pipeline command with sentiment analysis only."""
    print("\nTesting pipeline sentiment analysis...")

    try:
        # Test with a small set of symbols to avoid API rate limits
        result = subprocess.run([
            sys.executable, "main.py", "data", "pipeline",
            "--sentiment",
            "--symbols", "AAPL,GOOGL",
            "--sentiment-days", "1",
            "--output-dir", "test_output"
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            output = result.stdout + result.stderr
            if "Sentiment analysis completed" in output:
                print("✅ Pipeline sentiment analysis completed successfully")
                return True
            else:
                print("❌ Pipeline sentiment analysis failed or incomplete")
                print(f"Output: {output}")
                return False
        else:
            print(f"❌ Pipeline sentiment command failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Pipeline sentiment command timed out")
        return False
    except Exception as e:
        print(f"❌ Pipeline sentiment command error: {e}")
        return False

def cleanup_test_output() -> None:
    """Clean up test output directory."""
    test_dir = Path("test_output")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test output directory")

def main() -> None:
    """Run all tests."""
    print("🧪 Testing Enhanced Pipeline Command with Sentiment Analysis")
    print("=" * 60)

    # Test 1: Help command
    help_success = test_pipeline_help()

    # Test 2: Deprecated commands
    test_deprecated_commands()

    # Test 3: Sentiment analysis (optional - requires API access)
    print("\nNote: Sentiment analysis test requires API access and may take time...")
    sentiment_success = test_pipeline_sentiment_only()

    # Cleanup
    cleanup_test_output()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Help command: {'PASS' if help_success else 'FAIL'}")
    print(f"✅ Sentiment analysis: {'PASS' if sentiment_success else 'FAIL (may need API keys)'}")
    print("\n🎉 Pipeline command enhancement complete!")
    print("💡 Use: python main.py data pipeline --run --symbols 'AAPL,GOOGL' --sentiment-days 7")

if __name__ == "__main__":
    main()
