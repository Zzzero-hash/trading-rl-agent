#!/usr/bin/env python3
"""
Test runner script for different test suites.
Provides fine-grained control over test execution with proper setup/teardown.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run_command(cmd, description, cwd=None):
    """Run a command with proper error handling."""
    print(f"\n🔧 {description}")
    print(f"📝 Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=cwd or Path.cwd(), check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"📄 Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with code {e.returncode}")
        if e.stdout:
            print(f"📄 STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"📄 STDERR:\n{e.stderr}")
        return False


def setup_ray_cluster():
    """Setup Ray cluster for distributed tests."""
    print("\n🚀 Setting up Ray cluster...")

    # Check if Ray is already running
    try:
        result = subprocess.run(
            ["ray", "status"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Ray cluster already running")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # Stop any existing Ray processes
    subprocess.run(["ray", "stop"], capture_output=True)

    # Start new Ray cluster
    return run_command(
        ["ray", "start", "--head", "--num-cpus=4", "--num-gpus=1", "--port=6379"],
        "Starting Ray cluster",
    )


def teardown_ray_cluster():
    """Teardown Ray cluster."""
    print("\n🛑 Tearing down Ray cluster...")
    return run_command(["ray", "stop"], "Stopping Ray cluster")


def run_unit_tests():
    """Run unit tests only."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-m",
        "unit or not (integration or slow or gpu or network or ray)",
        "--tb=short",
        "-v",
    ]
    return run_command(cmd, "Running unit tests")


def run_integration_tests():
    """Run integration tests with Ray setup."""
    if not setup_ray_cluster():
        print("❌ Failed to setup Ray cluster for integration tests")
        return False

    try:
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "integration",
            "--tb=short",
            "-v",
        ]
        return run_command(cmd, "Running integration tests")
    finally:
        teardown_ray_cluster()


def run_smoke_tests():
    """Run smoke tests for CI."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-m",
        "smoke",
        "--tb=line",
        "-x",  # Stop on first failure
    ]
    return run_command(cmd, "Running smoke tests")


def run_all_tests():
    """Run all tests with proper setup."""
    if not setup_ray_cluster():
        print("❌ Failed to setup Ray cluster")
        return False

    try:
        cmd = ["python", "-m", "pytest", "tests/", "--tb=short", "-v"]
        return run_command(cmd, "Running all tests")
    finally:
        teardown_ray_cluster()


def run_coverage_tests():
    """Run tests with coverage reporting."""
    if not setup_ray_cluster():
        print("❌ Failed to setup Ray cluster")
        return False

    try:
        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-fail-under=90",
            "--tb=short",
            "-v",
        ]
        success = run_command(cmd, "Running tests with coverage")

        if success:
            print("\n📊 Coverage report generated:")
            print("   • Terminal: coverage summary above")
            print("   • HTML: htmlcov/index.html")
            print("   • XML: coverage.xml")

        return success
    finally:
        teardown_ray_cluster()


def run_performance_tests():
    """Run performance and benchmark tests."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-m",
        "slow",
        "--tb=short",
        "-v",
        "--durations=0",
    ]
    return run_command(cmd, "Running performance tests")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Trading RL Agent Test Runner")
    parser.add_argument(
        "suite",
        choices=["unit", "integration", "smoke", "all", "coverage", "performance"],
        help="Test suite to run",
    )
    parser.add_argument(
        "--keep-ray", action="store_true", help="Keep Ray cluster running after tests"
    )

    args = parser.parse_args()

    print("🧪 Trading RL Agent Test Runner")
    print("=" * 50)

    # Set environment variables for testing
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

    # Run the requested test suite
    suite_runners = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "smoke": run_smoke_tests,
        "all": run_all_tests,
        "coverage": run_coverage_tests,
        "performance": run_performance_tests,
    }

    success = suite_runners[args.suite]()

    if not args.keep_ray:
        teardown_ray_cluster()

    print("\n" + "=" * 50)
    if success:
        print("✅ Test suite completed successfully!")
        sys.exit(0)
    else:
        print("❌ Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
