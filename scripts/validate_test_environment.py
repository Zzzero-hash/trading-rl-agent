#!/usr/bin/env python3
"""
Test Environment Validation Script

This script validates the test environment setup and identifies potential issues
that could cause inconsistent test runs. It checks:

1. Environment configuration
2. Dependency compatibility
3. Test data availability
4. Resource availability
5. Configuration consistency
6. Test isolation setup
"""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestEnvironmentValidator:
    """Validates test environment setup and configuration."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.validation_results = {
            "environment": {},
            "dependencies": {},
            "configuration": {},
            "test_data": {},
            "resources": {},
            "overall_status": "PASS",
        }

    def validate_environment_variables(self) -> dict[str, Any]:
        """Validate environment variables for consistent test execution."""
        logger.info("Validating environment variables...")

        required_vars = {
            "TRADING_RL_AGENT_ENVIRONMENT": "test",
            "TRADING_RL_AGENT_DEBUG": "false",
            "RAY_DISABLE_IMPORT_WARNING": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }

        results = {"status": "PASS", "issues": [], "warnings": []}

        for var, expected_value in required_vars.items():
            actual_value = os.environ.get(var)
            if actual_value != expected_value:
                if actual_value is None:
                    results["warnings"].append(f"Missing environment variable: {var}")
                else:
                    results["warnings"].append(
                        f"Environment variable {var} has unexpected value: {actual_value} (expected: {expected_value})"
                    )

        if results["warnings"]:
            results["status"] = "WARNING"

        self.validation_results["environment"] = results
        return results

    def validate_dependencies(self) -> dict[str, Any]:
        """Validate required dependencies and their versions."""
        logger.info("Validating dependencies...")

        required_packages = {
            "pytest": "7.4.0",
            "pytest-cov": "4.1.0",
            "pytest-mock": "3.11.0",
            "pytest-asyncio": "0.21.0",
            "pytest-timeout": "2.1.0",
            "pytest-xdist": "3.3.0",
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "hypothesis": "6.75.0",
        }

        results = {"status": "PASS", "issues": [], "warnings": []}

        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")

                # Simple version comparison (can be enhanced)
                if version == "unknown":
                    results["warnings"].append(f"Could not determine version for {package}")
                elif version < min_version:
                    results["issues"].append(f"{package} version {version} is below minimum {min_version}")

            except ImportError:
                results["issues"].append(f"Missing required package: {package}")

        if results["issues"]:
            results["status"] = "FAIL"
        elif results["warnings"]:
            results["status"] = "WARNING"

        self.validation_results["dependencies"] = results
        return results

    def validate_configuration_files(self) -> dict[str, Any]:
        """Validate configuration files for consistency."""
        logger.info("Validating configuration files...")

        config_files = [
            "pytest.ini",
            "conftest.py",
            ".coveragerc",
            "requirements-test.txt",
        ]

        results = {"status": "PASS", "issues": [], "warnings": []}

        for config_file in config_files:
            file_path = self.project_root / config_file
            if not file_path.exists():
                results["issues"].append(f"Missing configuration file: {config_file}")
            # Check file size and basic content
            elif file_path.stat().st_size == 0:
                results["warnings"].append(f"Empty configuration file: {config_file}")

        # Check for duplicate pytest configurations
        pytest_configs = list(self.project_root.glob("pytest*.ini"))
        if len(pytest_configs) > 1:
            results["warnings"].append(f"Multiple pytest configuration files found: {[f.name for f in pytest_configs]}")

        if results["issues"]:
            results["status"] = "FAIL"
        elif results["warnings"]:
            results["status"] = "WARNING"

        self.validation_results["configuration"] = results
        return results

    def validate_test_data(self) -> dict[str, Any]:
        """Validate test data availability and consistency."""
        logger.info("Validating test data...")

        results = {"status": "PASS", "issues": [], "warnings": []}

        # Check test directories
        test_dirs = [
            "tests/unit",
            "tests/integration",
            "tests/performance",
            "tests/smoke",
        ]
        for test_dir in test_dirs:
            dir_path = self.project_root / test_dir
            if not dir_path.exists():
                results["warnings"].append(f"Test directory missing: {test_dir}")
            elif not any(dir_path.glob("test_*.py")):
                results["warnings"].append(f"No test files found in: {test_dir}")

        # Check data directory
        data_dir = self.project_root / "data"
        if not data_dir.exists():
            results["warnings"].append("Data directory missing")
        else:
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                results["warnings"].append("No CSV data files found in data directory")

        # Check test data utilities
        test_data_utils = self.project_root / "tests/unit/test_data_utils.py"
        if not test_data_utils.exists():
            results["issues"].append("Test data utilities file missing")

        if results["issues"]:
            results["status"] = "FAIL"
        elif results["warnings"]:
            results["status"] = "WARNING"

        self.validation_results["test_data"] = results
        return results

    def validate_resources(self) -> dict[str, Any]:
        """Validate system resources and availability."""
        logger.info("Validating system resources...")

        results = {"status": "PASS", "issues": [], "warnings": []}

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 10):
            results["issues"].append(
                f"Python version {python_version.major}.{python_version.minor} is below minimum 3.10"
            )

        # Check available memory (basic check)
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.available < 1_000_000_000:  # 1GB
                results["warnings"].append(f"Low available memory: {memory.available / 1_000_000_000:.1f}GB")
        except ImportError:
            results["warnings"].append("psutil not available for memory validation")

        # Check disk space
        try:
            disk_usage = os.statvfs(self.project_root)
            free_space = disk_usage.f_frsize * disk_usage.f_bavail
            if free_space < 1_000_000_000:  # 1GB
                results["warnings"].append(f"Low disk space: {free_space / 1_000_000_000:.1f}GB")
        except OSError:
            results["warnings"].append("Could not check disk space")

        if results["issues"]:
            results["status"] = "FAIL"
        elif results["warnings"]:
            results["status"] = "WARNING"

        self.validation_results["resources"] = results
        return results

    def run_basic_test_collection(self) -> dict[str, Any]:
        """Run basic test collection to validate test discovery."""
        logger.info("Running test collection validation...")

        results = {"status": "PASS", "issues": [], "warnings": []}

        try:
            # Run pytest collection only
            cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                results["issues"].append(f"Test collection failed: {result.stderr}")
            else:
                # Count collected tests
                lines = result.stdout.split("\n")
                test_count = len([line for line in lines if "test_" in line and "::" in line])
                if test_count == 0:
                    results["issues"].append("No tests collected")
                elif test_count < 100:
                    results["warnings"].append(f"Low test count: {test_count}")
                else:
                    logger.info(f"Successfully collected {test_count} tests")

        except Exception as e:
            results["issues"].append(f"Test collection error: {e!s}")

        if results["issues"]:
            results["status"] = "FAIL"
        elif results["warnings"]:
            results["status"] = "WARNING"

        return results

    def generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Environment recommendations
        env_results = self.validation_results["environment"]
        if env_results["warnings"]:
            recommendations.append("Set required environment variables for consistent test execution")

        # Dependency recommendations
        dep_results = self.validation_results["dependencies"]
        if dep_results["issues"]:
            recommendations.append("Install missing dependencies: pip install -r requirements-test.txt")
        if dep_results["warnings"]:
            recommendations.append("Update dependencies to recommended versions")

        # Configuration recommendations
        config_results = self.validation_results["configuration"]
        if config_results["warnings"]:
            recommendations.append("Review and consolidate pytest configuration files")

        # Test data recommendations
        data_results = self.validation_results["test_data"]
        if data_results["warnings"]:
            recommendations.append("Ensure test data files are available in data/ directory")

        # Resource recommendations
        resource_results = self.validation_results["resources"]
        if resource_results["warnings"]:
            recommendations.append("Ensure adequate system resources for test execution")

        return recommendations

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting comprehensive test environment validation...")

        # Run all validation checks
        self.validate_environment_variables()
        self.validate_dependencies()
        self.validate_configuration_files()
        self.validate_test_data()
        self.validate_resources()

        # Run test collection
        collection_results = self.run_basic_test_collection()
        self.validation_results["test_collection"] = collection_results

        # Determine overall status
        all_results = [
            self.validation_results["environment"]["status"],
            self.validation_results["dependencies"]["status"],
            self.validation_results["configuration"]["status"],
            self.validation_results["test_data"]["status"],
            self.validation_results["resources"]["status"],
            collection_results["status"],
        ]

        if "FAIL" in all_results:
            self.validation_results["overall_status"] = "FAIL"
        elif "WARNING" in all_results:
            self.validation_results["overall_status"] = "WARNING"

        # Add recommendations
        self.validation_results["recommendations"] = self.generate_recommendations()

        return self.validation_results

    def print_results(self):
        """Print validation results in a readable format."""
        print("\n" + "=" * 60)
        print("TEST ENVIRONMENT VALIDATION RESULTS")
        print("=" * 60)

        for category, results in self.validation_results.items():
            if category == "overall_status":
                continue
            if category == "recommendations":
                continue

            print(f"\n{category.upper()}: {results['status']}")
            print("-" * 40)

            if results["issues"]:
                print("ISSUES:")
                for issue in results["issues"]:
                    print(f"  ❌ {issue}")

            if results["warnings"]:
                print("WARNINGS:")
                for warning in results["warnings"]:
                    print(f"  ⚠️  {warning}")

        print(f"\nOVERALL STATUS: {self.validation_results['overall_status']}")

        if self.validation_results["recommendations"]:
            print("\nRECOMMENDATIONS:")
            for rec in self.validation_results["recommendations"]:
                print(f"  💡 {rec}")

        print("\n" + "=" * 60)


def main():
    """Main entry point for the validation script."""
    validator = TestEnvironmentValidator()
    results = validator.run_full_validation()
    validator.print_results()

    # Exit with appropriate code
    if results["overall_status"] == "FAIL":
        sys.exit(1)
    elif results["overall_status"] == "WARNING":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
