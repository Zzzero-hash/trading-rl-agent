#!/usr/bin/env python3
"""
Dependency Validation Script for Trading RL Agent

This script validates:
1. All required dependencies are installed
2. Version compatibility between packages
3. Environment-specific requirements
4. Test environment setup
5. Ray compatibility and fallback mechanisms
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pkg_resources


class DependencyValidator:
    """Comprehensive dependency validation for the trading system."""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0

    def log_issue(self, category: str, message: str, severity: str = "ERROR") -> None:
        """Log an issue or warning."""
        if severity == "ERROR":
            self.issues.append(f"[{category}] {message}")
        else:
            self.warnings.append(f"[{category}] {message}")

    def log_success(self, category: str, message: str) -> None:
        """Log a successful check."""
        self.success_count += 1
        print(f"✅ [{category}] {message}")

    def check_package_installed(self, package_name: str, min_version: str | None = None) -> bool:
        """Check if a package is installed and optionally verify version."""
        self.total_checks += 1
        try:
            # Try to import the package
            importlib.import_module(package_name)

            if min_version:
                # Map import names to package names for version checking
                package_map = {
                    "yaml": "PyYAML",
                    "sklearn": "scikit-learn",
                }
                pkg_name = package_map.get(package_name, package_name)

                try:
                    # Get installed version
                    installed_version = pkg_resources.get_distribution(pkg_name).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                        self.log_issue(
                            "VERSION",
                            f"{package_name} version {installed_version} < {min_version}",
                        )
                        return False
                    self.log_success(
                        "VERSION",
                        f"{package_name} {installed_version} >= {min_version}",
                    )
                except pkg_resources.DistributionNotFound:
                    # Package not found in pkg_resources, but import worked
                    self.log_success(
                        "IMPORT",
                        f"{package_name} imported successfully (version check skipped)",
                    )

            self.log_success("IMPORT", f"{package_name} imported successfully")
            return True

        except ImportError as e:
            self.log_issue("IMPORT", f"Failed to import {package_name}: {e}")
            return False
        except Exception as e:
            self.log_issue("VERSION", f"Error checking {package_name} version: {e}")
            return False

    def check_ray_compatibility(self) -> bool:
        """Check Ray compatibility and fallback mechanisms."""
        self.total_checks += 1

        try:
            import ray

            ray_version = ray.__version__
            self.log_success("RAY", f"Ray {ray_version} installed")

            # Check if Ray can be initialized
            if not ray.is_initialized():
                try:
                    ray.init(ignore_reinit_error=True, local_mode=True)
                    self.log_success("RAY", "Ray initialization successful")
                    ray.shutdown()
                except Exception as e:
                    self.log_issue("RAY", f"Ray initialization failed: {e}")
                    return False
            else:
                self.log_success("RAY", "Ray already initialized")

            # Check RLlib availability
            try:
                # Test RLlib availability without importing specific algorithms
                importlib.import_module("ray.rllib")
                self.log_success("RAY", "RLlib available")
            except ImportError:
                self.log_issue("RAY", "RLlib not available")
                return False

            return True

        except ImportError:
            self.log_issue("RAY", "Ray not installed")
            return False

    def check_structlog_functionality(self) -> bool:
        """Check structlog functionality in test environments."""
        self.total_checks += 1

        try:
            import structlog

            # Test basic structlog functionality
            structlog.get_logger("test")
            self.log_success("STRUCTLOG", "structlog logger creation successful")

            # Test configuration
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            self.log_success("STRUCTLOG", "structlog configuration successful")

            return True

        except Exception as e:
            self.log_issue("STRUCTLOG", f"structlog functionality test failed: {e}")
            return False

    def check_environment_specific_deps(self) -> bool:
        """Check environment-specific dependencies."""
        self.total_checks += 1

        # Core dependencies (always required)
        core_deps = [
            ("numpy", "1.24.0"),
            ("pandas", "2.0.0"),
            ("structlog", "23.1.0"),
            ("yaml", "6.0"),  # PyYAML is imported as yaml
            ("requests", "2.31.0"),
        ]

        # ML dependencies (optional but recommended)
        ml_deps = [
            ("torch", "2.0.0"),
            ("sklearn", "1.3.0"),  # scikit-learn is imported as sklearn
            ("gymnasium", "0.29.0"),
        ]

        # Development dependencies (for testing)
        dev_deps = [
            ("pytest", "7.4.0"),
            ("black", "23.7.0"),
            ("ruff", "0.0.284"),
        ]

        all_passed = True

        # Check core dependencies
        for package, min_version in core_deps:
            if not self.check_package_installed(package, min_version):
                all_passed = False

        # Check ML dependencies (warn if missing)
        for package, min_version in ml_deps:
            if not self.check_package_installed(package, min_version):
                self.log_issue("ML_DEPS", f"ML dependency {package} not available")
                all_passed = False

        # Check dev dependencies (warn if missing)
        for package, min_version in dev_deps:
            if not self.check_package_installed(package, min_version):
                self.log_issue("DEV_DEPS", f"Dev dependency {package} not available")
                all_passed = False

        return all_passed

    def check_test_environment(self) -> bool:
        """Check test environment setup."""
        self.total_checks += 1

        try:
            # Check if test directories exist
            test_dirs = ["tests/unit", "tests/integration", "tests/performance"]
            for test_dir in test_dirs:
                if not Path(test_dir).exists():
                    self.log_issue("TEST_ENV", f"Test directory {test_dir} not found")
                    return False

            # Check if pytest configuration exists
            if not Path("pytest.ini").exists():
                self.log_issue("TEST_ENV", "pytest.ini not found")
                return False

            # Try to run a simple test
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.log_success("TEST_ENV", "pytest available and working")
                else:
                    self.log_issue("TEST_ENV", f"pytest test failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                self.log_issue("TEST_ENV", "pytest test timed out")
                return False
            except Exception as e:
                self.log_issue("TEST_ENV", f"pytest test error: {e}")
                return False

            return True

        except Exception as e:
            self.log_issue("TEST_ENV", f"Test environment check failed: {e}")
            return False

    def check_parallel_processing(self) -> bool:
        """Check parallel processing capabilities."""
        self.total_checks += 1

        try:
            import concurrent.futures
            import multiprocessing as mp

            # Check multiprocessing
            cpu_count = mp.cpu_count()
            self.log_success("PARALLEL", f"CPU count: {cpu_count}")

            # Test basic parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(lambda: "test")
                result = future.result()
                if result == "test":
                    self.log_success("PARALLEL", "ThreadPoolExecutor working")
                else:
                    self.log_issue("PARALLEL", "ThreadPoolExecutor test failed")
                    return False

            # Test Ray parallel processing if available
            if self.check_ray_compatibility():
                self.log_success("PARALLEL", "Ray parallel processing available")
            else:
                self.log_issue("PARALLEL", "Ray parallel processing not available")
                return False

            return True

        except Exception as e:
            self.log_issue("PARALLEL", f"Parallel processing check failed: {e}")
            return False

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            "summary": {
                "total_checks": self.total_checks,
                "successful_checks": self.success_count,
                "failed_checks": len(self.issues),
                "warnings": len(self.warnings),
                "success_rate": ((self.success_count / self.total_checks * 100) if self.total_checks > 0 else 0),
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": [],
        }

        # Generate recommendations based on issues
        if any("RAY" in issue for issue in self.issues):
            report["recommendations"].append("Install Ray with RLlib: pip install 'ray[rllib,tune]>=2.6.0'")

        if any("STRUCTLOG" in issue for issue in self.issues):
            report["recommendations"].append("Install structlog: pip install 'structlog>=23.1.0'")

        if any("ML_DEPS" in issue for issue in self.issues):
            report["recommendations"].append("Install ML dependencies: pip install -r requirements-ml.txt")

        if any("DEV_DEPS" in issue for issue in self.issues):
            report["recommendations"].append("Install dev dependencies: pip install -r requirements-dev.txt")

        return report

    def run_full_validation(self) -> dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Starting comprehensive dependency validation...")
        print("=" * 60)

        # Run all checks
        checks = [
            ("Core Dependencies", self.check_environment_specific_deps),
            ("Ray Compatibility", self.check_ray_compatibility),
            ("Structlog Functionality", self.check_structlog_functionality),
            ("Test Environment", self.check_test_environment),
            ("Parallel Processing", self.check_parallel_processing),
        ]

        for check_name, check_func in checks:
            print(f"\n📋 Running {check_name} check...")
            try:
                check_func()
            except Exception as e:
                self.log_issue("VALIDATION", f"{check_name} check failed: {e}")

        # Generate and display report
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)

        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"Successful: {report['summary']['successful_checks']}")
        print(f"Failed: {report['summary']['failed_checks']}")
        print(f"Warnings: {report['summary']['warnings']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")

        if report["issues"]:
            print(f"\n❌ ISSUES FOUND ({len(report['issues'])}):")
            for issue in report["issues"]:
                print(f"  • {issue}")

        if report["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
            for warning in report["warnings"]:
                print(f"  • {warning}")

        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")

        # Save report to file
        report_file = Path("dependency_validation_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_file}")

        return report


def main():
    """Main entry point."""
    validator = DependencyValidator()
    report = validator.run_full_validation()

    # Exit with appropriate code
    if report["summary"]["failed_checks"] > 0:
        print("\n❌ Validation failed - please fix the issues above")
        sys.exit(1)
    elif report["summary"]["warnings"] > 0:
        print("\n⚠️  Validation completed with warnings")
        sys.exit(0)
    else:
        print("\n✅ All validations passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
