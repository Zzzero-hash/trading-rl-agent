#!/usr/bin/env python3
"""
Advanced Testing Framework Runner

This script orchestrates and runs all advanced testing types:
- Property-based testing with Hypothesis
- Chaos engineering tests
- Load testing with Locust
- Contract testing with Pact
- Data quality testing with Great Expectations and Pandera
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AdvancedTestRunner:
    """Runner for advanced testing framework."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.results = {}
        self.start_time = time.time()

    def run_property_tests(self) -> bool:
        """Run property-based tests using Hypothesis."""
        logger.info("Running property-based tests...")

        try:
            cmd = [
                "pytest",
                "tests/property/",
                "-v",
                "-m",
                "property",
                "--tb=short",
                "--durations=10",
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.results["property"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                logger.info("✅ Property-based tests passed")
                return True
            logger.error("❌ Property-based tests failed")
            logger.error(result.stderr)
            return False

        except Exception as exc:
            logger.exception(f"Error running property tests: {exc}")
            self.results["property"] = {"success": False, "error": str(exc)}
            return False

    def run_chaos_tests(self) -> bool:
        """Run chaos engineering tests."""
        logger.info("Running chaos engineering tests...")

        try:
            cmd = [
                "pytest",
                "tests/chaos/",
                "-v",
                "-m",
                "chaos",
                "--tb=short",
                "--durations=10",
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.results["chaos"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                logger.info("✅ Chaos engineering tests passed")
                return True
            logger.error("❌ Chaos engineering tests failed")
            logger.error(result.stderr)
            return False

        except Exception as exc:
            logger.exception(f"Error running chaos tests: {exc}")
            self.results["chaos"] = {"success": False, "error": str(exc)}
            return False

    def run_load_tests(self) -> bool:
        """Run load tests using Locust."""
        logger.info("Running load tests...")

        try:
            # Start mock server in background
            logger.info("Starting mock server for load testing...")
            mock_server_process = subprocess.Popen(
                ["python", "tests/load/mock_server.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server to start
            import time

            time.sleep(3)

            # Run Locust in headless mode
            cmd = [
                "locust",
                "-f",
                "tests/load/locustfile.py",
                "--headless",
                "-u",
                "10",  # 10 users
                "-r",
                "1",  # 1 user per second spawn rate
                "--run-time",
                "30s",  # Run for 30 seconds
                "--html",
                "locust-report.html",
                "--csv",
                "locust-stats",
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            # Stop mock server
            mock_server_process.terminate()
            mock_server_process.wait()

            self.results["load"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                logger.info("✅ Load tests completed")
                return True
            logger.error("❌ Load tests failed")
            logger.error(result.stderr)
            return False

        except Exception as exc:
            logger.exception(f"Error running load tests: {exc}")
            self.results["load"] = {"success": False, "error": str(exc)}
            return False

    def run_contract_tests(self) -> bool:
        """Run contract tests using Pact."""
        logger.info("Running contract tests...")

        try:
            cmd = [
                "pytest",
                "tests/contract/",
                "-v",
                "-m",
                "contract",
                "--tb=short",
                "--durations=10",
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.results["contract"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                logger.info("✅ Contract tests passed")
                return True
            logger.error("❌ Contract tests failed")
            logger.error(result.stderr)
            return False

        except Exception as exc:
            logger.exception(f"Error running contract tests: {exc}")
            self.results["contract"] = {"success": False, "error": str(exc)}
            return False

    def run_data_quality_tests(self) -> bool:
        """Run data quality tests."""
        logger.info("Running data quality tests...")

        try:
            cmd = [
                "pytest",
                "tests/data_quality/",
                "-v",
                "-m",
                "data_quality",
                "--tb=short",
                "--durations=10",
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            self.results["data_quality"] = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                logger.info("✅ Data quality tests passed")
                return True
            logger.error("❌ Data quality tests failed")
            logger.error(result.stderr)
            return False

        except Exception as exc:
            logger.exception(f"Error running data quality tests: {exc}")
            self.results["data_quality"] = {"success": False, "error": str(exc)}
            return False

    def run_all_tests(self) -> bool:
        """Run all advanced tests."""
        logger.info("🚀 Starting Advanced Testing Framework")
        logger.info("=" * 50)

        test_functions = [
            ("Property-based Tests", self.run_property_tests),
            ("Chaos Engineering Tests", self.run_chaos_tests),
            ("Load Tests", self.run_load_tests),
            ("Contract Tests", self.run_contract_tests),
            ("Data Quality Tests", self.run_data_quality_tests),
        ]

        all_passed = True

        for test_name, test_func in test_functions:
            logger.info(f"\n📋 Running {test_name}...")
            logger.info("-" * 30)

            if not test_func():
                all_passed = False
                logger.warning(f"⚠️  {test_name} failed")
            else:
                logger.info(f"✅ {test_name} passed")

        return all_passed

    def generate_report(self) -> dict:
        """Generate comprehensive test report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "summary": self._generate_summary(),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if not self.results.get("property", {}).get("success", False):
            recommendations.append("Fix property-based test failures to ensure mathematical invariants")

        if not self.results.get("chaos", {}).get("success", False):
            recommendations.append("Address chaos engineering failures to improve system resilience")

        if not self.results.get("load", {}).get("success", False):
            recommendations.append("Optimize system performance based on load test results")

        if not self.results.get("contract", {}).get("success", False):
            recommendations.append("Fix API contract violations to ensure service compatibility")

        if not self.results.get("data_quality", {}).get("success", False):
            recommendations.append("Improve data quality based on validation test results")

        if not recommendations:
            recommendations.append("All tests passed! System is ready for production.")

        return recommendations

    def save_report(self, filename: str = "advanced_test_report.json"):
        """Save test report to file."""
        report = self.generate_report()

        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📊 Test report saved to {filename}")

    def print_summary(self):
        """Print test summary to console."""
        report = self.generate_report()

        logger.info("\n" + "=" * 50)
        logger.info("📊 ADVANCED TESTING FRAMEWORK SUMMARY")
        logger.info("=" * 50)

        # Overall status
        overall_success = report["summary"]["overall_success"]
        status_emoji = "✅" if overall_success else "❌"
        logger.info(f"{status_emoji} Overall Status: {'PASSED' if overall_success else 'FAILED'}")

        # Duration
        duration = report["summary"]["total_duration"]
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")

        # Individual test results
        logger.info("\n📋 Test Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")

        # Recommendations
        logger.info("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            logger.info(f"  • {rec}")

        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced Testing Framework Runner")
    parser.add_argument(
        "--test-type",
        choices=["property", "chaos", "load", "contract", "data-quality", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--save-report", action="store_true", help="Save detailed report to file")
    parser.add_argument("--report-file", default="advanced_test_report.json", help="Report filename")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create runner
    runner = AdvancedTestRunner()

    # Run tests based on type
    success = False

    if args.test_type == "all":
        success = runner.run_all_tests()
    elif args.test_type == "property":
        success = runner.run_property_tests()
    elif args.test_type == "chaos":
        success = runner.run_chaos_tests()
    elif args.test_type == "load":
        success = runner.run_load_tests()
    elif args.test_type == "contract":
        success = runner.run_contract_tests()
    elif args.test_type == "data-quality":
        success = runner.run_data_quality_tests()

    # Generate and display results
    runner.print_summary()

    # Save report if requested
    if args.save_report:
        runner.save_report(args.report_file)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
