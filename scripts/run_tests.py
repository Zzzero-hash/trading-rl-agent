#!/usr/bin/env python3
"""
Robust Test Execution Script

This script provides consistent test execution with:

1. Environment validation before test runs
2. Test data setup and cleanup
3. Isolated test execution
4. Comprehensive reporting
5. Coverage tracking
6. Performance monitoring
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestRunner:
    """Manages robust test execution with validation and isolation."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)

        # Test execution history
        self.execution_history = []

    def validate_environment(self) -> dict[str, Any]:
        """Validate test environment before execution."""
        logger.info("Validating test environment...")

        # Run environment validation script
        validation_script = self.project_root / "scripts" / "validate_test_environment.py"
        if not validation_script.exists():
            return {"valid": False, "error": "Validation script not found"}

        try:
            result = subprocess.run(
                [sys.executable, str(validation_script)],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return {"valid": True, "status": "PASS"}
            if result.returncode == 2:
                return {"valid": True, "status": "WARNING", "warnings": result.stdout}
            return {"valid": False, "status": "FAIL", "errors": result.stderr}

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def setup_test_data(self) -> dict[str, Any]:
        """Setup test data for consistent execution."""
        logger.info("Setting up test data...")

        # Run test data management script
        data_script = self.project_root / "scripts" / "manage_test_data.py"
        if not data_script.exists():
            return {"success": False, "error": "Data management script not found"}

        try:
            result = subprocess.run(
                [sys.executable, str(data_script), "--action", "create"],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_tests(
        self,
        test_paths: list[str] | None = None,
        markers: list[str] | None = None,
        exclude_markers: list[str] | None = None,
        parallel: bool = False,
        coverage: bool = True,
        timeout: int = 300,
        max_failures: int = 5,
    ) -> dict[str, Any]:
        """Run tests with robust configuration."""
        logger.info("Starting test execution...")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]

        # Add test paths
        if test_paths:
            cmd.extend(test_paths)
        else:
            cmd.extend(["tests/unit", "tests/integration", "tests/smoke"])

        # Add markers
        if markers:
            marker_expr = " and ".join(markers)
            cmd.extend(["-m", marker_expr])

        # Exclude markers
        if exclude_markers:
            exclude_expr = " and ".join([f"not {marker}" for marker in exclude_markers])
            if markers:
                cmd.extend(["-m", f"({marker_expr}) and ({exclude_expr})"])
            else:
                cmd.extend(["-m", exclude_expr])

        # Add execution options
        cmd.extend(
            [
                "-v",
                "--tb=short",
                f"--timeout={timeout}",
                f"--maxfail={max_failures}",
                "--junitxml=test-results.xml",
                "--json-report",
                "--json-report-file=test-results.json",
            ]
        )

        # Add coverage options
        if coverage:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml:coverage.xml",
                    "--cov-report=json:coverage.json",
                    "--cov-fail-under=95",
                ]
            )

        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadfile"])

        # Set environment variables for consistent execution
        env = os.environ.copy()
        env.update(
            {
                "TRADING_RL_AGENT_ENVIRONMENT": "test",
                "TRADING_RL_AGENT_DEBUG": "false",
                "RAY_DISABLE_IMPORT_WARNING": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": str(self.project_root / "src"),
            }
        )

        # Execute tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=self.project_root,
                timeout=timeout * 2,  # Double timeout for safety
            )

            execution_time = time.time() - start_time

            # Parse results
            test_results = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

            # Parse JSON report if available
            json_report_path = self.project_root / "test-results.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path) as f:
                        test_results["json_report"] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not parse JSON report: {e}")

            # Parse coverage if available
            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                try:
                    with open(coverage_json_path) as f:
                        test_results["coverage"] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not parse coverage report: {e}")

            return test_results

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "execution_time": timeout * 2,
                "command": " ".join(cmd),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "command": " ".join(cmd)}

    def cleanup_test_data(self):
        """Clean up test data after execution."""
        logger.info("Cleaning up test data...")

        data_script = self.project_root / "scripts" / "manage_test_data.py"
        if data_script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(data_script), "--action", "cleanup"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

    def generate_report(self, test_results: dict[str, Any]) -> str:
        """Generate comprehensive test execution report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TEST EXECUTION REPORT")
        report_lines.append("=" * 60)

        # Basic execution info
        report_lines.append(f"Execution Time: {test_results.get('execution_time', 'N/A'):.2f} seconds")
        report_lines.append(f"Success: {'✓' if test_results.get('success') else '✗'}")
        report_lines.append(f"Return Code: {test_results.get('return_code', 'N/A')}")

        # Test summary from JSON report
        if "json_report" in test_results:
            json_report = test_results["json_report"]
            summary = json_report.get("summary", {})
            report_lines.append("\nTest Summary:")
            report_lines.append(f"  Total: {summary.get('total', 'N/A')}")
            report_lines.append(f"  Passed: {summary.get('passed', 'N/A')}")
            report_lines.append(f"  Failed: {summary.get('failed', 'N/A')}")
            report_lines.append(f"  Skipped: {summary.get('skipped', 'N/A')}")

        # Coverage summary
        if "coverage" in test_results:
            coverage = test_results["coverage"]
            report_lines.append("\nCoverage Summary:")
            report_lines.append(f"  Overall: {coverage.get('totals', {}).get('percent_covered', 'N/A')}%")
            report_lines.append(
                f"  Lines: {coverage.get('totals', {}).get('covered_lines', 'N/A')}/{coverage.get('totals', {}).get('num_statements', 'N/A')}"
            )

        # Errors and warnings
        if test_results.get("stderr"):
            report_lines.append("\nErrors/Warnings:")
            report_lines.append(
                test_results["stderr"][:500] + "..." if len(test_results["stderr"]) > 500 else test_results["stderr"]
            )

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    def save_results(self, test_results: dict[str, Any], report: str):
        """Save test results and report to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        # Save report
        report_file = self.results_dir / f"test_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")

    def run_comprehensive_test_suite(self, **kwargs) -> dict[str, Any]:
        """Run comprehensive test suite with validation and cleanup."""
        logger.info("Starting comprehensive test suite...")

        # Step 1: Validate environment
        env_validation = self.validate_environment()
        if not env_validation.get("valid"):
            logger.error("Environment validation failed")
            return {
                "success": False,
                "error": "Environment validation failed",
                "details": env_validation,
            }

        # Step 2: Setup test data
        data_setup = self.setup_test_data()
        if not data_setup.get("success"):
            logger.warning("Test data setup failed, continuing with existing data")

        # Step 3: Run tests
        test_results = self.run_tests(**kwargs)

        # Step 4: Generate report
        report = self.generate_report(test_results)

        # Step 5: Save results
        self.save_results(test_results, report)

        # Step 6: Cleanup
        self.cleanup_test_data()

        # Print report
        print(report)

        return {
            "success": test_results.get("success", False),
            "test_results": test_results,
            "report": report,
            "environment_validation": env_validation,
            "data_setup": data_setup,
        }


def main():
    """Main entry point for test execution."""
    parser = argparse.ArgumentParser(description="Run Trading RL Agent tests with robust execution")
    parser.add_argument("--test-paths", nargs="+", help="Specific test paths to run")
    parser.add_argument("--markers", nargs="+", help="Test markers to include")
    parser.add_argument("--exclude-markers", nargs="+", help="Test markers to exclude")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--max-failures", type=int, default=5, help="Maximum test failures")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment")
    parser.add_argument("--setup-data-only", action="store_true", help="Only setup test data")

    args = parser.parse_args()

    runner = TestRunner()

    if args.validate_only:
        validation = runner.validate_environment()
        print(json.dumps(validation, indent=2))
        sys.exit(0 if validation.get("valid") else 1)

    if args.setup_data_only:
        setup = runner.setup_test_data()
        print(json.dumps(setup, indent=2))
        sys.exit(0 if setup.get("success") else 1)

    # Run comprehensive test suite
    results = runner.run_comprehensive_test_suite(
        test_paths=args.test_paths,
        markers=args.markers,
        exclude_markers=args.exclude_markers,
        parallel=args.parallel,
        coverage=not args.no_coverage,
        timeout=args.timeout,
        max_failures=args.max_failures,
    )

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
