#!/usr/bin/env python3
"""
Comprehensive test runner for the trading RL agent project.
Provides flexible test execution with detailed reporting and coverage analysis.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple


class ComprehensiveTestRunner:
    """Comprehensive test runner with advanced features."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = time.time()

        # Ensure we're in the project root
        os.chdir(self.project_root)

        # Set environment variables for testing
        os.environ["TESTING"] = "true"
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def setup_environment(self) -> bool:
        """Set up the testing environment."""
        print("🔧 Setting up testing environment...")

        try:
            # Check Python version
            if sys.version_info < (3, 9):
                print("❌ Python 3.9+ required")
                return False

            python_version = (
                f"✅ Python {sys.version_info.major}."
                f"{sys.version_info.minor}.{sys.version_info.micro}"
            )
            print(python_version)

            # Check if pytest is available
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("❌ pytest not available")
                return False

            print(f"✅ {result.stdout.strip()}")

            # Check if coverage is available
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
            else:
                print("⚠️ coverage not available - coverage reports will be skipped")

            return True

        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False

    def run_smoke_tests(self) -> bool:
        """Run smoke tests for basic functionality."""
        print("\n🔥 Running smoke tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "smoke",
            "-v",
            "--tb=short",
            "--maxfail=5",
            "--timeout=60",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        self.test_results["smoke"] = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "duration": 0,  # Will be updated later
        }

        if result.returncode == 0:
            print("✅ Smoke tests passed")
            return True
        else:
            print("❌ Smoke tests failed")
            print(f"Error output: {result.stderr}")
            return False

    def run_unit_tests(
        self, pattern: Optional[str] = None, coverage: bool = True
    ) -> bool:
        """Run unit tests with optional coverage."""
        print(
            f"\n🧪 Running unit tests{f' (pattern: {pattern})' if pattern else ''}..."
        )

        cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "unit"]

        if pattern:
            cmd.extend(["-k", pattern])

        if coverage:
            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=term-missing",
                    "--cov-report=html:htmlcov",
                    "--cov-report=xml:coverage.xml",
                    "--cov-report=json:coverage.json",
                ]
            )

        cmd.extend(
            ["-v", "--tb=short", "--maxfail=10", "--junitxml=test-results-unit.xml"]
        )

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        self.test_results["unit"] = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "duration": duration,
        }

        if result.returncode == 0:
            print(f"✅ Unit tests passed ({duration:.2f}s)")
            return True
        else:
            print(f"❌ Unit tests failed ({duration:.2f}s)")
            print(f"Error output: {result.stderr}")
            return False

    def run_integration_tests(self, pattern: Optional[str] = None) -> bool:
        """Run integration tests."""
        print(
            f"\n🔗 Running integration tests{f' (pattern: {pattern})' if pattern else ''}..."
        )

        # Start Ray cluster if available
        ray_started = self._start_ray_cluster()

        try:
            cmd = [sys.executable, "-m", "pytest", "tests/", "-m", "integration"]

            if pattern:
                cmd.extend(["-k", pattern])

            cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=xml:coverage-integration.xml",
                    "-v",
                    "--tb=short",
                    "--maxfail=5",
                    "--timeout=300",
                    "--junitxml=test-results-integration.xml",
                ]
            )

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time

            self.test_results["integration"] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "duration": duration,
            }

            if result.returncode == 0:
                print(f"✅ Integration tests passed ({duration:.2f}s)")
                return True
            else:
                print(f"❌ Integration tests failed ({duration:.2f}s)")
                print(f"Error output: {result.stderr}")
                return False

        finally:
            if ray_started:
                self._stop_ray_cluster()

    def run_performance_tests(self) -> bool:
        """Run performance and benchmark tests."""
        print("\n⚡ Running performance tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-json=benchmark-results.json",
            "-v",
            "--tb=short",
            "--timeout=600",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        self.test_results["performance"] = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "duration": duration,
        }

        if result.returncode == 0:
            print(f"✅ Performance tests passed ({duration:.2f}s)")
            return True
        else:
            print(f"❌ Performance tests failed ({duration:.2f}s)")
            print(f"Error output: {result.stderr}")
            return False

    def run_memory_tests(self) -> bool:
        """Run memory usage tests."""
        print("\n🧠 Running memory tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "memory",
            "-v",
            "--tb=short",
            "--timeout=300",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        self.test_results["memory"] = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "duration": duration,
        }

        if result.returncode == 0:
            print(f"✅ Memory tests passed ({duration:.2f}s)")
            return True
        else:
            print(f"❌ Memory tests failed ({duration:.2f}s)")
            print(f"Error output: {result.stderr}")
            return False

    def run_end_to_end_tests(self) -> bool:
        """Run end-to-end tests."""
        print("\n🎯 Running end-to-end tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "e2e",
            "-v",
            "--tb=short",
            "--timeout=600",
            "--junitxml=test-results-e2e.xml",
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        self.test_results["e2e"] = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "errors": result.stderr,
            "duration": duration,
        }

        if result.returncode == 0:
            print(f"✅ End-to-end tests passed ({duration:.2f}s)")
            return True
        else:
            print(f"❌ End-to-end tests failed ({duration:.2f}s)")
            print(f"Error output: {result.stderr}")
            return False

    def check_coverage(self, target: float = 92.0) -> bool:
        """Check if coverage meets target."""
        print(f"\n📊 Checking coverage target ({target}%)...")

        coverage_file = Path("coverage.json")
        if not coverage_file.exists():
            print("⚠️ No coverage file found")
            return False

        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)

            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

            print(f"📈 Total coverage: {total_coverage:.2f}%")

            if total_coverage >= target:
                print(f"✅ Coverage target met ({total_coverage:.2f}% >= {target}%)")
                return True
            else:
                print(f"❌ Coverage below target ({total_coverage:.2f}% < {target}%)")
                return False

        except Exception as e:
            print(f"❌ Error reading coverage data: {e}")
            return False

    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        print("\n🔍 Running code quality checks...")

        checks = {
            "black": [
                sys.executable,
                "-m",
                "black",
                "--check",
                "--diff",
                "src/",
                "tests/",
            ],
            "isort": [
                sys.executable,
                "-m",
                "isort",
                "--check-only",
                "--diff",
                "src/",
                "tests/",
            ],
            "flake8": [
                sys.executable,
                "-m",
                "flake8",
                "src/",
                "tests/",
                "--max-line-length=100",
            ],
        }

        all_passed = True

        for check_name, cmd in checks.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"✅ {check_name} passed")
                else:
                    print(f"❌ {check_name} failed")
                    print(f"Output: {result.stdout}")
                    all_passed = False
            except subprocess.TimeoutExpired:
                print(f"⏰ {check_name} timed out")
                all_passed = False
            except FileNotFoundError:
                print(f"⚠️ {check_name} not available")

        return all_passed

    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n📋 Generating test report...")

        total_duration = time.time() - self.start_time

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results.values() if r["success"]),
                "failed": sum(
                    1 for r in self.test_results.values() if not r["success"]
                ),
            },
        }

        # Save JSON report
        with open("test-report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        md_report = self._generate_markdown_report(report)
        with open("test-report.md", "w") as f:
            f.write(md_report)

        print("✅ Test report generated: test-report.json, test-report.md")

    def _generate_markdown_report(self, report: dict) -> str:
        """Generate markdown test report."""
        md = f"""# Comprehensive Test Report

**Generated:** {report['timestamp']}
**Total Duration:** {report['total_duration']:.2f} seconds
**Tests Run:** {report['summary']['total_tests']}
**Passed:** {report['summary']['passed']}
**Failed:** {report['summary']['failed']}

## Test Results

| Test Suite | Status | Duration |
|------------|--------|----------|
"""

        for test_name, result in report["results"].items():
            status = "✅ Passed" if result["success"] else "❌ Failed"
            duration = f"{result['duration']:.2f}s" if "duration" in result else "N/A"
            md += f"| {test_name.title()} | {status} | {duration} |\n"

        md += """
## Coverage

Coverage reports are available in:
- `htmlcov/index.html` - HTML coverage report
- `coverage.xml` - XML coverage report (for CI)
- `coverage.json` - JSON coverage report

## Artifacts

- `test-results-*.xml` - JUnit test results
- `benchmark-results.json` - Performance benchmarks
- Test logs available in individual test outputs

## Status

"""

        if report["summary"]["failed"] == 0:
            md += "🎉 **All tests passed!** The codebase is ready for deployment."
        else:
            failed_count = report["summary"]["failed"]
            md += (
                f"⚠️ **{failed_count} test suite(s) failed.** "
                "Please review and fix issues before deployment."
            )

        return md

    def _start_ray_cluster(self) -> bool:
        """Start Ray cluster for integration tests."""
        try:
            result = subprocess.run(
                [
                    "ray",
                    "start",
                    "--head",
                    "--num-cpus=2",
                    "--num-gpus=0",
                    "--disable-usage-stats",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("✅ Ray cluster started")
                return True
            else:
                print("⚠️ Could not start Ray cluster")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Ray not available")
            return False

    def _stop_ray_cluster(self) -> None:
        """Stop Ray cluster."""
        try:
            subprocess.run(["ray", "stop"], capture_output=True, timeout=10)
            print("🛑 Ray cluster stopped")
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up test artifacts."""
        print("\n🧹 Cleaning up...")

        # Clean up temporary files
        temp_files = [
            ".coverage*",
            "*.pyc",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
        ]

        for pattern in temp_files:
            for file in Path(".").rglob(pattern):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
                except Exception:
                    pass

        print("✅ Cleanup completed")


def main() -> None:
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for trading RL agent"
    )

    parser.add_argument(
        "--suite",
        choices=["smoke", "unit", "integration", "performance", "memory", "e2e", "all"],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument("--pattern", help="Test pattern to match")
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage collection"
    )
    parser.add_argument(
        "--no-quality", action="store_true", help="Skip code quality checks"
    )
    parser.add_argument(
        "--target-coverage", type=float, default=92.0, help="Coverage target percentage"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up artifacts after tests"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = ComprehensiveTestRunner()

    if not runner.setup_environment():
        sys.exit(1)

    print(f"\n🚀 Starting comprehensive test run (suite: {args.suite})")

    try:
        success = True

        # Run code quality checks
        if not args.no_quality and args.suite in ["all"]:
            if not runner.run_code_quality_checks():
                success = False

        # Run specific test suites
        if args.suite in ["smoke", "all"]:
            if not runner.run_smoke_tests():
                success = False

        if args.suite in ["unit", "all"]:
            if not runner.run_unit_tests(args.pattern, not args.no_coverage):
                success = False

        if args.suite in ["integration", "all"]:
            if not runner.run_integration_tests(args.pattern):
                success = False

        if args.suite in ["performance", "all"]:
            if not runner.run_performance_tests():
                success = False

        if args.suite in ["memory", "all"]:
            if not runner.run_memory_tests():
                success = False

        if args.suite in ["e2e", "all"]:
            if not runner.run_end_to_end_tests():
                success = False

        # Check coverage if not disabled
        if not args.no_coverage and args.suite in ["unit", "integration", "all"]:
            if not runner.check_coverage(args.target_coverage):
                success = False

        # Generate report
        runner.generate_report()

        if success:
            print("\n🎉 All tests completed successfully!")
            exit_code = 0
        else:
            print("\n❌ Some tests failed. Check the report for details.")
            exit_code = 1

    finally:
        if args.cleanup:
            runner.cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
