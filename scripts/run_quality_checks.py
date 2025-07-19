#!/usr/bin/env python3
"""Comprehensive quality assurance script for trading RL agent."""

import argparse
import json
import subprocess
import sys
import time


class QualityChecker:
    """Comprehensive quality assurance checker for trading RL agent."""

    def __init__(self, verbose: bool = False):
        """Initialize the quality checker."""
        self.verbose = verbose
        self.results = {}
        self.start_time = time.time()

    def run_command(self, command: list[str], description: str) -> tuple[bool, str]:
        """Run a command and return success status and output."""
        if self.verbose:
            print(f"Running: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if self.verbose:
                print(f"{description}: {'✓ PASS' if success else '✗ FAIL'}")
                if output.strip():
                    print(f"Output: {output.strip()}")

            return success, output
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"{description}: ✗ TIMEOUT")
            return False, "Command timed out"
        except Exception as e:
            if self.verbose:
                print(f"{description}: ✗ ERROR - {e}")
            return False, str(e)

    def run_mutation_testing(self) -> bool:
        """Run mutation testing using mutmut."""
        print("\n🔬 Running Mutation Testing...")

        # Run mutmut
        success, output = self.run_command(["mutmut", "run"], "Mutation testing")

        if success:
            # Get mutation results
            results_success, results_output = self.run_command(["mutmut", "results"], "Mutation results")

            if results_success:
                # Parse results
                lines = results_output.strip().split("\n")
                total_mutations = 0
                killed_mutations = 0

                for line in lines:
                    if "killed" in line.lower():
                        killed_mutations += 1
                    if "survived" in line.lower():
                        total_mutations += 1

                if total_mutations > 0:
                    mutation_score = (killed_mutations / total_mutations) * 100
                    print(f"Mutation Score: {mutation_score:.1f}% ({killed_mutations}/{total_mutations})")

                    if mutation_score < 80:
                        print("⚠️  Warning: Mutation score below 80%")
                        success = False

        self.results["mutation_testing"] = success
        return success

    def run_security_tests(self) -> bool:
        """Run security tests."""
        print("\n🔒 Running Security Tests...")

        # Run bandit security scanner
        bandit_success, bandit_output = self.run_command(["bandit", "-r", "src/", "-f", "json"], "Bandit security scan")

        # Run safety check
        safety_success, safety_output = self.run_command(["safety", "check", "--json"], "Safety dependency check")

        # Run semgrep
        semgrep_success, semgrep_output = self.run_command(
            ["semgrep", "scan", "--config=auto", "--json"], "Semgrep security scan"
        )

        # Run pip-audit
        pip_audit_success, pip_audit_output = self.run_command(["pip-audit", "--format=json"], "Pip audit")

        success = all([bandit_success, safety_success, semgrep_success, pip_audit_success])
        self.results["security_tests"] = success
        return success

    def run_code_quality_tests(self) -> bool:
        """Run code quality tests."""
        print("\n📊 Running Code Quality Tests...")

        # Run ruff linting
        ruff_success, ruff_output = self.run_command(["ruff", "check", "src/"], "Ruff linting")

        # Run mypy type checking
        mypy_success, mypy_output = self.run_command(["mypy", "src/"], "MyPy type checking")

        # Run black formatting check
        black_success, black_output = self.run_command(["black", "--check", "src/"], "Black formatting check")

        # Run isort import sorting check
        isort_success, isort_output = self.run_command(["isort", "--check-only", "src/"], "Isort import sorting")

        success = all([ruff_success, mypy_success, black_success, isort_success])
        self.results["code_quality"] = success
        return success

    def run_documentation_tests(self) -> bool:
        """Run documentation tests."""
        print("\n📚 Running Documentation Tests...")

        # Run pydocstyle
        pydocstyle_success, pydocstyle_output = self.run_command(
            ["pydocstyle", "src/"], "Pydocstyle documentation check"
        )

        # Run doc8 (if available)
        doc8_success, doc8_output = self.run_command(["doc8", "docs/"], "Doc8 documentation check")

        # Run custom documentation tests
        doc_tests_success, doc_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_documentation_accuracy.py", "-v"],
            "Documentation accuracy tests",
        )

        success = all([pydocstyle_success, doc_tests_success])
        self.results["documentation"] = success
        return success

    def run_type_hints_tests(self) -> bool:
        """Run type hints validation tests."""
        print("\n🔍 Running Type Hints Validation...")

        # Run custom type hints tests
        type_hints_success, type_hints_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_type_hints_validation.py", "-v"],
            "Type hints validation tests",
        )

        self.results["type_hints"] = type_hints_success
        return type_hints_success

    def run_security_validation_tests(self) -> bool:
        """Run security validation tests."""
        print("\n🛡️ Running Security Validation Tests...")

        # Run input validation tests
        input_validation_success, input_validation_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/security/test_input_validation.py", "-v"],
            "Input validation security tests",
        )

        # Run authentication tests
        auth_success, auth_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/security/test_authentication_authorization.py", "-v"],
            "Authentication and authorization tests",
        )

        # Run data sanitization tests
        sanitization_success, sanitization_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/security/test_data_sanitization.py", "-v"],
            "Data sanitization tests",
        )

        # Run API security tests
        api_security_success, api_security_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/security/test_api_security.py", "-v"], "API security tests"
        )

        success = all([input_validation_success, auth_success, sanitization_success, api_security_success])
        self.results["security_validation"] = success
        return success

    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("\n⚡ Running Performance Tests...")

        # Run pytest with profiling
        perf_success, perf_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/performance/", "-v"], "Performance tests"
        )

        # Run memory profiling
        memory_success, memory_output = self.run_command(
            [sys.executable, "-m", "memory_profiler", "src/trading_rl_agent/main.py"], "Memory profiling"
        )

        success = perf_success
        self.results["performance"] = success
        return success

    def run_error_message_tests(self) -> bool:
        """Run error message accuracy tests."""
        print("\n⚠️ Running Error Message Tests...")

        # Run custom error message tests
        error_tests_success, error_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_error_messages.py", "-v"],
            "Error message accuracy tests",
        )

        self.results["error_messages"] = error_tests_success
        return error_tests_success

    def run_logging_security_tests(self) -> bool:
        """Run logging security tests."""
        print("\n📝 Running Logging Security Tests...")

        # Run custom logging security tests
        logging_tests_success, logging_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_logging_security.py", "-v"], "Logging security tests"
        )

        self.results["logging_security"] = logging_tests_success
        return logging_tests_success

    def run_configuration_validation_tests(self) -> bool:
        """Run configuration validation tests."""
        print("\n⚙️ Running Configuration Validation Tests...")

        # Run custom configuration validation tests
        config_tests_success, config_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_configuration_validation.py", "-v"],
            "Configuration validation tests",
        )

        self.results["configuration_validation"] = config_tests_success
        return config_tests_success

    def run_environment_variable_tests(self) -> bool:
        """Run environment variable handling tests."""
        print("\n🌍 Running Environment Variable Tests...")

        # Run custom environment variable tests
        env_tests_success, env_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_environment_variables.py", "-v"],
            "Environment variable handling tests",
        )

        self.results["environment_variables"] = env_tests_success
        return env_tests_success

    def run_dependency_security_tests(self) -> bool:
        """Run dependency security tests."""
        print("\n📦 Running Dependency Security Tests...")

        # Run safety check
        safety_success, safety_output = self.run_command(["safety", "check", "--json"], "Safety dependency check")

        # Run pip-audit
        pip_audit_success, pip_audit_output = self.run_command(["pip-audit", "--format=json"], "Pip audit")

        # Run custom dependency tests
        dep_tests_success, dep_tests_output = self.run_command(
            [sys.executable, "-m", "pytest", "tests/quality/test_dependency_security.py", "-v"],
            "Dependency security tests",
        )

        success = all([safety_success, pip_audit_success, dep_tests_success])
        self.results["dependency_security"] = success
        return success

    def generate_report(self) -> None:
        """Generate a comprehensive quality report."""
        print("\n" + "=" * 80)
        print("📋 QUALITY ASSURANCE REPORT")
        print("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        failed_tests = total_tests - passed_tests

        print("\nOverall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ✓")
        print(f"  Failed: {failed_tests} ✗")
        print(f"  Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")

        print(f"\nExecution Time: {time.time() - self.start_time:.2f} seconds")

        # Save results to JSON file
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": time.time() - self.start_time,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "results": self.results,
        }

        with open("quality_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print("\nReport saved to: quality_report.json")

        if failed_tests > 0:
            print("\n❌ Quality checks failed! Please review the issues above.")
            sys.exit(1)
        else:
            print("\n✅ All quality checks passed!")

    def run_all_checks(self) -> None:
        """Run all quality assurance checks."""
        print("🚀 Starting Comprehensive Quality Assurance Checks...")
        print("=" * 80)

        # Run all checks
        checks = [
            ("Mutation Testing", self.run_mutation_testing),
            ("Security Tests", self.run_security_tests),
            ("Code Quality", self.run_code_quality_tests),
            ("Documentation", self.run_documentation_tests),
            ("Type Hints", self.run_type_hints_tests),
            ("Security Validation", self.run_security_validation_tests),
            ("Performance", self.run_performance_tests),
            ("Error Messages", self.run_error_message_tests),
            ("Logging Security", self.run_logging_security_tests),
            ("Configuration Validation", self.run_configuration_validation_tests),
            ("Environment Variables", self.run_environment_variable_tests),
            ("Dependency Security", self.run_dependency_security_tests),
        ]

        for check_name, check_func in checks:
            try:
                check_func()
            except Exception as e:
                print(f"❌ Error running {check_name}: {e}")
                self.results[check_name.lower().replace(" ", "_")] = False

        # Generate final report
        self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive quality assurance checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--check",
        "-c",
        choices=[
            "mutation",
            "security",
            "quality",
            "docs",
            "types",
            "validation",
            "performance",
            "errors",
            "logging",
            "config",
            "env",
            "deps",
        ],
        help="Run specific check only",
    )

    args = parser.parse_args()

    checker = QualityChecker(verbose=args.verbose)

    if args.check:
        # Run specific check
        check_map = {
            "mutation": checker.run_mutation_testing,
            "security": checker.run_security_tests,
            "quality": checker.run_code_quality_tests,
            "docs": checker.run_documentation_tests,
            "types": checker.run_type_hints_tests,
            "validation": checker.run_security_validation_tests,
            "performance": checker.run_performance_tests,
            "errors": checker.run_error_message_tests,
            "logging": checker.run_logging_security_tests,
            "config": checker.run_configuration_validation_tests,
            "env": checker.run_environment_variable_tests,
            "deps": checker.run_dependency_security_tests,
        }

        check_func = check_map.get(args.check)
        if check_func:
            check_func()
            checker.generate_report()
        else:
            print(f"Unknown check: {args.check}")
            sys.exit(1)
    else:
        # Run all checks
        checker.run_all_checks()


if __name__ == "__main__":
    main()
