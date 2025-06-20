#!/usr/bin/env python3
"""
Development automation script for code quality, testing, and documentation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


class DevAutomation:
    """Development automation utilities."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.tests_dir = project_root / "tests"
        self.docs_dir = project_root / "docs"

    def format_code(self, check_only: bool = False) -> bool:
        """Format code with Black and isort."""
        print("🎨 Formatting code...")

        success = True

        # Black formatting
        black_cmd = ["black"]
        if check_only:
            black_cmd.extend(["--check", "--diff"])
        black_cmd.extend([str(self.src_dir), str(self.tests_dir)])

        try:
            result = subprocess.run(black_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(
                    "✅ Black formatting"
                    + (" check passed" if check_only else " applied")
                )
            else:
                print("❌ Black formatting failed:")
                print(result.stdout)
                success = False
        except FileNotFoundError:
            print("❌ Black not found. Install with: pip install black")
            return False

        # isort import sorting
        isort_cmd = ["isort"]
        if check_only:
            isort_cmd.extend(["--check-only", "--diff"])
        isort_cmd.extend([str(self.src_dir), str(self.tests_dir)])

        try:
            result = subprocess.run(isort_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ isort" + (" check passed" if check_only else " applied"))
            else:
                print("❌ isort failed:")
                print(result.stdout)
                success = False
        except FileNotFoundError:
            print("❌ isort not found. Install with: pip install isort")
            return False

        return success

    def lint_code(self) -> bool:
        """Run code linting with flake8."""
        print("🔍 Linting code...")

        try:
            cmd = [
                "flake8",
                str(self.src_dir),
                str(self.tests_dir),
                "--max-line-length=88",
                "--extend-ignore=E203,W503",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Linting passed")
                return True
            else:
                print("❌ Linting failed:")
                print(result.stdout)
                return False
        except FileNotFoundError:
            print("❌ flake8 not found. Install with: pip install flake8")
            return False

    def type_check(self) -> bool:
        """Run type checking with mypy."""
        print("🔍 Type checking...")

        try:
            cmd = ["mypy", str(self.src_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Type checking passed")
                return True
            else:
                print("❌ Type checking failed:")
                print(result.stdout)
                return False
        except FileNotFoundError:
            print("❌ mypy not found. Install with: pip install mypy")
            return False

    def security_check(self) -> bool:
        """Run security checks with bandit."""
        print("🔒 Security checking...")

        try:
            cmd = [
                "bandit",
                "-r",
                str(self.src_dir),
                "-f",
                "json",
                "-o",
                "bandit-report.json",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse bandit results
            try:
                with open("bandit-report.json") as f:
                    report = json.load(f)

                high_issues = [
                    issue
                    for issue in report.get("results", [])
                    if issue.get("issue_severity") == "HIGH"
                ]
                medium_issues = [
                    issue
                    for issue in report.get("results", [])
                    if issue.get("issue_severity") == "MEDIUM"
                ]

                if high_issues:
                    print(f"❌ Found {len(high_issues)} high-severity security issues")
                    for issue in high_issues[:3]:  # Show first 3
                        print(
                            f"  - {issue.get('test_name')}: {issue.get('issue_text')}"
                        )
                    return False
                elif medium_issues:
                    print(
                        f"⚠️ Found {len(medium_issues)} medium-severity security issues"
                    )
                    return True  # Don't fail on medium issues
                else:
                    print("✅ Security check passed")
                    return True

            except (json.JSONDecodeError, FileNotFoundError):
                print("⚠️ Could not parse security report")
                return True

        except FileNotFoundError:
            print("❌ bandit not found. Install with: pip install bandit")
            return False

    def run_tests(self, test_type: str = "unit", coverage: bool = True) -> bool:
        """Run tests with pytest."""
        print(f"🧪 Running {test_type} tests...")

        cmd = ["python", "-m", "pytest"]

        # Add test selection based on type
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "fast":
            cmd.extend(["-m", "not slow"])
        elif test_type == "all":
            pass  # Run all tests

        # Add coverage
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])

        cmd.extend(["-v", "--tb=short"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ {test_type.capitalize()} tests passed")
                return True
            else:
                print(f"❌ {test_type.capitalize()} tests failed:")
                print(result.stdout)
                return False
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False

    def build_docs(self) -> bool:
        """Build documentation."""
        print("📚 Building documentation...")

        try:
            build_script = self.project_root / "scripts" / "build_docs.py"
            if build_script.exists():
                result = subprocess.run(
                    [sys.executable, str(build_script)], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("✅ Documentation built successfully")
                    return True
                else:
                    print("❌ Documentation build failed:")
                    print(result.stdout)
                    return False
            else:
                print("⚠️ Documentation build script not found")
                return False
        except Exception as e:
            print(f"❌ Documentation build error: {e}")
            return False

    def check_dependencies(self) -> bool:
        """Check for dependency issues."""
        print("📦 Checking dependencies...")

        try:
            # Check for security vulnerabilities
            result = subprocess.run(
                ["safety", "check", "--json"], capture_output=True, text=True
            )

            if result.returncode == 0:
                print("✅ No known security vulnerabilities")
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    if vulnerabilities:
                        print(
                            f"⚠️ Found {len(vulnerabilities)} security vulnerabilities"
                        )
                        for vuln in vulnerabilities[:3]:  # Show first 3
                            print(f"  - {vuln.get('package')}: {vuln.get('advisory')}")
                except json.JSONDecodeError:
                    print("⚠️ Could not parse security report")

            return True  # Don't fail build on security issues

        except FileNotFoundError:
            print("⚠️ safety not found. Install with: pip install safety")
            return True

    def pre_commit_check(self) -> bool:
        """Run all pre-commit checks."""
        print("🚀 Running pre-commit checks...")

        checks = [
            ("Format Check", lambda: self.format_code(check_only=True)),
            ("Lint", self.lint_code),
            ("Type Check", self.type_check),
            ("Security Check", self.security_check),
            ("Unit Tests", lambda: self.run_tests("unit", coverage=False)),
        ]

        failed_checks = []

        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                if not check_func():
                    failed_checks.append(check_name)
            except Exception as e:
                print(f"❌ {check_name} failed with exception: {e}")
                failed_checks.append(check_name)

        if failed_checks:
            print(f"\n❌ Pre-commit checks failed: {failed_checks}")
            return False
        else:
            print("\n✅ All pre-commit checks passed!")
            return True

    def setup_dev_environment(self) -> bool:
        """Set up development environment."""
        print("🔧 Setting up development environment...")

        # Install pre-commit hooks
        try:
            subprocess.run(["pre-commit", "install"], check=True)
            print("✅ Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Could not install pre-commit hooks")

        # Create necessary directories
        dirs_to_create = [
            self.project_root / ".vscode",
            self.project_root / "logs",
            self.project_root / "experiments",
            self.project_root / "models",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {dir_path.name}")

        # Create VS Code settings
        vscode_settings = {
            "python.formatting.provider": "black",
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "editor.formatOnSave": True,
            "python.sortImports.args": ["--profile", "black"],
            "[python]": {"editor.codeActionsOnSave": {"source.organizeImports": True}},
        }

        vscode_settings_path = self.project_root / ".vscode" / "settings.json"
        with open(vscode_settings_path, "w") as f:
            json.dump(vscode_settings, f, indent=2)
        print("✅ VS Code settings configured")

        return True

    def clean_project(self) -> bool:
        """Clean project artifacts."""
        print("🧹 Cleaning project...")

        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.coverage",
            "**/htmlcov",
            "**/coverage.xml",
            "**/test-results.xml",
            "**/bandit-report.json",
        ]

        cleaned_count = 0

        for pattern in patterns_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    path.unlink()
                    cleaned_count += 1
                elif path.is_dir():
                    import shutil

                    shutil.rmtree(path)
                    cleaned_count += 1

        print(f"✅ Cleaned {cleaned_count} files/directories")
        return True


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Development automation tools")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Format command
    format_parser = subparsers.add_parser("format", help="Format code")
    format_parser.add_argument(
        "--check", action="store_true", help="Check formatting only"
    )

    # Lint command
    subparsers.add_parser("lint", help="Lint code")

    # Type check command
    subparsers.add_parser("typecheck", help="Run type checking")

    # Security check command
    subparsers.add_parser("security", help="Run security checks")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--type",
        choices=["unit", "integration", "fast", "all"],
        default="unit",
        help="Type of tests to run",
    )
    test_parser.add_argument("--no-coverage", action="store_true", help="Skip coverage")

    # Documentation command
    subparsers.add_parser("docs", help="Build documentation")

    # Pre-commit command
    subparsers.add_parser("precommit", help="Run pre-commit checks")

    # Setup command
    subparsers.add_parser("setup", help="Set up development environment")

    # Clean command
    subparsers.add_parser("clean", help="Clean project artifacts")

    # Full quality check
    subparsers.add_parser("quality", help="Run all quality checks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    project_root = Path(__file__).parent.parent
    dev = DevAutomation(project_root)

    # Execute command
    if args.command == "format":
        success = dev.format_code(check_only=args.check)
    elif args.command == "lint":
        success = dev.lint_code()
    elif args.command == "typecheck":
        success = dev.type_check()
    elif args.command == "security":
        success = dev.security_check()
    elif args.command == "test":
        success = dev.run_tests(args.type, coverage=not args.no_coverage)
    elif args.command == "docs":
        success = dev.build_docs()
    elif args.command == "precommit":
        success = dev.pre_commit_check()
    elif args.command == "setup":
        success = dev.setup_dev_environment()
    elif args.command == "clean":
        success = dev.clean_project()
    elif args.command == "quality":
        # Run all quality checks
        checks = [
            dev.format_code(check_only=True),
            dev.lint_code(),
            dev.type_check(),
            dev.security_check(),
            dev.run_tests("unit", coverage=True),
            dev.build_docs(),
        ]
        success = all(checks)
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
