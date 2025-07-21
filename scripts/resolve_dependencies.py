#!/usr/bin/env python3
"""
Dependency Resolution Script for Trading RL Agent

This script automatically resolves common dependency conflicts and ensures
compatibility across different environments.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class DependencyResolver:
    """Automated dependency resolution for the trading system."""

    def __init__(self):
        self.resolved_issues = []
        self.failed_resolutions = []
        self.actions_taken = []

    def log_action(self, action: str, details: str) -> None:
        """Log an action taken during resolution."""
        self.actions_taken.append(f"{action}: {details}")
        print(f"🔧 {action}: {details}")

    def log_resolution(self, issue: str, solution: str) -> None:
        """Log a successful resolution."""
        self.resolved_issues.append(f"{issue} -> {solution}")
        print(f"✅ Resolved: {issue}")

    def log_failure(self, issue: str, error: str) -> None:
        """Log a failed resolution."""
        self.failed_resolutions.append(f"{issue}: {error}")
        print(f"❌ Failed: {issue} - {error}")

    def check_pip_install(self, package: str, version_constraint: str | None = None) -> bool:
        """Check if a package can be installed with pip."""
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--dry-run"]
            if version_constraint:
                cmd.append(f"{package}{version_constraint}")
            else:
                cmd.append(package)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            self.log_failure(f"pip install check for {package}", str(e))
            return False

    def install_package(self, package: str, version_constraint: str | None = None) -> bool:
        """Install a package with pip."""
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if version_constraint:
                cmd.append(f"{package}{version_constraint}")
            else:
                cmd.append(package)

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.log_resolution(f"Install {package}", "Successfully installed")
                return True
            self.log_failure(f"Install {package}", result.stderr)
            return False
        except Exception as e:
            self.log_failure(f"Install {package}", str(e))
            return False

    def resolve_ray_compatibility(self) -> bool:
        """Resolve Ray compatibility issues."""
        self.log_action("Ray Compatibility", "Checking Ray installation and compatibility")

        try:
            import ray

            current_version = ray.__version__
            self.log_action("Ray Version", f"Current: {current_version}")

            # Check if current version is compatible
            if current_version.startswith("2."):
                self.log_resolution("Ray Version", f"Version {current_version} is compatible")
                return True
            # Try to install compatible version
            return self.install_package("ray[rllib,tune]", ">=2.6.0,<3.0.0")

        except ImportError:
            # Ray not installed, install it
            return self.install_package("ray[rllib,tune]", ">=2.6.0,<3.0.0")

    def resolve_structlog_issues(self) -> bool:
        """Resolve structlog import issues."""
        self.log_action("Structlog", "Checking structlog installation")

        try:
            import structlog

            version = structlog.__version__
            self.log_resolution("Structlog", f"Version {version} installed and working")
            return True
        except ImportError:
            # Install structlog
            return self.install_package("structlog", ">=23.1.0")

    def resolve_test_dependencies(self) -> bool:
        """Resolve test environment dependencies."""
        self.log_action("Test Dependencies", "Installing test-specific dependencies")

        # Install test requirements
        test_req_file = Path("requirements-test.txt")
        if test_req_file.exists():
            if self.install_requirements_file(test_req_file):
                self.log_resolution("Test Dependencies", "Test requirements installed")
                return True
            return False
        # Fallback to core test packages
        test_packages = [
            ("pytest", ">=7.4.0"),
            ("pytest-cov", ">=4.1.0"),
            ("pytest-mock", ">=3.11.0"),
        ]

        all_installed = True
        for package, version in test_packages:
            if not self.install_package(package, version):
                all_installed = False

        return all_installed

    def install_requirements_file(self, req_file: Path) -> bool:
        """Install dependencies from a requirements file."""
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                self.log_resolution(f"Requirements file {req_file.name}", "Successfully installed")
                return True
            self.log_failure(f"Requirements file {req_file.name}", result.stderr)
            return False
        except Exception as e:
            self.log_failure(f"Requirements file {req_file.name}", str(e))
            return False

    def resolve_version_conflicts(self) -> bool:
        """Resolve common version conflicts."""
        self.log_action("Version Conflicts", "Checking for version conflicts")

        # Common conflict resolutions
        conflict_resolutions = [
            ("numpy", ">=1.24.0,<2.0.0"),
            ("pandas", ">=2.0.0,<3.0.0"),
            ("torch", ">=2.0.0,<3.0.0"),
            ("scikit-learn", ">=1.3.0,<2.0.0"),
        ]

        all_resolved = True
        for package, version in conflict_resolutions:
            if not self.check_pip_install(package, version):
                if self.install_package(package, version):
                    self.log_resolution(f"Version conflict {package}", f"Installed {version}")
                else:
                    all_resolved = False

        return all_resolved

    def create_environment_summary(self) -> dict[str, Any]:
        """Create a summary of the current environment."""
        summary = {
            "python_version": sys.version,
            "resolved_issues": self.resolved_issues,
            "failed_resolutions": self.failed_resolutions,
            "actions_taken": self.actions_taken,
        }

        # Check key packages
        key_packages = ["numpy", "pandas", "torch", "ray", "structlog", "pytest"]
        installed_packages = {}

        for package in key_packages:
            try:
                module = __import__(package)
                if hasattr(module, "__version__"):
                    installed_packages[package] = module.__version__
                else:
                    installed_packages[package] = "installed"
            except ImportError:
                installed_packages[package] = "not_installed"

        summary["installed_packages"] = installed_packages
        return summary

    def run_full_resolution(self) -> dict[str, Any]:
        """Run the complete dependency resolution process."""
        print("🔧 Starting dependency resolution...")
        print("=" * 60)

        # Run all resolution steps
        resolution_steps = [
            ("Ray Compatibility", self.resolve_ray_compatibility),
            ("Structlog Issues", self.resolve_structlog_issues),
            ("Test Dependencies", self.resolve_test_dependencies),
            ("Version Conflicts", self.resolve_version_conflicts),
        ]

        for step_name, step_func in resolution_steps:
            print(f"\n📋 Running {step_name} resolution...")
            try:
                step_func()
            except Exception as e:
                self.log_failure(step_name, str(e))

        # Generate summary
        summary = self.create_environment_summary()

        print("\n" + "=" * 60)
        print("📊 RESOLUTION SUMMARY")
        print("=" * 60)

        print(f"Resolved Issues: {len(self.resolved_issues)}")
        print(f"Failed Resolutions: {len(self.failed_resolutions)}")
        print(f"Actions Taken: {len(self.actions_taken)}")

        if self.resolved_issues:
            print("\n✅ RESOLVED ISSUES:")
            for issue in self.resolved_issues:
                print(f"  • {issue}")

        if self.failed_resolutions:
            print("\n❌ FAILED RESOLUTIONS:")
            for failure in self.failed_resolutions:
                print(f"  • {failure}")

        if self.actions_taken:
            print("\n🔧 ACTIONS TAKEN:")
            for action in self.actions_taken:
                print(f"  • {action}")

        # Save summary to file
        summary_file = Path("dependency_resolution_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📄 Summary saved to: {summary_file}")

        return summary


def main():
    """Main entry point."""
    resolver = DependencyResolver()
    summary = resolver.run_full_resolution()

    # Exit with appropriate code
    if summary["failed_resolutions"]:
        print(f"\n⚠️  Resolution completed with {len(summary['failed_resolutions'])} failures")
        print("Please review the failed resolutions above")
        sys.exit(1)
    else:
        print("\n✅ All dependency resolutions completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
