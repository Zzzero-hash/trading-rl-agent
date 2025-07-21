#!/usr/bin/env python3
"""
Dependency Analysis Tool for Trading RL Agent
Shows detailed information about dependencies and their sizes
"""

import json
import subprocess
import sys
from pathlib import Path


class DependencyAnalyzer:
    """Analyze dependencies and their characteristics"""

    def __init__(self) -> None:
        self.requirements_files = {
            "core": "requirements-core.txt",
            "ml": "requirements-ml.txt",
            "full": "requirements-full.txt",
            "dev": "requirements-dev.txt",
            "production": "requirements-production.txt",
        }

        # Known package sizes (approximate, in MB)
        self.package_sizes = {
            "numpy": 25,
            "pandas": 15,
            "torch": 800,
            "scikit-learn": 8,
            "matplotlib": 20,
            "seaborn": 2,
            "scipy": 30,
            "gymnasium": 5,
            "stable-baselines3": 15,
            "ray": 200,
            "yfinance": 1,
            "mlflow": 50,
            "tensorboard": 30,
            "wandb": 20,
            "fastapi": 5,
            "uvicorn": 3,
            "gunicorn": 2,
            "pytest": 8,
            "black": 5,
            "ruff": 3,
            "mypy": 15,
            "jupyter": 40,
            "notebook": 30,
        }

    def parse_requirements_file(self, filename: str) -> list[str]:
        """Parse a requirements file and extract package names"""
        if not Path(filename).exists():
            return []

        packages = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-r"):
                    # Extract package name (remove version constraints)
                    package = line.split(">=")[0].split("==")[0].split("<")[0].strip()
                    packages.append(package)
        return packages

    def estimate_size(self, packages: list[str]) -> int:
        """Estimate total size of packages in MB"""
        total_size = 0
        for package in packages:
            if package in self.package_sizes:
                total_size += self.package_sizes[package]
            else:
                # Default estimate for unknown packages
                total_size += 5
        return total_size

    def analyze_profile(self, profile: str) -> dict:
        """Analyze a specific profile"""
        requirements_file = self.requirements_files.get(profile)
        if not requirements_file:
            return {}

        packages = self.parse_requirements_file(requirements_file)
        estimated_size = self.estimate_size(packages)

        return {
            "profile": profile,
            "requirements_file": requirements_file,
            "packages": packages,
            "package_count": len(packages),
            "estimated_size_mb": estimated_size,
            "exists": Path(requirements_file).exists(),
        }

    def show_profile_analysis(self, profile: str) -> None:
        """Show detailed analysis for a profile"""
        analysis = self.analyze_profile(profile)
        if not analysis:
            print(f"❌ Profile '{profile}' not found")
            return

        print(f"\n📊 Analysis for '{profile}' profile:")
        print("=" * 50)
        print(f"Requirements file: {analysis['requirements_file']}")
        print(f"Package count: {analysis['package_count']}")
        print(f"Estimated size: ~{analysis['estimated_size_mb']} MB")
        print(f"File exists: {'✅' if analysis['exists'] else '❌'}")

        if analysis["packages"]:
            print("\n📦 Packages:")
            for i, package in enumerate(analysis["packages"], 1):
                size = self.package_sizes.get(package, "~5")
                print(f"  {i:2d}. {package:<20} (~{size} MB)")

    def show_all_profiles(self) -> None:
        """Show analysis for all profiles"""
        print("🚀 Trading RL Agent - Dependency Analysis")
        print("=" * 50)

        all_analyses = []
        for profile in self.requirements_files:
            analysis = self.analyze_profile(profile)
            if analysis:
                all_analyses.append(analysis)

        # Sort by size
        all_analyses.sort(key=lambda x: x["estimated_size_mb"])

        print("\n📋 Profile Summary:")
        print("-" * 50)
        for analysis in all_analyses:
            status = "✅" if analysis["exists"] else "❌"
            print(
                f"{status} {analysis['profile']:<12} - "
                f"{analysis['package_count']:2d} packages, ~{analysis['estimated_size_mb']:4d} MB"
            )

        print("\n💡 Recommendations:")
        print("-" * 50)
        print("• Start with 'core' for basic functionality")
        print("• Add 'ml' when you need machine learning")
        print("• Use 'dev' for development and testing")
        print("• Choose 'full' or 'production' for complete setup")

    def show_installation_commands(self) -> None:
        """Show installation commands for all profiles"""
        print("\n🔧 Installation Commands:")
        print("=" * 50)

        for profile in self.requirements_files:
            analysis = self.analyze_profile(profile)
            if analysis and analysis["exists"]:
                print(f"\n{profile.upper()} profile (~{analysis['estimated_size_mb']} MB):")
                print(f"  pip install -r {analysis['requirements_file']}")
                print(f"  # or: pip install -e .[{profile}]")

    def check_current_installation(self) -> None:
        """Check what's currently installed"""
        print("\n🔍 Current Installation Check:")
        print("=" * 50)

        try:
            # Get list of installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )

            installed_packages = json.loads(result.stdout)
            installed_names = {pkg["name"].lower() for pkg in installed_packages}

            # Check against our profiles
            for profile in self.requirements_files:
                analysis = self.analyze_profile(profile)
                if not analysis:
                    continue

                profile_packages = set(analysis["packages"])
                installed_count = len(profile_packages.intersection(installed_names))
                coverage = (installed_count / len(profile_packages)) * 100 if profile_packages else 0

                print(f"{profile:<12}: {installed_count:2d}/{len(profile_packages):2d} packages ({coverage:5.1f}%)")

        except subprocess.CalledProcessError:
            print("❌ Could not check current installation")
        except json.JSONDecodeError:
            print("❌ Could not parse pip list output")


def main() -> None:
    """Main entry point"""
    analyzer = DependencyAnalyzer()

    if len(sys.argv) > 1:
        profile = sys.argv[1]
        analyzer.show_profile_analysis(profile)
    else:
        analyzer.show_all_profiles()
        analyzer.show_installation_commands()
        analyzer.check_current_installation()

        print("\n💡 Usage:")
        print(f"  python {sys.argv[0]} [profile]  # Analyze specific profile")
        print(f"  python {sys.argv[0]}            # Show all profiles")


if __name__ == "__main__":
    main()
