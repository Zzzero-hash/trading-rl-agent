#!/usr/bin/env python3
"""Check requirements files for consistency and security issues."""  # noqa: D212

from pathlib import Path
import re
import sys
from typing import Dict, List, Set


def parse_requirements(file_path: Path) -> set[str]:
    """Parse requirements file and extract package names."""
    if not file_path.exists():
        return set()

    packages = set()
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("-r "):
                    # Handle included requirements files
                    included_file = file_path.parent / line[3:].strip()
                    packages.update(parse_requirements(included_file))
                elif not line.startswith("-"):
                    # Extract package name (before version specifiers)
                    package = re.split(r"[>=<!=]", line)[0].strip()
                    if package:
                        packages.add(package.lower())

    return packages


def check_requirements_consistency() -> bool:
    """Check consistency between different requirements files."""
    project_root = Path(__file__).parent.parent

    # Define requirements files and their relationships
    requirements_files = {
        "core": project_root / "requirements-core.txt",
        "test": project_root / "requirements-test.txt",
        "ml": project_root / "requirements-ml.txt",
        "full": project_root / "requirements-full.txt",
        "main": project_root / "requirements.txt",
    }

    # Parse all files
    parsed_reqs = {}
    for name, path in requirements_files.items():
        parsed_reqs[name] = parse_requirements(path)

    errors = []

    # Check that core packages are included in other files
    core_packages = parsed_reqs.get("core", set())
    for name, packages in parsed_reqs.items():
        if name != "core" and core_packages:
            missing = core_packages - packages
            if (
                missing and name != "test"
            ):  # Test requirements might not include all core
                errors.append(
                    f"Missing core packages in {name}: {missing}"
                )  # Check for duplicate packages across files

    all_packages: dict[str, list[str]] = {}
    for name, packages in parsed_reqs.items():
        for package in packages:
            if package not in all_packages:
                all_packages[package] = []
            all_packages[package].append(name)

    # Print warnings for packages in multiple files
    for package, files in all_packages.items():
        if len(files) > 2:  # Allow some overlap
            print(f"Warning: {package} appears in multiple files: {files}")

    if errors:
        print("Requirements consistency errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Requirements files are consistent")
    return True


def check_pinned_versions() -> bool:
    """Check that critical packages have pinned versions."""
    project_root = Path(__file__).parent.parent
    main_req_file = project_root / "requirements.txt"

    if not main_req_file.exists():
        print("Warning: requirements.txt not found")
        return True

    critical_packages = {
        "torch",
        "numpy",
        "pandas",
        "ray",
        "gymnasium",
        "pytest",
        "black",
        "isort",
        "flake8",
        "mypy",
    }

    unpinned: list[str] = []
    with open(main_req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            pkg = re.split(r"[>=<!=]", line)[0].strip().lower()
            if pkg not in critical_packages:
                continue

            # Check if package has version constraint
            has_version_constraint = re.search(r"[>=<!=]", line) is not None
            if not has_version_constraint:
                unpinned.append(pkg)

    if unpinned:
        print(f"Warning: Critical packages without version pins: {unpinned}")
        return False

    print("✅ Critical packages have version constraints")
    return True


def main() -> int:
    """Main function."""
    success = True

    success &= check_requirements_consistency()
    success &= check_pinned_versions()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
