#!/usr/bin/env python3
"""
Verify the documentation and code quality setup.
"""

from pathlib import Path
import subprocess
import sys


def check_tool(tool_name: str, command: list) -> bool:
    """Check if a tool is installed and working."""
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {tool_name}: Available")
            return True
        else:
            print(f"❌ {tool_name}: Error - {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print(f"❌ {tool_name}: Not found")
        return False


def check_files(project_root: Path) -> bool:
    """Check if required configuration files exist."""
    required_files = [
        ".pre-commit-config.yaml",
        "pyproject.toml",
        "docs/conf.py",
        "docs/Makefile",
        "CONTRIBUTING.md",
        "scripts/dev.py",
        "scripts/build_docs.py",
        "scripts/check_type_coverage.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required configuration files present")
        return True


def main() -> int:
    """Main verification function."""
    print("🔍 Verifying Documentation and Code Quality Setup")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    # Check tools
    tools = [
        ("Black", ["black", "--version"]),
        ("isort", ["isort", "--version"]),
        ("flake8", ["flake8", "--version"]),
        ("mypy", ["mypy", "--version"]),
        ("Sphinx", ["sphinx-build", "--version"]),
        ("pre-commit", ["pre-commit", "--version"]),
        ("bandit", ["bandit", "--version"]),
    ]

    tool_results = []
    for tool_name, command in tools:
        tool_results.append(check_tool(tool_name, command))

    # Check files
    files_ok = check_files(project_root)

    # Check pre-commit hooks
    hooks_installed = (project_root / ".git" / "hooks" / "pre-commit").exists()
    if hooks_installed:
        print("✅ Pre-commit hooks: Installed")
    else:
        print("❌ Pre-commit hooks: Not installed")

    print("\n" + "=" * 60)

    # Summary
    tools_ok = all(tool_results)
    all_ok = tools_ok and files_ok and hooks_installed

    if all_ok:
        print("🎉 Setup verification PASSED!")
        print("\nNext steps:")
        print(
            "1. Run 'python scripts/dev.py setup' to configure development environment"
        )
        print("2. Run 'python scripts/dev.py format' to format existing code")
        print("3. Run 'python scripts/build_docs.py' to build documentation")
        print("4. Run 'python scripts/dev.py quality' for full quality check")
        return 0
    else:
        print("❌ Setup verification FAILED!")
        print("\nPlease address the issues above and run this script again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
