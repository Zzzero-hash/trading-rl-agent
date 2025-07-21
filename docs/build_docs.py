#!/usr/bin/env python3
"""
Build and serve Sphinx documentation for the Trading RL Agent project.

This script provides a convenient way to build, check, and serve the documentation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        if check:
            sys.exit(1)
        return e


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx_autodoc_typehints",
        "myst_parser",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements-dev.txt")
        return False
    return True


def build_docs(clean=True, fast=False):
    """Build the documentation."""
    if not check_dependencies():
        return False

    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)

    if clean:
        print("Cleaning previous builds...")
        run_command(["make", "clean"], "Cleaning build directory")

    if fast:
        run_command(["make", "html-fast"], "Building HTML documentation (fast)")
    else:
        run_command(["make", "html"], "Building HTML documentation")

    return True


def serve_docs(port=8000):
    """Serve the documentation on a local server."""
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "_build" / "html"

    if not build_dir.exists():
        print("Documentation not built yet. Building first...")
        if not build_docs():
            return False

    print(f"\nServing documentation at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        os.chdir(build_dir)
        subprocess.run(["python", "-m", "http.server", str(port)], check=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error serving docs: {e}")
        return False

    return True


def check_docs():
    """Run documentation quality checks."""
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)

    print("Running documentation quality checks...")

    # Check links
    run_command(["make", "linkcheck"], "Checking external links", check=False)

    # Run doctests
    run_command(["make", "doctest"], "Running doctests", check=False)

    # Check coverage
    run_command(["make", "coverage"], "Checking documentation coverage", check=False)

    print("\nDocumentation checks completed!")


def main():
    parser = argparse.ArgumentParser(description="Build and serve Sphinx documentation")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before building")
    parser.add_argument("--fast", action="store_true", help="Fast build (incremental)")
    parser.add_argument("--serve", action="store_true", help="Serve documentation after building")
    parser.add_argument("--port", type=int, default=8000, help="Port for serving docs (default: 8000)")
    parser.add_argument("--check", action="store_true", help="Run documentation quality checks")
    parser.add_argument("--all", action="store_true", help="Build, check, and serve documentation")

    args = parser.parse_args()

    if args.all:
        print("Building, checking, and serving documentation...")
        if build_docs(clean=True):
            check_docs()
            serve_docs(args.port)
    elif args.check:
        check_docs()
    elif args.serve:
        serve_docs(args.port)
    else:
        build_docs(clean=args.clean, fast=args.fast)


if __name__ == "__main__":
    main()
