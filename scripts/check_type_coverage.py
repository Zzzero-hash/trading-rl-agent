"""Checks the type coverage of the trading-rl-agent project using mypy."""

#!/usr/bin/env python3
"""
Type checking script to ensure comprehensive type coverage.
"""

import ast
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple, Union


class TypeHintChecker(ast.NodeVisitor):
    """AST visitor to check for type hints."""

    def __init__(self) -> None:
        self.functions: list[tuple[str, int, bool]] = []
        self.classes: list[tuple[str, int]] = []
        self.methods: list[tuple[str, int, bool]] = []
        self.current_class: str = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        self.classes.append((node.name, node.lineno))
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        has_return_annotation = node.returns is not None
        has_arg_annotations = all(
            arg.annotation is not None for arg in node.args.args if arg.arg != "self"
        )

        is_typed = has_return_annotation and has_arg_annotations

        if self.current_class:
            method_name = f"{self.current_class}.{node.name}"
            self.methods.append((method_name, node.lineno, is_typed))
        else:
            self.functions.append((node.name, node.lineno, is_typed))

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        # Handle async functions the same way as regular functions
        has_return_annotation = node.returns is not None
        has_arg_annotations = all(
            arg.annotation is not None for arg in node.args.args if arg.arg != "self"
        )

        is_typed = has_return_annotation and has_arg_annotations

        if self.current_class:
            method_name = f"{self.current_class}.{node.name}"
            self.methods.append((method_name, node.lineno, is_typed))
        else:
            self.functions.append((node.name, node.lineno, is_typed))

        self.generic_visit(node)


def check_file_type_coverage(file_path: Path) -> dict[str, Any]:
    """Check type hint coverage for a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        checker = TypeHintChecker()
        checker.visit(tree)

        # Calculate coverage
        total_functions = len(checker.functions)
        typed_functions = sum(1 for _, _, typed in checker.functions if typed)

        total_methods = len(checker.methods)
        typed_methods = sum(1 for _, _, typed in checker.methods if typed)

        total_callables = total_functions + total_methods
        typed_callables = typed_functions + typed_methods

        coverage = (
            (typed_callables / total_callables * 100) if total_callables > 0 else 100.0
        )

        return {
            "coverage": coverage,
            "total_callables": total_callables,
            "typed_callables": typed_callables,
            "functions": checker.functions,
            "methods": checker.methods,
            "classes": checker.classes,
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"coverage": 0.0, "total_callables": 0, "typed_callables": 0}


def check_project_type_coverage(src_dir: Path, min_coverage: float = 80.0) -> bool:
    """Check type hint coverage for the entire project."""
    python_files = list(src_dir.rglob("*.py"))

    if not python_files:
        print("No Python files found in src directory")
        return False

    total_callables = 0
    total_typed = 0
    files_below_threshold = []

    print(f"Checking type hint coverage for {len(python_files)} files...")
    print("=" * 60)

    for file_path in python_files:
        if "__pycache__" in str(file_path):
            continue

        result = check_file_type_coverage(file_path)

        total_callables += result["total_callables"]
        total_typed += result["typed_callables"]

        rel_path = file_path.relative_to(src_dir)

        if result["total_callables"] == 0:
            print(f"📄 {rel_path}: No callables found")
        else:
            coverage = result["coverage"]
            status = "✅" if coverage >= min_coverage else "❌"
            print(
                f"{status} {rel_path}: {coverage:.1f}% ({result['typed_callables']}/{result['total_callables']})"
            )

            if coverage < min_coverage:
                files_below_threshold.append((rel_path, result))

    # Overall statistics
    overall_coverage = (
        (total_typed / total_callables * 100) if total_callables > 0 else 100.0
    )

    print("=" * 60)
    print(
        f"Overall Coverage: {overall_coverage:.1f}% ({total_typed}/{total_callables})"
    )

    if files_below_threshold:
        print(
            f"\n❌ {len(files_below_threshold)} files below {min_coverage}% threshold:"
        )
        for file_path, result in files_below_threshold:
            print(f"  - {file_path}: {result['coverage']:.1f}%")

            # Show untyped functions
            untyped_functions = [
                f"    • {name} (line {line})"
                for name, line, typed in result["functions"]
                if not typed
            ]
            untyped_methods = [
                f"    • {name} (line {line})"
                for name, line, typed in result["methods"]
                if not typed
            ]

            if untyped_functions:
                print("    Untyped functions:")
                for func in untyped_functions[:5]:  # Show first 5
                    print(func)
                if len(untyped_functions) > 5:
                    print(f"    ... and {len(untyped_functions) - 5} more")

            if untyped_methods:
                print("    Untyped methods:")
                for method in untyped_methods[:5]:  # Show first 5
                    print(method)
                if len(untyped_methods) > 5:
                    print(f"    ... and {len(untyped_methods) - 5} more")
            print()

    success = overall_coverage >= min_coverage

    if success:
        print(f"✅ Type hint coverage meets minimum requirement ({min_coverage}%)")
    else:
        print(f"❌ Type hint coverage below minimum requirement ({min_coverage}%)")

    return success


def run_mypy_check(src_dir: Path) -> bool:
    """Run mypy type checking."""
    print("\n🔍 Running mypy type checking...")

    try:
        result = subprocess.run(
            ["mypy", str(src_dir)], capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            print("✅ mypy type checking passed")
            return True
        else:
            print("❌ mypy type checking failed:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ mypy check timed out")
        return False
    except FileNotFoundError:
        print("⚠️ mypy not found, skipping type checking")
        return True  # Don't fail if mypy is not installed


def main() -> int:
    """Main function."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return 1

    # Check type hint coverage
    min_coverage = 80.0  # Require 80% type hint coverage
    coverage_ok = check_project_type_coverage(src_dir, min_coverage)

    # Run mypy type checking
    mypy_ok = run_mypy_check(src_dir)

    if coverage_ok and mypy_ok:
        print("\n🎉 All type checking passed!")
        return 0
    else:
        print("\n💥 Type checking failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
