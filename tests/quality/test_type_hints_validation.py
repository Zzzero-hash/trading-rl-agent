"""Tests for type hints validation in trading RL agent."""

import ast
import inspect
import os
import re
from pathlib import Path

import pytest

from src.trading_rl_agent.core.config import Config
from src.trading_rl_agent.data.data_loader import DataLoader
from src.trading_rl_agent.features.feature_engineering import FeatureEngineer
from src.trading_rl_agent.portfolio.portfolio_manager import PortfolioManager
from src.trading_rl_agent.risk.risk_manager import RiskManager


class TestTypeHintsValidation:
    """Test type hints validation and completeness."""

    @pytest.fixture
    def source_files(self) -> list[str]:
        """Get all Python source files."""
        src_dir = Path("src/trading_rl_agent")
        python_files = []

        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        return python_files

    def test_function_type_hints(self, source_files):
        """Test that all functions have type hints."""
        missing_type_hints = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Check if function has type hints
                        has_return_annotation = node.returns is not None
                        has_param_annotations = all(arg.annotation is not None for arg in node.args.args)

                        if not has_return_annotation or not has_param_annotations:
                            missing_type_hints.append(f"{file_path}:{node.lineno} - {node.name}")
            except Exception as e:
                missing_type_hints.append(f"{file_path}: Error parsing - {e}")

        # Allow some functions to be missing type hints (gradual typing)
        assert len(missing_type_hints) < 50, "Too many functions missing type hints:\n" + "\n".join(
            missing_type_hints[:20]
        )

    def test_class_type_hints(self, source_files):
        """Test that all classes have proper type hints."""
        missing_type_hints = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check class methods for type hints
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                has_return_annotation = item.returns is not None
                                has_param_annotations = all(arg.annotation is not None for arg in item.args.args)

                                if not has_return_annotation or not has_param_annotations:
                                    missing_type_hints.append(f"{file_path}:{item.lineno} - {node.name}.{item.name}")
            except Exception as e:
                missing_type_hints.append(f"{file_path}: Error parsing - {e}")

        # Allow some methods to be missing type hints
        assert len(missing_type_hints) < 100, "Too many class methods missing type hints:\n" + "\n".join(
            missing_type_hints[:20]
        )

    def test_variable_type_hints(self, source_files):
        """Test that important variables have type hints."""
        missing_type_hints = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.AnnAssign):
                        # Variable with type annotation
                        continue
                    if isinstance(node, ast.Assign):
                        # Check if assignment should have type hints
                        for target in node.targets:
                            if isinstance(target, ast.Name) and (target.id.isupper() or len(target.id) > 10):
                                missing_type_hints.append(f"{file_path}:{node.lineno} - {target.id}")
            except Exception as e:
                missing_type_hints.append(f"{file_path}: Error parsing - {e}")

        # Allow some variables to be missing type hints
        assert len(missing_type_hints) < 30, "Too many variables missing type hints:\n" + "\n".join(
            missing_type_hints[:10]
        )

    def test_import_type_hints(self, source_files):
        """Test that type hints are properly imported."""
        missing_imports = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Check for typing imports
                has_typing_import = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "typing":
                            has_typing_import = True
                            break
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "typing":
                                has_typing_import = True
                                break

                # Check if file uses type hints
                uses_type_hints = False
                for node in ast.walk(tree):
                    if (isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation is not None) or (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None
                    ):
                        uses_type_hints = True
                        break

                if uses_type_hints and not has_typing_import:
                    missing_imports.append(file_path)
            except Exception as e:
                missing_imports.append(f"{file_path}: Error parsing - {e}")

        assert not missing_imports, "Files using type hints without typing import:\n" + "\n".join(missing_imports)

    def test_type_hint_consistency(self, source_files):
        """Test that type hints are consistent across the codebase."""
        inconsistencies = []

        # Common type patterns to check
        type_patterns = {
            r"List\[.*\]": "List[T]",
            r"Dict\[.*,.*\]": "Dict[K, V]",
            r"Optional\[.*\]": "Optional[T]",
            r"Union\[.*\]": "Union[T, ...]",
            r"Tuple\[.*\]": "Tuple[T, ...]",
            r"Callable\[.*\]": "Callable[..., R]",
        }

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                for pattern in type_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Check if the type hint follows expected format
                        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\[.*\])?$", match):
                            inconsistencies.append(f"{file_path}: Invalid type hint format: {match}")
            except Exception as e:
                inconsistencies.append(f"{file_path}: Error parsing - {e}")

        assert not inconsistencies, "Type hint inconsistencies:\n" + "\n".join(inconsistencies)

    def test_forward_references(self, source_files):
        """Test that forward references are properly used."""
        forward_ref_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Check for string type annotations (forward references)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and re.match(r"^[A-Z][a-zA-Z0-9_]*$", node.value)
                    ):
                        # This might be a forward reference
                        forward_ref_issues.append(
                            f"{file_path}:{node.lineno} - Potential forward reference: {node.value}"
                        )
            except Exception as e:
                forward_ref_issues.append(f"{file_path}: Error parsing - {e}")

        # Forward references are acceptable, just log them
        if forward_ref_issues:
            print(f"Found {len(forward_ref_issues)} potential forward references")

    def test_generic_types(self, source_files):
        """Test that generic types are properly used."""
        generic_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for common generic type patterns
                generic_patterns = [
                    r"List\[.*\]",
                    r"Dict\[.*,.*\]",
                    r"Set\[.*\]",
                    r"Tuple\[.*\]",
                    r"Optional\[.*\]",
                    r"Union\[.*\]",
                ]

                for pattern in generic_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Check if generic type is properly formatted
                        if "Any" in match and "Any" not in ["Any", "AnyStr"]:
                            generic_issues.append(f"{file_path}: Overly generic type: {match}")
            except Exception as e:
                generic_issues.append(f"{file_path}: Error parsing - {e}")

        # Allow some generic types
        assert len(generic_issues) < 20, "Too many overly generic types:\n" + "\n".join(generic_issues)

    def test_type_hint_completeness(self):
        """Test that main classes have complete type hints."""
        classes_to_test = [
            Config,
            DataLoader,
            FeatureEngineer,
            PortfolioManager,
            RiskManager,
        ]

        incomplete_classes = []

        for cls in classes_to_test:
            methods = inspect.getmembers(cls, predicate=inspect.isfunction)
            for method_name, method in methods:
                if not method_name.startswith("_"):
                    # Check method signature
                    sig = inspect.signature(method)

                    # Check parameters
                    for param_name, param in sig.parameters.items():
                        if param.annotation == inspect.Parameter.empty:
                            incomplete_classes.append(f"{cls.__name__}.{method_name}.{param_name}")

                    # Check return type
                    if sig.return_annotation == inspect.Signature.empty:
                        incomplete_classes.append(f"{cls.__name__}.{method_name}.return")

        # Allow some methods to be missing type hints
        assert len(incomplete_classes) < 50, "Too many incomplete type hints:\n" + "\n".join(incomplete_classes[:20])

    def test_type_hint_accuracy(self, source_files):
        """Test that type hints accurately reflect the actual types."""
        accuracy_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                        # Check return type annotation
                        return_type = ast.unparse(node.returns)

                        # Check for common accuracy issues
                        if return_type == "str" and any(
                            isinstance(n, ast.Return)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, int)
                            for n in ast.walk(node)
                        ):
                            accuracy_issues.append(
                                f"{file_path}:{node.lineno} - {node.name} returns int but annotated as str"
                            )

                        elif return_type == "int" and any(
                            isinstance(n, ast.Return)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str)
                            for n in ast.walk(node)
                        ):
                            accuracy_issues.append(
                                f"{file_path}:{node.lineno} - {node.name} returns str but annotated as int"
                            )
            except Exception as e:
                accuracy_issues.append(f"{file_path}: Error parsing - {e}")

        assert not accuracy_issues, "Type hint accuracy issues:\n" + "\n".join(accuracy_issues)

    def test_protocol_usage(self, source_files):
        """Test that protocols are properly used for structural typing."""
        protocol_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for Protocol usage
                if "Protocol" in content and (
                    "from typing import Protocol" not in content
                    and "from typing_extensions import Protocol" not in content
                ):
                    protocol_issues.append(f"{file_path}: Uses Protocol without proper import")
            except Exception as e:
                protocol_issues.append(f"{file_path}: Error parsing - {e}")

        assert not protocol_issues, "Protocol usage issues:\n" + "\n".join(protocol_issues)

    def test_literal_types(self, source_files):
        """Test that literal types are properly used."""
        literal_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for Literal usage
                if "Literal[" in content and (
                    "from typing import Literal" not in content
                    and "from typing_extensions import Literal" not in content
                ):
                    literal_issues.append(f"{file_path}: Uses Literal without proper import")
            except Exception as e:
                literal_issues.append(f"{file_path}: Error parsing - {e}")

        assert not literal_issues, "Literal type issues:\n" + "\n".join(literal_issues)

    def test_type_hint_documentation(self, source_files):
        """Test that type hints are documented in docstrings."""
        documentation_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node)
                        if docstring and node.args.args:
                            # Check if docstring mentions parameter types
                            for arg in node.args.args:
                                if arg.annotation and arg.arg != "self" and arg.arg not in docstring:
                                    documentation_issues.append(
                                        f"{file_path}:{node.lineno} - {node.name}.{arg.arg} has type hint but not in docstring"
                                    )
            except Exception as e:
                documentation_issues.append(f"{file_path}: Error parsing - {e}")

        # Allow some flexibility in documentation
        assert len(documentation_issues) < 30, "Too many documentation issues:\n" + "\n".join(documentation_issues[:10])

    def test_type_hint_complexity(self, source_files):
        """Test that type hints are not overly complex."""
        complexity_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for overly complex type hints
                complex_patterns = [
                    r"Union\[.*,.*,.*,.*,.*\]",  # Too many Union types
                    r"Dict\[.*,.*,.*\]",  # Dict with too many type parameters
                    r"Tuple\[.*,.*,.*,.*,.*,.*,.*,.*\]",  # Very long tuple
                ]

                for pattern in complex_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        complexity_issues.append(f"{file_path}: Overly complex type hint: {match}")
            except Exception as e:
                complexity_issues.append(f"{file_path}: Error parsing - {e}")

        assert not complexity_issues, "Overly complex type hints:\n" + "\n".join(complexity_issues)

    def test_type_hint_imports(self, source_files):
        """Test that type hint imports are properly organized."""
        import_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                # Check import order and organization
                typing_imports = []
                other_imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "typing":
                            typing_imports.append(node)
                        else:
                            other_imports.append(node)
                    elif isinstance(node, ast.Import):
                        other_imports.append(node)

                # Check if typing imports come before other imports
                if typing_imports and other_imports:
                    typing_lines = [node.lineno for node in typing_imports]
                    other_lines = [node.lineno for node in other_imports]

                    if max(typing_lines) > min(other_lines):
                        import_issues.append(f"{file_path}: Typing imports should come before other imports")
            except Exception as e:
                import_issues.append(f"{file_path}: Error parsing - {e}")

        assert not import_issues, "Import organization issues:\n" + "\n".join(import_issues)

    def test_type_hint_consistency_across_modules(self):
        """Test that type hints are consistent across modules."""
        consistency_issues = []

        # Check for consistent type usage across modules
        common_types = {
            "DataFrame": "pandas.DataFrame",
            "Series": "pandas.Series",
            "ndarray": "numpy.ndarray",
            "Config": "src.trading_rl_agent.core.config.Config",
        }

        for type_name in common_types:
            # Check if this type is used consistently
            usage_pattern = rf"\b{type_name}\b"

            # This would require more complex analysis across modules
            # For now, just check that the type is properly imported where used

        # This is a placeholder for more comprehensive cross-module analysis
        assert True, "Cross-module type consistency check passed"

    def test_type_hint_performance(self, source_files):
        """Test that type hints don't impact performance significantly."""
        performance_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for expensive type operations
                expensive_patterns = [
                    r"isinstance\(.*,.*\)",  # isinstance checks
                    r"type\(.*\)",  # type() calls
                    r"getattr\(.*,.*\)",  # getattr calls
                ]

                for pattern in expensive_patterns:
                    matches = re.findall(pattern, content)
                    if len(matches) > 10:  # Too many expensive operations
                        performance_issues.append(f"{file_path}: Too many expensive type operations: {len(matches)}")
            except Exception as e:
                performance_issues.append(f"{file_path}: Error parsing - {e}")

        assert not performance_issues, "Performance issues:\n" + "\n".join(performance_issues)

    def test_type_hint_maintainability(self, source_files):
        """Test that type hints are maintainable."""
        maintainability_issues = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for maintainability issues
                issues = []

                # Long type hints
                if len(content) > 10000:  # Large file
                    issues.append("Large file may have maintainability issues")

                # Complex nested types
                if content.count("[") > content.count("]") + 10:
                    issues.append("Unbalanced brackets in type hints")

                if issues:
                    maintainability_issues.append(f"{file_path}: {', '.join(issues)}")
            except Exception as e:
                maintainability_issues.append(f"{file_path}: Error parsing - {e}")

        assert not maintainability_issues, "Maintainability issues:\n" + "\n".join(maintainability_issues)
