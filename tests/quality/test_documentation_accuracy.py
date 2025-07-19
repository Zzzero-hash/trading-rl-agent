"""Tests for documentation accuracy in trading RL agent."""

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


class TestDocumentationAccuracy:
    """Test documentation accuracy and completeness."""

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

    def test_function_docstrings(self, source_files):
        """Test that all functions have docstrings."""
        missing_docstrings = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                        missing_docstrings.append(f"{file_path}:{node.lineno} - {node.name}")
            except Exception as e:
                missing_docstrings.append(f"{file_path}: Error parsing - {e}")

        assert not missing_docstrings, "Functions missing docstrings:\n" + "\n".join(missing_docstrings)

    def test_class_docstrings(self, source_files):
        """Test that all classes have docstrings."""
        missing_docstrings = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                        missing_docstrings.append(f"{file_path}:{node.lineno} - {node.name}")
            except Exception as e:
                missing_docstrings.append(f"{file_path}: Error parsing - {e}")

        assert not missing_docstrings, "Classes missing docstrings:\n" + "\n".join(missing_docstrings)

    def test_module_docstrings(self, source_files):
        """Test that all modules have docstrings."""
        missing_docstrings = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                if not ast.get_docstring(tree):
                    missing_docstrings.append(file_path)
            except Exception as e:
                missing_docstrings.append(f"{file_path}: Error parsing - {e}")

        assert not missing_docstrings, "Modules missing docstrings:\n" + "\n".join(missing_docstrings)

    def test_docstring_format(self, source_files):
        """Test that docstrings follow Google format."""
        malformed_docstrings = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node)
                        if (
                            docstring
                            and not re.search(r"Args:|Parameters:|Returns:|Raises:|Yields:", docstring)
                            and len(docstring.split("\n")) > 3
                        ):  # Multi-line docstring
                            malformed_docstrings.append(f"{file_path}:{node.lineno} - {node.name}")
            except Exception as e:
                malformed_docstrings.append(f"{file_path}: Error parsing - {e}")

        # Allow some flexibility in docstring format
        assert len(malformed_docstrings) < 10, "Too many malformed docstrings:\n" + "\n".join(malformed_docstrings)

    def test_readme_completeness(self):
        """Test README completeness."""
        readme_path = Path("README.md")
        assert readme_path.exists(), "README.md file not found"

        with open(readme_path, encoding="utf-8") as f:
            content = f.read()

        # Check for required sections
        required_sections = [
            "# Trading RL Agent",
            "## Installation",
            "## Usage",
            "## Features",
            "## Configuration",
            "## Testing",
            "## Contributing",
            "## License",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)

        assert not missing_sections, "Missing README sections:\n" + "\n".join(missing_sections)

    def test_api_documentation(self):
        """Test API documentation completeness."""
        # Test that main classes have comprehensive docstrings
        classes_to_test = [
            Config,
            DataLoader,
            FeatureEngineer,
            PortfolioManager,
            RiskManager,
        ]

        for cls in classes_to_test:
            docstring = cls.__doc__
            assert docstring is not None, f"Class {cls.__name__} missing docstring"
            assert len(docstring.strip()) > 50, f"Class {cls.__name__} has too short docstring"

            # Check for method documentation
            methods = inspect.getmembers(cls, predicate=inspect.isfunction)
            for method_name, method in methods:
                if not method_name.startswith("_"):
                    method_doc = method.__doc__
                    assert method_doc is not None, f"Method {cls.__name__}.{method_name} missing docstring"

    def test_example_code_accuracy(self):
        """Test that example code in documentation is accurate."""
        # Check for example code blocks in README
        readme_path = Path("README.md")
        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                content = f.read()

            # Find code blocks
            code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

            for i, code_block in enumerate(code_blocks):
                try:
                    # Try to parse the code
                    ast.parse(code_block)
                except SyntaxError as e:
                    pytest.fail(f"Invalid Python syntax in README code block {i + 1}: {e}")

    def test_config_documentation(self):
        """Test configuration documentation accuracy."""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                content = f.read()

            # Check for comments explaining configuration options
            lines = content.split("\n")
            commented_lines = [line for line in lines if line.strip().startswith("#")]

            # Should have reasonable number of comments
            assert len(commented_lines) > 5, "Configuration file lacks sufficient documentation"

    def test_dependency_documentation(self):
        """Test dependency documentation accuracy."""
        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-ml.txt",
            "requirements-production.txt",
        ]

        for req_file in requirements_files:
            req_path = Path(req_file)
            if req_path.exists():
                with open(req_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for comments explaining dependencies
                lines = content.split("\n")
                commented_lines = [line for line in lines if line.strip().startswith("#")]

                # Should have some comments explaining dependencies
                assert len(commented_lines) > 0, f"Requirements file {req_file} lacks documentation"

    def test_test_documentation(self):
        """Test that test files have proper documentation."""
        test_files = []
        test_dir = Path("tests")

        if test_dir.exists():
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith(".py") and file.startswith("test_"):
                        test_files.append(os.path.join(root, file))

        missing_test_docs = []
        for test_file in test_files:
            try:
                with open(test_file, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                # Check for module docstring
                if not ast.get_docstring(tree):
                    missing_test_docs.append(test_file)

                # Check for test class docstrings
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.startswith("Test") and not ast.get_docstring(node):
                        missing_test_docs.append(f"{test_file}:{node.lineno} - {node.name}")
            except Exception:
                missing_test_docs.append(f"{test_file}: Error parsing")

        assert not missing_test_docs, "Test files missing documentation:\n" + "\n".join(missing_test_docs)

    def test_changelog_documentation(self):
        """Test changelog documentation."""
        changelog_files = [
            "CHANGELOG.md",
            "CHANGELOG.txt",
            "HISTORY.md",
            "HISTORY.txt",
        ]

        changelog_found = False
        for changelog_file in changelog_files:
            changelog_path = Path(changelog_file)
            if changelog_path.exists():
                changelog_found = True
                with open(changelog_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for version headers
                version_headers = re.findall(r"^#+\s+v?\d+\.\d+\.\d+", content, re.MULTILINE)
                assert len(version_headers) > 0, f"Changelog {changelog_file} lacks version headers"
                break

        # Changelog is optional but recommended
        if not changelog_found:
            pytest.skip("No changelog file found (optional)")

    def test_license_documentation(self):
        """Test license documentation."""
        license_files = [
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
        ]

        license_found = False
        for license_file in license_files:
            license_path = Path(license_file)
            if license_path.exists():
                license_found = True
                with open(license_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for license content
                assert len(content.strip()) > 100, f"License file {license_file} seems too short"
                break

        assert license_found, "No license file found"

    def test_contributing_documentation(self):
        """Test contributing documentation."""
        contributing_files = [
            "CONTRIBUTING.md",
            "CONTRIBUTING.txt",
            "CONTRIBUTING.rst",
        ]

        contributing_found = False
        for contributing_file in contributing_files:
            contributing_path = Path(contributing_file)
            if contributing_path.exists():
                contributing_found = True
                with open(contributing_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for required sections
                required_sections = [
                    "Contributing",
                    "Development",
                    "Testing",
                    "Pull Request",
                ]

                missing_sections = []
                for section in required_sections:
                    if section not in content:
                        missing_sections.append(section)

                assert not missing_sections, f"Contributing file missing sections: {missing_sections}"
                break

        # Contributing guide is optional but recommended
        if not contributing_found:
            pytest.skip("No contributing file found (optional)")

    def test_docstring_parameter_accuracy(self, source_files):
        """Test that docstring parameters match function signatures."""
        parameter_mismatches = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Extract parameters from function signature
                            func_params = [arg.arg for arg in node.args.args if arg.arg != "self"]

                            # Extract parameters from docstring
                            doc_params = re.findall(r"(\w+):\s*[^,\n]+", docstring)

                            # Check for mismatches
                            missing_in_doc = set(func_params) - set(doc_params)
                            extra_in_doc = set(doc_params) - set(func_params)

                            if missing_in_doc or extra_in_doc:
                                parameter_mismatches.append(
                                    f"{file_path}:{node.lineno} - {node.name} - "
                                    f"Missing: {missing_in_doc}, Extra: {extra_in_doc}"
                                )
            except Exception as e:
                parameter_mismatches.append(f"{file_path}: Error parsing - {e}")

        assert not parameter_mismatches, "Parameter mismatches:\n" + "\n".join(parameter_mismatches)

    def test_docstring_return_accuracy(self, source_files):
        """Test that docstring return statements match function behavior."""
        return_mismatches = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Check if function has return statements
                            has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))

                            # Check if docstring mentions returns
                            mentions_return = "return" in docstring.lower() or "returns" in docstring.lower()

                            if has_return and not mentions_return:
                                return_mismatches.append(
                                    f"{file_path}:{node.lineno} - {node.name} - "
                                    f"Function returns but docstring doesn't mention it"
                                )
            except Exception as e:
                return_mismatches.append(f"{file_path}: Error parsing - {e}")

        assert not return_mismatches, "Return documentation mismatches:\n" + "\n".join(return_mismatches)

    def test_docstring_exception_accuracy(self, source_files):
        """Test that docstring exception statements match function behavior."""
        exception_mismatches = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Check if function has raise statements
                            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))

                            # Check if docstring mentions exceptions
                            mentions_exception = "raises" in docstring.lower() or "exception" in docstring.lower()

                            if has_raise and not mentions_exception:
                                exception_mismatches.append(
                                    f"{file_path}:{node.lineno} - {node.name} - "
                                    f"Function raises but docstring doesn't mention it"
                                )
            except Exception as e:
                exception_mismatches.append(f"{file_path}: Error parsing - {e}")

        assert not exception_mismatches, "Exception documentation mismatches:\n" + "\n".join(exception_mismatches)

    def test_docstring_example_accuracy(self, source_files):
        """Test that docstring examples are syntactically correct."""
        example_errors = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Find code examples in docstring
                            code_examples = re.findall(r"```python\n(.*?)\n```", docstring, re.DOTALL)
                            code_examples.extend(re.findall(r">>>\s+(.*?)(?=\n>>>|\n\n|\n$)", docstring, re.DOTALL))

                            for i, example in enumerate(code_examples):
                                try:
                                    # Try to parse the example code
                                    ast.parse(example)
                                except SyntaxError as e:
                                    example_errors.append(
                                        f"{file_path}:{node.lineno} - {node.name} - "
                                        f"Example {i + 1} has syntax error: {e}"
                                    )
            except Exception as e:
                example_errors.append(f"{file_path}: Error parsing - {e}")

        assert not example_errors, "Docstring example errors:\n" + "\n".join(example_errors)

    def test_docstring_link_accuracy(self, source_files):
        """Test that docstring links are valid."""
        link_errors = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Find links in docstring
                            links = re.findall(
                                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                                docstring,
                            )

                            for link in links:
                                # Basic URL validation
                                if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", link):
                                    link_errors.append(
                                        f"{file_path}:{node.lineno} - {node.name} - Invalid link: {link}"
                                    )
            except Exception as e:
                link_errors.append(f"{file_path}: Error parsing - {e}")

        assert not link_errors, "Docstring link errors:\n" + "\n".join(link_errors)

    def test_docstring_consistency(self, source_files):
        """Test that docstrings follow consistent formatting."""
        consistency_errors = []

        for file_path in source_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            # Check for consistent indentation
                            lines = docstring.split("\n")
                            for i, line in enumerate(lines[1:], 1):  # Skip first line
                                if line.strip() and not line.startswith("    "):
                                    consistency_errors.append(
                                        f"{file_path}:{node.lineno} - {node.name} - "
                                        f"Line {i + 1} has inconsistent indentation"
                                    )

                            # Check for proper sentence endings
                            if not docstring.strip().endswith((".", "!", "?")):
                                consistency_errors.append(
                                    f"{file_path}:{node.lineno} - {node.name} - "
                                    f"Docstring doesn't end with proper punctuation"
                                )
            except Exception as e:
                consistency_errors.append(f"{file_path}: Error parsing - {e}")

        # Allow some flexibility in formatting
        assert len(consistency_errors) < 20, "Too many consistency errors:\n" + "\n".join(consistency_errors)
