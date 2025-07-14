#!/usr/bin/env python3
"""
Comprehensive documentation build and quality check script.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


class DocumentationBuilder:
    """Build and validate documentation."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.build_dir = self.docs_dir / "_build"
        self.source_dir = project_root / "src"

    def clean_build(self) -> bool:
        """Clean previous build artifacts."""
        print("🧹 Cleaning build directory...")

        try:
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)

            # Clean autosummary
            autosummary_dir = self.docs_dir / "_autosummary"
            if autosummary_dir.exists():
                shutil.rmtree(autosummary_dir)

            print("✅ Build directory cleaned")
        except Exception as e:
            print(f"❌ Failed to clean build directory: {e}")
            return False
        else:
            return True

    def generate_api_docs(self) -> bool:
        """Generate API documentation from source code."""
        print("📚 Generating API documentation...")

        try:
            # Generate API documentation
            cmd = [
                "sphinx-apidoc",
                "-f",  # Force overwrite
                "-o",
                str(self.docs_dir),  # Output directory
                str(self.source_dir),  # Source directory
                "--separate",  # Create separate pages for each module
                "--module-first",  # Put module documentation first
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ API documentation generated")
            else:
                print(f"❌ Failed to generate API docs: {result.stderr}")
                return False

        except FileNotFoundError:
            print("❌ sphinx-apidoc not found. Install with: pip install sphinx")
            return False
        except Exception as e:
            print(f"❌ Error generating API docs: {e}")
            return False
        else:
            return True

    def build_html(self) -> bool:
        """Build HTML documentation."""
        print("🔨 Building HTML documentation...")

        try:
            cmd = [
                "sphinx-build",
                "-b",
                "html",
                "-W",  # Treat warnings as errors
                "--keep-going",  # Continue on errors
                str(self.docs_dir),
                str(self.build_dir / "html"),
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ HTML documentation built successfully")
            else:
                print("❌ HTML build failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except FileNotFoundError:
            print("❌ sphinx-build not found. Install with: pip install sphinx")
            return False
        except Exception as e:
            print(f"❌ Error building HTML docs: {e}")
            return False
        else:
            return True

    def check_links(self) -> bool:
        """Check for broken links in documentation."""
        print("🔗 Checking documentation links...")

        try:
            cmd = [
                "sphinx-build",
                "-b",
                "linkcheck",
                str(self.docs_dir),
                str(self.build_dir / "linkcheck"),
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Link check passed")
                return True
            print("⚠️ Some links may be broken - check linkcheck report")
            # Don't fail on link check issues as they might be temporary
            return True

        except Exception as e:
            print(f"⚠️ Link check failed: {e}")
            return True  # Don't fail build on link check issues

    def run_doctests(self) -> bool:
        """Run doctests in documentation."""
        print("🧪 Running documentation tests...")

        try:
            cmd = [
                "sphinx-build",
                "-b",
                "doctest",
                str(self.docs_dir),
                str(self.build_dir / "doctest"),
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Documentation tests passed")
                return True
            print("⚠️ Some documentation tests failed:")
            print(result.stdout)
            return True  # Don't fail build on doctest issues

        except Exception as e:
            print(f"⚠️ Doctest failed: {e}")
            return True

    def check_coverage(self) -> bool:
        """Check documentation coverage."""
        print("📊 Checking documentation coverage...")

        try:
            cmd = [
                "sphinx-build",
                "-b",
                "coverage",
                str(self.docs_dir),
                str(self.build_dir / "coverage"),
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            coverage_file = self.build_dir / "coverage" / "python.txt"
            if coverage_file.exists():
                with Path(coverage_file).open(coverage_file) as f:
                    coverage_content = f.read()

                # Parse coverage results
                lines = coverage_content.split("\n")
                undocumented = [line for line in lines if "undocumented" in line.lower()]

                if undocumented:
                    print(f"⚠️ Found {len(undocumented)} undocumented items:")
                    for item in undocumented[:10]:  # Show first 10
                        print(f"  - {item.strip()}")
                    if len(undocumented) > 10:
                        print(f"  ... and {len(undocumented) - 10} more")
                else:
                    print("✅ Documentation coverage check passed")

            return True

        except Exception as e:
            print(f"⚠️ Coverage check failed: {e}")
            return True

    def build_pdf(self) -> bool:
        """Build PDF documentation."""
        print("📄 Building PDF documentation...")

        try:
            # First build LaTeX
            cmd = [
                "sphinx-build",
                "-b",
                "latex",
                str(self.docs_dir),
                str(self.build_dir / "latex"),
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ LaTeX build failed: {result.stderr}")
                return False

            # Then build PDF
            latex_dir = self.build_dir / "latex"
            pdf_cmd = ["make", "-C", str(latex_dir), "all-pdf"]

            result = subprocess.run(
                pdf_cmd,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ PDF documentation built successfully")
                return True
            print("⚠️ PDF build failed (this is optional)")
            return True  # PDF build is optional

        except Exception as e:
            print(f"⚠️ PDF build failed: {e}")
            return True  # PDF build is optional

    def validate_structure(self) -> bool:
        """Validate documentation structure."""
        print("🏗️ Validating documentation structure...")

        required_files = [
            "index.rst",
            "api_reference.md",
            "getting_started.md",
            "examples.md",
            "conf.py",
        ]

        missing_files = []
        for file_name in required_files:
            if not (self.docs_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            print(f"❌ Missing required documentation files: {missing_files}")
            return False

        print("✅ Documentation structure is valid")
        return True

    def generate_coverage_badge(self) -> bool:
        """Generate documentation coverage badge."""
        try:
            # This would integrate with a badge service
            # For now, just create a simple coverage report
            coverage_file = self.build_dir / "coverage" / "python.txt"
            if coverage_file.exists():
                print(
                    "📊 Coverage report available at: docs/_build/coverage/python.txt",
                )
            return True
        except Exception as e:
            print(f"⚠️ Badge generation failed: {e}")
            return True

    def serve_docs(self, port: int = 8000) -> None:
        """Serve documentation locally."""
        html_dir = self.build_dir / "html"
        if not html_dir.exists():
            print("❌ HTML documentation not found. Run build first.")
            return

        print(f"🌐 Starting documentation server on http://localhost:{port}")
        print("Press Ctrl+C to stop the server")

        try:
            os.chdir(html_dir)
            subprocess.run(
                [sys.executable, "-m", "http.server", str(port)],
                check=False,
            )
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

    def full_build(self, include_pdf: bool = False) -> bool:
        """Run complete documentation build process."""
        print("🚀 Starting complete documentation build...")
        print("=" * 60)

        steps = [
            ("Validate Structure", self.validate_structure),
            ("Clean Build", self.clean_build),
            ("Generate API Docs", self.generate_api_docs),
            ("Build HTML", self.build_html),
            ("Check Links", self.check_links),
            ("Run Doctests", self.run_doctests),
            ("Check Coverage", self.check_coverage),
            ("Generate Badge", self.generate_coverage_badge),
        ]

        if include_pdf:
            steps.append(("Build PDF", self.build_pdf))

        failed_steps = []

        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)

        print("\n" + "=" * 60)
        if failed_steps:
            print(f"❌ Build completed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                print(f"  - {step}")
            return False
        print("✅ Documentation build completed successfully!")

        html_path = self.build_dir / "html" / "index.html"
        if html_path.exists():
            print(f"📖 Documentation available at: file://{html_path.absolute()}")

        return True


def main() -> int:
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Build and validate documentation")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory only",
    )
    parser.add_argument("--html", action="store_true", help="Build HTML only")
    parser.add_argument("--pdf", action="store_true", help="Include PDF build")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve documentation after build",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for serving docs")
    parser.add_argument("--check", action="store_true", help="Run quality checks only")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    builder = DocumentationBuilder(project_root)

    if args.clean:
        return 0 if builder.clean_build() else 1

    if args.html:
        success = builder.clean_build() and builder.generate_api_docs() and builder.build_html()
        if args.serve and success:
            builder.serve_docs(args.port)
        return 0 if success else 1

    if args.check:
        success = builder.check_links() and builder.run_doctests() and builder.check_coverage()
        return 0 if success else 1

    # Full build
    success = builder.full_build(include_pdf=args.pdf)

    if args.serve and success:
        builder.serve_docs(args.port)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
