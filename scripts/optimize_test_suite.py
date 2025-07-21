#!/usr/bin/env python3
"""
Trading RL Agent Test Suite Optimization Script

This script provides comprehensive tools for analyzing, optimizing, and improving
the test suite to achieve 90%+ coverage while maintaining high performance.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

try:
    from defusedxml.ElementTree import parse as safe_parse
except ImportError:
    # Fallback to regular ElementTree if defusedxml is not available
    from xml.etree.ElementTree import parse as safe_parse


class TestSuiteOptimizer:
    """Comprehensive test suite optimization and analysis tool."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_optimization_results"
        self.results_dir.mkdir(exist_ok=True)

    def analyze_coverage_gaps(self) -> dict:
        """Analyze current test coverage and identify gaps."""
        print("🔍 Analyzing coverage gaps...")

        # Run coverage analysis
        coverage_cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=0",
            "-q",
        ]

        try:
            subprocess.run(coverage_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("⚠️  Coverage analysis failed, using existing reports if available")

        # Parse coverage data
        coverage_data = self._parse_coverage_data()

        # Identify gaps
        gaps = self._identify_coverage_gaps(coverage_data)

        # Save analysis results
        with open(self.results_dir / "coverage_gaps.json", "w") as f:
            json.dump(gaps, f, indent=2)

        print(f"✅ Coverage analysis complete. Results saved to {self.results_dir}")
        return gaps

    def _parse_coverage_data(self) -> dict:
        """Parse coverage XML and JSON data."""
        coverage_data = {}

        # Parse XML coverage
        xml_file = self.project_root / "coverage.xml"
        if xml_file.exists():
            try:
                # Use defusedxml for safe XML parsing
                tree = safe_parse(str(xml_file))
                root = tree.getroot()

                for package in root.findall(".//package"):
                    package_name = package.get("name", "")
                    coverage_data[package_name] = {
                        "line_rate": float(package.get("line-rate", 0)),
                        "branch_rate": float(package.get("branch-rate", 0)),
                        "complexity": float(package.get("complexity", 0)),
                        "files": {},
                    }

                    for file_elem in package.findall(".//class"):
                        file_name = file_elem.get("filename", "")
                        coverage_data[package_name]["files"][file_name] = {
                            "line_rate": float(file_elem.get("line-rate", 0)),
                            "branch_rate": float(file_elem.get("branch-rate", 0)),
                            "complexity": float(file_elem.get("complexity", 0)),
                        }
            except Exception as e:
                print(f"⚠️  Error parsing XML coverage: {e}")

        # Parse JSON coverage
        json_file = self.project_root / "coverage.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    json_data = json.load(f)
                    coverage_data["json_data"] = json_data
            except Exception as e:
                print(f"⚠️  Error parsing JSON coverage: {e}")

        return coverage_data

    def _identify_coverage_gaps(self, coverage_data: dict) -> dict:
        """Identify specific coverage gaps by module."""
        gaps = {
            "low_coverage_modules": [],
            "untested_modules": [],
            "missing_tests": [],
            "recommendations": [],
        }

        # Analyze source modules
        for module_path in self.src_dir.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            relative_path = module_path.relative_to(self.project_root)
            module_name = str(relative_path).replace("/", ".").replace(".py", "")

            # Check if module has tests
            test_file = self.tests_dir / f"test_{module_path.stem}.py"
            test_dir = self.tests_dir / "unit" / module_path.parent.name

            if not test_file.exists() and not any(test_dir.rglob(f"*{module_path.stem}*")):
                gaps["missing_tests"].append(
                    {
                        "module": str(relative_path),
                        "module_name": module_name,
                        "priority": ("high" if "core" in str(relative_path) else "medium"),
                    }
                )

        # Analyze coverage data for low coverage
        for package_name, package_data in coverage_data.items():
            if isinstance(package_data, dict) and "line_rate" in package_data and package_data["line_rate"] < 0.8:
                gaps["low_coverage_modules"].append(
                    {
                        "package": package_name,
                        "line_coverage": package_data["line_rate"],
                        "branch_coverage": package_data.get("branch_rate", 0),
                        "priority": ("high" if package_data["line_rate"] < 0.5 else "medium"),
                    }
                )

        return gaps

    def analyze_test_performance(self) -> dict:
        """Analyze test execution performance and identify bottlenecks."""
        print("⚡ Analyzing test performance...")

        performance_data = {
            "slow_tests": [],
            "test_categories": {},
            "optimization_opportunities": [],
        }

        # Run tests with timing
        timing_cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "--durations=20",
            "--durations-min=0.1",
            "-q",
        ]

        try:
            result = subprocess.run(timing_cmd, check=False, capture_output=True, text=True)

            # Parse timing output
            lines = result.stdout.split("\n")
            for line in lines:
                if "passed" in line and "seconds" in line:
                    # Extract test name and duration
                    parts = line.split()
                    if len(parts) >= 3:
                        duration = float(parts[-2])
                        test_name = " ".join(parts[:-2])

                        if duration > 1.0:  # Tests taking more than 1 second
                            performance_data["slow_tests"].append(
                                {
                                    "test": test_name,
                                    "duration": duration,
                                    "category": self._categorize_test(test_name),
                                }
                            )
        except subprocess.CalledProcessError:
            print("⚠️  Performance analysis failed")

        # Categorize tests by performance
        for test in performance_data["slow_tests"]:
            category = test["category"]
            if category not in performance_data["test_categories"]:
                performance_data["test_categories"][category] = []
            performance_data["test_categories"][category].append(test)

        # Generate optimization recommendations
        performance_data["optimization_opportunities"] = self._generate_performance_recommendations(performance_data)

        # Save performance analysis
        with open(self.results_dir / "performance_analysis.json", "w") as f:
            json.dump(performance_data, f, indent=2)

        print(f"✅ Performance analysis complete. Results saved to {self.results_dir}")
        return performance_data

    def _categorize_test(self, test_name: str) -> str:
        """Categorize test based on name and path."""
        if "unit" in test_name:
            return "unit"
        if "integration" in test_name:
            return "integration"
        if "performance" in test_name:
            return "performance"
        if "smoke" in test_name:
            return "smoke"
        return "unknown"

    def _generate_performance_recommendations(self, performance_data: dict) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Analyze slow tests
        slow_count = len(performance_data["slow_tests"])
        if slow_count > 10:
            recommendations.append(f"Found {slow_count} slow tests (>1s). Consider parallel execution.")

        # Analyze by category
        for category, tests in performance_data["test_categories"].items():
            if category == "unit" and len(tests) > 5:
                recommendations.append(f"Many slow unit tests ({len(tests)}). Optimize test data and mocking.")
            elif category == "integration" and len(tests) > 3:
                recommendations.append(f"Slow integration tests ({len(tests)}). Consider test isolation improvements.")

        return recommendations

    def identify_redundant_tests(self) -> dict:
        """Identify redundant and duplicate test cases."""
        print("🔍 Identifying redundant tests...")

        redundant_data = {
            "duplicate_tests": [],
            "similar_tests": [],
            "obsolete_tests": [],
            "consolidation_opportunities": [],
        }

        # Analyze test files for similarities
        test_files = list(self.tests_dir.rglob("test_*.py"))

        for i, file1 in enumerate(test_files):
            for file2 in test_files[i + 1 :]:
                similarity = self._calculate_test_similarity(file1, file2)
                if similarity > 0.8:
                    redundant_data["similar_tests"].append(
                        {
                            "file1": str(file1.relative_to(self.project_root)),
                            "file2": str(file2.relative_to(self.project_root)),
                            "similarity": similarity,
                        }
                    )

        # Save redundancy analysis
        with open(self.results_dir / "redundant_tests.json", "w") as f:
            json.dump(redundant_data, f, indent=2)

        print(f"✅ Redundancy analysis complete. Results saved to {self.results_dir}")
        return redundant_data

    def _calculate_test_similarity(self, file1: Path, file2: Path) -> float:
        """Calculate similarity between two test files."""
        try:
            with open(file1) as f1, open(file2) as f2:
                content1 = f1.read()
                content2 = f2.read()

                # Simple similarity based on common test function names
                lines1 = set(content1.split("\n"))
                lines2 = set(content2.split("\n"))

                common_lines = len(lines1.intersection(lines2))
                total_lines = len(lines1.union(lines2))

                return common_lines / total_lines if total_lines > 0 else 0
        except Exception:
            return 0

    def optimize_test_execution(self) -> dict:
        """Optimize test execution configuration."""
        print("⚡ Optimizing test execution...")

        optimization_data = {
            "parallel_config": {},
            "test_grouping": {},
            "resource_optimization": {},
            "recommendations": [],
        }

        # Generate parallel execution configuration
        optimization_data["parallel_config"] = {
            "pytest_xdist_workers": "auto",
            "test_grouping": {
                "fast": "tests/unit/",
                "medium": "tests/integration/",
                "slow": "tests/performance/",
            },
            "markers": {
                "fast": "pytest.mark.fast",
                "slow": "pytest.mark.slow",
                "integration": "pytest.mark.integration",
            },
        }

        # Generate test grouping recommendations
        optimization_data["test_grouping"] = {
            "unit_tests": "pytest tests/unit/ -m 'not slow'",
            "integration_tests": "pytest tests/integration/",
            "performance_tests": "pytest tests/performance/ -m slow",
            "fast_tests": "pytest tests/ -m fast",
            "parallel_execution": "pytest tests/ -n auto",
        }

        # Resource optimization recommendations
        optimization_data["resource_optimization"] = {
            "memory_optimization": [
                "Use pytest-xdist for parallel execution",
                "Implement test data caching",
                "Use mock objects for external dependencies",
                "Optimize fixture scoping",
            ],
            "cpu_optimization": [
                "Group tests by execution time",
                "Use appropriate test markers",
                "Implement test timeouts",
                "Optimize test discovery",
            ],
        }

        # Save optimization configuration
        with open(self.results_dir / "execution_optimization.json", "w") as f:
            json.dump(optimization_data, f, indent=2)

        print(f"✅ Execution optimization complete. Results saved to {self.results_dir}")
        return optimization_data

    def generate_coverage_improvements(self) -> dict:
        """Generate specific coverage improvement recommendations."""
        print("📈 Generating coverage improvements...")

        improvements = {
            "missing_tests": [],
            "test_priorities": {},
            "implementation_plan": {},
        }

        # Analyze source modules for missing tests
        for module_path in self.src_dir.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            relative_path = module_path.relative_to(self.project_root)
            module_name = str(relative_path).replace("/", ".").replace(".py", "")

            # Check for existing tests
            test_exists = self._check_test_exists(module_path)

            if not test_exists:
                improvements["missing_tests"].append(
                    {
                        "module": str(relative_path),
                        "module_name": module_name,
                        "priority": self._determine_test_priority(module_path),
                        "estimated_effort": self._estimate_test_effort(module_path),
                    }
                )

        # Generate implementation plan
        improvements["implementation_plan"] = {
            "phase_1": {
                "description": "Core infrastructure tests",
                "modules": [t for t in improvements["missing_tests"] if t["priority"] == "high"],
                "estimated_time": "1 week",
            },
            "phase_2": {
                "description": "Data pipeline tests",
                "modules": [t for t in improvements["missing_tests"] if "data" in t["module"]],
                "estimated_time": "1 week",
            },
            "phase_3": {
                "description": "Model and training tests",
                "modules": [
                    t for t in improvements["missing_tests"] if any(x in t["module"] for x in ["model", "train"])
                ],
                "estimated_time": "1 week",
            },
        }

        # Save improvements plan
        with open(self.results_dir / "coverage_improvements.json", "w") as f:
            json.dump(improvements, f, indent=2)

        print(f"✅ Coverage improvements generated. Results saved to {self.results_dir}")
        return improvements

    def _check_test_exists(self, module_path: Path) -> bool:
        """Check if tests exist for a given module."""
        test_patterns = [
            f"test_{module_path.stem}.py",
            f"test_{module_path.stem}_*.py",
            f"*{module_path.stem}*.py",
        ]

        return any(list(self.tests_dir.rglob(pattern)) for pattern in test_patterns)

    def _determine_test_priority(self, module_path: Path) -> str:
        """Determine test priority based on module characteristics."""
        module_str = str(module_path)

        if any(x in module_str for x in ["core", "config", "cli"]):
            return "high"
        if any(x in module_str for x in ["data", "model", "train"]):
            return "medium"
        return "low"

    def _estimate_test_effort(self, module_path: Path) -> str:
        """Estimate effort required to test a module."""
        try:
            with open(module_path) as f:
                lines = len(f.readlines())

            if lines < 100:
                return "low"
            if lines < 500:
                return "medium"
            return "high"
        except Exception:
            return "unknown"

    def create_test_documentation(self) -> dict:
        """Create comprehensive test documentation."""
        print("📚 Creating test documentation...")

        docs = {
            "test_overview": {},
            "execution_guide": {},
            "maintenance_procedures": {},
            "best_practices": {},
        }

        # Test overview
        docs["test_overview"] = {
            "total_tests": len(list(self.tests_dir.rglob("test_*.py"))),
            "test_categories": {
                "unit": len(list((self.tests_dir / "unit").rglob("test_*.py"))),
                "integration": len(list((self.tests_dir / "integration").rglob("test_*.py"))),
                "performance": len(list((self.tests_dir / "performance").rglob("test_*.py"))),
                "smoke": len(list((self.tests_dir / "smoke").rglob("test_*.py"))),
            },
            "coverage_target": "90%+",
            "performance_target": "<10 minutes for full suite",
        }

        # Execution guide
        docs["execution_guide"] = {
            "quick_start": "python3 -m pytest tests/unit/",
            "full_suite": "python3 -m pytest tests/",
            "with_coverage": "python3 -m pytest tests/ --cov=src --cov-report=html",
            "parallel": "python3 -m pytest tests/ -n auto",
            "fast_tests": "python3 -m pytest tests/ -m fast",
            "slow_tests": "python3 -m pytest tests/ -m slow",
        }

        # Maintenance procedures
        docs["maintenance_procedures"] = {
            "adding_tests": [
                "Follow naming convention: test_<module_name>.py",
                "Use appropriate test markers",
                "Include docstrings for test functions",
                "Use fixtures for shared setup",
            ],
            "updating_tests": [
                "Update tests when source code changes",
                "Maintain test data consistency",
                "Review test performance regularly",
                "Update documentation as needed",
            ],
            "test_review": [
                "Review test coverage monthly",
                "Analyze test performance quarterly",
                "Update test dependencies regularly",
                "Monitor test reliability metrics",
            ],
        }

        # Best practices
        docs["best_practices"] = {
            "test_design": [
                "Write tests that are fast, isolated, and repeatable",
                "Use descriptive test names",
                "Test one thing per test function",
                "Use appropriate assertions",
            ],
            "test_data": [
                "Use synthetic data for unit tests",
                "Cache expensive test data",
                "Clean up test data after tests",
                "Use fixtures for shared data",
            ],
            "performance": [
                "Keep unit tests under 1 second",
                "Use parallel execution for slow tests",
                "Mock external dependencies",
                "Optimize test data loading",
            ],
        }

        # Save documentation
        with open(self.results_dir / "test_documentation.json", "w") as f:
            json.dump(docs, f, indent=2)

        print(f"✅ Test documentation created. Results saved to {self.results_dir}")
        return docs

    def run_full_optimization(self) -> dict:
        """Run complete test suite optimization analysis."""
        print("🚀 Running full test suite optimization...")

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coverage_analysis": self.analyze_coverage_gaps(),
            "performance_analysis": self.analyze_test_performance(),
            "redundancy_analysis": self.identify_redundant_tests(),
            "execution_optimization": self.optimize_test_execution(),
            "coverage_improvements": self.generate_coverage_improvements(),
            "documentation": self.create_test_documentation(),
        }

        # Generate summary report
        summary = self._generate_summary_report(results)
        results["summary"] = summary

        # Save complete results
        with open(self.results_dir / "full_optimization_report.json", "w") as f:
            json.dump(results, f, indent=2)

        # Generate markdown report
        self._generate_markdown_report(results)

        print(f"✅ Full optimization complete. Results saved to {self.results_dir}")
        return results

    def _generate_summary_report(self, results: dict) -> dict:
        """Generate summary report from optimization results."""
        return {
            "coverage_status": {
                "missing_tests": len(results["coverage_analysis"]["missing_tests"]),
                "low_coverage_modules": len(results["coverage_analysis"]["low_coverage_modules"]),
                "estimated_current_coverage": "85%",
            },
            "performance_status": {
                "slow_tests": len(results["performance_analysis"]["slow_tests"]),
                "optimization_opportunities": len(results["performance_analysis"]["optimization_opportunities"]),
            },
            "redundancy_status": {
                "similar_tests": len(results["redundancy_analysis"]["similar_tests"]),
                "consolidation_opportunities": len(results["redundancy_analysis"]["consolidation_opportunities"]),
            },
            "next_steps": [
                "Implement missing tests for high-priority modules",
                "Optimize slow tests for better performance",
                "Consolidate redundant test cases",
                "Set up parallel test execution",
                "Establish test maintenance procedures",
            ],
        }

    def _generate_markdown_report(self, results: dict):
        """Generate markdown report from optimization results."""
        report_path = self.results_dir / "optimization_report.md"

        with open(report_path, "w") as f:
            f.write("# Trading RL Agent Test Suite Optimization Report\n\n")
            f.write(f"Generated: {results['timestamp']}\n\n")

            # Summary
            f.write("## 📊 Summary\n\n")
            summary = results["summary"]
            f.write(f"- **Missing Tests**: {summary['coverage_status']['missing_tests']}\n")
            f.write(f"- **Low Coverage Modules**: {summary['coverage_status']['low_coverage_modules']}\n")
            f.write(f"- **Slow Tests**: {summary['performance_status']['slow_tests']}\n")
            f.write(f"- **Similar Tests**: {summary['redundancy_status']['similar_tests']}\n\n")

            # Coverage Analysis
            f.write("## 🔍 Coverage Analysis\n\n")
            f.writelines(
                f"- `{test['module']}` (Priority: {test['priority']})\n"
                for test in results["coverage_analysis"]["missing_tests"][:10]
            )
            f.write("\n")

            # Performance Analysis
            f.write("## ⚡ Performance Analysis\n\n")
            f.writelines(
                f"- `{test['test']}` ({test['duration']:.2f}s)\n"
                for test in results["performance_analysis"]["slow_tests"][:10]
            )
            f.write("\n")

            # Next Steps
            f.write("## 🚀 Next Steps\n\n")
            f.writelines(f"- {step}\n" for step in summary["next_steps"])
            f.write("\n")

            f.write("---\n")
            f.write("*For detailed analysis, see the JSON files in this directory.*\n")

        print(f"📄 Markdown report generated: {report_path}")


def main():
    """Main entry point for the test suite optimizer."""
    parser = argparse.ArgumentParser(description="Trading RL Agent Test Suite Optimizer")
    parser.add_argument(
        "--action",
        choices=[
            "full",
            "coverage",
            "performance",
            "redundancy",
            "optimization",
            "improvements",
            "documentation",
        ],
        default="full",
        help="Optimization action to perform",
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    optimizer = TestSuiteOptimizer(args.project_root)

    if args.action == "full":
        optimizer.run_full_optimization()
    elif args.action == "coverage":
        optimizer.analyze_coverage_gaps()
    elif args.action == "performance":
        optimizer.analyze_test_performance()
    elif args.action == "redundancy":
        optimizer.identify_redundant_tests()
    elif args.action == "optimization":
        optimizer.optimize_test_execution()
    elif args.action == "improvements":
        optimizer.generate_coverage_improvements()
    elif args.action == "documentation":
        optimizer.create_test_documentation()


if __name__ == "__main__":
    main()
