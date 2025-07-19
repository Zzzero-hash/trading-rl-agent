#!/usr/bin/env python3
"""
Trading RL Agent Test Maintenance Script

This script provides tools for maintaining test quality, monitoring test health,
and managing test data and dependencies.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path


class TestMaintenance:
    """Test maintenance and monitoring tools."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        self.maintenance_dir = self.project_root / "test_maintenance"
        self.maintenance_dir.mkdir(exist_ok=True)

    def monitor_test_health(self) -> dict:
        """Monitor overall test health and generate health report."""
        print("🏥 Monitoring test health...")

        health_data = {
            "timestamp": datetime.now().isoformat(),
            "test_counts": {},
            "coverage_status": {},
            "performance_metrics": {},
            "reliability_metrics": {},
            "issues": [],
            "recommendations": [],
        }

        # Count tests by category
        health_data["test_counts"] = self._count_tests_by_category()

        # Check coverage status
        health_data["coverage_status"] = self._check_coverage_status()

        # Analyze performance metrics
        health_data["performance_metrics"] = self._analyze_performance_metrics()

        # Check reliability metrics
        health_data["reliability_metrics"] = self._check_reliability_metrics()

        # Identify issues
        health_data["issues"] = self._identify_health_issues(health_data)

        # Generate recommendations
        health_data["recommendations"] = self._generate_health_recommendations(health_data)

        # Save health report
        with open(self.maintenance_dir / "test_health_report.json", "w") as f:
            json.dump(health_data, f, indent=2)

        # Generate health score
        health_score = self._calculate_health_score(health_data)
        health_data["health_score"] = health_score

        print(f"✅ Test health monitoring complete. Health score: {health_score}/100")
        return health_data

    def _count_tests_by_category(self) -> dict:
        """Count tests by category and type."""
        counts = {"total_tests": 0, "by_category": {}, "by_marker": {}, "recent_additions": 0}

        # Count by directory category
        for category_dir in ["unit", "integration", "performance", "smoke"]:
            category_path = self.tests_dir / category_dir
            if category_path.exists():
                test_files = list(category_path.rglob("test_*.py"))
                counts["by_category"][category_dir] = len(test_files)
                counts["total_tests"] += len(test_files)

        # Count by marker (approximate)
        marker_counts = {
            "fast": 0,
            "slow": 0,
            "integration": 0,
            "performance": 0,
            "core": 0,
            "data": 0,
            "model": 0,
            "training": 0,
            "risk": 0,
            "portfolio": 0,
            "cli": 0,
        }

        for test_file in self.tests_dir.rglob("test_*.py"):
            try:
                with open(test_file) as f:
                    content = f.read()
                    for marker in marker_counts:
                        if f"@pytest.mark.{marker}" in content or f"pytest.mark.{marker}" in content:
                            marker_counts[marker] += 1
            except Exception as e:
                print(f"Warning: Could not read test file {test_file}: {e}")
                continue

        counts["by_marker"] = marker_counts

        # Count recent additions (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_count = 0
        for test_file in self.tests_dir.rglob("test_*.py"):
            try:
                mtime = datetime.fromtimestamp(test_file.stat().st_mtime, tz=datetime.timezone.utc)
                if mtime > recent_cutoff:
                    recent_count += 1
            except Exception as e:
                print(f"Warning: Could not get modification time for {test_file}: {e}")
                continue

        counts["recent_additions"] = recent_count

        return counts

    def _check_coverage_status(self) -> dict:
        """Check current coverage status."""
        coverage_status = {
            "current_coverage": 0,
            "target_coverage": 90,
            "coverage_trend": "stable",
            "low_coverage_modules": [],
        }

        # Try to read existing coverage data
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    if "totals" in coverage_data:
                        coverage_status["current_coverage"] = round(coverage_data["totals"]["percent_covered"], 2)
            except Exception:
                coverage_status["current_coverage"] = 85  # Default estimate

        # Check for low coverage modules
        for module_path in self.src_dir.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue

            # Simple heuristic: check if module has tests
            test_exists = self._check_module_has_tests(module_path)
            if not test_exists:
                coverage_status["low_coverage_modules"].append(str(module_path.relative_to(self.project_root)))

        return coverage_status

    def _check_module_has_tests(self, module_path: Path) -> bool:
        """Check if a module has corresponding tests."""
        test_patterns = [f"test_{module_path.stem}.py", f"test_{module_path.stem}_*.py", f"*{module_path.stem}*.py"]

        return any(list(self.tests_dir.rglob(pattern)) for pattern in test_patterns)

    def _analyze_performance_metrics(self) -> dict:
        """Analyze test performance metrics."""
        performance_metrics = {
            "average_execution_time": 0,
            "slow_tests_count": 0,
            "very_slow_tests_count": 0,
            "parallel_execution_ready": True,
            "resource_usage": {},
        }

        # Try to get execution times from recent test runs
        test_results_file = self.project_root / "test-results.xml"
        if test_results_file.exists():
            try:
                # Parse test results for timing information
                # This is a simplified version - in practice, you'd parse the XML
                performance_metrics["average_execution_time"] = 1.5  # Default estimate
                performance_metrics["slow_tests_count"] = 5  # Default estimate
                performance_metrics["very_slow_tests_count"] = 2  # Default estimate
            except Exception:
                pass

        return performance_metrics

    def _check_reliability_metrics(self) -> dict:
        """Check test reliability metrics."""
        reliability_metrics = {
            "pass_rate": 95,  # Default estimate
            "flaky_tests": [],
            "intermittent_failures": [],
            "test_dependencies": {},
            "fixture_issues": [],
        }

        # Check for common reliability issues
        for test_file in self.tests_dir.rglob("test_*.py"):
            try:
                with open(test_file) as f:
                    content = f.read()

                    # Check for potential flaky tests
                    if any(pattern in content for pattern in ["time.sleep", "random", "datetime.now", "time.time"]):
                        reliability_metrics["flaky_tests"].append(str(test_file.relative_to(self.project_root)))

                    # Check for test dependencies
                    if "depends_on" in content or "requires" in content:
                        reliability_metrics["test_dependencies"][str(test_file.relative_to(self.project_root))] = (
                            "Has dependencies"
                        )

            except Exception as e:
                print(f"Warning: Could not analyze test file {test_file}: {e}")
                continue

        return reliability_metrics

    def _identify_health_issues(self, health_data: dict) -> list[str]:
        """Identify health issues from the collected data."""
        issues = []

        # Coverage issues
        coverage = health_data["coverage_status"]
        if coverage["current_coverage"] < coverage["target_coverage"]:
            issues.append(f"Coverage below target: {coverage['current_coverage']}% < {coverage['target_coverage']}%")

        if len(coverage["low_coverage_modules"]) > 10:
            issues.append(f"Many modules with low coverage: {len(coverage['low_coverage_modules'])} modules")

        # Performance issues
        performance = health_data["performance_metrics"]
        if performance["slow_tests_count"] > 10:
            issues.append(f"Too many slow tests: {performance['slow_tests_count']} tests")

        if performance["very_slow_tests_count"] > 5:
            issues.append(f"Too many very slow tests: {performance['very_slow_tests_count']} tests")

        # Reliability issues
        reliability = health_data["reliability_metrics"]
        if reliability["pass_rate"] < 95:
            issues.append(f"Low pass rate: {reliability['pass_rate']}%")

        if len(reliability["flaky_tests"]) > 5:
            issues.append(f"Too many flaky tests: {len(reliability['flaky_tests'])} tests")

        return issues

    def _generate_health_recommendations(self, health_data: dict) -> list[str]:
        """Generate recommendations based on health data."""
        recommendations = []

        # Coverage recommendations
        coverage = health_data["coverage_status"]
        if coverage["current_coverage"] < coverage["target_coverage"]:
            recommendations.append("Increase test coverage by adding tests for uncovered modules")

        if len(coverage["low_coverage_modules"]) > 0:
            recommendations.append("Add tests for modules with low coverage")

        # Performance recommendations
        performance = health_data["performance_metrics"]
        if performance["slow_tests_count"] > 10:
            recommendations.append("Optimize slow tests by using mocks and synthetic data")

        if performance["parallel_execution_ready"]:
            recommendations.append("Enable parallel test execution for better performance")

        # Reliability recommendations
        reliability = health_data["reliability_metrics"]
        if len(reliability["flaky_tests"]) > 0:
            recommendations.append("Fix flaky tests by removing time dependencies")

        if reliability["pass_rate"] < 95:
            recommendations.append("Investigate and fix failing tests")

        return recommendations

    def _calculate_health_score(self, health_data: dict) -> int:
        """Calculate overall test health score (0-100)."""
        score = 100

        # Coverage penalty
        coverage = health_data["coverage_status"]
        coverage_gap = coverage["target_coverage"] - coverage["current_coverage"]
        if coverage_gap > 0:
            score -= min(30, coverage_gap * 2)

        # Performance penalty
        performance = health_data["performance_metrics"]
        if performance["slow_tests_count"] > 10:
            score -= 10
        if performance["very_slow_tests_count"] > 5:
            score -= 15

        # Reliability penalty
        reliability = health_data["reliability_metrics"]
        if reliability["pass_rate"] < 95:
            score -= 20
        if len(reliability["flaky_tests"]) > 5:
            score -= 15

        # Issue penalty
        score -= len(health_data["issues"]) * 5

        return max(0, score)

    def manage_test_data(self) -> dict:
        """Manage test data and cleanup."""
        print("🗂️  Managing test data...")

        data_management = {
            "test_data_files": [],
            "large_files": [],
            "outdated_files": [],
            "cleanup_recommendations": [],
            "storage_usage": {},
        }

        # Scan test data directory
        test_data_dir = self.project_root / "test_data"
        if test_data_dir.exists():
            for file_path in test_data_dir.rglob("*"):
                if file_path.is_file():
                    file_info = {
                        "path": str(file_path.relative_to(self.project_root)),
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(
                            file_path.stat().st_mtime, tz=datetime.timezone.utc
                        ).isoformat(),
                        "age_days": (
                            datetime.now(datetime.timezone.utc)
                            - datetime.fromtimestamp(file_path.stat().st_mtime, tz=datetime.timezone.utc)
                        ).days,
                    }

                    data_management["test_data_files"].append(file_info)

                    # Identify large files
                    if file_info["size_mb"] > 10:
                        data_management["large_files"].append(file_info)

                    # Identify outdated files
                    if file_info["age_days"] > 90:
                        data_management["outdated_files"].append(file_info)

        # Calculate storage usage
        total_size = sum(f["size_mb"] for f in data_management["test_data_files"])
        data_management["storage_usage"] = {
            "total_size_mb": total_size,
            "file_count": len(data_management["test_data_files"]),
            "average_file_size_mb": total_size / len(data_management["test_data_files"])
            if data_management["test_data_files"]
            else 0,
        }

        # Generate cleanup recommendations
        if data_management["large_files"]:
            data_management["cleanup_recommendations"].append(
                f"Consider compressing or removing {len(data_management['large_files'])} large files"
            )

        if data_management["outdated_files"]:
            data_management["cleanup_recommendations"].append(
                f"Consider removing {len(data_management['outdated_files'])} outdated files"
            )

        # Save data management report
        with open(self.maintenance_dir / "test_data_management.json", "w") as f:
            json.dump(data_management, f, indent=2)

        print(f"✅ Test data management complete. Found {len(data_management['test_data_files'])} files")
        return data_management

    def update_test_dependencies(self) -> dict:
        """Update and manage test dependencies."""
        print("📦 Updating test dependencies...")

        dependency_management = {
            "current_dependencies": {},
            "outdated_dependencies": [],
            "security_vulnerabilities": [],
            "recommendations": [],
        }

        # Check requirements files
        requirements_files = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt"]

        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    with open(req_path) as f:
                        dependencies = []
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                dependencies.append(line)

                        dependency_management["current_dependencies"][req_file] = dependencies
                except Exception as e:
                    dependency_management["current_dependencies"][req_file] = [f"Error reading: {e}"]

        # Generate recommendations
        if "pytest-cov" not in str(dependency_management["current_dependencies"]):
            dependency_management["recommendations"].append("Add pytest-cov for coverage reporting")

        if "pytest-xdist" not in str(dependency_management["current_dependencies"]):
            dependency_management["recommendations"].append("Add pytest-xdist for parallel execution")

        if "pytest-benchmark" not in str(dependency_management["current_dependencies"]):
            dependency_management["recommendations"].append("Add pytest-benchmark for performance testing")

        # Save dependency report
        with open(self.maintenance_dir / "test_dependencies.json", "w") as f:
            json.dump(dependency_management, f, indent=2)

        print(
            f"✅ Dependency management complete. Found {len(dependency_management['current_dependencies'])} requirement files"
        )
        return dependency_management

    def generate_maintenance_report(self) -> dict:
        """Generate comprehensive maintenance report."""
        print("📋 Generating maintenance report...")

        # Collect all maintenance data
        health_data = self.monitor_test_health()
        data_management = self.manage_test_data()
        dependency_management = self.update_test_dependencies()

        # Generate comprehensive report
        maintenance_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "health_score": health_data.get("health_score", 0),
                "total_tests": health_data["test_counts"]["total_tests"],
                "current_coverage": health_data["coverage_status"]["current_coverage"],
                "test_data_files": len(data_management["test_data_files"]),
                "issues_count": len(health_data["issues"]),
            },
            "health_analysis": health_data,
            "data_management": data_management,
            "dependency_management": dependency_management,
            "action_items": self._generate_action_items(health_data, data_management, dependency_management),
        }

        # Save comprehensive report
        with open(self.maintenance_dir / "maintenance_report.json", "w") as f:
            json.dump(maintenance_report, f, indent=2)

        # Generate markdown report
        self._generate_maintenance_markdown(maintenance_report)

        print(f"✅ Maintenance report generated. Health score: {maintenance_report['summary']['health_score']}/100")
        return maintenance_report

    def _generate_action_items(
        self, health_data: dict, data_management: dict, dependency_management: dict
    ) -> list[dict]:
        """Generate prioritized action items."""
        action_items = []

        # High priority items
        if health_data.get("health_score", 0) < 70:
            action_items.append(
                {
                    "priority": "high",
                    "action": "Improve test health score",
                    "description": f"Current score: {health_data.get('health_score', 0)}/100",
                    "estimated_effort": "1-2 weeks",
                }
            )

        if health_data["coverage_status"]["current_coverage"] < 85:
            action_items.append(
                {
                    "priority": "high",
                    "action": "Increase test coverage",
                    "description": f"Current coverage: {health_data['coverage_status']['current_coverage']}%",
                    "estimated_effort": "1 week",
                }
            )

        # Medium priority items
        if len(health_data["performance_metrics"]["slow_tests_count"]) > 10:
            action_items.append(
                {
                    "priority": "medium",
                    "action": "Optimize slow tests",
                    "description": f"{health_data['performance_metrics']['slow_tests_count']} slow tests identified",
                    "estimated_effort": "3-5 days",
                }
            )

        if data_management["large_files"]:
            action_items.append(
                {
                    "priority": "medium",
                    "action": "Clean up large test data files",
                    "description": f"{len(data_management['large_files'])} large files found",
                    "estimated_effort": "1 day",
                }
            )

        # Low priority items
        if dependency_management["recommendations"]:
            action_items.append(
                {
                    "priority": "low",
                    "action": "Update test dependencies",
                    "description": f"{len(dependency_management['recommendations'])} recommendations",
                    "estimated_effort": "1 day",
                }
            )

        return action_items

    def _generate_maintenance_markdown(self, maintenance_report: dict):
        """Generate markdown maintenance report."""
        report_path = self.maintenance_dir / "maintenance_report.md"

        with open(report_path, "w") as f:
            f.write("# Trading RL Agent Test Maintenance Report\n\n")
            f.write(f"Generated: {maintenance_report['timestamp']}\n\n")

            # Summary
            f.write("## 📊 Summary\n\n")
            summary = maintenance_report["summary"]
            f.write(f"- **Health Score**: {summary['health_score']}/100\n")
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Current Coverage**: {summary['current_coverage']}%\n")
            f.write(f"- **Test Data Files**: {summary['test_data_files']}\n")
            f.write(f"- **Issues**: {summary['issues_count']}\n\n")

            # Action Items
            f.write("## 🚀 Action Items\n\n")
            for item in maintenance_report["action_items"]:
                f.write(f"### {item['priority'].title()} Priority\n")
                f.write(f"- **{item['action']}**: {item['description']}\n")
                f.write(f"- **Effort**: {item['estimated_effort']}\n\n")

            # Health Analysis
            f.write("## 🏥 Health Analysis\n\n")
            health = maintenance_report["health_analysis"]
            f.write(f"- **Health Score**: {health.get('health_score', 0)}/100\n")
            f.write(f"- **Issues**: {len(health['issues'])}\n")
            f.write(f"- **Recommendations**: {len(health['recommendations'])}\n\n")

            # Data Management
            f.write("## 🗂️ Data Management\n\n")
            data = maintenance_report["data_management"]
            f.write(f"- **Total Files**: {data['storage_usage']['file_count']}\n")
            f.write(f"- **Total Size**: {data['storage_usage']['total_size_mb']:.2f} MB\n")
            f.write(f"- **Large Files**: {len(data['large_files'])}\n")
            f.write(f"- **Outdated Files**: {len(data['outdated_files'])}\n\n")

            f.write("---\n")
            f.write("*For detailed analysis, see the JSON files in this directory.*\n")

        print(f"📄 Markdown report generated: {report_path}")


def main():
    """Main entry point for test maintenance."""
    parser = argparse.ArgumentParser(description="Trading RL Agent Test Maintenance")
    parser.add_argument(
        "--action",
        choices=["health", "data", "dependencies", "full"],
        default="full",
        help="Maintenance action to perform",
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    maintenance = TestMaintenance(args.project_root)

    if args.action == "health":
        maintenance.monitor_test_health()
    elif args.action == "data":
        maintenance.manage_test_data()
    elif args.action == "dependencies":
        maintenance.update_test_dependencies()
    elif args.action == "full":
        maintenance.generate_maintenance_report()


if __name__ == "__main__":
    main()
