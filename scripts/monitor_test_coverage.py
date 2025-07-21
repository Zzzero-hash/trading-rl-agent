#!/usr/bin/env python3
"""
Test Coverage Monitoring Script

This script monitors test coverage and ensures the 95%+ target is maintained:

1. Tracks coverage trends over time
2. Identifies coverage gaps and regressions
3. Generates coverage improvement recommendations
4. Maintains coverage history and reporting
5. Alerts on coverage drops below thresholds
"""

import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CoverageMonitor:
    """Monitors and tracks test coverage over time."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.coverage_dir = self.project_root / "coverage_history"
        self.coverage_dir.mkdir(exist_ok=True)

        # Initialize coverage database
        self.db_path = self.coverage_dir / "coverage_history.db"
        self._init_database()

        # Coverage thresholds
        self.min_coverage = 95.0
        self.warning_threshold = 97.0

    def _init_database(self):
        """Initialize SQLite database for coverage tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_coverage REAL NOT NULL,
                lines_covered INTEGER NOT NULL,
                lines_total INTEGER NOT NULL,
                branches_covered INTEGER,
                branches_total INTEGER,
                functions_covered INTEGER,
                functions_total INTEGER,
                test_count INTEGER,
                test_duration REAL,
                commit_hash TEXT,
                branch TEXT,
                coverage_data TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def run_coverage_analysis(self) -> dict[str, Any]:
        """Run coverage analysis and collect metrics."""
        logger.info("Running coverage analysis...")

        # Run pytest with coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "tests/integration",
            "tests/smoke",
            "--cov=src",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "--cov-fail-under=0",  # Don't fail, just collect
            "--json-report",
            "--json-report-file=test-results.json",
            "-v",
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600,  # 10 minute timeout
            )

            # Parse coverage data
            coverage_data = self._parse_coverage_data()
            test_data = self._parse_test_results()

            return {
                "success": result.returncode == 0,
                "coverage": coverage_data,
                "tests": test_data,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_coverage_data(self) -> dict[str, Any]:
        """Parse coverage.json file."""
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            return {}

        try:
            with open(coverage_file) as f:
                data = json.load(f)

            # Extract key metrics
            totals = data.get("totals", {})
            return {
                "overall_coverage": totals.get("percent_covered", 0),
                "lines_covered": totals.get("covered_lines", 0),
                "lines_total": totals.get("num_statements", 0),
                "branches_covered": totals.get("covered_branches", 0),
                "branches_total": totals.get("num_branches", 0),
                "functions_covered": totals.get("covered_functions", 0),
                "functions_total": totals.get("num_functions", 0),
                "raw_data": data,
            }
        except Exception as exc:
            logger.exception(f"Error parsing coverage data: {exc}")
            return {}

    def _parse_test_results(self) -> dict[str, Any]:
        """Parse test results from JSON report."""
        test_file = self.project_root / "test-results.json"
        if not test_file.exists():
            return {}

        try:
            with open(test_file) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            return {
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "duration": data.get("duration", 0),
            }
        except Exception as exc:
            logger.exception(f"Error parsing test results: {exc}")
            return {}

    def save_coverage_data(self, coverage_data: dict[str, Any], test_data: dict[str, Any]):
        """Save coverage data to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current git info
        commit_hash = self._get_git_commit_hash()
        branch = self._get_git_branch()

        cursor.execute(
            """
            INSERT INTO coverage_history (
                timestamp, overall_coverage, lines_covered, lines_total,
                branches_covered, branches_total, functions_covered, functions_total,
                test_count, test_duration, commit_hash, branch, coverage_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                coverage_data.get("overall_coverage", 0),
                coverage_data.get("lines_covered", 0),
                coverage_data.get("lines_total", 0),
                coverage_data.get("branches_covered", 0),
                coverage_data.get("branches_total", 0),
                coverage_data.get("functions_covered", 0),
                coverage_data.get("functions_total", 0),
                test_data.get("total", 0),
                test_data.get("duration", 0),
                commit_hash,
                branch,
                json.dumps(coverage_data.get("raw_data", {})),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Saved coverage data: {coverage_data.get('overall_coverage', 0):.2f}%")

    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def get_coverage_history(self, days: int = 30) -> pd.DataFrame:
        """Get coverage history for the specified number of days."""
        conn = sqlite3.connect(self.db_path)

        # Use parameterized query to avoid SQL injection
        query = """
            SELECT * FROM coverage_history
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        """

        df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def analyze_coverage_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyze coverage trends and identify issues."""
        df = self.get_coverage_history(days)

        if df.empty:
            return {"error": "No coverage data available"}

        analysis = {
            "current_coverage": df.iloc[0]["overall_coverage"],
            "average_coverage": df["overall_coverage"].mean(),
            "min_coverage": df["overall_coverage"].min(),
            "max_coverage": df["overall_coverage"].max(),
            "trend": "stable",
            "issues": [],
            "recommendations": [],
        }

        # Analyze trend
        if len(df) > 1:
            recent_avg = df.head(7)["overall_coverage"].mean()
            older_avg = df.tail(7)["overall_coverage"].mean()

            if recent_avg > older_avg + 1:
                analysis["trend"] = "improving"
            elif recent_avg < older_avg - 1:
                analysis["trend"] = "declining"

        # Check for issues
        current_coverage = analysis["current_coverage"]

        if current_coverage < self.min_coverage:
            analysis["issues"].append(
                f"Coverage below minimum threshold: {current_coverage:.2f}% < {self.min_coverage}%"
            )

        if current_coverage < self.warning_threshold:
            analysis["issues"].append(
                f"Coverage below warning threshold: {current_coverage:.2f}% < {self.warning_threshold}%"
            )

        # Generate recommendations
        if current_coverage < self.min_coverage:
            analysis["recommendations"].append("Immediate action required: Add tests to reach 95% coverage")

        if analysis["trend"] == "declining":
            analysis["recommendations"].append("Investigate recent coverage drops and add missing tests")

        if df["test_count"].iloc[0] < 100:
            analysis["recommendations"].append("Consider adding more test cases for better coverage")

        return analysis

    def generate_coverage_report(self, analysis: dict[str, Any]) -> str:
        """Generate comprehensive coverage report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TEST COVERAGE MONITORING REPORT")
        report_lines.append("=" * 60)

        # Current status
        current_coverage = analysis.get("current_coverage", 0)
        status_icon = "✓" if current_coverage >= self.min_coverage else "✗"
        report_lines.append(f"Current Coverage: {current_coverage:.2f}% {status_icon}")
        report_lines.append(f"Target Coverage: {self.min_coverage}%")
        report_lines.append(f"Warning Threshold: {self.warning_threshold}%")

        # Trend analysis
        report_lines.append("\nTrend Analysis:")
        report_lines.append(f"  Trend: {analysis.get('trend', 'unknown')}")
        report_lines.append(f"  Average (30 days): {analysis.get('average_coverage', 0):.2f}%")
        report_lines.append(f"  Range: {analysis.get('min_coverage', 0):.2f}% - {analysis.get('max_coverage', 0):.2f}%")

        # Issues
        issues = analysis.get("issues", [])
        if issues:
            report_lines.append("\nIssues Found:")
            for issue in issues:
                report_lines.append(f"  ⚠️  {issue}")
        else:
            report_lines.append("\n✓ No issues found")

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            report_lines.append("\nRecommendations:")
            for rec in recommendations:
                report_lines.append(f"  💡 {rec}")

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    def create_coverage_chart(self, days: int = 30) -> Path | None:
        """Create a coverage trend chart."""
        df = self.get_coverage_history(days)

        if df.empty:
            logger.warning("No data available for chart generation")
            return None

        plt.figure(figsize=(12, 8))

        # Coverage trend
        plt.subplot(2, 1, 1)
        plt.plot(df["timestamp"], df["overall_coverage"], marker="o", linewidth=2)
        plt.axhline(
            y=self.min_coverage,
            color="r",
            linestyle="--",
            label=f"Target ({self.min_coverage}%)",
        )
        plt.axhline(
            y=self.warning_threshold,
            color="orange",
            linestyle="--",
            label=f"Warning ({self.warning_threshold}%)",
        )
        plt.title("Test Coverage Trend")
        plt.ylabel("Coverage (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Test count trend
        plt.subplot(2, 1, 2)
        plt.plot(df["timestamp"], df["test_count"], marker="s", color="green", linewidth=2)
        plt.title("Test Count Trend")
        plt.ylabel("Number of Tests")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        chart_path = self.coverage_dir / f"coverage_trend_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def run_full_monitoring(self) -> dict[str, Any]:
        """Run complete coverage monitoring workflow."""
        logger.info("Starting coverage monitoring...")

        # Step 1: Run coverage analysis
        analysis_result = self.run_coverage_analysis()
        if not analysis_result.get("success"):
            return {
                "success": False,
                "error": "Coverage analysis failed",
                "details": analysis_result,
            }

        # Step 2: Save coverage data
        coverage_data = analysis_result.get("coverage", {})
        test_data = analysis_result.get("tests", {})
        self.save_coverage_data(coverage_data, test_data)

        # Step 3: Analyze trends
        trend_analysis = self.analyze_coverage_trends()

        # Step 4: Generate report
        report = self.generate_coverage_report(trend_analysis)

        # Step 5: Create chart
        chart_path = self.create_coverage_chart()

        # Save report
        report_path = self.coverage_dir / f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Print report
        print(report)

        return {
            "success": True,
            "coverage_data": coverage_data,
            "trend_analysis": trend_analysis,
            "report": report,
            "report_path": report_path,
            "chart_path": chart_path,
        }


def main():
    """Main entry point for coverage monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor test coverage for Trading RL Agent")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--chart-only", action="store_true", help="Only generate chart")
    parser.add_argument("--report-only", action="store_true", help="Only generate report")
    parser.add_argument("--threshold", type=float, default=95.0, help="Minimum coverage threshold")

    args = parser.parse_args()

    monitor = CoverageMonitor()
    monitor.min_coverage = args.threshold

    if args.chart_only:
        chart_path = monitor.create_coverage_chart(args.days)
        if chart_path:
            print(f"Chart saved to: {chart_path}")
        else:
            print("No data available for chart generation")
        sys.exit(0)

    if args.report_only:
        analysis = monitor.analyze_coverage_trends(args.days)
        report = monitor.generate_coverage_report(analysis)
        print(report)
        sys.exit(0)

    # Run full monitoring
    results = monitor.run_full_monitoring()

    if results["success"]:
        print(f"\nReport saved to: {results['report_path']}")
        if results["chart_path"]:
            print(f"Chart saved to: {results['chart_path']}")
        sys.exit(0)
    else:
        print(f"Monitoring failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
