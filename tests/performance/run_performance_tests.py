#!/usr/bin/env python3
"""
Performance Testing Runner for Trading RL Agent.

This script provides a comprehensive interface for running performance tests,
generating reports, and integrating with CI/CD pipelines.

Usage:
    python run_performance_tests.py --test-type all
    python run_performance_tests.py --test-type data --benchmark
    python run_performance_tests.py --test-type stress --load-level heavy
    python run_performance_tests.py --generate-report
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from test_performance_regression import PerformanceRegressionDetector, PerformanceMetric

console = Console()


class PerformanceTestRunner:
    """Comprehensive performance testing runner."""
    
    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.regression_detector = PerformanceRegressionDetector(str(self.output_dir / "metrics"))
        self.results = []
        
        # Test categories
        self.test_categories = {
            "data": "tests/performance/test_data_processing_performance.py",
            "training": "tests/performance/test_model_training_performance.py",
            "risk": "tests/performance/test_risk_calculation_performance.py",
            "stress": "tests/performance/test_stress_testing.py",
            "load": "tests/performance/test_load_testing.py",
            "regression": "tests/performance/test_performance_regression.py"
        }
    
    def run_tests(self, test_type: str, benchmark: bool = False, 
                  load_level: Optional[str] = None, parallel: bool = False) -> Dict:
        """Run performance tests based on configuration."""
        console.print(f"[bold blue]Running {test_type} performance tests...[/bold blue]")
        
        start_time = time.time()
        
        # Prepare pytest arguments
        pytest_args = [
            "--tb=short",
            "--durations=10",
            "-v"
        ]
        
        if benchmark:
            pytest_args.extend(["--benchmark-only", "--benchmark-sort=mean"])
        
        if parallel:
            pytest_args.extend(["-n", "auto"])
        
        # Add test file
        if test_type == "all":
            test_files = list(self.test_categories.values())
        elif test_type in self.test_categories:
            test_files = [self.test_categories[test_type]]
        else:
            console.print(f"[red]Unknown test type: {test_type}[/red]")
            return {"error": f"Unknown test type: {test_type}"}
        
        # Add load level filter if specified
        if load_level:
            pytest_args.extend(["-k", f"test_{load_level}_load"])
        
        # Run tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running performance tests...", total=None)
            
            try:
                # Run pytest
                exit_code = pytest.main(pytest_args + test_files)
                
                if exit_code == 0:
                    progress.update(task, description="✅ Tests completed successfully")
                else:
                    progress.update(task, description="❌ Tests failed")
                
            except Exception as e:
                progress.update(task, description=f"❌ Error running tests: {e}")
                return {"error": str(e)}
        
        end_time = time.time()
        
        # Collect results
        results = {
            "test_type": test_type,
            "benchmark": benchmark,
            "load_level": load_level,
            "parallel": parallel,
            "duration": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
            "success": exit_code == 0
        }
        
        self.results.append(results)
        
        console.print(f"[green]✅ Performance tests completed in {results['duration']:.2f} seconds[/green]")
        
        return results
    
    def run_benchmark_suite(self) -> Dict:
        """Run comprehensive benchmark suite."""
        console.print("[bold blue]Running comprehensive benchmark suite...[/bold blue]")
        
        benchmark_results = {}
        
        # Run benchmarks for each category
        for category in self.test_categories.keys():
            console.print(f"[yellow]Benchmarking {category}...[/yellow]")
            
            result = self.run_tests(
                test_type=category,
                benchmark=True,
                parallel=False  # Benchmarks should run sequentially
            )
            
            benchmark_results[category] = result
            
            if "error" in result:
                console.print(f"[red]❌ {category} benchmarks failed: {result['error']}[/red]")
            else:
                console.print(f"[green]✅ {category} benchmarks completed[/green]")
        
        return benchmark_results
    
    def run_stress_suite(self, load_levels: List[str] = None) -> Dict:
        """Run comprehensive stress testing suite."""
        if load_levels is None:
            load_levels = ["light", "medium", "heavy", "stress"]
        
        console.print("[bold blue]Running comprehensive stress testing suite...[/bold blue]")
        
        stress_results = {}
        
        for level in load_levels:
            console.print(f"[yellow]Stress testing with {level} load...[/yellow]")
            
            result = self.run_tests(
                test_type="load",
                load_level=level,
                parallel=True
            )
            
            stress_results[level] = result
            
            if "error" in result:
                console.print(f"[red]❌ {level} stress tests failed: {result['error']}[/red]")
            else:
                console.print(f"[green]✅ {level} stress tests completed[/green]")
        
        return stress_results
    
    def run_load_suite(self) -> Dict:
        """Run comprehensive load testing suite."""
        console.print("[bold blue]Running comprehensive load testing suite...[/bold blue]")
        
        load_results = {}
        
        # Test different load scenarios
        load_scenarios = [
            ("light", "light_load"),
            ("medium", "medium_load"),
            ("heavy", "heavy_load"),
            ("stress", "stress_load"),
            ("scalability", "scalability_testing"),
            ("concurrent", "concurrent_user_simulation")
        ]
        
        for scenario_name, test_filter in load_scenarios:
            console.print(f"[yellow]Testing {scenario_name} load scenario...[/yellow]")
            
            result = self.run_tests(
                test_type="load",
                load_level=test_filter,
                parallel=True
            )
            
            load_results[scenario_name] = result
            
            if "error" in result:
                console.print(f"[red]❌ {scenario_name} load tests failed: {result['error']}[/red]")
            else:
                console.print(f"[green]✅ {scenario_name} load tests completed[/green]")
        
        return load_results
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        console.print("[bold blue]Generating performance report...[/bold blue]")
        
        # Get regression detector report
        regression_report = self.regression_detector.generate_performance_report()
        
        # Create comprehensive report
        report = {
            "generated_at": datetime.now().isoformat(),
            "test_results": self.results,
            "regression_analysis": regression_report,
            "summary": {
                "total_tests_run": len(self.results),
                "successful_tests": len([r for r in self.results if r.get("success", False)]),
                "failed_tests": len([r for r in self.results if not r.get("success", True)]),
                "total_duration": sum(r.get("duration", 0) for r in self.results),
                "recent_regressions": len(regression_report.get("critical_alerts", []))
            }
        }
        
        # Save report
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        console.print(f"[green]✅ Performance report generated: {report_file}[/green]")
        
        return report
    
    def _generate_html_report(self, report: Dict) -> None:
        """Generate HTML performance report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading RL Agent Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .warning {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading RL Agent Performance Report</h1>
        <p>Generated: {report['generated_at']}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Tests: {report['summary']['total_tests_run']}</div>
        <div class="metric success">Successful: {report['summary']['successful_tests']}</div>
        <div class="metric error">Failed: {report['summary']['failed_tests']}</div>
        <div class="metric">Total Duration: {report['summary']['total_duration']:.2f}s</div>
        <div class="metric warning">Recent Regressions: {report['summary']['recent_regressions']}</div>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test Type</th>
                <th>Benchmark</th>
                <th>Load Level</th>
                <th>Duration</th>
                <th>Status</th>
            </tr>
"""
        
        for result in report['test_results']:
            status_class = "success" if result.get("success", False) else "error"
            status_text = "✅ Success" if result.get("success", False) else "❌ Failed"
            
            html_content += f"""
            <tr>
                <td>{result.get('test_type', 'N/A')}</td>
                <td>{'Yes' if result.get('benchmark', False) else 'No'}</td>
                <td>{result.get('load_level', 'N/A')}</td>
                <td>{result.get('duration', 0):.2f}s</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Baselines</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Sample Count</th>
                <th>Mean Execution Time</th>
                <th>Mean Memory Usage</th>
                <th>Mean CPU Usage</th>
            </tr>
"""
        
        for test_name, baseline in report['regression_analysis'].get('baselines', {}).items():
            html_content += f"""
            <tr>
                <td>{test_name}</td>
                <td>{baseline.get('sample_count', 0)}</td>
                <td>{baseline.get('execution_time', {}).get('mean', 0):.2f}s</td>
                <td>{baseline.get('memory_peak_mb', {}).get('mean', 0):.2f} MB</td>
                <td>{baseline.get('cpu_usage_percent', {}).get('mean', 0):.2f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Recent Performance Trends</h2>
"""
        
        for test_name, trends in report['regression_analysis'].get('recent_trends', {}).items():
            html_content += f"""
        <h3>{test_name}</h3>
        <div class="metric">Execution Time: {trends.get('execution_time', {}).get('trend_direction', 'N/A')}</div>
        <div class="metric">Memory Usage: {trends.get('memory_peak_mb', {}).get('trend_direction', 'N/A')}</div>
        <div class="metric">CPU Usage: {trends.get('cpu_usage_percent', {}).get('trend_direction', 'N/A')}</div>
"""
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Critical Alerts</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Metric Type</th>
                <th>Description</th>
                <th>Timestamp</th>
            </tr>
"""
        
        for alert in report['regression_analysis'].get('critical_alerts', []):
            html_content += f"""
            <tr>
                <td>{alert.get('test_name', 'N/A')}</td>
                <td>{alert.get('metric_type', 'N/A')}</td>
                <td>{alert.get('description', 'N/A')}</td>
                <td>{alert.get('timestamp', 'N/A')}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        html_file = self.output_dir / "performance_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        console.print(f"[green]✅ HTML report generated: {html_file}[/green]")
    
    def display_results_table(self) -> None:
        """Display results in a formatted table."""
        table = Table(title="Performance Test Results")
        
        table.add_column("Test Type", style="cyan")
        table.add_column("Benchmark", style="magenta")
        table.add_column("Load Level", style="yellow")
        table.add_column("Duration", style="green")
        table.add_column("Status", style="bold")
        
        for result in self.results:
            status = "✅ Success" if result.get("success", False) else "❌ Failed"
            table.add_row(
                result.get("test_type", "N/A"),
                "Yes" if result.get("benchmark", False) else "No",
                result.get("load_level", "N/A"),
                f"{result.get('duration', 0):.2f}s",
                status
            )
        
        console.print(table)
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        # Clean up any temporary files created during testing
        temp_files = [
            "test_outputs",
            ".pytest_cache",
            "htmlcov",
            "coverage.xml"
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                import shutil
                shutil.rmtree(temp_file, ignore_errors=True)
        
        console.print("[yellow]🧹 Cleanup completed[/yellow]")


def main():
    """Main entry point for performance testing."""
    parser = argparse.ArgumentParser(description="Trading RL Agent Performance Testing")
    
    parser.add_argument(
        "--test-type",
        choices=["all", "data", "training", "risk", "stress", "load", "regression"],
        default="all",
        help="Type of performance tests to run"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests"
    )
    
    parser.add_argument(
        "--load-level",
        choices=["light", "medium", "heavy", "stress"],
        help="Load level for stress/load testing"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help="Run comprehensive benchmark suite"
    )
    
    parser.add_argument(
        "--stress-suite",
        action="store_true",
        help="Run comprehensive stress testing suite"
    )
    
    parser.add_argument(
        "--load-suite",
        action="store_true",
        help="Run comprehensive load testing suite"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate performance report"
    )
    
    parser.add_argument(
        "--output-dir",
        default="performance_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after testing"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PerformanceTestRunner(output_dir=args.output_dir)
    
    try:
        # Run tests based on arguments
        if args.benchmark_suite:
            console.print("[bold blue]Running benchmark suite...[/bold blue]")
            runner.run_benchmark_suite()
        
        elif args.stress_suite:
            console.print("[bold blue]Running stress suite...[/bold blue]")
            runner.run_stress_suite()
        
        elif args.load_suite:
            console.print("[bold blue]Running load suite...[/bold blue]")
            runner.run_load_suite()
        
        else:
            # Run specific test type
            runner.run_tests(
                test_type=args.test_type,
                benchmark=args.benchmark,
                load_level=args.load_level,
                parallel=args.parallel
            )
        
        # Generate report if requested
        if args.generate_report or args.benchmark_suite or args.stress_suite or args.load_suite:
            report = runner.generate_performance_report()
            
            # Display summary
            console.print("\n[bold green]Performance Testing Summary:[/bold green]")
            console.print(f"Total Tests: {report['summary']['total_tests_run']}")
            console.print(f"Successful: {report['summary']['successful_tests']}")
            console.print(f"Failed: {report['summary']['failed_tests']}")
            console.print(f"Total Duration: {report['summary']['total_duration']:.2f}s")
            console.print(f"Recent Regressions: {report['summary']['recent_regressions']}")
            
            # Display results table
            runner.display_results_table()
        
        # Cleanup if requested
        if args.cleanup:
            runner.cleanup()
        
        console.print("[bold green]🎉 Performance testing completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Performance testing interrupted by user[/yellow]")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"\n[red]❌ Performance testing failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()