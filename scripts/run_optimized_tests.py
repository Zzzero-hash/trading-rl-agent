#!/usr/bin/env python3
"""
Trading RL Agent Optimized Test Execution Script

This script provides optimized test execution with parallel processing,
test grouping, performance monitoring, and intelligent test selection.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing

import psutil


class OptimizedTestRunner:
    """Optimized test execution with parallel processing and intelligent grouping."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.tests_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Get system information
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def run_fast_tests(self, parallel: bool = True) -> Dict:
        """Run fast unit tests with optional parallel execution."""
        print("⚡ Running fast tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/unit/",
            "-m", "fast and not slow and not very_slow",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
            "--durations=10",
            "--maxfail=5"
        ]
        
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadfile"])
        
        start_time = time.time()
        result = self._run_command(cmd, "fast_tests")
        end_time = time.time()
        
        return {
            "test_type": "fast",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", 0)
        }
    
    def run_integration_tests(self, parallel: bool = False) -> Dict:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "--durations=15",
            "--maxfail=3"
        ]
        
        if parallel:
            cmd.extend(["-n", "2"])  # Limited parallel for integration tests
        
        start_time = time.time()
        result = self._run_command(cmd, "integration_tests")
        end_time = time.time()
        
        return {
            "test_type": "integration",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", 0)
        }
    
    def run_performance_tests(self) -> Dict:
        """Run performance tests."""
        print("📊 Running performance tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/performance/",
            "-m", "performance",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-skip",
            "--durations=20",
            "--maxfail=2"
        ]
        
        start_time = time.time()
        result = self._run_command(cmd, "performance_tests")
        end_time = time.time()
        
        return {
            "test_type": "performance",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "benchmarks": result.get("benchmarks", [])
        }
    
    def run_smoke_tests(self) -> Dict:
        """Run smoke tests for CI/CD."""
        print("💨 Running smoke tests...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/smoke/",
            "-m", "smoke",
            "--tb=short",
            "--durations=5",
            "--maxfail=1"
        ]
        
        start_time = time.time()
        result = self._run_command(cmd, "smoke_tests")
        end_time = time.time()
        
        return {
            "test_type": "smoke",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0)
        }
    
    def run_full_suite(self, parallel: bool = True, coverage: bool = True) -> Dict:
        """Run the complete test suite with optimizations."""
        print("🚀 Running full test suite...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/",
            "-m", "not very_slow",
            "--tb=short",
            "--durations=20",
            "--maxfail=10",
            "--junitxml=test-results.xml"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-report=json:coverage.json",
                "--cov-fail-under=85"
            ])
        
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadfile"])
        
        start_time = time.time()
        result = self._run_command(cmd, "full_suite")
        end_time = time.time()
        
        return {
            "test_type": "full_suite",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", 0)
        }
    
    def run_coverage_analysis(self) -> Dict:
        """Run comprehensive coverage analysis."""
        print("📈 Running coverage analysis...")
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=0",
            "-q"
        ]
        
        start_time = time.time()
        result = self._run_command(cmd, "coverage_analysis")
        end_time = time.time()
        
        # Parse coverage data
        coverage_data = self._parse_coverage_data()
        
        return {
            "test_type": "coverage_analysis",
            "duration": end_time - start_time,
            "success": result["success"],
            "coverage_data": coverage_data
        }
    
    def run_parallel_optimized(self) -> Dict:
        """Run tests with optimal parallel configuration."""
        print("⚡ Running parallel optimized tests...")
        
        # Determine optimal worker count
        optimal_workers = min(self.cpu_count, 8)  # Cap at 8 workers
        
        cmd = [
            "python3", "-m", "pytest",
            "tests/",
            "-m", "not very_slow",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=85",
            "--durations=20",
            "--maxfail=10",
            "--junitxml=test-results.xml",
            "-n", str(optimal_workers),
            "--dist=loadfile"
        ]
        
        start_time = time.time()
        result = self._run_command(cmd, "parallel_optimized")
        end_time = time.time()
        
        return {
            "test_type": "parallel_optimized",
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", 0),
            "workers_used": optimal_workers
        }
    
    def run_selective_tests(self, modules: List[str], parallel: bool = True) -> Dict:
        """Run tests for specific modules."""
        print(f"🎯 Running selective tests for: {', '.join(modules)}")
        
        test_paths = []
        for module in modules:
            # Find test files for the module
            module_tests = list(self.tests_dir.rglob(f"*{module}*"))
            test_paths.extend([str(t.relative_to(self.project_root)) for t in module_tests])
        
        if not test_paths:
            return {
                "test_type": "selective",
                "success": False,
                "error": f"No tests found for modules: {modules}"
            }
        
        cmd = [
            "python3", "-m", "pytest"
        ] + test_paths + [
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
            "--durations=15",
            "--maxfail=5"
        ]
        
        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadfile"])
        
        start_time = time.time()
        result = self._run_command(cmd, "selective_tests")
        end_time = time.time()
        
        return {
            "test_type": "selective",
            "modules": modules,
            "duration": end_time - start_time,
            "success": result["success"],
            "tests_run": result.get("tests_run", 0),
            "tests_passed": result.get("tests_passed", 0),
            "tests_failed": result.get("tests_failed", 0),
            "coverage": result.get("coverage", 0)
        }
    
    def _run_command(self, cmd: List[str], test_name: str) -> Dict:
        """Run a command and capture results."""
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=3600  # 1 hour timeout
            )
            
            # Parse output for test results
            output = result.stdout
            error_output = result.stderr
            
            # Extract test counts
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            coverage = 0
            
            for line in output.split('\n'):
                if 'passed' in line and 'failed' in line:
                    # Parse pytest summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed':
                            tests_passed = int(parts[i-1])
                        elif part == 'failed':
                            tests_failed = int(parts[i-1])
                    tests_run = tests_passed + tests_failed
                
                if 'TOTAL' in line and '%' in line:
                    # Parse coverage line
                    try:
                        coverage_str = line.split()[-1].replace('%', '')
                        coverage = float(coverage_str)
                    except (ValueError, IndexError):
                        pass
            
            return {
                "success": result.returncode == 0,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "coverage": coverage,
                "output": output,
                "error_output": error_output,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "output": "",
                "error_output": "Timeout after 1 hour"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "error_output": str(e)
            }
    
    def _parse_coverage_data(self) -> Dict:
        """Parse coverage data from generated reports."""
        coverage_data = {
            "overall_coverage": 0,
            "module_coverage": {},
            "missing_lines": []
        }
        
        # Parse JSON coverage
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    data = json.load(f)
                    if "totals" in data:
                        coverage_data["overall_coverage"] = data["totals"]["percent_covered"]
                    
                    if "files" in data:
                        for file_path, file_data in data["files"].items():
                            if "summary" in file_data:
                                coverage_data["module_coverage"][file_path] = {
                                    "percent_covered": file_data["summary"]["percent_covered"],
                                    "num_statements": file_data["summary"]["num_statements"],
                                    "missing_lines": file_data.get("missing_lines", [])
                                }
            except Exception as e:
                coverage_data["error"] = f"Error parsing coverage: {e}"
        
        return coverage_data
    
    def generate_execution_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive execution report."""
        print("📋 Generating execution report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": self.cpu_count,
                "memory_gb": round(self.memory_gb, 2),
                "platform": sys.platform
            },
            "execution_summary": {
                "total_tests_run": sum(r.get("tests_run", 0) for r in results),
                "total_tests_passed": sum(r.get("tests_passed", 0) for r in results),
                "total_tests_failed": sum(r.get("tests_failed", 0) for r in results),
                "total_duration": sum(r.get("duration", 0) for r in results),
                "overall_success": all(r.get("success", False) for r in results)
            },
            "test_results": results,
            "performance_metrics": {
                "average_test_time": 0,
                "slowest_test_type": None,
                "fastest_test_type": None
            },
            "recommendations": []
        }
        
        # Calculate performance metrics
        if results:
            durations = [r.get("duration", 0) for r in results if r.get("duration", 0) > 0]
            if durations:
                report["performance_metrics"]["average_test_time"] = sum(durations) / len(durations)
                
                slowest = max(results, key=lambda x: x.get("duration", 0))
                fastest = min(results, key=lambda x: x.get("duration", 0))
                
                report["performance_metrics"]["slowest_test_type"] = slowest.get("test_type", "unknown")
                report["performance_metrics"]["fastest_test_type"] = fastest.get("test_type", "unknown")
        
        # Generate recommendations
        if report["execution_summary"]["total_tests_failed"] > 0:
            report["recommendations"].append("Fix failing tests to improve reliability")
        
        if report["performance_metrics"]["average_test_time"] > 300:  # 5 minutes
            report["recommendations"].append("Consider optimizing slow tests")
        
        if not any("parallel" in r.get("test_type", "") for r in results):
            report["recommendations"].append("Enable parallel execution for better performance")
        
        # Save report
        with open(self.results_dir / "execution_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        print(f"✅ Execution report generated. Total tests: {report['execution_summary']['total_tests_run']}")
        return report
    
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown execution report."""
        report_path = self.results_dir / "execution_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Trading RL Agent Test Execution Report\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            # Summary
            f.write("## 📊 Execution Summary\n\n")
            summary = report["execution_summary"]
            f.write(f"- **Total Tests**: {summary['total_tests_run']}\n")
            f.write(f"- **Passed**: {summary['total_tests_passed']}\n")
            f.write(f"- **Failed**: {summary['total_tests_failed']}\n")
            f.write(f"- **Success Rate**: {summary['total_tests_passed']/summary['total_tests_run']*100:.1f}%\n")
            f.write(f"- **Total Duration**: {summary['total_duration']:.2f}s\n")
            f.write(f"- **Overall Success**: {'✅' if summary['overall_success'] else '❌'}\n\n")
            
            # System Info
            f.write("## 💻 System Information\n\n")
            sys_info = report["system_info"]
            f.write(f"- **CPU Cores**: {sys_info['cpu_count']}\n")
            f.write(f"- **Memory**: {sys_info['memory_gb']} GB\n")
            f.write(f"- **Platform**: {sys_info['platform']}\n\n")
            
            # Test Results
            f.write("## 🧪 Test Results\n\n")
            for result in report["test_results"]:
                f.write(f"### {result['test_type'].replace('_', ' ').title()}\n")
                f.write(f"- **Duration**: {result.get('duration', 0):.2f}s\n")
                f.write(f"- **Tests Run**: {result.get('tests_run', 0)}\n")
                f.write(f"- **Passed**: {result.get('tests_passed', 0)}\n")
                f.write(f"- **Failed**: {result.get('tests_failed', 0)}\n")
                if result.get('coverage'):
                    f.write(f"- **Coverage**: {result['coverage']:.1f}%\n")
                f.write(f"- **Status**: {'✅' if result.get('success') else '❌'}\n\n")
            
            # Performance Metrics
            f.write("## ⚡ Performance Metrics\n\n")
            perf = report["performance_metrics"]
            f.write(f"- **Average Test Time**: {perf['average_test_time']:.2f}s\n")
            f.write(f"- **Slowest Test Type**: {perf['slowest_test_type']}\n")
            f.write(f"- **Fastest Test Type**: {perf['fastest_test_type']}\n\n")
            
            # Recommendations
            if report["recommendations"]:
                f.write("## 🚀 Recommendations\n\n")
                for rec in report["recommendations"]:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*For detailed results, see the JSON files in this directory.*\n")
        
        print(f"📄 Markdown report generated: {report_path}")


def main():
    """Main entry point for optimized test execution."""
    parser = argparse.ArgumentParser(description="Trading RL Agent Optimized Test Runner")
    parser.add_argument("--mode", choices=[
        "fast", "integration", "performance", "smoke", "full", "coverage", "parallel", "selective"
    ], default="fast", help="Test execution mode")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--coverage", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--modules", nargs="+", help="Specific modules to test (for selective mode)")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    runner = OptimizedTestRunner(args.project_root)
    results = []
    
    if args.mode == "fast":
        results.append(runner.run_fast_tests(parallel=args.parallel))
    elif args.mode == "integration":
        results.append(runner.run_integration_tests(parallel=args.parallel))
    elif args.mode == "performance":
        results.append(runner.run_performance_tests())
    elif args.mode == "smoke":
        results.append(runner.run_smoke_tests())
    elif args.mode == "full":
        results.append(runner.run_full_suite(parallel=args.parallel, coverage=args.coverage))
    elif args.mode == "coverage":
        results.append(runner.run_coverage_analysis())
    elif args.mode == "parallel":
        results.append(runner.run_parallel_optimized())
    elif args.mode == "selective":
        if not args.modules:
            print("❌ Error: --modules required for selective mode")
            sys.exit(1)
        results.append(runner.run_selective_tests(args.modules, parallel=args.parallel))
    
    # Generate execution report
    runner.generate_execution_report(results)


if __name__ == "__main__":
    main()