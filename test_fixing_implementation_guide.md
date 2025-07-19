# Test Fixing Implementation Guide

## Quick Start Workflow

### 1. Initial Assessment Script
```bash
#!/bin/bash
# initial_assessment.sh

echo "=== Test Assessment Report ==="
echo "Timestamp: $(date)"
echo ""

# Run tests and capture output
echo "Running all tests..."
pytest --verbose --tb=short --durations=10 > test_output.log 2>&1

# Parse results
TOTAL_TESTS=$(grep -c "test_" test_output.log || echo "0")
PASSED_TESTS=$(grep -c "PASSED" test_output.log || echo "0")
FAILED_TESTS=$(grep -c "FAILED" test_output.log || echo "0")
ERROR_TESTS=$(grep -c "ERROR" test_output.log || echo "0")

echo "Test Summary:"
echo "- Total tests found: $TOTAL_TESTS"
echo "- Passed: $PASSED_TESTS"
echo "- Failed: $FAILED_TESTS"
echo "- Errors: $ERROR_TESTS"
echo ""

# Show failure details
if [ $FAILED_TESTS -gt 0 ]; then
    echo "=== FAILED TESTS ==="
    grep -A 5 "FAILED" test_output.log
fi

if [ $ERROR_TESTS -gt 0 ]; then
    echo "=== ERROR TESTS ==="
    grep -A 5 "ERROR" test_output.log
fi

# Check for common issues
echo ""
echo "=== Common Issues Check ==="
echo "Import errors: $(grep -c "ImportError\|ModuleNotFoundError" test_output.log || echo "0")"
echo "Syntax errors: $(grep -c "SyntaxError" test_output.log || echo "0")"
echo "Assertion errors: $(grep -c "AssertionError" test_output.log || echo "0")"
echo "Database errors: $(grep -c "DatabaseError\|OperationalError" test_output.log || echo "0")"
```

### 2. Environment Setup Script
```bash
#!/bin/bash
# setup_test_env.sh

echo "Setting up test environment..."

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing main dependencies..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install pytest and common plugins
echo "Installing pytest and plugins..."
pip install pytest pytest-cov pytest-mock pytest-asyncio pytest-xdist

# Check for configuration files
echo "Checking configuration files..."
if [ ! -f "pytest.ini" ] && [ ! -f "pyproject.toml" ]; then
    echo "Creating basic pytest configuration..."
    cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
EOF
fi

# Set up test database
echo "Setting up test database..."
if [ -f "setup_test_db.py" ]; then
    python setup_test_db.py
fi

# Create test directories if they don't exist
mkdir -p tests/{unit,integration,api,ui,performance}

echo "Test environment setup complete!"
```

### 3. Progressive Test Fixing Script
```bash
#!/bin/bash
# fix_tests_progressive.sh

# Step 1: Fix syntax and import errors
echo "Step 1: Fixing syntax and import errors..."
python -m py_compile $(find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*")
if [ $? -eq 0 ]; then
    echo "✓ No syntax errors found"
else
    echo "✗ Syntax errors detected - fix them first"
    exit 1
fi

# Step 2: Run unit tests only
echo "Step 2: Running unit tests..."
pytest tests/unit/ -v --tb=short
UNIT_EXIT_CODE=$?

if [ $UNIT_EXIT_CODE -eq 0 ]; then
    echo "✓ All unit tests passing"
else
    echo "✗ Unit tests failing - fix them before proceeding"
    exit 1
fi

# Step 3: Run integration tests
echo "Step 3: Running integration tests..."
pytest tests/integration/ -v --tb=short
INTEGRATION_EXIT_CODE=$?

if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ All integration tests passing"
else
    echo "✗ Integration tests failing - fix them before proceeding"
    exit 1
fi

# Step 4: Run all tests
echo "Step 4: Running all tests..."
pytest --verbose --tb=short
ALL_EXIT_CODE=$?

if [ $ALL_EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passing!"
else
    echo "✗ Some tests still failing"
    exit 1
fi
```

## Python Helper Scripts

### Test Analysis Script
```python
#!/usr/bin/env python3
# test_analyzer.py

import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Any

class TestAnalyzer:
    def __init__(self):
        self.test_results = {}
        self.failure_patterns = {}
        
    def run_tests(self) -> Dict[str, Any]:
        """Run tests and capture detailed output"""
        try:
            result = subprocess.run(
                ["pytest", "--verbose", "--tb=long", "--json-report"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse JSON report if available
            if result.returncode == 0:
                try:
                    with open(".report.json", "r") as f:
                        report = json.load(f)
                    return self._analyze_json_report(report)
                except FileNotFoundError:
                    return self._analyze_text_output(result.stdout, result.stderr)
            else:
                return self._analyze_text_output(result.stdout, result.stderr)
                
        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out"}
    
    def _analyze_json_report(self, report: Dict) -> Dict[str, Any]:
        """Analyze pytest JSON report"""
        summary = report.get("summary", {})
        tests = report.get("tests", [])
        
        analysis = {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errors": summary.get("errors", 0),
            "skipped": summary.get("skipped", 0),
            "duration": summary.get("duration", 0),
            "failures": [],
            "errors": [],
            "slow_tests": []
        }
        
        for test in tests:
            if test.get("outcome") == "failed":
                analysis["failures"].append({
                    "name": test.get("nodeid"),
                    "message": test.get("call", {}).get("longrepr", ""),
                    "duration": test.get("call", {}).get("duration", 0)
                })
            elif test.get("outcome") == "error":
                analysis["errors"].append({
                    "name": test.get("nodeid"),
                    "message": test.get("call", {}).get("longrepr", ""),
                    "duration": test.get("call", {}).get("duration", 0)
                })
            
            # Identify slow tests (>1 second)
            duration = test.get("call", {}).get("duration", 0)
            if duration > 1.0:
                analysis["slow_tests"].append({
                    "name": test.get("nodeid"),
                    "duration": duration
                })
        
        return analysis
    
    def categorize_failures(self, analysis: Dict[str, Any]) -> Dict[str, List]:
        """Categorize failures by type"""
        categories = {
            "syntax_errors": [],
            "import_errors": [],
            "assertion_failures": [],
            "database_errors": [],
            "api_errors": [],
            "timeout_errors": [],
            "other": []
        }
        
        for failure in analysis.get("failures", []) + analysis.get("errors", []):
            message = failure["message"].lower()
            
            if "syntaxerror" in message:
                categories["syntax_errors"].append(failure)
            elif "importerror" in message or "modulenotfounderror" in message:
                categories["import_errors"].append(failure)
            elif "assertionerror" in message:
                categories["assertion_failures"].append(failure)
            elif "database" in message or "sql" in message:
                categories["database_errors"].append(failure)
            elif "timeout" in message or "timed out" in message:
                categories["timeout_errors"].append(failure)
            elif "api" in message or "http" in message:
                categories["api_errors"].append(failure)
            else:
                categories["other"].append(failure)
        
        return categories
    
    def generate_report(self) -> str:
        """Generate comprehensive test analysis report"""
        analysis = self.run_tests()
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        categories = self.categorize_failures(analysis)
        
        report = f"""
=== Test Analysis Report ===
Total Tests: {analysis['total']}
Passed: {analysis['passed']}
Failed: {analysis['failed']}
Errors: {analysis['errors']}
Skipped: {analysis['skipped']}
Duration: {analysis['duration']:.2f}s

=== Failure Categories ===
Syntax Errors: {len(categories['syntax_errors'])}
Import Errors: {len(categories['import_errors'])}
Assertion Failures: {len(categories['assertion_failures'])}
Database Errors: {len(categories['database_errors'])}
API Errors: {len(categories['api_errors'])}
Timeout Errors: {len(categories['timeout_errors'])}
Other: {len(categories['other'])}

=== Slow Tests (>1s) ===
"""
        
        for test in analysis.get("slow_tests", []):
            report += f"- {test['name']}: {test['duration']:.2f}s\n"
        
        return report

if __name__ == "__main__":
    analyzer = TestAnalyzer()
    print(analyzer.generate_report())
```

### Test Fixing Helper
```python
#!/usr/bin/env python3
# test_fixer.py

import subprocess
import re
import os
from typing import List, Dict, Any

class TestFixer:
    def __init__(self):
        self.fixes_applied = []
        
    def fix_import_errors(self, error_output: str) -> List[str]:
        """Attempt to fix common import errors"""
        fixes = []
        
        # Find missing modules
        missing_modules = re.findall(r"No module named '([^']+)'", error_output)
        for module in missing_modules:
            try:
                subprocess.run(["pip", "install", module], check=True)
                fixes.append(f"Installed missing module: {module}")
            except subprocess.CalledProcessError:
                fixes.append(f"Failed to install module: {module}")
        
        return fixes
    
    def fix_syntax_errors(self, error_output: str) -> List[str]:
        """Identify and suggest fixes for syntax errors"""
        fixes = []
        
        # Common syntax error patterns
        patterns = {
            r"SyntaxError: invalid syntax": "Check for missing colons, parentheses, or brackets",
            r"IndentationError": "Fix indentation - use consistent spaces or tabs",
            r"NameError: name '([^']+)' is not defined": "Define the variable or import the module",
            r"AttributeError: '([^']+)' object has no attribute '([^']+)'": "Check object type and available methods"
        }
        
        for pattern, suggestion in patterns.items():
            if re.search(pattern, error_output):
                fixes.append(suggestion)
        
        return fixes
    
    def fix_database_errors(self, error_output: str) -> List[str]:
        """Suggest fixes for database-related errors"""
        fixes = []
        
        if "database" in error_output.lower():
            fixes.extend([
                "Check database connection settings",
                "Ensure test database exists and is accessible",
                "Verify database user permissions",
                "Check for pending migrations"
            ])
        
        return fixes
    
    def run_test_with_fixes(self, test_path: str) -> Dict[str, Any]:
        """Run a specific test and attempt to fix issues"""
        result = {
            "test_path": test_path,
            "initial_status": "unknown",
            "fixes_applied": [],
            "final_status": "unknown"
        }
        
        # Run test initially
        try:
            output = subprocess.run(
                ["pytest", test_path, "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if output.returncode == 0:
                result["initial_status"] = "passed"
                result["final_status"] = "passed"
                return result
            else:
                result["initial_status"] = "failed"
                
                # Apply fixes based on error type
                error_output = output.stdout + output.stderr
                
                if "ImportError" in error_output or "ModuleNotFoundError" in error_output:
                    result["fixes_applied"].extend(self.fix_import_errors(error_output))
                
                if "SyntaxError" in error_output:
                    result["fixes_applied"].extend(self.fix_syntax_errors(error_output))
                
                if "database" in error_output.lower():
                    result["fixes_applied"].extend(self.fix_database_errors(error_output))
                
                # Run test again after fixes
                if result["fixes_applied"]:
                    try:
                        output2 = subprocess.run(
                            ["pytest", test_path, "-v"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        
                        if output2.returncode == 0:
                            result["final_status"] = "passed"
                        else:
                            result["final_status"] = "still_failing"
                    except subprocess.TimeoutExpired:
                        result["final_status"] = "timeout"
                
        except subprocess.TimeoutExpired:
            result["initial_status"] = "timeout"
            result["final_status"] = "timeout"
        
        return result

if __name__ == "__main__":
    fixer = TestFixer()
    
    # Example usage
    test_files = [
        "tests/test_example.py",
        "tests/integration/test_database.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = fixer.run_test_with_fixes(test_file)
            print(f"Test: {result['test_path']}")
            print(f"Status: {result['initial_status']} -> {result['final_status']}")
            if result['fixes_applied']:
                print("Fixes applied:")
                for fix in result['fixes_applied']:
                    print(f"  - {fix}")
            print()
```

## Pytest Configuration Examples

### Basic pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    api: API tests
    ui: UI tests
    performance: Performance tests
    security: Security tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Advanced pyproject.toml
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
    "--color=yes",
    "--durations=10",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
    "api: API tests",
    "ui: UI tests",
    "performance: Performance tests",
    "security: Security tests"
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning"
]
asyncio_mode = "auto"
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --verbose --tb=short --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### GitLab CI Pipeline
```yaml
stages:
  - test

test:
  stage: test
  image: python:3.11
  before_script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
  script:
    - pytest --verbose --tb=short --cov=src --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
```

## Usage Instructions

1. **Start with Assessment**: Run `./initial_assessment.sh` to understand current state
2. **Setup Environment**: Run `./setup_test_env.sh` to configure testing environment
3. **Progressive Fixing**: Use `./fix_tests_progressive.sh` for systematic fixing
4. **Analysis**: Use `python test_analyzer.py` for detailed failure analysis
5. **Automated Fixes**: Use `python test_fixer.py` for common issue resolution
6. **CI Integration**: Use the provided CI/CD configurations for automated testing

## Best Practices

1. **Fix in Order**: Always fix syntax/import errors first, then unit tests, then integration tests
2. **Isolate Issues**: Run individual test files to isolate problems
3. **Use Markers**: Tag tests appropriately for selective execution
4. **Monitor Performance**: Track test execution time and optimize slow tests
5. **Document Changes**: Keep track of fixes applied and configuration changes
6. **Automate**: Use CI/CD to prevent regressions