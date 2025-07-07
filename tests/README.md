# Test Suite Overview

This directory contains all automated tests for the Trading RL Agent project. Tests are grouped by purpose and use standard pytest markers for selection.

## Directory layout

- `unit/` - Fast unit tests for individual components
- `integration/` - End-to-end tests of multiple components working together
- `performance/` - Benchmark and optimization tests
- `smoke/` - Lightweight smoke tests for CI
- `conftest.py` and `conftest_extra.py` - Shared fixtures

## Pytest markers

Key markers defined in `pytest.ini`:

- `e2e`
- `gpu`
- `integration`
- `memory`
- `ml`
- `network`
- `performance`
- `ray`
- `regression`
- `security`
- `slow`
- `smoke`
- `unit`

## Running tests

Example commands:

```bash
# run everything
pytest

# quick subsets
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m smoke

# generate coverage report
pytest --cov=src --cov-report=html
```

Expected coverage thresholds are defined in the centralized configuration file (`config.py`). Please refer to that file for the latest coverage expectations.

Utility scripts are available at the repository root:

```bash
./test-fast.sh    # core functionality
./test-ml.sh      # ML specific tests
./test-all.sh     # full suite
```
