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

- `unit`
- `integration`
- `performance`
- `smoke`
- `slow`
- `gpu`
- `network`
- `ray`
- `ml`
- `e2e`
- `regression`
- `memory`
- `security`

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

Expected coverage is **>92% overall** with **>95%** on critical modules. New features should include tests reaching **100%** coverage.

Utility scripts are available at the repository root:

```bash
./test-fast.sh    # core functionality
./test-ml.sh      # ML specific tests
./test-all.sh     # full suite
```
