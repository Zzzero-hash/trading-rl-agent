# Testing Guide

This project includes an extensive test suite to ensure trading algorithms and utilities work as expected.

## Installing test dependencies

Use Python 3.9 or newer and install the base and testing requirements:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

`nltk>=3.8` is included in the base requirements and is required for the
sentiment analysis tests.

For machine-learning features (PyTorch and RLlib) install additional packages:

```bash
pip install -r requirements-ml.txt
```

The optional file `requirements-test-comprehensive.txt` provides extra plugins for the full coverage suite:

```bash
pip install -r requirements-test-comprehensive.txt
```

## Running tests with pytest

Tests live under the `tests/` directory. See [`tests/README.md`](tests/README.md) for directory layout and available markers.
Example commands:

```bash
# run all tests
pytest -v

# only unit tests
pytest -m unit

# integration tests
pytest -m integration

# performance benchmarks
pytest -m performance

# lightweight smoke tests
pytest -m smoke
```

## Running tests with `run_tests.py`

The `run_tests.py` helper script wraps pytest and handles Ray cluster
setup where required. Usage:

```bash
# unit tests
./run_tests.py unit

# integration tests
./run_tests.py integration

# fast smoke test
./run_tests.py smoke

# full suite with coverage
./run_tests.py coverage
```

The script prints progress and manages teardown automatically.

## Coverage targets

The project aims for **92% or higher** total coverage with critical modules above
**95%**. New features should include tests reaching **100%** coverage.
