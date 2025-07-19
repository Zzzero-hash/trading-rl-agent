# Quality Assurance Makefile for Trading RL Agent

.PHONY: help quality security mutation docs types lint format test clean install

# Default target
help:
	@echo "Quality Assurance Commands:"
	@echo "  quality     - Run all quality checks"
	@echo "  security    - Run security tests only"
	@echo "  mutation    - Run mutation testing only"
	@echo "  docs        - Run documentation tests only"
	@echo "  types       - Run type hints validation only"
	@echo "  lint        - Run linting and formatting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  test        - Run all tests"
	@echo "  install     - Install development dependencies"
	@echo "  clean       - Clean up generated files"

# Install development dependencies
install:
	pip install -r requirements-dev.txt

# Run all quality checks
quality:
	python scripts/run_quality_checks.py --verbose

# Run security tests only
security:
	python scripts/run_quality_checks.py --check security --verbose

# Run mutation testing only
mutation:
	python scripts/run_quality_checks.py --check mutation --verbose

# Run documentation tests only
docs:
	python scripts/run_quality_checks.py --check docs --verbose

# Run type hints validation only
types:
	python scripts/run_quality_checks.py --check types --verbose

# Run linting and formatting checks
lint:
	ruff check src/
	mypy src/
	black --check src/
	isort --check-only src/

# Format code
format:
	black src/
	isort src/
	ruff format src/

# Run all tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Clean up generated files
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf quality_report.json
	rm -rf mutmut-cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
