# Trading RL Platform - Comprehensive Makefile
# Combines Quality Assurance with Production Deployment Commands

.PHONY: help quality security mutation docs types lint format test clean install property chaos load contract data-quality deploy run-demo

# Default target
help:
	@echo "🚀 Trading RL Platform - Available Commands:"
	@echo ""
	@echo "🎯 QUICK START:"
	@echo "  quickstart       - Complete quickstart setup"
	@echo "  demo             - Run complete demo"
	@echo "  run-demo         - Run CLI demo"
	@echo ""
	@echo "🔧 TEST ENVIRONMENT SETUP:"
	@echo "  setup-test-env   - Setup robust test environment"
	@echo "  validate-env     - Validate test environment"
	@echo "  setup-test-data  - Setup test data"
	@echo "  clean-test-data  - Clean test data"
	@echo ""
	@echo "🧪 TEST EXECUTION:"
	@echo "  test             - Run comprehensive test suite"
	@echo "  test-fast        - Run fast unit tests only"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-smoke       - Run smoke tests only"
	@echo "  test-performance - Run performance tests only"
	@echo "  test-coverage    - Monitor test coverage"
	@echo ""
	@echo "🔍 QUALITY CHECKS:"
	@echo "  quality          - Run all quality checks"
	@echo "  security         - Run security tests only"
	@echo "  mutation         - Run mutation testing only"
	@echo "  docs             - Run documentation tests only"
	@echo "  types            - Run type hints validation only"
	@echo "  lint             - Run linting and formatting checks"
	@echo "  format           - Format code with black and isort"
	@echo ""
	@echo "🧪 ADVANCED TESTING:"
	@echo "  property         - Run property-based tests"
	@echo "  chaos            - Run chaos engineering tests"
	@echo "  load             - Run load tests"
	@echo "  contract         - Run contract tests"
	@echo "  data-quality     - Run data quality tests"
	@echo ""
	@echo "🛠️  UTILITIES:"
	@echo "  install          - Install development dependencies"
	@echo "  clean            - Clean up generated files"

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

# =============================================================================
# ROBUST TEST EXECUTION
# =============================================================================

# Setup robust test environment
setup-test-env:
	@echo "🔧 Setting up robust test environment..."
	@chmod +x scripts/setup_test_environment.sh
	@./scripts/setup_test_environment.sh

# Validate test environment
validate-env:
	@echo "🔍 Validating test environment..."
	@python3 scripts/validate_test_environment.py

# Run all tests with robust execution
test:
	@echo "🎯 Running comprehensive test suite..."
	@python3 scripts/run_tests.py --parallel --timeout 600

# Run fast tests (unit tests only)
test-fast:
	@echo "🏃 Running fast tests..."
	@python3 scripts/run_tests.py --test-paths tests/unit --markers fast --no-coverage

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	@python3 scripts/run_tests.py --test-paths tests/unit

# Run integration tests only
test-integration:
	@echo "🔗 Running integration tests..."
	@python3 scripts/run_tests.py --test-paths tests/integration

# Run smoke tests only
test-smoke:
	@echo "💨 Running smoke tests..."
	@python3 scripts/run_tests.py --test-paths tests/smoke --no-coverage

# Run performance tests
test-performance:
	@echo "⚡ Running performance tests..."
	@python3 scripts/run_tests.py --test-paths tests/performance --markers performance

# Monitor test coverage
test-coverage:
	@echo "📊 Monitoring test coverage..."
	@python3 scripts/monitor_test_coverage.py

# Setup test data
setup-test-data:
	@echo "📁 Setting up test data..."
	@python3 scripts/manage_test_data.py --action create

# Clean test data
clean-test-data:
	@echo "🧹 Cleaning test data..."
	@python3 scripts/manage_test_data.py --action cleanup

# Run property-based tests
property:
	pytest tests/property/ -v -m property

# Run chaos engineering tests
chaos:
	pytest tests/chaos/ -v -m chaos

# Run load tests
load:
	locust -f tests/load/locustfile.py --headless -u 10 -r 1 --run-time 30s

# Run contract tests
contract:
	pytest tests/contract/ -v -m contract

# Run data quality tests
data-quality:
	pytest tests/data_quality/ -v -m data_quality

# Verify advanced testing setup
verify-setup:
	python scripts/test_advanced_setup.py

# Run all advanced tests
advanced-tests: verify-setup property chaos load contract data-quality

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
	rm -rf locust-report.html
	rm -rf locust-stats.csv
	rm -rf pact-results/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# PRODUCTION COMMANDS
# =============================================================================

# Quick Start Commands
quickstart: setup-test-env install
	@echo "🎉 Trading RL Platform - Quick Start Setup"
	@echo "Setting up environment..."
	mkdir -p data/raw data/processed models logs config
	cp env.example .env 2>/dev/null || echo "env.example not found, skipping"
	@echo "✅ Quick start setup complete!"
	@echo "   1. Edit .env file with your API keys"
	@echo "   2. Run 'make demo' for a complete demonstration"

demo: run-demo cli-data cli-train
	@echo "🎉 Complete demo finished!"

run-demo:
	@echo "🚀 Running Trading CLI Demo..."
	python trading_cli.py --help
	python trading_cli.py status
	python trading_cli.py config --init

cli-data:
	@echo "📊 Demo: Data Collection and Processing"
	python trading_cli.py data collect --source yahoo --symbols AAPL,GOOGL --period 6m
	python trading_cli.py data preprocess --features technical --clean
	python trading_cli.py data split --train-ratio 0.8 --validation-ratio 0.1

cli-train:
	@echo "🧠 Demo: Model Training and Evaluation"
	python trading_cli.py train --algorithm PPO --episodes 1000 --save-best
	@echo "Note: Training with 1000 episodes for demo purposes"

deploy-docker:
	@echo "🐳 Building and deploying with Docker..."
	docker build -t trading-rl:latest -f Dockerfile.production --target production .
	docker-compose up -d

deploy-production:
	@echo "🚀 Deploying to production..."
	docker-compose --profile production up -d
