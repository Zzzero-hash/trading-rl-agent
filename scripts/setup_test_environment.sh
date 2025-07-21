#!/bin/bash
"""
Comprehensive Test Environment Setup Script

This script sets up a robust test environment for the Trading RL Agent:

1. Validates system requirements
2. Sets up test directories and configurations
3. Initializes test data management
4. Runs environment validation
5. Performs initial test execution
6. Sets up coverage monitoring
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "Setting up robust test environment for Trading RL Agent"
log_info "Project root: $PROJECT_ROOT"

# Function to check Python version
check_python_version() {
    log_info "Checking Python version..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Found Python $PYTHON_VERSION"

        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            log_success "Python version is compatible (>= 3.10)"
        else
            log_error "Python version $PYTHON_VERSION is too old. Required: >= 3.10"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.10 or higher."
        exit 1
    fi
}

# Function to check required dependencies
check_dependencies() {
    log_info "Checking required dependencies..."

    local missing_deps=()

    # Check for required Python packages
    python3 -c "import pytest" 2>/dev/null || missing_deps+=("pytest")
    python3 -c "import numpy" 2>/dev/null || missing_deps+=("numpy")
    python3 -c "import pandas" 2>/dev/null || missing_deps+=("pandas")
    python3 -c "import hypothesis" 2>/dev/null || missing_deps+=("hypothesis")

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_warning "Missing dependencies: ${missing_deps[*]}"
        log_info "Installing missing dependencies..."
        pip install "${missing_deps[@]}"
    else
        log_success "All required dependencies are installed"
    fi
}

# Function to create test directories
create_test_directories() {
    log_info "Creating test directories..."

    local dirs=(
        "test_data"
        "test_data/synthetic"
        "test_data/fixtures"
        "test_data/temp"
        "test_data/cache"
        "test_results"
        "coverage_history"
        "test_logs"
        "test_models"
        "test_output"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        log_info "Created directory: $dir"
    done

    log_success "Test directories created"
}

# Function to setup environment variables
setup_environment_variables() {
    log_info "Setting up environment variables..."

    # Create .env file for test environment
    cat > "$PROJECT_ROOT/.env.test" << EOF
# Test Environment Configuration
TRADING_RL_AGENT_ENVIRONMENT=test
TRADING_RL_AGENT_DEBUG=false
RAY_DISABLE_IMPORT_WARNING=1
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
PYTHONPATH=$PROJECT_ROOT/src
EOF

    log_success "Environment variables configured"
}

# Function to validate test environment
validate_environment() {
    log_info "Validating test environment..."

    cd "$PROJECT_ROOT"

    if python3 scripts/validate_test_environment.py; then
        log_success "Environment validation passed"
    else
        log_error "Environment validation failed"
        exit 1
    fi
}

# Function to setup test data
setup_test_data() {
    log_info "Setting up test data..."

    cd "$PROJECT_ROOT"

    if python3 scripts/manage_test_data.py --action create; then
        log_success "Test data setup completed"
    else
        log_warning "Test data setup failed, continuing with existing data"
    fi
}

# Function to run initial test execution
run_initial_tests() {
    log_info "Running initial test execution..."

    cd "$PROJECT_ROOT"

    # Run a small subset of tests to validate setup
    if python3 scripts/run_tests.py --test-paths tests/smoke --no-coverage --timeout 60; then
        log_success "Initial test execution passed"
    else
        log_warning "Initial test execution failed, but setup continues"
    fi
}

# Function to setup coverage monitoring
setup_coverage_monitoring() {
    log_info "Setting up coverage monitoring..."

    cd "$PROJECT_ROOT"

    # Create initial coverage baseline
    if python3 scripts/monitor_test_coverage.py --report-only; then
        log_success "Coverage monitoring setup completed"
    else
        log_warning "Coverage monitoring setup failed, but setup continues"
    fi
}

# Function to create convenience scripts
create_convenience_scripts() {
    log_info "Creating convenience scripts..."

    cd "$PROJECT_ROOT"

    # Create test execution aliases
    cat > "test-fast.sh" << 'EOF'
#!/bin/bash
# Fast test execution (unit tests only)
echo "🏃 Running fast tests..."
python3 scripts/run_tests.py --test-paths tests/unit --markers fast --no-coverage
EOF

    cat > "test-full.sh" << 'EOF'
#!/bin/bash
# Full test execution with coverage
echo "🎯 Running full test suite..."
python3 scripts/run_tests.py --parallel --timeout 600
EOF

    cat > "test-coverage.sh" << 'EOF'
#!/bin/bash
# Coverage monitoring
echo "📊 Running coverage monitoring..."
python3 scripts/monitor_test_coverage.py
EOF

    cat > "validate-env.sh" << 'EOF'
#!/bin/bash
# Environment validation
echo "🔍 Validating test environment..."
python3 scripts/validate_test_environment.py
EOF

    # Make scripts executable
    chmod +x test-fast.sh test-full.sh test-coverage.sh validate-env.sh

    log_success "Convenience scripts created"
}

# Function to display setup summary
display_summary() {
    log_info "Test environment setup completed!"
    echo
    echo "=========================================="
    echo "TEST ENVIRONMENT SETUP SUMMARY"
    echo "=========================================="
    echo
    echo "Available commands:"
    echo "  ./test-fast.sh          - Run fast unit tests"
    echo "  ./test-full.sh          - Run full test suite with coverage"
    echo "  ./test-coverage.sh      - Monitor test coverage"
    echo "  ./validate-env.sh       - Validate test environment"
    echo
    echo "Python scripts:"
    echo "  python3 scripts/run_tests.py --help"
    echo "  python3 scripts/manage_test_data.py --help"
    echo "  python3 scripts/monitor_test_coverage.py --help"
    echo "  python3 scripts/validate_test_environment.py"
    echo
    echo "Direct pytest commands:"
    echo "  python3 -m pytest tests/unit -v"
    echo "  python3 -m pytest tests/integration -v"
    echo "  python3 -m pytest tests/smoke -v"
    echo
    echo "Coverage target: 95%+"
    echo "Test isolation: Enabled"
    echo "Data management: Automated"
    echo "Environment validation: Active"
    echo
    echo "=========================================="
}

# Main execution
main() {
    log_info "Starting comprehensive test environment setup..."

    check_python_version
    check_dependencies
    create_test_directories
    setup_environment_variables
    validate_environment
    setup_test_data
    run_initial_tests
    setup_coverage_monitoring
    create_convenience_scripts
    display_summary

    log_success "Test environment setup completed successfully!"
}

# Run main function
main "$@"
