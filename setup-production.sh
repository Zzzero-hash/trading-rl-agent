#!/bin/bash

# Trading RL Agent - Production Setup Script
# Comprehensive setup for the restructured trading system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."

    if command_exists python3; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        major_version=$(echo $python_version | cut -d'.' -f1)
        minor_version=$(echo $python_version | cut -d'.' -f2)

        if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 9 ]; then
            print_success "Python $python_version is supported"
            return 0
        else
            print_error "Python 3.9+ required, found $python_version"
            return 1
        fi
    else
        print_error "Python3 not found. Please install Python 3.9 or higher."
        return 1
    fi
}

# Function to setup Python virtual environment
setup_virtual_env() {
    print_status "Setting up Python virtual environment..."

    # Remove existing venv if it exists
    if [ -d "venv" ]; then
        print_warning "Removing existing virtual environment..."
        rm -rf venv
    fi

    # Create new virtual environment
    python3 -m venv venv

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    print_success "Virtual environment created and activated"
}

# Function to install core dependencies
install_core_dependencies() {
    print_status "Installing core dependencies..."

    # Install production requirements
    if [ -f "requirements-production.txt" ]; then
        pip install -r requirements-production.txt
    else
        print_warning "requirements-production.txt not found, installing basic requirements..."
        pip install -r requirements.txt
    fi

    print_success "Core dependencies installed"
}

# Function to install optional dependencies based on setup type
install_optional_dependencies() {
    local setup_type=$1

    case $setup_type in
        "minimal")
            print_status "Minimal setup - skipping optional dependencies"
            ;;
        "development")
            print_status "Installing development dependencies..."
            pip install pytest pytest-cov black isort flake8 mypy pre-commit
            ;;
        "production")
            print_status "Installing production dependencies..."
            pip install gunicorn uvicorn prometheus-client
            ;;
        "full")
            print_status "Installing all dependencies (development + production)..."
            pip install pytest pytest-cov black isort flake8 mypy pre-commit
            pip install gunicorn uvicorn prometheus-client
            ;;
    esac
}

# Function to install system dependencies (Linux/Ubuntu)
install_system_dependencies() {
    print_status "Checking system dependencies..."

    if command_exists apt-get; then
        print_status "Installing system dependencies via apt-get..."

        # Update package list
        sudo apt-get update

        # Install essential build tools
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            curl \
            wget \
            unzip \
            pkg-config

        # Install TA-Lib dependencies
        sudo apt-get install -y \
            libatlas-base-dev \
            libopenblas-dev \
            liblapack-dev \
            gfortran

        # Install Redis (for caching)
        sudo apt-get install -y redis-server

        # Install PostgreSQL (for data storage)
        sudo apt-get install -y postgresql postgresql-contrib

        print_success "System dependencies installed"
    else
        print_warning "apt-get not available. Please install system dependencies manually."
    fi
}

# Function to install TA-Lib
install_talib() {
    print_status "Installing TA-Lib..."

    # Check if TA-Lib is already installed
    if pip show TA-Lib >/dev/null 2>&1; then
        print_success "TA-Lib already installed"
        return 0
    fi

    # Download and install TA-Lib
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd -

    # Install Python wrapper
    pip install TA-Lib

    print_success "TA-Lib installed successfully"
}

# Function to setup configuration files
setup_configurations() {
    print_status "Setting up configuration files..."

    # Create configs directory if it doesn't exist
    mkdir -p configs

    # Create default configuration if it doesn't exist
    if [ ! -f "configs/config.yaml" ]; then
        cat > configs/config.yaml << EOF
# Trading RL Agent - Default Configuration

environment: development
debug: true

data:
  data_sources:
    primary: yfinance
    backup: null
  data_path: data/
  cache_enabled: true
  cache_ttl: 3600
  feature_window: 50
  technical_indicators: true
  alternative_data: false
  real_time_enabled: false
  update_frequency: 60

model:
  cnn_filters: [32, 64, 128]
  cnn_kernel_size: 3
  lstm_units: 256
  lstm_layers: 2
  dropout_rate: 0.2
  batch_normalization: true
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  model_save_path: models/
  checkpoint_frequency: 10

agent:
  agent_type: sac
  sac_learning_rate: 0.0003
  sac_buffer_size: 1000000
  sac_tau: 0.005
  sac_gamma: 0.99
  sac_alpha: 0.2
  total_timesteps: 1000000
  eval_frequency: 10000
  save_frequency: 50000

risk:
  max_position_size: 0.1
  max_leverage: 1.0
  max_drawdown: 0.1
  var_confidence_level: 0.05
  var_time_horizon: 1
  kelly_fraction: 0.25
  risk_per_trade: 0.02

execution:
  broker: alpaca
  paper_trading: true
  order_timeout: 60
  max_slippage: 0.001
  commission_rate: 0.0
  execution_frequency: 5
  market_hours_only: true

monitoring:
  log_level: INFO
  log_file: logs/trading_system.log
  structured_logging: true
  metrics_enabled: true
  metrics_frequency: 300
  alerts_enabled: true
  email_alerts: false
  slack_alerts: false
  mlflow_enabled: false
  mlflow_tracking_uri: http://localhost:5000

use_gpu: false
max_workers: 4
memory_limit: 8GB
EOF
        print_success "Default configuration created"
    fi

    # Create environment-specific configs
    for env in development staging production; do
        if [ ! -f "configs/$env.yaml" ]; then
            cp configs/config.yaml "configs/$env.yaml"
            # Modify environment-specific settings
            sed -i "s/environment: development/environment: $env/" "configs/$env.yaml"
            if [ "$env" = "production" ]; then
                sed -i "s/debug: true/debug: false/" "configs/$env.yaml"
                sed -i "s/paper_trading: true/paper_trading: false/" "configs/$env.yaml"
            fi
        fi
    done

    print_success "Configuration files setup complete"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up project directories..."

    # Create essential directories
    mkdir -p data/{raw,processed,cache}
    mkdir -p logs
    mkdir -p models/{checkpoints,artifacts}
    mkdir -p outputs/{experiments,reports}
    mkdir -p configs/hydra

    # Create .gitkeep files to preserve empty directories
    for dir in data/raw data/processed data/cache logs models/checkpoints models/artifacts outputs/experiments outputs/reports; do
        touch "$dir/.gitkeep"
    done

    print_success "Project directories created"
}

# Function to setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."

    if command_exists pre-commit; then
        # Create .pre-commit-config.yaml if it doesn't exist
        if [ ! -f ".pre-commit-config.yaml" ]; then
            cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
EOF
        fi

        # Install pre-commit hooks
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not installed, skipping hook setup"
    fi
}

# Function to run tests
run_tests() {
    print_status "Running test suite..."

    if command_exists pytest; then
        # Run a quick test to verify installation
        python -c "
import sys
print('Python version:', sys.version)

try:
    import numpy
    print('✓ NumPy:', numpy.__version__)
except ImportError:
    print('✗ NumPy not found')

try:
    import pandas
    print('✓ Pandas:', pandas.__version__)
except ImportError:
    print('✗ Pandas not found')

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError:
    print('✗ PyTorch not found')

try:
    import stable_baselines3
    print('✓ Stable Baselines3:', stable_baselines3.__version__)
except ImportError:
    print('✗ Stable Baselines3 not found')

print('\\n✅ Basic installation verification complete')
"

        # Run actual tests if they exist
        if [ -d "tests" ]; then
            pytest tests/ -v --tb=short || true
        fi

        print_success "Tests completed"
    else
        print_warning "pytest not installed, skipping tests"
    fi
}

# Function to display usage information
show_usage() {
    echo "Trading RL Agent - Production Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  minimal      - Install only core dependencies"
    echo "  development  - Install core + development dependencies"
    echo "  production   - Install core + production dependencies"
    echo "  full         - Install all dependencies (default)"
    echo "  --help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full installation"
    echo "  $0 development        # Development setup"
    echo "  $0 production         # Production setup"
}

# Main setup function
main() {
    local setup_type="${1:-full}"

    # Show usage if requested
    if [ "$setup_type" = "--help" ] || [ "$setup_type" = "-h" ]; then
        show_usage
        exit 0
    fi

    print_status "🚀 Starting Trading RL Agent setup ($setup_type)..."
    echo ""

    # Validate setup type
    case $setup_type in
        "minimal"|"development"|"production"|"full")
            ;;
        *)
            print_error "Invalid setup type: $setup_type"
            show_usage
            exit 1
            ;;
    esac

    # Run setup steps
    check_python_version || exit 1

    # Install system dependencies (skip in minimal mode)
    if [ "$setup_type" != "minimal" ]; then
        install_system_dependencies
        install_talib
    fi

    setup_virtual_env
    setup_directories
    setup_configurations
    install_core_dependencies
    install_optional_dependencies "$setup_type"

    if [ "$setup_type" = "development" ] || [ "$setup_type" = "full" ]; then
        setup_pre_commit
    fi

    run_tests

    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Review configuration: configs/config.yaml"
    echo "  3. Run tests: pytest tests/"
    echo "  4. Start development: python -m trading_rl_agent.main"
    echo ""
    print_status "For more information, see README.md and docs/"
}

# Run main function with all arguments
main "$@"
