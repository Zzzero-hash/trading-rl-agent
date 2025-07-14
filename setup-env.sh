#!/bin/sh
# Trading RL Agent - Environment Setup Script
# Fixes Python interpreter issues and sets up development environment

set -e

# Fix locale issues in dev container - use available locale
export LANG=C.utf8
export LC_ALL=C.utf8

echo "🚀 Setting up Trading RL Agent Development Environment"
echo "======================================================="

# Function to print status
print_status() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

# Check Python versions available
echo "\n🔍 Checking Python installations..."
python3.10 --version 2>/dev/null && echo "  - Python 3.10: Available" || echo "  - Python 3.10: Not found"
python3.12 --version 2>/dev/null && echo "  - Python 3.12: Available" || echo "  - Python 3.12: Not found"
/opt/conda/bin/python3.12 --version 2>/dev/null && echo "  - Conda Python 3.12: Available" || echo "  - Conda Python 3.12: Not found"

# Use the best available Python
if [ -f "/opt/conda/bin/python3.12" ]; then
    PYTHON_BIN="/opt/conda/bin/python3.12"
    PIP_BIN="/opt/conda/bin/pip"
    print_status "Using Conda Python 3.12"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
    PIP_BIN="pip3.12"
    print_status "Using system Python 3.12"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
    PIP_BIN="pip3.10"
    print_status "Using system Python 3.10"
else
    PYTHON_BIN="python3"
    PIP_BIN="pip3"
    print_status "Using default Python 3"
fi

echo "Selected Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Create shell aliases for consistent usage
echo "\n📝 Creating shell aliases..."
cat >> ~/.bashrc << 'EOF'

# Trading RL Agent aliases
alias pytrading="${PYTHON_BIN}"
alias piptrading="${PIP_BIN}"
alias pytest-trading="${PYTHON_BIN} -m pytest"
EOF

# Make aliases available in current session
alias pytrading="$PYTHON_BIN"
alias piptrading="$PIP_BIN"
alias pytest-trading="$PYTHON_BIN -m pytest"

print_status "Created aliases: pytrading, piptrading, pytest-trading"

# Function to install dependencies based on development phase
install_deps() {
    phase=$1
    case $phase in
        "core")
            echo "\n📦 Installing core dependencies..."
            $PIP_BIN install -r requirements-core.txt
            print_status "Core dependencies installed (~111MB)"
            ;;
        "ml")
            echo "\n📦 Installing ML dependencies..."
            $PIP_BIN install -r requirements-ml.txt
            print_status "ML dependencies installed (~2.1GB)"
            ;;
        "full")
            echo "\n📦 Installing full dependencies..."
            $PIP_BIN install -r requirements-full.txt
            print_status "Full dependencies installed (~2.6GB)"
            ;;
        *)
            echo "Usage: $0 [core|ml|full]"
            echo "  core: Fast installation for basic development"
            echo "  ml:   Includes PyTorch for neural network development"
            echo "  full: Production-ready with all features"
            exit 1
            ;;
    esac
}

# Install dependencies based on argument
if [ $# -eq 0 ]; then
    echo "\n🤔 Which dependencies would you like to install?"
    echo "1) core - Fast development (111MB)"
    echo "2) ml   - Neural networks (2.1GB)"
    echo "3) full - Production ready (2.6GB)"
    echo "4) skip - Just fix environment"
    printf "Choose [1-4]: "
    read choice
    case $choice in
        1) install_deps "core" ;;
        2) install_deps "ml" ;;
        3) install_deps "full" ;;
        4) echo "Skipping dependency installation" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    install_deps "$1"
fi

# Test the installation
echo "\n🧪 Testing installation..."
if $PYTHON_BIN minimal_test.py; then
    print_status "Environment setup successful!"
else
    print_error "Environment test failed"
    exit 1
fi

# Create convenient development scripts
echo "\n📝 Creating development scripts..."

# Fast test script
cat > test-fast.sh << EOF
#!/bin/sh
# Fast tests without heavy dependencies
echo "🏃 Running fast tests..."
$PYTHON_BIN -m pytest tests/test_data_pipeline.py tests/test_trading_env.py tests/test_features.py -v
EOF
chmod +x test-fast.sh

# ML test script
cat > test-ml.sh << EOF
#!/bin/sh
# ML tests requiring PyTorch
echo "🤖 Running ML tests..."
$PYTHON_BIN -m pytest tests/test_cnn_lstm.py tests/test_sac_agent.py tests/test_td3_agent.py -v
EOF
chmod +x test-ml.sh

# Full test script
cat > test-all.sh << EOF
#!/bin/sh
# All tests including Ray integration
echo "🎯 Running all tests..."
$PYTHON_BIN -m pytest tests/ -v --tb=short
EOF
chmod +x test-all.sh

print_status "Created test scripts: test-fast.sh, test-ml.sh, test-all.sh"

echo "\n🎉 Setup Complete!"
echo "================================"
echo "Quick start commands:"
echo "  pytrading minimal_test.py     # Test environment"
echo "  ./test-fast.sh                # Run fast tests"
echo "  pytrading src/agents/sac_agent.py  # Test SAC agent"
echo ""
echo "To activate aliases in current session:"
echo "  source ~/.bashrc"
echo ""
echo "Development phases:"
echo "  ./setup-env.sh core   # Fast development setup"
echo "  ./setup-env.sh ml     # Add neural networks"
echo "  ./setup-env.sh full   # Production ready"
