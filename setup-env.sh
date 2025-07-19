#!/bin/bash
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

print_warning() {
    echo "⚠️  $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python versions available
echo -e "\n🔍 Checking Python installations..."

# Try multiple Python versions in order of preference
PYTHON_VERSIONS=("python3.12" "python3.11" "python3.10" "python3.9" "python3")
CONDA_PATHS=("/opt/conda/bin/python3.12" "/opt/conda/bin/python3.11" "/opt/conda/bin/python3.10" "/opt/conda/bin/python3.9" "/opt/conda/bin/python3")

# Check system Python versions
for version in "${PYTHON_VERSIONS[@]}"; do
    if command_exists "$version"; then
        echo "  - $version: Available ($($version --version 2>&1 | head -1))"
    else
        echo "  - $version: Not found"
    fi
done

# Check conda Python versions
for conda_path in "${CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        echo "  - $conda_path: Available ($($conda_path --version 2>&1 | head -1))"
    fi
done

# Use the best available Python
PYTHON_BIN=""
PIP_BIN=""

# First try conda Python
for conda_path in "${CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        PYTHON_BIN="$conda_path"
        PIP_BIN="$(dirname "$conda_path")/pip"
        print_status "Using Conda Python: $PYTHON_BIN"
        break
    fi
done

# If no conda Python, try system Python
if [ -z "$PYTHON_BIN" ]; then
    for version in "${PYTHON_VERSIONS[@]}"; do
        if command_exists "$version"; then
            PYTHON_BIN="$version"
            PIP_BIN="$(echo "$version" | sed 's/python/pip/')"
            print_status "Using system Python: $PYTHON_BIN"
            break
        fi
    done
fi

# Fallback to python3 if nothing else works
if [ -z "$PYTHON_BIN" ]; then
    if command_exists python3; then
        PYTHON_BIN="python3"
        PIP_BIN="pip3"
        print_status "Using fallback Python: $PYTHON_BIN"
    else
        print_error "No Python installation found. Please install Python 3.9 or higher."
        exit 1
    fi
fi

echo "Selected Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Verify pip is available
if ! command_exists "$PIP_BIN"; then
    print_warning "Pip not found at $PIP_BIN, trying to use pip from Python module"
    PIP_BIN="$PYTHON_BIN -m pip"
fi

# Create shell aliases for consistent usage
echo -e "\n📝 Creating shell aliases..."

# Create a temporary file for the aliases
cat > /tmp/trading_aliases << EOF
# Trading RL Agent aliases
alias pytrading="$PYTHON_BIN"
alias piptrading="$PIP_BIN"
alias pytest-trading="$PYTHON_BIN -m pytest"
EOF

# Append to bashrc if not already present
if ! grep -q "Trading RL Agent aliases" ~/.bashrc 2>/dev/null; then
    cat /tmp/trading_aliases >> ~/.bashrc
    print_status "Added aliases to ~/.bashrc"
else
    print_status "Aliases already present in ~/.bashrc"
fi

# Make aliases available in current session
alias pytrading="$PYTHON_BIN"
alias piptrading="$PIP_BIN"
alias pytest-trading="$PYTHON_BIN -m pytest"

print_status "Created aliases: pytrading, piptrading, pytest-trading"

# Function to install dependencies based on development phase
install_deps() {
    local phase=$1
    case $phase in
        "core")
            echo -e "\n📦 Installing core dependencies..."
            if [ -f "requirements-core.txt" ]; then
                $PIP_BIN install -r requirements-core.txt
                print_status "Core dependencies installed (~50MB)"
            else
                print_error "requirements-core.txt not found"
                return 1
            fi
            ;;
        "ml")
            echo -e "\n📦 Installing ML dependencies..."
            if [ -f "requirements-ml.txt" ]; then
                $PIP_BIN install -r requirements-ml.txt
                print_status "ML dependencies installed (~2.1GB)"
            else
                print_error "requirements-ml.txt not found"
                return 1
            fi
            ;;
        "full")
            echo -e "\n📦 Installing full dependencies..."
            if [ -f "requirements-full.txt" ]; then
                $PIP_BIN install -r requirements-full.txt
                print_status "Full dependencies installed (~2.6GB)"
            else
                print_error "requirements-full.txt not found"
                return 1
            fi
            ;;
        "dev")
            echo -e "\n📦 Installing development dependencies..."
            if [ -f "requirements-dev.txt" ]; then
                $PIP_BIN install -r requirements-dev.txt
                print_status "Development dependencies installed (~500MB)"
            else
                print_error "requirements-dev.txt not found"
                return 1
            fi
            ;;
        "production")
            echo -e "\n📦 Installing production dependencies..."
            if [ -f "requirements-production.txt" ]; then
                $PIP_BIN install -r requirements-production.txt
                print_status "Production dependencies installed (~2.6GB)"
            else
                print_error "requirements-production.txt not found"
                return 1
            fi
            ;;
        *)
            echo "Usage: $0 [core|ml|full|dev|production]"
            echo "  core:       Fast installation for basic development (~50MB)"
            echo "  ml:         Includes PyTorch for neural network development (~2.1GB)"
            echo "  full:       Production-ready with all features (~2.6GB)"
            echo "  dev:        Development tools and testing (~500MB)"
            echo "  production: Optimized for deployment (~2.6GB)"
            exit 1
            ;;
    esac
}

# Install dependencies based on argument
if [ $# -eq 0 ]; then
    echo -e "\n🤔 Which dependencies would you like to install?"
    echo "1) core       - Fast development (~50MB)"
    echo "2) ml         - Neural networks (~2.1GB)"
    echo "3) full       - Production ready (~2.6GB)"
    echo "4) dev        - Development tools (~500MB)"
    echo "5) production - Optimized deployment (~2.6GB)"
    echo "6) skip       - Just fix environment"
    printf "Choose [1-6]: "
    read -r choice
    case $choice in
        1) install_deps "core" ;;
        2) install_deps "ml" ;;
        3) install_deps "full" ;;
        4) install_deps "dev" ;;
        5) install_deps "production" ;;
        6) echo "Skipping dependency installation" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
else
    install_deps "$1"
fi

# Test the installation
echo -e "\n🧪 Testing installation..."
if [ -f "minimal_test.py" ]; then
    if $PYTHON_BIN minimal_test.py; then
        print_status "Environment setup successful!"
    else
        print_error "Environment test failed"
        exit 1
    fi
else
    print_warning "minimal_test.py not found, skipping test"
    # Run a basic Python test instead
    if $PYTHON_BIN -c "import sys; print('Python version:', sys.version); print('✅ Basic Python test passed')"; then
        print_status "Basic Python test passed"
    else
        print_error "Basic Python test failed"
        exit 1
    fi
fi

# Create convenient development scripts
echo -e "\n📝 Creating development scripts..."

# Fast test script
cat > test-fast.sh << 'EOF'
#!/bin/bash
# Fast tests without heavy dependencies
echo "🏃 Running fast tests..."
EOF
echo "$PYTHON_BIN -m pytest tests/test_data_pipeline.py tests/test_trading_env.py tests/test_features.py -v" >> test-fast.sh
chmod +x test-fast.sh

# ML test script
cat > test-ml.sh << 'EOF'
#!/bin/bash
# ML tests requiring PyTorch
echo "🤖 Running ML tests..."
EOF
echo "$PYTHON_BIN -m pytest tests/test_cnn_lstm.py tests/test_sac_agent.py tests/test_td3_agent.py -v" >> test-ml.sh
chmod +x test-ml.sh

# Full test script
cat > test-all.sh << 'EOF'
#!/bin/bash
# All tests including Ray integration
echo "🎯 Running all tests..."
EOF
echo "$PYTHON_BIN -m pytest tests/ -v --tb=short" >> test-all.sh
chmod +x test-all.sh

print_status "Created test scripts: test-fast.sh, test-ml.sh, test-all.sh"

echo -e "\n🎉 Setup Complete!"
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
