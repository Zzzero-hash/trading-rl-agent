#!/bin/bash
# Dashboard Installation Script
# Installs and runs the real-time P&L and performance dashboard

set -e

echo "📊 Trading RL Agent - Dashboard Installation"
echo "============================================="

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

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_BIN="python3"
PIP_BIN="pip3"

# Check if pip is available
if ! command -v "$PIP_BIN" &> /dev/null; then
    print_warning "Pip not found, trying python3 -m pip"
    PIP_BIN="$PYTHON_BIN -m pip"
fi

print_status "Using Python: $($PYTHON_BIN --version)"
print_status "Using Pip: $($PIP_BIN --version)"

# Install dashboard dependencies
echo -e "\n📦 Installing dashboard dependencies..."

if [ -f "requirements-dashboard.txt" ]; then
    $PIP_BIN install -r requirements-dashboard.txt
    print_status "Dashboard dependencies installed"
else
    print_warning "requirements-dashboard.txt not found, installing from main requirements"
    $PIP_BIN install streamlit plotly websockets
    print_status "Core dashboard packages installed"
fi

# Test the installation
echo -e "\n🧪 Testing dashboard installation..."

# Test imports
if $PYTHON_BIN -c "
import streamlit
import plotly
import websockets
print('✅ All dashboard dependencies imported successfully')
"; then
    print_status "Dashboard dependencies test passed"
else
    print_error "Dashboard dependencies test failed"
    exit 1
fi

# Test dashboard components
if $PYTHON_BIN -c "
import sys
sys.path.insert(0, 'src')
try:
    from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard
    from trading_rl_agent.monitoring.streaming_dashboard import StreamingDashboard
    print('✅ Dashboard components imported successfully')
except ImportError as e:
    print(f'❌ Dashboard component import failed: {e}')
    sys.exit(1)
"; then
    print_status "Dashboard components test passed"
else
    print_error "Dashboard components test failed"
    exit 1
fi

# Create dashboard launcher script
echo -e "\n📝 Creating dashboard launcher..."

cat > run-dashboard.sh << 'EOF'
#!/bin/bash
# Dashboard Launcher Script

echo "🚀 Starting Trading Performance Dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Run the dashboard
python3 -m trading_rl_agent.cli_dashboard run "$@"
EOF

chmod +x run-dashboard.sh

# Create streaming dashboard launcher
cat > run-streaming-dashboard.sh << 'EOF'
#!/bin/bash
# Streaming Dashboard Launcher Script

echo "📡 Starting Streaming Dashboard..."
echo "Streaming server will be available at: ws://localhost:8765"
echo "Press Ctrl+C to stop the streaming server"
echo ""

# Run the streaming dashboard
python3 -m trading_rl_agent.cli_dashboard stream "$@"
EOF

chmod +x run-streaming-dashboard.sh

# Create example runner
cat > run-dashboard-example.sh << 'EOF'
#!/bin/bash
# Dashboard Example Launcher Script

echo "📊 Running Dashboard Example..."
echo "This will start a dashboard with sample data"
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the example"
echo ""

# Run the example
python3 examples/dashboard_example.py basic
EOF

chmod +x run-dashboard-example.sh

print_status "Created launcher scripts:"
print_status "  - run-dashboard.sh (basic dashboard)"
print_status "  - run-streaming-dashboard.sh (streaming server)"
print_status "  - run-dashboard-example.sh (example with sample data)"

echo -e "\n🎉 Dashboard Installation Complete!"
echo "====================================="
echo ""
echo "Quick start commands:"
echo "  ./run-dashboard.sh                    # Start basic dashboard"
echo "  ./run-dashboard.sh --streaming        # Start with streaming"
echo "  ./run-dashboard-example.sh            # Run with sample data"
echo "  ./run-streaming-dashboard.sh          # Start streaming server only"
echo ""
echo "CLI commands:"
echo "  python3 -m trading_rl_agent.cli_dashboard run"
echo "  python3 -m trading_rl_agent.cli_dashboard run --streaming"
echo "  python3 -m trading_rl_agent.cli_dashboard stream"
echo "  python3 -m trading_rl_agent.cli_dashboard status"
echo ""
echo "Configuration:"
echo "  python3 -m trading_rl_agent.cli_dashboard config --create my_config.json"
echo "  python3 -m trading_rl_agent.cli_dashboard run --config my_config.json"
echo ""
echo "Documentation:"
echo "  See docs/DASHBOARD_README.md for detailed usage"
echo ""
echo "Dashboard URLs:"
echo "  Web Dashboard:    http://localhost:8501"
echo "  Streaming Server: ws://localhost:8765"