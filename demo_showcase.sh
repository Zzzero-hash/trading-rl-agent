#!/bin/bash

# Trading RL Agent - Comprehensive Demo Showcase
# This script demonstrates the full capabilities of the trading system

set -e  # Exit on any error

echo "🚀 Trading RL Agent - Comprehensive Demo Showcase"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "Please run this script from the trading-rl-agent root directory"
    exit 1
fi

# Create demo directories
mkdir -p demo_outputs/{data,models,backtest,logs}

print_step "1. System Information and Health Check"
echo "-------------------------------------------"
python main.py version
python main.py info
echo ""

print_step "2. Data Pipeline Demo"
echo "-------------------------"
print_info "Downloading sample market data for AAPL, GOOGL, MSFT..."
python main.py data download \
    --symbols "AAPL,GOOGL,MSFT" \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --timeframe 1d \
    --output demo_outputs/data/

print_info "Processing and building datasets..."
python main.py data process \
    --symbols "AAPL,GOOGL,MSFT" \
    --force \
    --output demo_outputs/data/processed/

print_success "Data pipeline completed successfully!"
echo ""

print_step "3. Feature Engineering Demo"
echo "-------------------------------"
print_info "Running enhanced market patterns demo..."
python examples/enhanced_market_patterns_demo.py

print_info "Generated pattern visualizations saved to:"
ls -la *.png 2>/dev/null || print_warning "No pattern images generated"
echo ""

print_step "4. Model Training Demo"
echo "--------------------------"
print_info "Training CNN+LSTM model (small demo version)..."
python main.py train cnn-lstm \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output demo_outputs/models/cnn_lstm/ \
    --config configs/development.yaml

print_info "Training RL agent (SAC) for demo..."
python main.py train rl sac \
    --timesteps 10000 \
    --output demo_outputs/models/rl/ \
    --config configs/development.yaml

print_success "Model training completed!"
echo ""

print_step "5. Model Evaluation Demo"
echo "-----------------------------"
print_info "Evaluating trained models..."
python main.py evaluate \
    demo_outputs/models/cnn_lstm/best_model.pth \
    --data demo_outputs/data/processed/ \
    --output demo_outputs/evaluation/

print_success "Model evaluation completed!"
echo ""

print_step "6. Backtesting Demo"
echo "-----------------------"
print_info "Running backtesting with trained models..."
python main.py backtest strategy \
    --data-path demo_outputs/data/processed/AAPL_1d.csv \
    --model demo_outputs/models/rl/sac_agent.zip \
    --initial-capital 100000 \
    --commission 0.001 \
    --output demo_outputs/backtest/

print_success "Backtesting completed!"
echo ""

print_step "7. Risk Management Demo"
echo "----------------------------"
print_info "Running scenario evaluation..."
python examples/scenario_evaluation_example.py

print_success "Risk analysis completed!"
echo ""

print_step "8. Portfolio Management Demo"
echo "---------------------------------"
print_info "Running ensemble trading example..."
python examples/ensemble_trading_example.py

print_success "Portfolio management demo completed!"
echo ""

print_step "9. Configuration Management Demo"
echo "-------------------------------------"
print_info "Showing configuration examples..."
python examples/config_example.py

print_success "Configuration demo completed!"
echo ""

print_step "10. Performance Monitoring Demo"
echo "-----------------------------------"
print_info "Running comprehensive test suite..."
python -m pytest tests/unit/ -v --tb=short

print_success "Testing completed!"
echo ""

print_step "11. CLI Interface Demo"
echo "---------------------------"
echo "Available commands:"
python main.py --help

echo ""
echo "Data commands:"
python main.py data --help

echo ""
echo "Training commands:"
python main.py train --help

echo ""
echo "Backtesting commands:"
python main.py backtest --help

print_success "CLI interface demo completed!"
echo ""

print_step "12. Demo Summary"
echo "-------------------"
echo "Demo outputs created in:"
echo "  📁 demo_outputs/data/ - Downloaded and processed market data"
echo "  📁 demo_outputs/models/ - Trained CNN+LSTM and RL models"
echo "  📁 demo_outputs/backtest/ - Backtesting results"
echo "  📁 demo_outputs/evaluation/ - Model evaluation results"
echo "  📁 demo_outputs/logs/ - System logs"

echo ""
echo "Generated visualizations:"
ls -la *.png 2>/dev/null || echo "  No visualizations generated"

echo ""
print_success "🎉 Demo showcase completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review the generated outputs in demo_outputs/"
echo "  2. Examine the trained models and backtesting results"
echo "  3. Customize configurations for your specific use case"
echo "  4. Set up live trading with paper trading first"
echo ""
echo "For more information, see:"
echo "  📖 README.md - Main documentation"
echo "  📖 README_CLI_USAGE.md - Detailed CLI usage"
echo "  📖 CONTRIBUTING.md - Development guidelines"
echo "  📁 examples/ - Additional example scripts"