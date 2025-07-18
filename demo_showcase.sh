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

# Function to display file summary
display_file_summary() {
    local file_path="$1"
    local file_type="$2"

    if [ -f "$file_path" ]; then
        local size=$(du -h "$file_path" | cut -f1)
        local lines=$(wc -l < "$file_path" 2>/dev/null || echo "N/A")
        echo -e "  📄 ${CYAN}$(basename "$file_path")${NC} (${size}, ${lines} lines)"
    fi
}

# Function to display image summary
display_image_summary() {
    local image_path="$1"
    local description="$2"

    if [ -f "$image_path" ]; then
        local size=$(du -h "$image_path" | cut -f1)
        echo -e "  🖼️  ${CYAN}$(basename "$image_path")${NC} - $description (${size})"
    fi
}

# Function to display JSON summary
display_json_summary() {
    local json_file="$1"
    if [ -f "$json_file" ]; then
        echo -e "  📊 ${CYAN}$(basename "$json_file")${NC} - Summary data:"
        if command -v jq >/dev/null 2>&1; then
            # Pretty print JSON with jq if available
            jq -r 'to_entries[] | "    " + .key + ": " + (.value | tostring)' "$json_file" 2>/dev/null || echo "    (JSON content available)"
        else
            echo "    (JSON content available)"
        fi
    fi
}

# Function to run command with error handling
run_with_error_handling() {
    local cmd="$1"
    local step_name="$2"

    print_info "Running: $step_name"
    if eval "$cmd"; then
        print_success "$step_name completed successfully!"
    else
        print_warning "$step_name failed, continuing with demo..."
    fi
    echo ""
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
run_with_error_handling "python main.py version" "Version check"
run_with_error_handling "python main.py info" "System info"
echo ""

print_step "2. Data Pipeline Demo"
echo "-------------------------"
print_info "Downloading sample market data for AAPL, GOOGL, MSFT..."
run_with_error_handling "python main.py data download --symbols 'AAPL,GOOGL,MSFT' --start-date 2024-01-01 --end-date 2024-12-31 --output-dir demo_outputs/data/" "Data download"

print_info "Processing and building datasets..."
run_with_error_handling "python main.py data process --force-rebuild --output-dir demo_outputs/data/processed/" "Data processing"

# Display data outputs
echo -e "${PURPLE}📁 Data Pipeline Outputs:${NC}"
if [ -d "demo_outputs/data" ]; then
    find demo_outputs/data -type f -name "*.csv" | head -5 | while read file; do
        display_file_summary "$file" "Market data"
    done
fi
echo ""

print_step "3. Feature Engineering Demo"
echo "-------------------------------"
print_info "Running enhanced market patterns demo..."
run_with_error_handling "python examples/enhanced_market_patterns_demo.py" "Market patterns demo"

# Display generated visualizations
echo -e "${PURPLE}🖼️  Generated Pattern Visualizations:${NC}"
for pattern_file in *.png; do
    if [ -f "$pattern_file" ]; then
        case "$pattern_file" in
            "pattern_comparison.png")
                display_image_summary "$pattern_file" "Market pattern comparison analysis"
                ;;
            "arima_trends.png")
                display_image_summary "$pattern_file" "ARIMA trend forecasting"
                ;;
            "volatility_clustering.png")
                display_image_summary "$pattern_file" "Volatility clustering patterns"
                ;;
            "microstructure_effects.png")
                display_image_summary "$pattern_file" "Market microstructure analysis"
                ;;
            "correlated_assets.png")
                display_image_summary "$pattern_file" "Asset correlation matrix"
                ;;
            "regime_detection.png")
                display_image_summary "$pattern_file" "Market regime detection"
                ;;
            *)
                display_image_summary "$pattern_file" "Pattern analysis"
                ;;
        esac
    fi
done
echo ""

print_step "4. Model Training Demo"
echo "--------------------------"
print_info "Training CNN+LSTM model (small demo version)..."
run_with_error_handling "python main.py train cnn_lstm --epochs 10 --batch-size 32 --learning-rate 0.001 --output-dir demo_outputs/models/cnn_lstm/ --config-file configs/development.yaml" "CNN+LSTM training"

print_info "Training RL agent (SAC) for demo..."
run_with_error_handling "python main.py train rl --agent-type sac --timesteps 10000 --output-dir demo_outputs/models/rl/ --config-file configs/development.yaml" "RL training"

# Display model outputs
echo -e "${PURPLE}🤖 Model Training Outputs:${NC}"
if [ -d "demo_outputs/models" ]; then
    find demo_outputs/models -type f \( -name "*.pth" -o -name "*.zip" -o -name "*.json" \) | head -5 | while read file; do
        display_file_summary "$file" "Trained model"
    done
fi
echo ""

print_step "5. Model Evaluation Demo"
echo "-----------------------------"
print_info "Evaluating trained models..."
run_with_error_handling "python main.py backtest evaluate --model-path demo_outputs/models/cnn_lstm/best_model.pth --data-path demo_outputs/data/processed/ --output-dir demo_outputs/evaluation/" "Model evaluation"

# Display evaluation outputs
echo -e "${PURPLE}📊 Model Evaluation Outputs:${NC}"
if [ -d "demo_outputs/evaluation" ]; then
    find demo_outputs/evaluation -type f \( -name "*.json" -o -name "*.csv" \) | head -3 | while read file; do
        display_file_summary "$file" "Evaluation results"
    done
fi
echo ""

print_step "6. Backtesting Demo"
echo "-----------------------"
print_info "Running backtesting with trained models..."
run_with_error_handling "python main.py backtest strategy --data-path demo_outputs/data/processed/AAPL_1d.csv --model-path demo_outputs/models/rl/sac_agent.zip --initial-capital 100000 --commission 0.001 --output-dir demo_outputs/backtest/" "Backtesting"

# Display backtest outputs
echo -e "${PURPLE}📈 Backtesting Outputs:${NC}"
if [ -d "demo_outputs/backtest" ]; then
    find demo_outputs/backtest -type f \( -name "*.json" -o -name "*.csv" \) | head -3 | while read file; do
        display_file_summary "$file" "Backtest results"
    done
fi
echo ""

print_step "7. Risk Management Demo"
echo "----------------------------"
print_info "Running scenario evaluation..."
run_with_error_handling "python examples/scenario_evaluation_example.py" "Risk analysis"

# Display risk management outputs
echo -e "${PURPLE}⚠️  Risk Management Outputs:${NC}"
if [ -d "outputs/scenario_evaluation" ]; then
    find outputs/scenario_evaluation -type f \( -name "*.png" -o -name "*.md" \) | head -3 | while read file; do
        if [[ "$file" == *.png ]]; then
            display_image_summary "$file" "Risk analysis visualization"
        else
            display_file_summary "$file" "Risk analysis report"
        fi
    done
fi
echo ""

print_step "8. Portfolio Management Demo"
echo "---------------------------------"
print_info "Running ensemble trading example..."
run_with_error_handling "python examples/ensemble_trading_example.py" "Ensemble trading"

# Display ensemble outputs
echo -e "${PURPLE}🎯 Ensemble Trading Outputs:${NC}"
if [ -d "outputs/ensemble" ]; then
    for file in outputs/ensemble/*; do
        if [ -f "$file" ]; then
            case "$(basename "$file")" in
                "summary.json")
                    display_json_summary "$file"
                    ;;
                "equity_curve.csv")
                    display_file_summary "$file" "Portfolio equity curve"
                    ;;
                "trades.csv")
                    display_file_summary "$file" "Trading history"
                    ;;
                *)
                    display_file_summary "$file" "Ensemble data"
                    ;;
            esac
        fi
    done
fi
echo ""

print_step "9. Configuration Management Demo"
echo "-------------------------------------"
print_info "Showing configuration examples..."
run_with_error_handling "python examples/config_example.py" "Configuration demo"
echo ""

print_step "10. Performance Monitoring Demo"
echo "-----------------------------------"
print_info "Running comprehensive test suite..."
run_with_error_handling "python -m pytest tests/unit/ -v --tb=short" "Unit testing"
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
echo -e "${GREEN}🎉 Demo showcase completed successfully!${NC}"
echo ""

echo -e "${PURPLE}📁 Demo Outputs Created:${NC}"
echo "  📁 demo_outputs/data/ - Downloaded and processed market data"
echo "  📁 demo_outputs/models/ - Trained CNN+LSTM and RL models"
echo "  📁 demo_outputs/backtest/ - Backtesting results"
echo "  📁 demo_outputs/evaluation/ - Model evaluation results"
echo "  📁 demo_outputs/logs/ - System logs"
echo "  📁 outputs/ensemble/ - Ensemble trading results"
echo "  📁 outputs/scenario_evaluation/ - Risk analysis reports"

echo ""
echo -e "${PURPLE}🖼️  Generated Visualizations:${NC}"
total_images=0
for pattern_file in *.png; do
    if [ -f "$pattern_file" ]; then
        total_images=$((total_images + 1))
        size=$(du -h "$pattern_file" | cut -f1)
        echo -e "  🖼️  ${CYAN}$(basename "$pattern_file")${NC} (${size})"
    fi
done
echo -e "  📊 Total: ${GREEN}${total_images} visualization files${NC} generated"

echo ""
echo -e "${PURPLE}📊 Key Performance Metrics:${NC}"
if [ -f "outputs/ensemble/summary.json" ]; then
    if command -v jq >/dev/null 2>&1; then
        echo -e "  💰 Total Return: ${GREEN}$(jq -r '.total_return * 100 | "\(. | round * 0.01)%"' outputs/ensemble/summary.json 2>/dev/null || echo "N/A")${NC}"
        echo -e "  📈 Sharpe Ratio: ${GREEN}$(jq -r '.sharpe_ratio | round * 0.01' outputs/ensemble/summary.json 2>/dev/null || echo "N/A")${NC}"
        echo -e "  📉 Max Drawdown: ${RED}$(jq -r '.max_drawdown * 100 | "\(. | round * 0.01)%"' outputs/ensemble/summary.json 2>/dev/null || echo "N/A")${NC}"
        echo -e "  🔄 Number of Trades: ${CYAN}$(jq -r '.num_trades' outputs/ensemble/summary.json 2>/dev/null || echo "N/A")${NC}"
    else
        echo "  📊 Performance metrics available in outputs/ensemble/summary.json"
    fi
fi

echo ""
echo -e "${PURPLE}🚀 Next Steps:${NC}"
echo "  1. Review the generated outputs in demo_outputs/"
echo "  2. Examine the trained models and backtesting results"
echo "  3. Customize configurations for your specific use case"
echo "  4. Set up live trading with paper trading first"
echo ""
echo -e "${PURPLE}📖 Documentation:${NC}"
echo "  📖 README.md - Main documentation"
echo "  📖 README_CLI_USAGE.md - Detailed CLI usage"
echo "  📖 CONTRIBUTING.md - Development guidelines"
echo "  📁 examples/ - Additional example scripts"
