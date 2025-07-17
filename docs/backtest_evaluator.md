# Unified BacktestEvaluator

The Unified BacktestEvaluator is a comprehensive backtesting framework that integrates the existing backtesting engine with the model evaluation pipeline. It provides realistic transaction cost modeling, slippage simulation, detailed trade analysis, and automated performance attribution reports.

## Features

### 🎯 **Core Capabilities**

- **Realistic Transaction Cost Modeling**: Configurable commission, slippage, market impact, and bid-ask spread models
- **Slippage and Market Impact Simulation**: Multiple models (linear, square root) based on trade size and market conditions
- **Detailed Trade-by-Trade Analysis**: Complete trade history with cost breakdown and performance tracking
- **Performance Attribution Reports**: Comprehensive analysis of strategy performance and risk metrics
- **Risk-Adjusted Performance Metrics**: Sharpe, Sortino, Calmar ratios with statistical significance testing
- **Automated Model Comparison**: Side-by-side strategy evaluation with statistical validation

### 📊 **Performance Metrics**

The BacktestEvaluator calculates a comprehensive set of performance metrics:

#### **Return Metrics**

- Total Return
- Annualized Return
- Compound Annual Growth Rate (CAGR)

#### **Risk Metrics**

- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Value at Risk (VaR)
- Expected Shortfall (CVaR)
- Volatility (Annualized)

#### **Trading Metrics**

- Win Rate
- Profit Factor
- Average Win/Loss
- Largest Win/Loss
- Maximum Consecutive Wins/Losses
- Number of Trades
- Average Trade Return

#### **Cost Analysis**

- Total Commission
- Total Slippage
- Total Market Impact
- Total Spread Cost
- Cost Drag on Returns

## Architecture

### **Core Components**

```python
BacktestEvaluator
├── TransactionCostModel      # Realistic cost modeling
├── PortfolioManager          # Position and risk management
├── MetricsCalculator         # Performance metrics calculation
├── StatisticalTests          # Statistical validation
└── TradeRecord              # Detailed trade tracking
```

### **Transaction Cost Models**

The framework supports multiple transaction cost models:

#### **Commission Models**

- Fixed rate commission
- Minimum/maximum commission limits
- Tiered commission structures

#### **Slippage Models**

- **Linear**: Constant slippage rate
- **Square Root**: Slippage proportional to √(trade_size)
- **Volume Based**: Slippage based on market volume

#### **Market Impact Models**

- **Linear**: Impact proportional to trade size
- **Square Root**: Impact proportional to √(volume_ratio)

#### **Bid-Ask Spread**

- Configurable spread costs
- Dynamic spread modeling

## Usage

### **Basic Usage**

```python
from trading_rl_agent.eval.backtest_evaluator import (
    BacktestEvaluator,
    TransactionCostModel,
)
from trading_rl_agent.core.unified_config import BacktestConfig

# Configure backtest
config = BacktestConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    symbols=["AAPL", "GOOGL", "MSFT"],
    initial_capital=100000.0,
    commission_rate=0.001,
    slippage_rate=0.0001,
)

# Create transaction cost model
cost_model = TransactionCostModel(
    commission_rate=0.001,
    slippage_rate=0.0001,
    market_impact_rate=0.00005,
    bid_ask_spread=0.0002,
)

# Initialize evaluator
evaluator = BacktestEvaluator(config, cost_model)

# Run backtest
results = evaluator.run_backtest(
    data=price_data,
    strategy_signals=signals,
    strategy_name="momentum_strategy"
)

# Generate report
report = evaluator.generate_performance_report(results)
```

### **Strategy Comparison**

```python
# Define multiple strategies
strategies = {
    "momentum": momentum_signals,
    "mean_reversion": mean_reversion_signals,
    "volatility": volatility_signals,
}

# Compare strategies
comparison_results = evaluator.compare_strategies(
    data=price_data,
    strategies=strategies
)

# Access individual results
momentum_results = comparison_results["momentum"]
mean_rev_results = comparison_results["mean_reversion"]
```

### **CLI Usage**

The BacktestEvaluator is integrated into the CLI:

```bash
# Run single strategy backtest
python -m trading_rl_agent.cli_backtest run momentum \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --symbols AAPL,GOOGL,MSFT \
    --initial-capital 100000 \
    --commission-rate 0.001 \
    --slippage-rate 0.0001

# Compare multiple strategies
python -m trading_rl_agent.cli_backtest compare \
    momentum,mean_reversion,volatility \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --symbols AAPL,GOOGL,MSFT

# Batch backtesting across multiple periods
python -m trading_rl_agent.cli_backtest batch \
    momentum,mean_reversion \
    2023-01-01:2023-06-30,2023-07-01:2023-12-31 \
    --symbols AAPL,GOOGL,MSFT
```

## Configuration

### **BacktestConfig**

```python
@dataclass
class BacktestConfig:
    # Test period
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

    # Instruments
    symbols: list[str] = ["AAPL", "GOOGL", "MSFT"]

    # Capital and position sizing
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001

    # Risk management
    max_position_size: float = 0.1
    max_leverage: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

    # Evaluation metrics
    metrics: list[str] = ["total_return", "sharpe_ratio", "max_drawdown"]

    # Output settings
    output_dir: str = "backtest_results"
    save_trades: bool = True
    save_portfolio: bool = True
    generate_plots: bool = True
```

### **TransactionCostModel**

```python
@dataclass
class TransactionCostModel:
    # Commission structure
    commission_rate: float = 0.001
    min_commission: float = 1.0
    max_commission: float = 1000.0

    # Slippage model
    slippage_rate: float = 0.0001
    slippage_model: str = "linear"  # linear, square_root

    # Market impact model
    market_impact_rate: float = 0.00005
    impact_model: str = "linear"  # linear, square_root

    # Bid-ask spread
    bid_ask_spread: float = 0.0002
```

## Output and Reports

### **BacktestResult**

The evaluator returns a comprehensive `BacktestResult` object:

```python
@dataclass
class BacktestResult:
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Risk metrics
    volatility: float
    var_95: float
    expected_shortfall: float

    # Trading statistics
    num_trades: int
    win_rate: float
    profit_factor: float

    # Cost analysis
    total_transaction_costs: float
    cost_drag: float

    # Portfolio evolution
    equity_curve: pd.Series
    returns_series: pd.Series
    drawdown_series: pd.Series

    # Trade history
    trades: List[TradeRecord]
```

### **Performance Reports**

The evaluator generates detailed performance reports including:

- **Executive Summary**: Key performance metrics
- **Risk Analysis**: Risk-adjusted returns and drawdown analysis
- **Trading Statistics**: Win rate, profit factor, trade analysis
- **Cost Analysis**: Transaction cost breakdown and impact
- **Portfolio Evolution**: Equity curve and drawdown charts
- **Trade History**: Detailed trade-by-trade analysis

## Integration with Model Evaluation

### **Model Performance Assessment**

The BacktestEvaluator integrates with the existing model evaluation pipeline:

```python
from trading_rl_agent.eval.model_evaluator import ModelEvaluator

# Evaluate model predictions
model_evaluator = ModelEvaluator()
model_results = model_evaluator.evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name="cnn_lstm_model"
)

# Convert predictions to trading signals
signals = convert_predictions_to_signals(model_results["predictions"])

# Run backtest with model signals
backtest_results = evaluator.run_backtest(
    data=test_data,
    strategy_signals=signals,
    strategy_name="cnn_lstm_strategy"
)
```

### **Statistical Validation**

The framework includes statistical validation:

```python
# Perform statistical tests
statistical_tests = evaluator.statistical_tests.test_model_residuals(
    residuals=model_results["residuals"],
    predictions=model_results["predictions"]
)

# Calculate confidence intervals
confidence_intervals = evaluator.metrics_calculator.calculate_bootstrap_confidence_intervals(
    metric_values=returns_series,
    confidence_level=0.95
)
```

## Examples

### **Complete Example**

See `examples/backtest_evaluator_example.py` for a complete example demonstrating:

- Loading historical data
- Creating multiple strategies
- Comparing different transaction cost models
- Generating detailed performance reports
- Analyzing transaction costs and their impact

### **Running the Example**

```bash
# Install dependencies
pip install yfinance pandas numpy

# Run the example
python examples/backtest_evaluator_example.py
```

## Testing

The BacktestEvaluator includes comprehensive tests:

```bash
# Run all backtest evaluator tests
pytest tests/test_backtest_evaluator.py -v

# Run specific test categories
pytest tests/test_backtest_evaluator.py::TestTransactionCostModel -v
pytest tests/test_backtest_evaluator.py::TestBacktestEvaluator -v
```

## Best Practices

### **Transaction Cost Modeling**

1. **Use Realistic Costs**: Configure commission and slippage rates based on your broker and market conditions
2. **Consider Market Impact**: Include market impact for large trades
3. **Test Different Scenarios**: Compare performance with different cost models

### **Strategy Evaluation**

1. **Multiple Time Periods**: Test strategies across different market conditions
2. **Statistical Significance**: Use confidence intervals and hypothesis testing
3. **Risk-Adjusted Metrics**: Focus on Sharpe and Sortino ratios rather than just returns

### **Performance Analysis**

1. **Cost Impact**: Always analyze the impact of transaction costs on returns
2. **Drawdown Analysis**: Pay attention to maximum drawdown and recovery periods
3. **Trade Analysis**: Review individual trades for strategy insights

## Future Enhancements

### **Planned Features**

- **Multi-Asset Support**: Portfolio-level backtesting with correlation analysis
- **Advanced Risk Models**: Monte Carlo VaR and stress testing
- **Real-Time Integration**: Live trading integration with real-time data
- **Machine Learning Integration**: Automated strategy optimization
- **Visualization**: Interactive charts and dashboards

### **Extensibility**

The BacktestEvaluator is designed to be extensible:

- Custom transaction cost models
- New performance metrics
- Alternative data sources
- Custom risk management rules

## Troubleshooting

### **Common Issues**

1. **Import Errors**: Ensure all dependencies are installed
2. **Data Format**: Verify data has required OHLCV columns
3. **Memory Issues**: Use smaller date ranges for large datasets
4. **Performance**: Consider using parallel processing for multiple strategies

### **Getting Help**

- Check the test files for usage examples
- Review the example script for complete workflows
- Consult the main documentation for configuration details
