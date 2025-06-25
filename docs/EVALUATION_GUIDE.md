# Trading Agent Evaluation Guide

This guide explains how to evaluate trained trading agents using the comprehensive evaluation framework. The system supports evaluation of individual agents (SAC, TD3) and ensemble methods with detailed performance metrics.

## 🚀 Quick Evaluation

### Basic Usage

```bash
# Evaluate a trained SAC agent
python evaluate_agent.py \
    --data data/advanced_trading_dataset_*.csv \
    --checkpoint models/sac_agent_best.pth \
    --agent sac \
    --output results/sac_evaluation.json

# Evaluate ensemble performance
python evaluate_agent.py \
    --data data/advanced_trading_dataset_*.csv \
    --checkpoint models/ensemble_agent_best.pth \
    --agent ensemble \
    --output results/ensemble_evaluation.json
```

### Advanced Evaluation with Hybrid Features

```bash
# Evaluate with CNN+LSTM hybrid features enabled
python evaluate_agent.py \
    --data data/advanced_trading_dataset_*.csv \
    --checkpoint models/hybrid_sac_agent.pth \
    --agent sac \
    --output results/hybrid_evaluation.json \
    --use-hybrid-features \
    --window-size 50
```

## ⚙️ Configuration Options

### Command Line Arguments

- `--data` – CSV file(s) containing market data (supports wildcards for multiple files)
- `--checkpoint` – Path to saved agent parameters (.pth file)
- `--agent` – Agent type: `sac`, `td3`, or `ensemble`
- `--output` – Output file for metrics JSON (default: `results/evaluation.json`)
- `--window-size` – Observation window length (default: 50)
- `--use-hybrid-features` – Enable CNN+LSTM feature extraction
- `--episodes` – Number of evaluation episodes (default: 10)
- `--render` – Enable visualization during evaluation

### Configuration Files

```yaml
# config/evaluation_config.yaml
evaluation:
  episodes: 10
  max_steps: 1000
  render: false
  save_trajectories: true

metrics:
  calculate_all: true
  rolling_window: 252 # Trading days per year
  benchmark: "SPY" # Benchmark for comparison

output:
  format: "json"
  include_trajectories: false
  save_plots: true
```

## 📊 Performance Metrics

The evaluation system calculates comprehensive trading performance metrics:

### Core Metrics

**Risk-Adjusted Returns:**

- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
- **Sortino Ratio**: Downside risk-adjusted return (focuses on negative volatility)
- **Calmar Ratio**: Return to maximum drawdown ratio

**Profitability:**

- **Total Return**: Overall portfolio return percentage
- **Profit Factor**: Ratio of gross profit to gross loss
- **Win Rate**: Percentage of profitable trades

**Risk Management:**

- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Potential loss at 95% confidence level

### Sample Output

```json
{
  "agent_type": "sac",
  "evaluation_episodes": 10,
  "dataset_info": {
    "total_records": 1370000,
    "date_range": "2020-01-01 to 2025-06-15",
    "instruments": 19
  },
  "performance_metrics": {
    "sharpe_ratio": 1.45,
    "sortino_ratio": 2.18,
    "calmar_ratio": 1.67,
    "max_drawdown": 0.087,
    "total_return": 0.452,
    "volatility": 0.156,
    "profit_factor": 2.34,
    "win_rate": 0.634,
    "var_95": 0.023,
    "num_trades": 1247,
    "avg_trade_duration": 3.2
  },
  "hybrid_features": {
    "enabled": true,
    "cnn_lstm_contribution": 0.73,
    "feature_importance_score": 0.89
  },
  "benchmark_comparison": {
    "benchmark": "SPY",
    "outperformance": 0.127,
    "correlation": 0.34,
    "beta": 0.78
  }
}
```

## 🔍 Advanced Analysis

### Comparative Evaluation

```bash
# Compare multiple agents
python scripts/compare_agents.py \
    --agents sac td3 ensemble \
    --data data/advanced_trading_dataset_*.csv \
    --output results/agent_comparison.json
```

### Rolling Performance Analysis

```python
from src.evaluation.rolling_metrics import RollingMetricsCalculator

# Analyze performance over time
calculator = RollingMetricsCalculator(window=252)  # 1 year rolling
rolling_metrics = calculator.calculate(
    returns=evaluation_results['returns'],
    benchmark_returns=benchmark_data['returns']
)
```

### Risk Attribution Analysis

```python
from src.evaluation.risk_attribution import RiskAttributor

# Decompose risk sources
attributor = RiskAttributor()
risk_breakdown = attributor.analyze(
    portfolio_returns=results['returns'],
    factor_exposures=results['factor_exposures']
)
```

## 📈 Visualization

### Performance Plots

```python
from src.utils.plotting import create_performance_dashboard

# Generate comprehensive performance dashboard
dashboard = create_performance_dashboard(
    evaluation_results=results,
    save_path="results/performance_dashboard.html"
)
```

### Equity Curve Analysis

```bash
# Generate interactive equity curve
python scripts/plot_equity_curve.py \
    --results results/evaluation.json \
    --benchmark SPY \
    --output results/equity_curve.html
```

## 🎯 Best Practices

### Evaluation Methodology

1. **Out-of-Sample Testing**: Always evaluate on unseen data
2. **Multiple Episodes**: Run 10+ episodes for statistical significance
3. **Benchmark Comparison**: Compare against market indices (SPY, QQQ)
4. **Risk Adjustment**: Focus on risk-adjusted metrics, not just returns
5. **Temporal Stability**: Test across different market regimes

### Metric Interpretation

**Excellent Performance (Production Ready):**

- Sharpe Ratio > 1.5
- Maximum Drawdown < 10%
- Win Rate > 60%
- Calmar Ratio > 1.0

**Good Performance:**

- Sharpe Ratio > 1.0
- Maximum Drawdown < 15%
- Win Rate > 55%
- Positive total returns

### Production Deployment Criteria

Before deploying to live trading:

1. **Consistent Performance**: 20+ evaluation runs with stable metrics
2. **Risk Management**: Maximum drawdown < 10% across all tests
3. **Market Adaptation**: Performance maintained across different market conditions
4. **Stress Testing**: Evaluation on extreme market events (2020 COVID crash, etc.)

## 🔧 Troubleshooting

### Common Issues

**Low Sharpe Ratio (<0.5)**:

- Increase training episodes
- Review hyperparameter optimization
- Check data quality and feature engineering

**High Maximum Drawdown (>20%)**:

- Implement stricter risk management
- Reduce position sizing
- Add stop-loss mechanisms

**Inconsistent Performance**:

- Increase model complexity
- Add ensemble methods
- Improve data preprocessing

### Performance Debugging

```python
# Debug evaluation step-by-step
from src.evaluation.debug_evaluator import DebugEvaluator

debugger = DebugEvaluator()
debug_results = debugger.evaluate_with_diagnostics(
    agent=agent,
    env=env,
    episodes=1,
    detailed_logging=True
)
```

## Notebook Example

See [`evaluation_example.ipynb`](../evaluation_example.ipynb) for a brief
walkthrough of running the evaluation script inside a Jupyter notebook and
visualizing the resulting metrics.
