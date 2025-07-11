# Trading Agent Evaluation Guide

This guide explains how to evaluate trained trading agents using the built-in evaluation framework.

## 🚀 Basic Evaluation

The `Trainer` class includes an `evaluate` method that can be used to assess agent performance. While not yet implemented, the evaluation process will involve:

1.  **Loading a trained agent**: Load the agent's checkpoint from the `models/` directory.
2.  **Running on unseen data**: Evaluate the agent on a dataset it was not trained on.
3.  **Generating performance metrics**: Calculate key metrics to assess performance.

## ⚙️ Configuration

Evaluation will be configured through the `SystemConfig` in `src/trading_rl_agent/core/config.py`. Key options will include:

- **`episodes`**: The number of evaluation episodes to run.
- **`max_steps`**: The maximum number of steps per episode.
- **`benchmark`**: The benchmark to compare against (e.g., "SPY").

## 📊 Performance Metrics

The evaluation will produce a comprehensive set of trading performance metrics:

- **Sharpe Ratio**: Measures risk-adjusted return.
- **Sortino Ratio**: Focuses on downside risk-adjusted return.
- **Maximum Drawdown**: The largest peak-to-trough decline in portfolio value.
- **Total Return**: The overall percentage return of the portfolio.
- **Win Rate**: The percentage of profitable trades.
- **Value at Risk (VaR)**: The potential loss at a given confidence level.

## 🎯 Best Practices

- **Use Out-of-Sample Data**: Always evaluate agents on data they have not been trained on to get a true measure of performance.
- **Run Multiple Episodes**: Run at least 10 evaluation episodes to ensure statistical significance.
- **Compare to a Benchmark**: Always compare your agent's performance to a relevant market index like the S&P 500 (SPY).
- **Focus on Risk-Adjusted Returns**: Don't just look at total return; consider risk-adjusted metrics like the Sharpe and Sortino ratios.

## 🔧 Troubleshooting

- **Low Sharpe Ratio**: If the Sharpe ratio is low, consider increasing the number of training episodes, tuning hyperparameters, or improving feature engineering.
- **High Maximum Drawdown**: If the maximum drawdown is high, consider implementing stricter risk management rules or reducing position sizes.
- **Inconsistent Performance**: If performance is inconsistent across different evaluation runs, you may need to increase model complexity or explore ensemble methods.

---

For legal and safety notes see the [project disclaimer](disclaimer.md).
