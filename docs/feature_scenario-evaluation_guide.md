# Agent Scenario Evaluation Framework

## Overview

The Agent Scenario Evaluation Framework provides comprehensive testing of trading agents across different market regimes using synthetic data. This framework is designed to evaluate agent robustness, adaptation capabilities, and performance consistency across various market conditions.

## Key Features

- **Synthetic Data Generation**: Creates realistic market scenarios with different characteristics
- **Multiple Market Regimes**: Tests agents across trend following, mean reversion, volatility breakout, crisis, and regime change scenarios
- **Comprehensive Metrics**: Calculates performance, risk, and adaptation metrics
- **Automated Evaluation**: Streamlined testing process with detailed reporting
- **Visualization**: Rich charts and graphs for result analysis
- **CLI Integration**: Easy-to-use command-line interface

## Market Scenarios

### 1. Trend Following

- **Description**: Market with clear directional trends and moderate volatility
- **Characteristics**:
  - Consistent price movement in one direction
  - Moderate volatility (1.5% daily)
  - Trend changes every 84 days
- **Success Criteria**:
  - Sharpe Ratio ≥ 0.8
  - Max Drawdown ≤ 12%
  - Win Rate ≥ 45%
  - Profit Factor ≥ 1.3

### 2. Mean Reversion

- **Description**: Market that reverts to mean price levels
- **Characteristics**:
  - Prices oscillate around a mean level
  - Higher volatility (2% daily)
  - Strong reversion force
- **Success Criteria**:
  - Sharpe Ratio ≥ 0.6
  - Max Drawdown ≤ 10%
  - Win Rate ≥ 50%
  - Profit Factor ≥ 1.4

### 3. Volatility Breakout

- **Description**: Market with sudden volatility spikes and regime changes
- **Characteristics**:
  - Low base volatility (1% daily)
  - Sudden spikes to 5% volatility
  - 10% probability of breakout events
- **Success Criteria**:
  - Sharpe Ratio ≥ 0.4
  - Max Drawdown ≤ 20%
  - Win Rate ≥ 35%
  - Profit Factor ≥ 1.1

### 4. Market Crisis

- **Description**: Simulated market crisis with high volatility and negative returns
- **Characteristics**:
  - High volatility (5% daily)
  - Negative drift (-0.5% daily)
  - Crisis period in middle of evaluation
- **Success Criteria**:
  - Sharpe Ratio ≥ 0.2
  - Max Drawdown ≤ 25%
  - Win Rate ≥ 30%
  - Profit Factor ≥ 1.0

### 5. Regime Changes

- **Description**: Market with multiple regime transitions
- **Characteristics**:
  - Four distinct market regimes
  - Smooth transitions between regimes
  - Varying volatility and drift
- **Success Criteria**:
  - Sharpe Ratio ≥ 0.5
  - Max Drawdown ≤ 15%
  - Win Rate ≥ 40%
  - Profit Factor ≥ 1.2

## Usage

### Command Line Interface

#### Evaluate Single Agent

```bash
# Evaluate moving average agent across all scenarios
python -m trading_rl_agent scenario evaluate --agent-type moving_average

# Evaluate with custom parameters
python -m trading_rl_agent scenario evaluate \
    --agent-type momentum \
    --output-dir outputs/my_evaluation \
    --seed 123 \
    --save-reports \
    --save-visualizations
```

#### Compare Multiple Agents

```bash
# Compare all available agents
python -m trading_rl_agent scenario compare \
    --output-dir outputs/agent_comparison \
    --seed 42
```

#### Custom Scenario Testing

```bash
# Test agent on specific scenario
python -m trading_rl_agent scenario custom \
    --agent-type mean_reversion \
    --scenario-name strong_uptrend \
    --output-dir outputs/custom_test
```

### Python API

#### Basic Usage

```python
from trading_rl_agent.eval import AgentScenarioEvaluator

# Initialize evaluator
evaluator = AgentScenarioEvaluator(seed=42)

# Create a simple agent
def my_agent(features):
    # Your agent logic here
    return np.random.normal(0, 1, len(features))

# Evaluate agent
results = evaluator.evaluate_agent(
    agent=my_agent,
    agent_name="My Agent"
)

# Print summary
evaluator.print_evaluation_summary(results)

# Generate report
report = evaluator.generate_evaluation_report(results)

# Create visualization
evaluator.create_visualization(results)
```

#### Custom Scenarios

```python
from trading_rl_agent.eval import MarketScenario

# Create custom scenario
custom_scenario = MarketScenario(
    name="My Custom Scenario",
    description="Custom market conditions",
    duration_days=100,
    market_regime="trend_following",
    base_volatility=0.02,
    drift=0.001,
    min_sharpe_ratio=0.7,
    max_drawdown=0.10,
    min_win_rate=0.45,
    min_profit_factor=1.3,
)

# Evaluate with custom scenario
results = evaluator.evaluate_agent(
    agent=my_agent,
    agent_name="My Agent",
    custom_scenarios=[custom_scenario]
)
```

## Performance Metrics

### Overall Performance Score

- **Range**: 0.0 to 1.0
- **Calculation**: Weighted average of key metrics
- **Weights**:
  - Sharpe Ratio: 30%
  - Total Return: 25%
  - Max Drawdown: 20%
  - Win Rate: 15%
  - Profit Factor: 10%

### Robustness Score

- **Range**: 0.0 to 1.0
- **Calculation**: Inverse of coefficient of variation across scenarios
- **Interpretation**: Higher values indicate more consistent performance

### Adaptation Score

- **Range**: 0.0 to 1.0
- **Calculation**: Average performance in challenging scenarios
- **Interpretation**: Higher values indicate better adaptation to difficult conditions

### Scenario-Specific Metrics

- **Total Return**: Cumulative return over the scenario period
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Volatility**: Standard deviation of returns

## Available Agent Types

### 1. Moving Average Agent

- **Strategy**: Moving average crossover
- **Parameters**: Window size (default: 20)
- **Best For**: Trend following markets

### 2. Momentum Agent

- **Strategy**: Price momentum detection
- **Parameters**: Lookback period (default: 10)
- **Best For**: Trending markets with momentum

### 3. Mean Reversion Agent

- **Strategy**: Z-score based mean reversion
- **Parameters**: Lookback period (default: 20)
- **Best For**: Range-bound markets

### 4. Volatility Breakout Agent

- **Strategy**: Volatility spike detection
- **Parameters**: Volatility window (default: 20)
- **Best For**: High volatility markets

## Output Files

### Reports

- **Markdown Reports**: Detailed text-based analysis
- **JSON Data**: Raw evaluation data for further analysis
- **Comparison Reports**: Multi-agent comparison summaries

### Visualizations

- **Performance Charts**: Equity curves and drawdown charts
- **Radar Charts**: Multi-dimensional performance visualization
- **Scenario Comparison**: Side-by-side scenario performance
- **Pass/Fail Summary**: Scenario success rate visualization

### Example Output Structure

```
outputs/scenario_evaluation/
├── moving_average_evaluation_report.md
├── moving_average_evaluation.png
├── momentum_evaluation_report.md
├── momentum_evaluation.png
├── agent_comparison_report.md
└── comprehensive_report.md
```

## Configuration

### YAML Configuration

```yaml
# configs/scenario_evaluation.yaml
evaluation:
  seed: 42
  output_dir: "outputs/scenario_evaluation"
  save_reports: true
  save_visualizations: true

default_scenarios:
  trend_following:
    name: "Trend Following"
    duration_days: 252
    base_volatility: 0.015
    min_sharpe_ratio: 0.8
    max_drawdown: 0.12
    # ... more parameters

transaction_costs:
  commission_rate: 0.001
  slippage_rate: 0.0001
  market_impact_rate: 0.00005
```

## Best Practices

### 1. Agent Development

- Test agents across multiple scenarios before deployment
- Focus on robustness over peak performance
- Consider scenario-specific optimization
- Monitor adaptation to regime changes

### 2. Evaluation Process

- Use consistent random seeds for reproducibility
- Run multiple evaluations with different seeds
- Compare against baseline strategies
- Consider transaction costs in evaluation

### 3. Result Interpretation

- Focus on overall score and robustness
- Consider scenario-specific strengths
- Monitor adaptation scores for regime changes
- Use pass rates as minimum viability criteria

### 4. Continuous Evaluation

- Re-evaluate agents regularly
- Monitor performance drift
- Update scenarios based on market conditions
- Track long-term performance trends

## Troubleshooting

### Common Issues

#### 1. Agent Not Generating Predictions

```python
# Ensure agent returns numpy array
def my_agent(features):
    # Convert to numpy array if needed
    return np.array(predictions)
```

#### 2. Feature Dimension Mismatch

```python
# Check feature preparation
features = evaluator._prepare_features(data)
print(f"Feature shape: {features.shape}")
```

#### 3. Memory Issues with Large Scenarios

```python
# Reduce scenario duration
scenario = MarketScenario(
    duration_days=100,  # Reduce from 252
    # ... other parameters
)
```

#### 4. Visualization Errors

```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('Agg')  # For non-interactive environments
```

## Advanced Usage

### Custom Scenario Generation

```python
from trading_rl_agent.eval.scenario_evaluator import MarketScenarioGenerator

generator = MarketScenarioGenerator(seed=42)

# Generate custom scenario data
data = generator.generate_trend_following_scenario(
    duration_days=100,
    trend_strength=0.002,
    volatility=0.02,
    trend_changes=1
)
```

### Custom Metrics

```python
def custom_metric(returns, data):
    """Calculate custom performance metric."""
    return np.mean(returns) / np.std(returns)

scenario = MarketScenario(
    # ... other parameters
    custom_metrics={
        "custom_sharpe": custom_metric
    }
)
```

### Integration with Existing Models

```python
# For scikit-learn models
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

results = evaluator.evaluate_agent(
    agent=model,
    agent_name="Random Forest"
)

# For PyTorch models
import torch

class MyModel(torch.nn.Module):
    # ... model definition

model = MyModel()
model.load_state_dict(torch.load("model.pth"))

results = evaluator.evaluate_agent(
    agent=model,
    agent_name="PyTorch Model"
)
```

## Contributing

### Adding New Scenarios

1. Extend `MarketScenarioGenerator` class
2. Add scenario definition to default scenarios
3. Update documentation
4. Add tests

### Adding New Metrics

1. Implement metric calculation function
2. Add to `MetricsCalculator` class
3. Update evaluation pipeline
4. Add visualization support

### Adding New Agent Types

1. Implement agent function
2. Add to CLI agent registry
3. Update example scripts
4. Add documentation

## References

- [Market Regime Detection](https://en.wikipedia.org/wiki/Market_regime)
- [Risk-Adjusted Returns](https://en.wikipedia.org/wiki/Risk-adjusted_return)
- [Synthetic Data Generation](https://en.wikipedia.org/wiki/Synthetic_data)
- [Agent Evaluation Methods](https://en.wikipedia.org/wiki/Evaluation_of_artificial_intelligence)
