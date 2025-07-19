# Performance Attribution Analysis Guide

## Overview

The Performance Attribution Analysis system provides comprehensive tools for analyzing portfolio performance through multiple attribution methodologies. This system integrates seamlessly with the existing portfolio management framework and offers both programmatic and command-line interfaces.

## Features

### 1. Systematic vs Idiosyncratic Return Decomposition

- **Factor Model Analysis**: Uses Principal Component Analysis (PCA) to extract systematic factors
- **Return Decomposition**: Separates returns into systematic (factor-driven) and idiosyncratic (asset-specific) components
- **Model Quality Assessment**: Provides R-squared metrics to evaluate factor model fit

### 2. Factor Attribution Analysis

- **Multi-Factor Models**: Supports multiple factor extraction and analysis
- **Factor Loadings**: Calculates asset-specific factor exposures
- **Factor Contributions**: Quantifies each factor's contribution to portfolio performance
- **Market Factor Integration**: Includes market factor for comprehensive analysis

### 3. Brinson Attribution for Sector/Asset Allocation

- **Sector Attribution**: Analyzes sector allocation effects using Brinson methodology
- **Asset Allocation Effects**: Quantifies allocation, selection, and interaction effects
- **Flexible Grouping**: Supports custom sector classifications and asset groupings
- **Time-Series Analysis**: Provides attribution analysis over multiple periods

### 4. Risk-Adjusted Attribution Analysis

- **Comprehensive Risk Metrics**: Volatility, VaR, CVaR, maximum drawdown, skewness, kurtosis
- **Information Ratio**: Calculates risk-adjusted excess returns
- **Factor Risk Contributions**: Analyzes factor contributions to portfolio risk
- **Downside Risk Analysis**: Focuses on downside risk metrics

### 5. Interactive Attribution Dashboards

- **Plotly Integration**: Interactive charts and dashboards
- **Matplotlib Fallback**: Static visualizations when Plotly is unavailable
- **Multi-Panel Dashboards**: Comprehensive view of all attribution components
- **Export Capabilities**: Save dashboards as HTML files

### 6. Automated Attribution Reporting

- **Comprehensive Reports**: Detailed attribution analysis reports
- **Excel Export**: Export all attribution data to Excel format
- **Automated Workflows**: Scheduled attribution analysis and reporting
- **Customizable Outputs**: Configurable report formats and content

## Installation and Setup

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install plotly  # Optional, for interactive dashboards
pip install openpyxl  # For Excel export functionality
```

### Basic Usage

#### 1. Simple Attribution Analysis

```python
from trading_rl_agent.portfolio.attribution import PerformanceAttributor, AttributionConfig

# Initialize attribution system
config = AttributionConfig(
    risk_free_rate=0.02,
    confidence_level=0.95,
    use_plotly=True
)
attributor = PerformanceAttributor(config)

# Run comprehensive analysis
results = attributor.analyze_performance(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    asset_returns=asset_returns,
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights,
    sector_data=sector_data
)

# Generate report
report = attributor.generate_report("attribution_report.txt")

# Create dashboard
dashboard = attributor.create_dashboard()
```

#### 2. Integration with Portfolio Manager

```python
from trading_rl_agent.portfolio.attribution_integration import AttributionIntegration

# Create integration layer
integration = AttributionIntegration(portfolio_manager, config)

# Run analysis with automatic data preparation
results = integration.run_attribution_analysis(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Create interactive dashboard
dashboard = integration.create_attribution_dashboard()

# Export data to Excel
integration.export_attribution_data("attribution_data.xlsx")
```

#### 3. Automated Workflow

```python
from trading_rl_agent.portfolio.attribution_integration import AutomatedAttributionWorkflow

# Set up automated workflow
workflow = AutomatedAttributionWorkflow(portfolio_manager, config)
workflow.analysis_frequency = "monthly"
workflow.auto_generate_reports = True
workflow.report_output_dir = "attribution_reports"

# Run scheduled analysis
results = workflow.run_scheduled_analysis()

# Trigger analysis on portfolio rebalancing
results = workflow.on_portfolio_rebalance()

# Run analysis on performance milestones
results = workflow.on_performance_milestone("quarterly")
```

## Command-Line Interface

### Basic Commands

#### 1. Comprehensive Attribution Analysis

```bash
python -m trading_rl_agent.portfolio.cli_attribution analyze \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --output-dir attribution_output \
    --symbols AAPL GOOGL MSFT AMZN TSLA
```

#### 2. Factor Analysis

```bash
python -m trading_rl_agent.portfolio.cli_attribution factor-analysis \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

#### 3. Sector Attribution

```bash
python -m trading_rl_agent.portfolio.cli_attribution sector-attribution \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

#### 4. Risk Analysis

```bash
python -m trading_rl_agent.portfolio.cli_attribution risk-analysis \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

#### 5. Automated Workflow Setup

```bash
python -m trading_rl_agent.portfolio.cli_attribution setup-automation \
    --frequency monthly \
    --auto-reports \
    --output-dir attribution_reports
```

#### 6. Export Data

```bash
python -m trading_rl_agent.portfolio.cli_attribution export-data \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --output-path attribution_data.xlsx
```

### Configuration Files

Create a YAML configuration file for custom settings:

```yaml
# attribution_config.yaml
risk_free_rate: 0.02
confidence_level: 0.95
lookback_period: 252
sector_column: "sector"
asset_class_column: "asset_class"
use_plotly: true
figure_size: [12, 8]
min_observations: 60
max_factors: 10
```

Use the configuration file:

```bash
python -m trading_rl_agent.portfolio.cli_attribution analyze \
    --config-file attribution_config.yaml \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

## Advanced Usage

### 1. Custom Factor Models

```python
from trading_rl_agent.portfolio.attribution import FactorModel

# Create custom factor model
factor_model = FactorModel(config)

# Fit model with custom data
factor_model.fit(asset_returns, market_returns)

# Extract factors and loadings
factors = factor_model.factors
loadings = factor_model.factor_loadings
r_squared = factor_model.r_squared

# Decompose returns
decomposition = factor_model.decompose_returns(asset_returns)
systematic_returns = decomposition['systematic']
idiosyncratic_returns = decomposition['idiosyncratic']
```

### 2. Brinson Attribution Analysis

```python
from trading_rl_agent.portfolio.attribution import BrinsonAttributor

# Create Brinson attributor
brinson = BrinsonAttributor(config)

# Calculate attribution for specific period
attribution = brinson.calculate_attribution(
    portfolio_weights=portfolio_weights,
    benchmark_weights=benchmark_weights,
    returns=returns,
    grouping_column="sector"
)

print(f"Allocation Effect: {attribution['allocation']:.6f}")
print(f"Selection Effect: {attribution['selection']:.6f}")
print(f"Interaction Effect: {attribution['interaction']:.6f}")
print(f"Total Attribution: {attribution['total']:.6f}")
```

### 3. Risk-Adjusted Analysis

```python
from trading_rl_agent.portfolio.attribution import RiskAdjustedAttributor

# Create risk-adjusted attributor
risk_attributor = RiskAdjustedAttributor(config)

# Calculate comprehensive risk metrics
risk_metrics = risk_attributor.calculate_risk_metrics(returns)

print(f"Volatility: {risk_metrics['volatility']:.4f}")
print(f"VaR: {risk_metrics['var']:.4f}")
print(f"Max Drawdown: {risk_metrics['max_drawdown']:.4f}")
print(f"Skewness: {risk_metrics['skewness']:.4f}")
print(f"Kurtosis: {risk_metrics['kurtosis']:.4f}")

# Calculate risk-adjusted attribution
risk_attribution = risk_attributor.calculate_risk_adjusted_attribution(
    portfolio_returns, benchmark_returns, factor_returns
)

print(f"Information Ratio: {risk_attribution['information_ratio']:.4f}")
```

### 4. Custom Visualizations

```python
from trading_rl_agent.portfolio.attribution import AttributionVisualizer

# Create visualizer
visualizer = AttributionVisualizer(config)

# Create custom dashboard
dashboard = visualizer.create_attribution_dashboard(
    attribution_results, portfolio_returns, benchmark_returns
)

# Save dashboard
if hasattr(dashboard, 'write_html'):
    dashboard.write_html("custom_dashboard.html")
```

## Data Requirements

### Input Data Format

#### 1. Portfolio and Benchmark Returns

```python
# Time series of returns (daily frequency recommended)
portfolio_returns = pd.Series(
    [0.001, 0.002, -0.001, ...],
    index=pd.date_range('2023-01-01', periods=252)
)

benchmark_returns = pd.Series(
    [0.0008, 0.0015, -0.0005, ...],
    index=pd.date_range('2023-01-01', periods=252)
)
```

#### 2. Asset Returns Matrix

```python
# Assets x Time matrix
asset_returns = pd.DataFrame(
    [[0.001, 0.002, -0.001, ...],
     [0.002, 0.001, 0.003, ...],
     ...],
    index=['AAPL', 'GOOGL', 'MSFT', ...],
    columns=pd.date_range('2023-01-01', periods=252)
)
```

#### 3. Portfolio and Benchmark Weights

```python
# Assets x Time matrices
portfolio_weights = pd.DataFrame(
    [[0.2, 0.21, 0.19, ...],
     [0.15, 0.14, 0.16, ...],
     ...],
    index=['AAPL', 'GOOGL', 'MSFT', ...],
    columns=pd.date_range('2023-01-01', periods=252)
)

benchmark_weights = pd.DataFrame(
    [[0.18, 0.19, 0.17, ...],
     [0.12, 0.11, 0.13, ...],
     ...],
    index=['AAPL', 'GOOGL', 'MSFT', ...],
    columns=pd.date_range('2023-01-01', periods=252)
)
```

#### 4. Sector Data

```python
# Asset classification data
sector_data = pd.DataFrame({
    'sector': ['Technology', 'Technology', 'Technology', 'Consumer', ...],
    'asset_class': ['Equity', 'Equity', 'Equity', 'Equity', ...]
}, index=['AAPL', 'GOOGL', 'MSFT', 'AMZN', ...])
```

## Output Interpretation

### 1. Factor Attribution Results

```python
factor_attribution = results['factor_attribution']
# Positive values indicate positive contribution
# Negative values indicate negative contribution
# Magnitude indicates relative importance
```

### 2. Brinson Attribution Results

```python
brinson_attribution = results['brinson_attribution']
# Allocation: Effect of overweighting/underweighting sectors
# Selection: Effect of picking better/worse stocks within sectors
# Interaction: Combined effect of allocation and selection decisions
```

### 3. Risk Metrics

```python
risk_metrics = results['risk_adjusted']['portfolio_risk']
# Volatility: Annualized standard deviation of returns
# VaR: Value at Risk at specified confidence level
# Max Drawdown: Largest peak-to-trough decline
# Information Ratio: Excess return per unit of tracking error
```

## Best Practices

### 1. Data Quality

- Ensure data consistency across all inputs
- Handle missing values appropriately
- Use sufficient historical data (minimum 60 observations)
- Align time periods across all datasets

### 2. Factor Model Selection

- Start with default settings for initial analysis
- Adjust `max_factors` based on asset universe size
- Monitor R-squared values for model quality
- Consider economic interpretation of factors

### 3. Attribution Analysis

- Use consistent benchmark throughout analysis
- Consider transaction costs in attribution
- Analyze attribution over multiple time periods
- Combine quantitative and qualitative analysis

### 4. Risk Management

- Monitor factor risk contributions regularly
- Set appropriate confidence levels for VaR
- Consider tail risk measures beyond standard metrics
- Integrate attribution with portfolio optimization

## Troubleshooting

### Common Issues

#### 1. Insufficient Data

```
Error: "Could not calculate loadings for asset"
Solution: Ensure minimum 60 observations and sufficient data quality
```

#### 2. Factor Model Convergence

```
Error: "Could not calculate loadings for asset"
Solution: Check for multicollinearity and reduce max_factors
```

#### 3. Memory Issues

```
Error: MemoryError during analysis
Solution: Reduce data size or use chunked processing
```

#### 4. Visualization Issues

```
Error: Plotly not available
Solution: Install plotly or set use_plotly=False
```

### Performance Optimization

#### 1. Large Datasets

```python
# Use chunked processing for large datasets
for chunk in data_chunks:
    results = attributor.analyze_performance(chunk)
    # Aggregate results
```

#### 2. Caching

```python
# Enable caching for repeated analysis
integration.run_attribution_analysis(force_recompute=False)
```

#### 3. Parallel Processing

```python
# Use parallel processing for factor calculations
from concurrent.futures import ProcessPoolExecutor
# Implement parallel factor model fitting
```

## Integration Examples

### 1. Backtesting Integration

```python
# Integrate with backtesting framework
def attribution_analysis_callback(portfolio_state):
    integration = AttributionIntegration(portfolio_manager, config)
    results = integration.run_attribution_analysis()
    return results

# Add to backtesting loop
backtest.add_callback(attribution_analysis_callback)
```

### 2. Real-Time Monitoring

```python
# Set up real-time attribution monitoring
def monitor_attribution():
    workflow = AutomatedAttributionWorkflow(portfolio_manager, config)
    if workflow.should_run_analysis():
        results = workflow.run_scheduled_analysis()
        # Send alerts or notifications
        send_attribution_alert(results)

# Schedule monitoring
schedule.every().day.at("18:00").do(monitor_attribution)
```

### 3. Reporting Integration

```python
# Integrate with reporting system
def generate_monthly_report():
    integration = AttributionIntegration(portfolio_manager, config)

    # Generate attribution report
    report = integration.generate_attribution_report()

    # Create dashboard
    dashboard = integration.create_attribution_dashboard()

    # Export data
    integration.export_attribution_data("monthly_attribution.xlsx")

    # Send to stakeholders
    send_report_to_stakeholders(report, dashboard)
```

## Conclusion

The Performance Attribution Analysis system provides a comprehensive framework for understanding portfolio performance through multiple attribution methodologies. By integrating with the existing portfolio management system, it enables both ad-hoc analysis and automated workflows for ongoing performance monitoring.

For additional support and advanced usage examples, refer to the test suite and demonstration scripts included in the codebase.
