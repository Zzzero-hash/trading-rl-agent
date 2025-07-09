# Trading RL Agent - Production System

## Quick Start

### 1. Train a new agent

```bash
python train.py --data-samples 500 --training-steps 10000
```

### 2. Evaluate the trained agent

```bash
python evaluate.py
```

### 3. Use the CLI for advanced operations

```bash
python cli.py --help
```

## Key Files

- `train.py` - Production training script
- `evaluate.py` - Production evaluation script
- `cli.py` - Full CLI interface
- `main.py` - Production entry point
- `configs/production.yaml` - Production configuration

## Output Files

All outputs are saved to the `outputs/` directory:

- `production_rl_agent.zip` - Trained RL agent checkpoint
- `production_trading_data.csv` - Generated trading data with features
- `production_training_summary.json` - Training summary and metrics
- `production_evaluation_report.json` - Detailed evaluation results

## System Architecture

The production system includes:

- **Data Pipeline**: Synthetic market data generation with 65+ technical indicators
- **Feature Engineering**: RSI, MACD, Bollinger Bands, candlestick patterns, etc.
- **RL Agent**: PPO-based agent with continuous action space mapped to discrete actions
- **Backtesting**: Portfolio simulation with P&L tracking and performance metrics
- **Evaluation**: Comprehensive performance analysis with Sharpe ratio, drawdown, etc.

## Performance Example

Recent training results:

- **Data**: 167 samples with 65 features
- **Return**: 9.97% over simulation period
- **Trades**: 51 total trades
- **Win Rate**: 31.4%
- **Sharpe Ratio**: 0.94
- **Max Drawdown**: -5.36%

The system is designed for further development and can be extended with:

- Real market data feeds
- Advanced reward functions
- Multi-asset portfolios
- Risk management modules
- Production deployment infrastructure
