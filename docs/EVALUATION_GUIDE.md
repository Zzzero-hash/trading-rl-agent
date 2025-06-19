# Agent Evaluation Guide

This document explains how to evaluate a trained trading agent using the
`evaluate_agent.py` script introduced in this repository. The evaluation
process measures trading performance on historical data and stores metrics
for further analysis.

## Usage

```
python evaluate_agent.py --data path/to/market_data.csv \
    --checkpoint path/to/agent_checkpoint.pth \
    --agent sac \
    --output results/evaluation.json
```

Arguments:
- `--data` – CSV file containing the market data used for evaluation.
- `--checkpoint` – File path to the saved agent parameters.
- `--agent` – Type of agent to load (`sac`, `td3`, or `ensemble`).
- `--output` – Where to write the resulting metrics JSON (default `results/evaluation.json`).
- `--window-size` – Observation window length (defaults to 50).

The script will load the dataset, initialize a `TradingEnv`, run one full
trading episode using the selected agent, and compute metrics such as
Sharpe ratio and maximum drawdown. The resulting dictionary is written to
the specified output file.

## Output

An example output file `results/evaluation.json` looks like this:

```json
{
  "sharpe_ratio": 1.25,
  "sortino_ratio": 2.10,
  "max_drawdown": 0.12,
  "profit_factor": 1.8,
  "win_rate": 0.55,
  "calmar_ratio": 1.3,
  "total_return": 0.35,
  "volatility": 0.2,
  "num_trades": 120
}
```

These metrics come from `src/utils/metrics.py` and provide a quick snapshot
of strategy quality. Higher Sharpe and Sortino ratios generally indicate a
better risk-adjusted return.

## Notebook Example

See [`evaluation_example.ipynb`](../evaluation_example.ipynb) for a brief
walkthrough of running the evaluation script inside a Jupyter notebook and
visualizing the resulting metrics.
