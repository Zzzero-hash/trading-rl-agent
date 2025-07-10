#!/usr/bin/env python3
"""
Production Evaluation Script for Trading RL Agent
Usage: python evaluate.py [--agent AGENT_PATH] [--data DATA_PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.agents.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Trading RL Agent")
    parser.add_argument(
        "--agent",
        default="outputs/production_rl_agent.zip",
        help="Trained agent checkpoint path",
    )
    parser.add_argument(
        "--data",
        default="outputs/production_trading_data.csv",
        help="Trading data CSV file",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial trading capital",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    logger.info("🎯 Starting production evaluation...")
    logger.info(f"   Agent: {args.agent}")
    logger.info(f"   Data: {args.data}")
    logger.info(f"   Initial capital: ${args.initial_capital:,.2f}")

    try:
        # Step 1: Load data
        logger.info("📊 Loading trading data...")
        df = pd.read_csv(args.data)
        logger.info(f"Loaded {len(df)} samples with {df.shape[1]} features")

        # Step 2: Load agent
        logger.info("🤖 Loading trained agent...")

        state_dim = df.shape[1]
        agent = PPOAgent(state_dim=state_dim, action_dim=3)
        agent.load(args.agent)
        logger.info(f"Loaded agent with state_dim={state_dim}")

        # Step 3: Run trading simulation
        logger.info("📈 Running trading simulation...")

        # Simple backtesting setup
        capital = args.initial_capital
        position = 0  # 0 = no position, 1 = long position
        trades = []
        portfolio_values = [capital]

        action_names = ["BUY", "SELL", "HOLD"]

        for i in range(len(df)):
            # Get current state
            state = torch.FloatTensor(df.iloc[i].values)

            # Get agent action
            action = agent.select_action(state)

            # Convert continuous action to discrete
            if isinstance(action, np.ndarray):
                continuous_val = action[0] if len(action) > 0 else 0
                if continuous_val < -0.33:
                    discrete_action = 0  # BUY
                elif continuous_val > 0.33:
                    discrete_action = 1  # SELL
                else:
                    discrete_action = 2  # HOLD
            else:
                discrete_action = int(action) % 3

            # Simple trading logic (assuming we have price data)
            # For synthetic data, we'll simulate price changes
            price_change = np.random.normal(0, 0.01)  # 1% daily volatility

            if discrete_action == 0 and position == 0:  # BUY when no position
                position = 1
                entry_price = 100 * (1 + price_change)  # Base price of 100
                trades.append(
                    {
                        "type": "BUY",
                        "price": entry_price,
                        "step": i,
                        "action": action_names[discrete_action],
                    },
                )
            elif discrete_action == 1 and position == 1:  # SELL when holding position
                position = 0
                exit_price = 100 * (1 + price_change)
                profit = (exit_price - entry_price) / entry_price * capital
                capital += profit
                trades.append(
                    {
                        "type": "SELL",
                        "price": exit_price,
                        "step": i,
                        "profit": profit,
                        "action": action_names[discrete_action],
                    },
                )

            # Update portfolio value
            if position == 1:
                current_price = 100 * (1 + price_change)
                unrealized_pnl = (
                    (current_price - entry_price) / entry_price * capital if "entry_price" in locals() else 0
                )
                portfolio_values.append(capital + unrealized_pnl)
            else:
                portfolio_values.append(capital)

        # Step 4: Calculate performance metrics
        logger.info("📊 Calculating performance metrics...")

        final_value = portfolio_values[-1]
        total_return = (final_value - args.initial_capital) / args.initial_capital * 100

        # Calculate volatility
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Win rate
        profitable_trades = [t for t in trades if t.get("profit", 0) > 0]
        win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0

        # Step 5: Generate report
        logger.info("=" * 60)
        logger.info("🎯 PRODUCTION EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Capital:     ${args.initial_capital:,.2f}")
        logger.info(f"Final Portfolio:     ${final_value:,.2f}")
        logger.info(f"Total Return:        {total_return:.2f}%")
        logger.info(f"Total Trades:        {len(trades)}")
        logger.info(f"Win Rate:           {win_rate:.1f}%")
        logger.info(f"Volatility:         {volatility:.2f}")
        logger.info(f"Max Drawdown:       {max_drawdown:.2f}%")
        logger.info(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
        logger.info("=" * 60)

        # Step 6: Save detailed report
        report = {
            "evaluation_date": pd.Timestamp.now().isoformat(),
            "agent_path": args.agent,
            "data_path": args.data,
            "initial_capital": args.initial_capital,
            "final_value": final_value,
            "total_return_percent": total_return,
            "total_trades": len(trades),
            "win_rate_percent": win_rate,
            "volatility": volatility,
            "max_drawdown_percent": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
            "portfolio_values": portfolio_values,
        }

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / "production_evaluation_report.json"
        with report_file.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📊 Detailed report saved to: {report_file}")
        logger.info("✅ Production evaluation completed!")

    except Exception:
        logger.exception("❌ Evaluation failed")
        raise
    else:
        return report


if __name__ == "__main__":
    main()
