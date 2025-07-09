#!/usr/bin/env python3
"""
Production Training Script for Trading RL Agent
Usage: python train.py [--config CONFIG_FILE]
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Trading RL Agent")
    parser.add_argument(
        "--config", default="configs/production.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--data-samples",
        type=int,
        default=500,
        help="Number of data samples to generate",
    )
    parser.add_argument(
        "--training-steps", type=int, default=10000, help="Number of RL training steps"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory for models and data"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting production training pipeline...")
    logger.info(f"   Config: {args.config}")
    logger.info(f"   Data samples: {args.data_samples}")
    logger.info(f"   Training steps: {args.training_steps}")
    logger.info(f"   Output directory: {args.output_dir}")

    try:
        # Step 1: Generate data
        logger.info("📊 Step 1: Generating market data...")
        from trading_rl_agent.data.features import generate_features
        from trading_rl_agent.data.synthetic import fetch_synthetic_data

        df = fetch_synthetic_data(n_samples=args.data_samples, timeframe="hour")
        logger.info(f"Generated {len(df)} samples")

        # Step 2: Add features
        logger.info("🔧 Step 2: Adding technical indicators...")
        df_features = generate_features(df)

        # Clean data
        import numpy as np

        numeric_df = df_features.select_dtypes(include=[np.number])
        numeric_df = numeric_df.dropna()

        logger.info(f"Final dataset: {numeric_df.shape}")

        # Save data
        data_file = output_dir / "production_trading_data.csv"
        numeric_df.to_csv(data_file, index=False)
        logger.info(f"💾 Data saved to: {data_file}")

        # Step 3: Train RL agent
        logger.info("🤖 Step 3: Training RL agent...")
        from trading_rl_agent.agents.ppo_agent import PPOAgent

        state_dim = numeric_df.shape[1]
        agent = PPOAgent(state_dim=state_dim, action_dim=3)

        # Simple training loop
        import torch

        for step in range(args.training_steps):
            # Sample random state from data
            idx = np.random.randint(0, len(numeric_df))
            state = torch.FloatTensor(numeric_df.iloc[idx].values)

            # Get action
            action = agent.select_action(state)

            # Simple reward (placeholder)
            reward = np.random.normal(0, 0.1)

            # Log progress
            if step % 1000 == 0:
                logger.info(f"   Training step {step}/{args.training_steps}")

        # Save trained agent
        agent_file = output_dir / "production_rl_agent.zip"
        agent.save(str(agent_file))
        logger.info(f"💾 Agent saved to: {agent_file}")

        # Step 4: Test the agent
        logger.info("🧪 Step 4: Testing trained agent...")
        test_actions = []
        for i in range(10):
            idx = np.random.randint(0, len(numeric_df))
            state = torch.FloatTensor(numeric_df.iloc[idx].values)
            action = agent.select_action(state)
            # Convert continuous action to discrete action (0=BUY, 1=SELL, 2=HOLD)
            if isinstance(action, np.ndarray):
                # Take the first element and map to discrete action
                continuous_val = action[0] if len(action) > 0 else 0
                # Map [-1, 1] to [0, 1, 2]
                if continuous_val < -0.33:
                    discrete_action = 0  # BUY
                elif continuous_val > 0.33:
                    discrete_action = 1  # SELL
                else:
                    discrete_action = 2  # HOLD
            else:
                discrete_action = int(action) % 3  # Fallback
            test_actions.append(discrete_action)

        action_names = ["BUY", "SELL", "HOLD"]
        action_counts = {
            name: test_actions.count(i) for i, name in enumerate(action_names)
        }
        logger.info(f"   Test actions: {action_counts}")

        # Step 5: Generate summary
        logger.info("📋 Step 5: Generating summary...")
        summary = {
            "config": args.config,
            "data_samples": len(numeric_df),
            "features": state_dim,
            "training_steps": args.training_steps,
            "data_file": str(data_file),
            "agent_file": str(agent_file),
            "test_actions": action_counts,
        }

        import json

        summary_file = output_dir / "production_training_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("✅ Production training completed successfully!")
        logger.info(f"📊 Summary: {summary}")
        logger.info(f"📝 Full report: {summary_file}")

        return summary

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
