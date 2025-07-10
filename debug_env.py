#!/usr/bin/env python3
"""
Debug script to understand the environment termination issue.
"""

import tempfile
from pathlib import Path

import pandas as pd

from trading_rl_agent.envs.trading_env import TradingEnv


def debug_environment():
    """Debug the environment termination logic."""
    print("🔍 Debugging Environment Termination...")

    # Create minimal test data
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=20,
                freq="D",
            ),  # Only 20 days
            "symbol": ["TEST"] * 20,
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000] * 20,
        },
    )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.to_csv(f.name, index=False)
        csv_path = f.name

    try:
        # Create environment
        config = {
            "dataset_paths": [csv_path],
            "window_size": 5,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
            "include_features": False,
        }

        env = TradingEnv(config)
        print(f"Environment created. Data length: {len(env.data)}")
        print(f"Window size: {env.window_size}")
        print(f"Expected available steps: {len(env.data) - env.window_size}")

        # Reset environment
        obs, info = env.reset()
        print(f"Reset complete. Current step: {env.current_step}")

        # Step through environment
        step_count = 0
        max_steps = 50  # Should be more than enough

        while step_count < max_steps:
            obs, reward, done, truncated, info = env.step(0)  # Hold action
            step_count += 1
            print(
                f"Step {step_count}: current_step={env.current_step}, done={done}, truncated={truncated}",
            )

            if done or truncated:
                print(f"✅ Episode terminated after {step_count} steps")
                break
        else:
            print(f"❌ Episode did NOT terminate after {step_count} steps")

    finally:
        # Clean up
        Path(csv_path).unlink()


if __name__ == "__main__":
    debug_environment()
