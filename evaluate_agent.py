"""Evaluate a trained agent on historical data and report metrics.

This script loads a saved agent checkpoint, runs it through ``TradingEnv``
on the provided dataset, and computes common performance metrics such as
Sharpe ratio and maximum drawdown. Results are stored in a JSON file for
later analysis.

Example:
-------
```
python evaluate_agent.py --data data/sample_data.csv --checkpoint sac_agent.pth \
    --agent sac --output results/evaluation.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trading_rl_agent.agents.policy_utils import CallablePolicy, WeightedEnsembleAgent
from trading_rl_agent.agents.sac_agent import SACAgent
from trading_rl_agent.agents.td3_agent import TD3Agent
from trading_rl_agent.envs.trading_env import TradingEnv
from trading_rl_agent.utils import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained trading agent")
    parser.add_argument(
        "--data", type=str, required=True, help="CSV dataset for evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Agent checkpoint path"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="sac",
        choices=["sac", "td3", "ensemble"],
        help="Type of agent to load",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Path to save metrics JSON",
    )
    parser.add_argument(
        "--window-size", type=int, default=50, help="Observation window size"
    )
    return parser.parse_args()


def load_agent(agent_type: str, state_dim: int, action_dim: int, checkpoint: str):
    if agent_type == "sac":
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(checkpoint)
        return agent
    if agent_type == "td3":
        agent = TD3Agent(state_dim=state_dim, action_dim=action_dim)
        agent.load(checkpoint)
        return agent

    # Ensemble via RLlib policy mapping
    sac = SACAgent(state_dim=state_dim, action_dim=action_dim)
    td3 = TD3Agent(state_dim=state_dim, action_dim=action_dim)
    sac.load(checkpoint)
    td3.load(checkpoint)

    obs_space = sac.observation_space
    act_space = sac.action_space
    policies = {
        "sac": CallablePolicy(obs_space, act_space, sac.select_action),
        "td3": CallablePolicy(obs_space, act_space, td3.select_action),
    }
    return WeightedEnsembleAgent(policies, {"sac": 0.5, "td3": 0.5})


def run_episode(env: TradingEnv, agent) -> list[float]:
    state, _ = env.reset()
    rewards: list[float] = []
    done = False
    while not done:
        obs = state
        if isinstance(obs, dict):
            # flatten dictionary observation for agent
            market = obs.get("market_features", np.array([])).flatten()
            model = obs.get("model_pred", np.array([])).flatten()
            obs = np.concatenate([market, model])
        action = agent.select_action(np.asarray(obs), evaluate=True)
        state, reward, done, _, _ = env.step(action)
        rewards.append(float(reward))
    return rewards


def main() -> None:
    args = parse_args()

    env_cfg = {
        "dataset_paths": [args.data],
        "window_size": args.window_size,
        "continuous_actions": True,
        "model_path": None,
    }
    env = TradingEnv(env_cfg)

    if isinstance(env.observation_space, dict) or hasattr(
        env.observation_space, "spaces"
    ):
        state_dim = int(np.prod(env.observation_space["market_features"].shape))
        if env.model_output_size:
            state_dim += env.model_output_size
    else:
        state_dim = int(np.prod(env.observation_space.shape))
    action_dim = (
        env.action_space.shape[0]
        if hasattr(env.action_space, "shape")
        else env.action_space.n
    )

    agent = load_agent(args.agent, state_dim, action_dim, args.checkpoint)

    rewards = run_episode(env, agent)
    equity = np.cumprod(1 + np.array(rewards))
    results = metrics.calculate_risk_metrics(np.array(rewards), equity)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("Saved evaluation metrics to", output_path)


if __name__ == "__main__":
    main()
