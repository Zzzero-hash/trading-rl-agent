"""
Ensemble Evaluator for Multi-Agent RL Trading Systems.

This module provides comprehensive evaluation and diagnostics for ensemble agents including:
- Ensemble performance metrics
- Agent diversity analysis
- Consensus and disagreement measures
- Stability and robustness assessment
- Comparative analysis between agents
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from .policy_utils import EnsembleAgent


class EnsembleEvaluator:
    """Comprehensive evaluator for ensemble agents."""

    def __init__(self, ensemble: EnsembleAgent):
        self.ensemble = ensemble
        self.logger = logging.getLogger(self.__class__.__name__)

        # Evaluation history
        self.evaluation_history: dict[str, list[float]] = defaultdict(list)

        # Diversity metrics
        self.diversity_metrics: dict[str, list[float]] = {
            "action_diversity": [],
            "policy_diversity": [],
            "performance_diversity": [],
            "temporal_diversity": [],
        }

    def evaluate_ensemble(
        self,
        env: Any,
        num_episodes: int = 100,
        include_diagnostics: bool = True,
        save_results: bool = False,
        results_path: str | None = None,
    ) -> dict[str, Any]:
        """Comprehensive ensemble evaluation."""
        self.logger.info(f"Starting ensemble evaluation with {num_episodes} episodes")

        # Initialize results containers
        episode_results = []
        agent_actions = defaultdict(list)
        agent_rewards: dict[str, list[float]] = defaultdict(list)
        consensus_events = []

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_actions = []
            episode_consensus = []

            while True:
                # Get individual agent actions
                individual_actions = {}
                for name, policy in self.ensemble.policy_map.items():
                    action, _, _ = policy.compute_single_action(obs)
                    individual_actions[name] = action
                    agent_actions[name].append(action)

                # Get ensemble action
                ensemble_action = self.ensemble.select_action(obs)

                # Check consensus
                consensus_score = self._calculate_consensus_score(individual_actions)
                episode_consensus.append(consensus_score)

                # Take action in environment
                obs, reward, done, info = env.step(ensemble_action)
                episode_reward += reward
                episode_length += 1
                episode_actions.append(ensemble_action)

                if done:
                    break

            # Store episode results
            episode_results.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "avg_consensus": np.mean(episode_consensus),
                    "std_consensus": np.std(episode_consensus),
                },
            )

            consensus_events.extend(episode_consensus)

        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(episode_results, agent_actions, consensus_events)

        # Add diagnostics if requested
        if include_diagnostics:
            diagnostics = self._calculate_diagnostics(agent_actions, episode_results)
            results["diagnostics"] = diagnostics

        # Update evaluation history
        self._update_evaluation_history(results)

        # Save results if requested
        if save_results and results_path:
            self._save_evaluation_results(results, results_path)

        self.logger.info("Ensemble evaluation completed")
        return results

    def _calculate_consensus_score(self, individual_actions: dict[str, np.ndarray]) -> float:
        """Calculate consensus score between agent actions."""
        if len(individual_actions) < 2:
            return 1.0

        actions_array = np.array(list(individual_actions.values()))
        std_action = np.std(actions_array, axis=0)

        # Normalize by action magnitude
        mean_action = np.mean(actions_array, axis=0)
        action_magnitude = np.linalg.norm(mean_action)

        if action_magnitude > 0:
            normalized_std = np.linalg.norm(std_action) / action_magnitude
            consensus_score = max(0.0, 1.0 - float(normalized_std))
        else:
            consensus_score = 1.0

        return float(consensus_score) if consensus_score is not None else 1.0

    def _calculate_comprehensive_metrics(
        self,
        episode_results: list[dict[str, Any]],
        agent_actions: dict[str, list[np.ndarray]],
        consensus_events: list[float],
    ) -> dict[str, Any]:
        """Calculate comprehensive ensemble metrics."""
        # Basic performance metrics
        rewards = [ep["reward"] for ep in episode_results]
        lengths = [ep["length"] for ep in episode_results]

        performance_metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_episode_length": np.mean(lengths),
            "success_rate": np.mean([r > 0 for r in rewards]),
        }

        # Consensus metrics
        consensus_metrics = {
            "mean_consensus": np.mean(consensus_events),
            "std_consensus": np.std(consensus_events),
            "consensus_stability": 1.0 - np.std(consensus_events),
            "high_consensus_rate": np.mean([c > 0.8 for c in consensus_events]),
        }

        # Agent diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(agent_actions)

        # Ensemble stability metrics
        stability_metrics = self._calculate_stability_metrics(episode_results)

        return {
            "performance": performance_metrics,
            "consensus": consensus_metrics,
            "diversity": diversity_metrics,
            "stability": stability_metrics,
            "num_episodes": len(episode_results),
            "num_agents": len(self.ensemble.policy_map),
        }

    def _calculate_diversity_metrics(self, agent_actions: dict[str, list[np.ndarray]]) -> dict[str, float]:
        """Calculate diversity metrics between agents."""
        if len(agent_actions) < 2:
            return {"action_diversity": 0.0, "policy_diversity": 0.0}

        # Action diversity
        action_diversities = []
        for i in range(len(next(iter(agent_actions.values())))):
            actions_at_step = []
            for actions in agent_actions.values():
                if i < len(actions):
                    actions_at_step.append(actions[i])

            if len(actions_at_step) > 1:
                actions_array = np.array(actions_at_step)
                diversity = np.std(actions_array, axis=0)
                action_diversities.append(np.linalg.norm(diversity))

        action_diversity = np.mean(action_diversities) if action_diversities else 0.0

        # Policy diversity (based on action patterns)
        policy_diversities = []
        agent_names = list(agent_actions.keys())

        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                agent1_actions = agent_actions[agent_names[i]]
                agent2_actions = agent_actions[agent_names[j]]

                min_length = min(len(agent1_actions), len(agent2_actions))
                if min_length > 0:
                    actions1 = np.array(agent1_actions[:min_length])
                    actions2 = np.array(agent2_actions[:min_length])

                    # Calculate correlation between action sequences
                    correlation = np.corrcoef(actions1.flatten(), actions2.flatten())[0, 1]
                    if not np.isnan(correlation):
                        policy_diversities.append(1.0 - abs(correlation))

        policy_diversity = np.mean(policy_diversities) if policy_diversities else 0.0

        return {
            "action_diversity": action_diversity,
            "policy_diversity": policy_diversity,
            "overall_diversity": (action_diversity + policy_diversity) / 2.0,
        }

    def _calculate_stability_metrics(self, episode_results: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate stability metrics for the ensemble."""
        rewards = [ep["reward"] for ep in episode_results]
        consensus_scores = [ep["avg_consensus"] for ep in episode_results]

        # Reward stability
        reward_stability = 1.0 - (np.std(rewards) / (np.mean(rewards) + 1e-8))

        # Consensus stability
        consensus_stability = 1.0 - np.std(consensus_scores)

        # Performance consistency
        performance_consistency = 1.0 - np.std([r > np.mean(rewards) for r in rewards])

        return {
            "reward_stability": max(0.0, reward_stability),
            "consensus_stability": max(0.0, consensus_stability),
            "performance_consistency": performance_consistency,
            "overall_stability": (reward_stability + consensus_stability + performance_consistency) / 3.0,
        }

    def _calculate_diagnostics(
        self,
        agent_actions: dict[str, list[np.ndarray]],
        episode_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate detailed ensemble diagnostics."""
        diagnostics = {}

        # Agent-specific diagnostics
        agent_diagnostics = {}
        for agent_name, actions in agent_actions.items():
            if actions:
                actions_array = np.array(actions)
                agent_diagnostics[agent_name] = {
                    "action_mean": np.mean(actions_array, axis=0).tolist(),
                    "action_std": np.std(actions_array, axis=0).tolist(),
                    "action_range": (np.max(actions_array, axis=0) - np.min(actions_array, axis=0)).tolist(),
                    "action_consistency": 1.0 - np.std(actions_array, axis=0).mean(),
                }

        diagnostics["agent_diagnostics"] = agent_diagnostics

        # Ensemble diagnostics from the ensemble itself
        ensemble_diagnostics = self.ensemble.get_ensemble_diagnostics()
        diagnostics["ensemble_diagnostics"] = ensemble_diagnostics  # type: ignore

        # Weight analysis
        weight_analysis = {
            "current_weights": self.ensemble.weights.copy(),
            "weight_entropy": self._calculate_weight_entropy(),
            "weight_balance": self._calculate_weight_balance(),
        }
        diagnostics["weight_analysis"] = weight_analysis  # type: ignore

        return diagnostics

    def _calculate_weight_entropy(self) -> float:
        """Calculate entropy of agent weights (measure of diversity)."""
        weights = list(self.ensemble.weights.values())
        weights_array = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Calculate entropy
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_weight_balance(self) -> float:
        """Calculate balance of agent weights."""
        weights = list(self.ensemble.weights.values())
        mean_weight = np.mean(weights)

        if mean_weight > 0:
            balance = 1.0 - (np.std(weights) / mean_weight)
            return float(max(0.0, balance))
        return 0.0

    def _update_evaluation_history(self, results: dict[str, Any]) -> None:
        """Update evaluation history with new results."""
        # Update performance metrics
        self.evaluation_history["mean_reward"].append(results["performance"]["mean_reward"])
        self.evaluation_history["consensus"].append(results["consensus"]["mean_consensus"])
        self.evaluation_history["diversity"].append(results["diversity"]["overall_diversity"])
        self.evaluation_history["stability"].append(results["stability"]["overall_stability"])

        # Update diversity metrics
        self.diversity_metrics["action_diversity"].append(results["diversity"]["action_diversity"])
        self.diversity_metrics["policy_diversity"].append(results["diversity"]["policy_diversity"])

    def _save_evaluation_results(self, results: dict[str, Any], results_path: str) -> None:
        """Save evaluation results to file."""
        import json
        from pathlib import Path

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_serializable = convert_numpy(results)

        # Save to JSON
        results_file = Path(results_path)
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results_serializable, f, indent=2)

        self.logger.info(f"Evaluation results saved to {results_path}")

    def compare_agents(self, env: Any, num_episodes: int = 50) -> dict[str, dict[str, float]]:
        """Compare individual agents against the ensemble."""
        self.logger.info(f"Comparing agents with {num_episodes} episodes each")

        comparison_results = {}

        # Test ensemble
        ensemble_results = self._test_agent("ensemble", env, num_episodes, use_ensemble=True)
        comparison_results["ensemble"] = ensemble_results

        # Test individual agents
        for agent_name, policy in self.ensemble.policy_map.items():
            agent_results = self._test_agent(agent_name, env, num_episodes, policy=policy)
            comparison_results[agent_name] = agent_results

        return comparison_results

    def _test_agent(
        self,
        agent_name: str,
        env: Any,
        num_episodes: int,
        policy: Any | None = None,
        use_ensemble: bool = False,
    ) -> dict[str, float]:
        """Test a single agent or the ensemble."""
        rewards = []
        lengths = []

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                if use_ensemble:
                    action = self.ensemble.select_action(obs)
                elif policy is not None:
                    action, _, _ = policy.compute_single_action(obs)
                else:
                    raise ValueError(f"No policy provided for agent {agent_name}")

                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean([r > 0 for r in rewards])),
        }

    def generate_evaluation_report(self, results: dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("ENSEMBLE EVALUATION REPORT")
        report.append("=" * 60)

        # Performance summary
        perf = results["performance"]
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  Mean Reward: {perf['mean_reward']:.3f} ± {perf['std_reward']:.3f}")
        report.append(f"  Reward Range: [{perf['min_reward']:.3f}, {perf['max_reward']:.3f}]")
        report.append(f"  Success Rate: {perf['success_rate']:.1%}")
        report.append(f"  Mean Episode Length: {perf['mean_episode_length']:.1f}")

        # Consensus metrics
        cons = results["consensus"]
        report.append("\nCONSENSUS METRICS:")
        report.append(f"  Mean Consensus: {cons['mean_consensus']:.3f} ± {cons['std_consensus']:.3f}")
        report.append(f"  Consensus Stability: {cons['consensus_stability']:.3f}")
        report.append(f"  High Consensus Rate: {cons['high_consensus_rate']:.1%}")

        # Diversity metrics
        div = results["diversity"]
        report.append("\nDIVERSITY METRICS:")
        report.append(f"  Action Diversity: {div['action_diversity']:.3f}")
        report.append(f"  Policy Diversity: {div['policy_diversity']:.3f}")
        report.append(f"  Overall Diversity: {div['overall_diversity']:.3f}")

        # Stability metrics
        stab = results["stability"]
        report.append("\nSTABILITY METRICS:")
        report.append(f"  Reward Stability: {stab['reward_stability']:.3f}")
        report.append(f"  Consensus Stability: {stab['consensus_stability']:.3f}")
        report.append(f"  Performance Consistency: {stab['performance_consistency']:.3f}")
        report.append(f"  Overall Stability: {stab['overall_stability']:.3f}")

        # Agent weights
        if "diagnostics" in results and "weight_analysis" in results["diagnostics"]:
            weights = results["diagnostics"]["weight_analysis"]["current_weights"]
            report.append("\nAGENT WEIGHTS:")
            for agent, weight in weights.items():
                report.append(f"  {agent}: {weight:.3f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def get_evaluation_summary(self) -> dict[str, Any]:
        """Get a summary of all evaluations performed."""
        if not self.evaluation_history["mean_reward"]:
            return {"error": "No evaluations performed yet"}

        return {
            "num_evaluations": len(self.evaluation_history["mean_reward"]),
            "performance_trend": {
                "mean_reward": np.mean(self.evaluation_history["mean_reward"]),
                "reward_std": np.std(self.evaluation_history["mean_reward"]),
                "reward_trend": self._calculate_trend(self.evaluation_history["mean_reward"]),
            },
            "consensus_trend": {
                "mean_consensus": np.mean(self.evaluation_history["consensus"]),
                "consensus_std": np.std(self.evaluation_history["consensus"]),
                "consensus_trend": self._calculate_trend(self.evaluation_history["consensus"]),
            },
            "diversity_trend": {
                "mean_diversity": np.mean(self.evaluation_history["diversity"]),
                "diversity_std": np.std(self.evaluation_history["diversity"]),
                "diversity_trend": self._calculate_trend(self.evaluation_history["diversity"]),
            },
            "stability_trend": {
                "mean_stability": np.mean(self.evaluation_history["stability"]),
                "stability_std": np.std(self.evaluation_history["stability"]),
                "stability_trend": self._calculate_trend(self.evaluation_history["stability"]),
            },
        }

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        if slope < -0.01:
            return "declining"
        return "stable"
