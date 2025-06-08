"""
Ensemble Agent for Robust Trading Decisions

Combines multiple RL agents to make more robust trading decisions.
Features:
1. Multiple model voting
2. Dynamic weight adjustment based on performance
3. Diversity-based model selection
4. Risk-aware ensemble methods
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import yaml
from pathlib import Path
import logging
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from .configs import EnsembleConfig, SACConfig, TD3Config

from .sac_agent import SACAgent
from .td3_agent import TD3Agent


class EnsembleAgent:
    """
    Ensemble Agent combining multiple RL agents for robust trading.
    
    Features:
    - Multi-agent voting mechanisms
    - Performance-based weight adjustment
    - Diversity promotion
    - Risk-aware decision making
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 1,
                 config: Optional[Union[str, Dict, EnsembleConfig]] = None,
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Ensemble parameters
        self.ensemble_method = self.config.get("ensemble_method", "weighted_average")
        self.weight_update_frequency = self.config.get("weight_update_frequency", 1000)
        self.diversity_penalty = self.config.get("diversity_penalty", 0.1)
        self.min_weight = self.config.get("min_weight", 0.1)
        self.performance_window = self.config.get("performance_window", 100)
        
        # Initialize agents
        self.agents = {}
        self.agent_weights = {}
        self.agent_performance = defaultdict(lambda: deque(maxlen=self.performance_window))
        self.agent_predictions = defaultdict(list)
        
        # Create agents based on configuration
        self._initialize_agents()
        
        # Training metrics
        self.step_count = 0
        self.ensemble_metrics = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config: Optional[Union[str, Dict, EnsembleConfig]]) -> Dict:
        """Load configuration from dataclass, file, or dict."""
        if config is None:
            return {
                "agents": {
                    "sac": {"enabled": True, "config": None},
                    "td3": {"enabled": True, "config": None}
                },
                "ensemble_method": "weighted_average",  # Options: weighted_average, voting, risk_parity
                "weight_update_frequency": 1000,
                "diversity_penalty": 0.1,
                "min_weight": 0.1,
                "performance_window": 100,
                "risk_adjustment": True
            }
        elif isinstance(config, str):
            with open(config, 'r') as f:
                return yaml.safe_load(f) or {}
        elif is_dataclass(config):
            return asdict(config)
        else:
            return config
    
    def _initialize_agents(self):
        """Initialize individual agents based on configuration."""
        agent_configs = self.config.get("agents", {})
        
        # Initialize SAC agent
        if agent_configs.get("sac", {}).get("enabled", True):
            sac_config = agent_configs["sac"].get("config", None)
            self.agents["sac"] = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=sac_config,
                device=str(self.device)
            )
            self.agent_weights["sac"] = 1.0
            
        # Initialize TD3 agent
        if agent_configs.get("td3", {}).get("enabled", True):
            td3_config = agent_configs["td3"].get("config", None)
            self.agents["td3"] = TD3Agent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=td3_config,
                device=str(self.device)
            )
            self.agent_weights["td3"] = 1.0
        
        # Normalize weights
        self._normalize_weights()
        
        self.logger.info(f"Initialized ensemble with agents: {list(self.agents.keys())}")
    
    def _normalize_weights(self):
        """Normalize agent weights to sum to 1."""
        if not self.agent_weights:
            return
            
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_name in self.agent_weights:
                self.agent_weights[agent_name] /= total_weight
                # Ensure minimum weight
                self.agent_weights[agent_name] = max(
                    self.agent_weights[agent_name], 
                    self.min_weight / len(self.agent_weights)
                )
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using ensemble method."""
        if not self.agents:
            raise ValueError("No agents available in ensemble")
        
        # Get predictions from all agents
        agent_actions = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, evaluate=evaluate)
                agent_actions[agent_name] = action
                # Store for diversity calculation
                self.agent_predictions[agent_name].append(action)
            else:
                # Fallback for different agent interfaces
                action = agent.select_action(state, add_noise=not evaluate)
                agent_actions[agent_name] = action
                self.agent_predictions[agent_name].append(action)
        
        # Combine actions using ensemble method
        ensemble_action = self._combine_actions(agent_actions, state)
        
        return ensemble_action
    
    def _combine_actions(self, agent_actions: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Combine agent actions using the specified ensemble method."""
        if self.ensemble_method == "weighted_average":
            return self._weighted_average(agent_actions)
        elif self.ensemble_method == "voting":
            return self._majority_voting(agent_actions)
        elif self.ensemble_method == "risk_parity":
            return self._risk_parity_combination(agent_actions, state)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_average(self, agent_actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine actions using weighted average."""
        weighted_action = np.zeros(self.action_dim)
        
        for agent_name, action in agent_actions.items():
            weight = self.agent_weights.get(agent_name, 0.0)
            weighted_action += weight * action
            
        return weighted_action
    
    def _majority_voting(self, agent_actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine actions using majority voting (for discrete-like decisions)."""
        # Convert continuous actions to discrete votes
        votes = []
        for agent_name, action in agent_actions.items():
            # Simple thresholding: positive = buy, negative = sell, near-zero = hold
            if action[0] > 0.1:
                vote = 1  # Buy
            elif action[0] < -0.1:
                vote = -1  # Sell
            else:
                vote = 0  # Hold
            votes.append(vote)
        
        # Majority vote
        majority_vote = np.sign(np.sum(votes))
        
        # Convert back to continuous action
        if majority_vote > 0:
            return np.array([0.5])  # Moderate buy
        elif majority_vote < 0:
            return np.array([-0.5])  # Moderate sell
        else:
            return np.array([0.0])  # Hold
    
    def _risk_parity_combination(self, agent_actions: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Combine actions using risk parity approach."""
        # Calculate action volatility for each agent (simple proxy for risk)
        risk_adjusted_weights = {}
        
        for agent_name, action in agent_actions.items():
            # Use recent prediction variance as risk measure
            recent_predictions = list(self.agent_predictions[agent_name])[-20:]  # Last 20 predictions
            if len(recent_predictions) > 1:
                action_variance = np.var(recent_predictions)
                risk_weight = 1.0 / (action_variance + 1e-6)  # Inverse variance weighting
            else:
                risk_weight = 1.0
                
            risk_adjusted_weights[agent_name] = risk_weight
        
        # Normalize risk-adjusted weights
        total_risk_weight = sum(risk_adjusted_weights.values())
        if total_risk_weight > 0:
            for agent_name in risk_adjusted_weights:
                risk_adjusted_weights[agent_name] /= total_risk_weight
        
        # Combine actions
        ensemble_action = np.zeros(self.action_dim)
        for agent_name, action in agent_actions.items():
            weight = risk_adjusted_weights.get(agent_name, 0.0)
            ensemble_action += weight * action
            
        return ensemble_action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in all agents."""
        for agent in self.agents.values():
            agent.store_experience(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, Any]:
        """Update all agents and ensemble weights."""
        ensemble_metrics = {}
        
        # Update individual agents
        for agent_name, agent in self.agents.items():
            agent_metrics = agent.update()
            if agent_metrics:
                ensemble_metrics[f"{agent_name}_metrics"] = agent_metrics
                
                # Store performance for weight updates
                if "critic_loss" in agent_metrics:
                    # Use negative loss as performance measure (higher is better)
                    performance = -agent_metrics["critic_loss"]
                    self.agent_performance[agent_name].append(performance)
        
        # Update ensemble weights periodically
        if self.step_count % self.weight_update_frequency == 0:
            self._update_weights()
            ensemble_metrics["weights"] = dict(self.agent_weights)
        
        # Calculate ensemble diversity
        diversity = self._calculate_diversity()
        ensemble_metrics["diversity"] = diversity
        
        self.step_count += 1
        return ensemble_metrics
    
    def _update_weights(self):
        """Update agent weights based on recent performance."""
        if not self.agent_performance:
            return
        
        # Calculate average performance for each agent
        avg_performance = {}
        for agent_name, performance_history in self.agent_performance.items():
            if performance_history:
                avg_performance[agent_name] = np.mean(list(performance_history))
            else:
                avg_performance[agent_name] = 0.0
        
        # Update weights based on performance (softmax transformation)
        if avg_performance:
            performance_values = np.array(list(avg_performance.values()))
            # Softmax with temperature for smooth updates
            temperature = 1.0
            exp_values = np.exp(performance_values / temperature)
            softmax_weights = exp_values / np.sum(exp_values)
            
            # Update weights
            for i, agent_name in enumerate(avg_performance.keys()):
                self.agent_weights[agent_name] = softmax_weights[i]
        
        # Apply diversity penalty
        self._apply_diversity_penalty()
        
        # Ensure minimum weights
        self._normalize_weights()
        
        self.logger.info(f"Updated ensemble weights: {self.agent_weights}")
    
    def _apply_diversity_penalty(self):
        """Apply penalty to reduce correlation between agents."""
        if len(self.agent_predictions) < 2:
            return
        
        # Calculate pairwise correlations
        agent_names = list(self.agent_predictions.keys())
        correlations = {}
        
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names[i+1:], i+1):
                if (len(self.agent_predictions[agent1]) > 10 and 
                    len(self.agent_predictions[agent2]) > 10):
                    
                    pred1 = np.array(self.agent_predictions[agent1][-50:])  # Last 50 predictions
                    pred2 = np.array(self.agent_predictions[agent2][-50:])
                    
                    # Calculate correlation
                    if len(pred1) == len(pred2) and len(pred1) > 1:
                        correlation = np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1]
                        if not np.isnan(correlation):
                            correlations[(agent1, agent2)] = abs(correlation)
        
        # Apply penalty to highly correlated agents
        for (agent1, agent2), correlation in correlations.items():
            if correlation > 0.8:  # High correlation threshold
                penalty = self.diversity_penalty * correlation
                self.agent_weights[agent1] *= (1 - penalty)
                self.agent_weights[agent2] *= (1 - penalty)
    
    def _calculate_diversity(self) -> float:
        """Calculate ensemble diversity measure."""
        if len(self.agent_predictions) < 2:
            return 0.0
        
        # Calculate average pairwise disagreement
        agent_names = list(self.agent_predictions.keys())
        disagreements = []
        
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names[i+1:], i+1):
                if (len(self.agent_predictions[agent1]) > 0 and 
                    len(self.agent_predictions[agent2]) > 0):
                    
                    # Get last prediction from each agent
                    pred1 = self.agent_predictions[agent1][-1]
                    pred2 = self.agent_predictions[agent2][-1]
                    
                    # Calculate disagreement (L2 distance)
                    disagreement = np.linalg.norm(pred1 - pred2)
                    disagreements.append(disagreement)
        
        return np.mean(disagreements) if disagreements else 0.0
    
    def save(self, filepath: str):
        """Save ensemble state."""
        ensemble_state = {
            'config': self.config,
            'agent_weights': self.agent_weights,
            'step_count': self.step_count,
            'ensemble_method': self.ensemble_method
        }
        
        # Save individual agents
        save_path = Path(filepath)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            agent_path = save_path / f"{agent_name}_agent.pt"
            agent.save(str(agent_path))
        
        # Save ensemble metadata
        ensemble_path = save_path / "ensemble_state.pt"
        torch.save(ensemble_state, ensemble_path)
    
    def load(self, filepath: str):
        """Load ensemble state."""
        load_path = Path(filepath)
        
        # Load ensemble metadata
        ensemble_path = load_path / "ensemble_state.pt"
        if ensemble_path.exists():
            ensemble_state = torch.load(ensemble_path, map_location=self.device)
            self.agent_weights = ensemble_state.get('agent_weights', {})
            self.step_count = ensemble_state.get('step_count', 0)
            self.ensemble_method = ensemble_state.get('ensemble_method', self.ensemble_method)
        
        # Load individual agents
        for agent_name, agent in self.agents.items():
            agent_path = load_path / f"{agent_name}_agent.pt"
            if agent_path.exists():
                agent.load(str(agent_path))
                self.logger.info(f"Loaded {agent_name} agent from {agent_path}")


# Configuration example
EXAMPLE_CONFIG = {
    "agents": {
        "sac": {
            "enabled": True,
            "config": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "hidden_dims": [256, 256]
            }
        },
        "td3": {
            "enabled": True,
            "config": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "hidden_dims": [256, 256],
                "policy_delay": 2
            }
        }
    },
    "ensemble_method": "weighted_average",  # weighted_average, voting, risk_parity
    "weight_update_frequency": 1000,
    "diversity_penalty": 0.1,
    "min_weight": 0.1,
    "performance_window": 100,
    "risk_adjustment": True
}


if __name__ == "__main__":
    # Example usage
    state_dim = 100
    action_dim = 1
    
    ensemble = EnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=EXAMPLE_CONFIG,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test action selection
    dummy_state = np.random.randn(state_dim)
    action = ensemble.select_action(dummy_state)
    print(f"Ensemble action: {action[0]:.3f}")
    print(f"Agent weights: {ensemble.agent_weights}")
    
    # Test training
    for i in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        ensemble.store_experience(state, action, reward, next_state, done)
        
        if i % 100 == 0:
            metrics = ensemble.update()
            if metrics:
                print(f"Step {i} - Diversity: {metrics.get('diversity', 0):.3f}")
                if "weights" in metrics:
                    print(f"Updated weights: {metrics['weights']}")
