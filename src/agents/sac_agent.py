"""
Soft Actor-Critic (SAC) Agent Implementation for Trading RL

SAC is particularly well-suited for trading applications because:
1. Continuous action space - ideal for position sizing
2. Maximum entropy framework - encourages exploration
3. Off-policy learning - sample efficient
4. Stable training - less prone to catastrophic forgetting

TODO: provide an RLlib-based baseline version of SAC for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Union
import yaml
from pathlib import Path
from dataclasses import asdict, is_dataclass
from .configs import SACConfig


class Actor(nn.Module):
    """SAC Actor Network with reparameterization trick."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Build actor network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Constrain log_std to reasonable range
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # rsample for reparameterization
        
        # Apply tanh squashing for bounded actions
        action = torch.tanh(x_t)
        
        # Calculate log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """SAC Twin Critic Networks (Q1 and Q2)."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        # Q1 network
        q1_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(current_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            current_dim = hidden_dim
        q1_layers.append(nn.Linear(current_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 network
        q2_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(current_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            current_dim = hidden_dim
        q2_layers.append(nn.Linear(current_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q1 and Q2 values."""
        x = torch.cat([state, action], dim=1)
        q1_value = self.q1(x)
        q2_value = self.q2(x)
        return q1_value, q2_value


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic Agent for Continuous Trading Actions.
    
    Features:
    - Continuous position sizing (-1 to +1)
    - Maximum entropy framework for exploration
    - Twin critic networks for stable training
    - Automatic temperature tuning
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 1,  # Position size
                 config: Optional[Union[str, Dict, SACConfig]] = None,
                 device: str = "cpu"):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Hyperparameters
        self.lr = self.config.get("learning_rate", 3e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.tau = self.config.get("tau", 0.005)
        self.batch_size = self.config.get("batch_size", 256)
        self.buffer_capacity = self.config.get("buffer_capacity", 1000000)
        self.hidden_dims = self.config.get("hidden_dims", [256, 256])
        
        # Automatic temperature tuning
        self.automatic_entropy_tuning = self.config.get("automatic_entropy_tuning", True)
        self.target_entropy = self.config.get("target_entropy", -action_dim)
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.hidden_dims).to(self.device)
        
        # Copy critic parameters to target
        self._hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Temperature parameter
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = self.config.get("alpha", 0.2)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Training metrics
        self.training_step = 0
        
    def _load_config(self, config: Optional[Union[str, Dict, SACConfig]]) -> Dict:
        """Load configuration from dataclass, file, or dict."""
        if config is None:
            # Default configuration
            return {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "buffer_capacity": 1000000,
                "hidden_dims": [256, 256],
                "automatic_entropy_tuning": True,
                "target_entropy": -1.0  # -action_dim
            }
        elif isinstance(config, str):
            with open(config, 'r') as f:
                return yaml.safe_load(f) or {}
        elif is_dataclass(config):
            return asdict(config)
        else:
            return config
            
    def _hard_update(self, target, source):
        """Hard update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def _soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action for given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            # Deterministic action for evaluation
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            # Stochastic action for training
            action, _ = self.actor.sample(state)
            
        return action.cpu().data.numpy().flatten()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update SAC networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        
        # Current alpha value
        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
            
        # Update Critic Networks
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            target_q = reward + (1 - done.float()) * self.gamma * q_next
        
        q1_current, q2_current = self.critic(state, action)
        critic_loss = F.mse_loss(q1_current, target_q) + F.mse_loss(q2_current, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor Network
        new_action, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = 0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic_target, self.critic, self.tau)
        
        self.training_step += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.automatic_entropy_tuning else 0,
            "alpha": alpha.item() if self.automatic_entropy_tuning else self.alpha,
            "mean_q1": q1_current.mean().item(),
            "mean_q2": q2_current.mean().item(),
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
        self.training_step = checkpoint['training_step']


# Configuration example
EXAMPLE_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 1000000,
    "hidden_dims": [256, 256],
    "automatic_entropy_tuning": True,
    "target_entropy": -1.0,
    "alpha": 0.2  # Used only if automatic_entropy_tuning is False
}


if __name__ == "__main__":
    # Example usage
    state_dim = 100  # Example: flattened market features from CNN-LSTM
    action_dim = 1   # Position size (-1 to +1)
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=EXAMPLE_CONFIG,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test action selection
    dummy_state = np.random.randn(state_dim)
    action = agent.select_action(dummy_state)
    print(f"Selected action (position size): {action[0]:.3f}")
    
    # Test training step
    for i in range(10):
        # Dummy experience
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    # Update after collecting enough experiences
    if len(agent.replay_buffer) >= agent.batch_size:
        metrics = agent.update()
        print("Training metrics:", metrics)
