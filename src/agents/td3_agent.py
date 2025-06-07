"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

TD3 improvements over DDPG:
1. Twin Critic Networks - reduces overestimation bias
2. Delayed Policy Updates - update policy less frequently than critics
3. Target Policy Smoothing - add noise to target actions for regularization

Ideal for trading applications requiring stable continuous control.
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
import copy


class Actor(nn.Module):
    """TD3 Actor Network for deterministic policy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action."""
        return self.network(state)


class Critic(nn.Module):
    """TD3 Twin Critic Networks (Q1 and Q2)."""
    
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
        
        # Q2 network (twin)
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
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q1 only (used in actor loss)."""
        x = torch.cat([state, action], dim=1)
        return self.q1(x)


class ReplayBuffer:
    """Experience replay buffer for TD3."""
    
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


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient Agent.
    
    Features:
    - Twin critic networks to reduce overestimation
    - Delayed policy updates for stability
    - Target policy smoothing for regularization
    - Continuous action space for position sizing
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 1,
                 config: Optional[Union[str, Dict]] = None,
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
        
        # TD3 specific parameters
        self.policy_delay = self.config.get("policy_delay", 2)  # Delayed policy updates
        self.target_noise = self.config.get("target_noise", 0.2)  # Target policy smoothing noise
        self.noise_clip = self.config.get("noise_clip", 0.5)  # Noise clipping range
        self.exploration_noise = self.config.get("exploration_noise", 0.1)  # Exploration noise
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = Critic(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Training counters
        self.training_step = 0
        self.total_iterations = 0
        
    def _load_config(self, config: Optional[Union[str, Dict]]) -> Dict:
        """Load configuration from file or dict."""
        if config is None:
            return {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "buffer_capacity": 1000000,
                "hidden_dims": [256, 256],
                "policy_delay": 2,
                "target_noise": 0.2,
                "noise_clip": 0.5,
                "exploration_noise": 0.1
            }
        elif isinstance(config, str):
            with open(config, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            return config
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action for given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add exploration noise during training
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
            
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update TD3 networks."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.total_iterations += 1
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).unsqueeze(1).to(self.device)
        
        # Update Critic Networks
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(action) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            
            # Compute target Q-values (take minimum to reduce overestimation)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done.float()) * self.gamma * q_next
        
        # Current Q-values
        q1_current, q2_current = self.critic(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(q1_current, target_q) + F.mse_loss(q2_current, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        metrics = {
            "critic_loss": critic_loss.item(),
            "mean_q1": q1_current.mean().item(),
            "mean_q2": q2_current.mean().item(),
            "target_q_mean": target_q.mean().item()
        }
        
        # Delayed policy updates
        if self.total_iterations % self.policy_delay == 0:
            # Actor loss (maximize Q1)
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.critic_target, self.critic, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)
            
            metrics.update({
                "actor_loss": actor_loss.item(),
                "policy_update": True
            })
        else:
            metrics.update({
                "actor_loss": 0.0,
                "policy_update": False
            })
        
        self.training_step += 1
        return metrics
    
    def _soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'total_iterations': self.total_iterations,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.total_iterations = checkpoint['total_iterations']


# Configuration example
EXAMPLE_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_capacity": 1000000,
    "hidden_dims": [256, 256],
    "policy_delay": 2,        # Update policy every 2 critic updates
    "target_noise": 0.2,      # Noise added to target actions
    "noise_clip": 0.5,        # Clipping range for target noise
    "exploration_noise": 0.1   # Exploration noise during training
}


if __name__ == "__main__":
    # Example usage
    state_dim = 100  # Example: flattened market features
    action_dim = 1   # Position size (-1 to +1)
    
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=EXAMPLE_CONFIG,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test action selection
    dummy_state = np.random.randn(state_dim)
    action = agent.select_action(dummy_state)
    print(f"Selected action (position size): {action[0]:.3f}")
    
    # Test training
    for i in range(500):  # Collect some experiences
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    # Update networks
    for i in range(10):
        metrics = agent.update()
        if metrics:
            print(f"Step {i+1} - Critic Loss: {metrics['critic_loss']:.4f}, "
                  f"Actor Loss: {metrics['actor_loss']:.4f}, "
                  f"Policy Update: {metrics['policy_update']}")
