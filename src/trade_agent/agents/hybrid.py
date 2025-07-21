"""
Hybrid Agent - Integration of CNN+LSTM with Reinforcement Learning.

This module provides a hybrid agent that combines:
- CNN+LSTM for pattern recognition and feature extraction
- RL agents (PPO, SAC) for decision making
- Ensemble methods for improved performance
"""

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from src.trade_agent.models.cnn_lstm import CNNLSTMModel


class HybridAgent(nn.Module):
    """
    Hybrid agent that combines CNN+LSTM with RL for trading decisions.

    Architecture:
    1. CNN+LSTM processes market data and extracts features
    2. RL agent makes trading decisions based on extracted features
    3. Ensemble methods combine predictions for robustness
    """

    def __init__(
        self,
        cnn_lstm_model: CNNLSTMModel | None = None,
        state_dim: int = 50,
        action_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        device: str = "auto",
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # CNN+LSTM model for feature extraction
        if cnn_lstm_model is None:
            self.cnn_lstm_model = CNNLSTMModel(
                input_dim=10,  # OHLCV + technical indicators
                output_dim=hidden_dim,
            )
        else:
            self.cnn_lstm_model = cnn_lstm_model

        # Feature fusion layer
        self.feature_fusion = nn.Linear(self.cnn_lstm_model.output_dim + state_dim, hidden_dim)

        # Policy network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Output actions in [-1, 1]
        )

        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)

        # Move to device
        self.to(self.device)

        # Training state
        self.training_mode = True
        self.epsilon = 0.1  # For exploration

        self.logger.info(f"Hybrid agent initialized on {self.device}")

    def forward(
        self,
        state: torch.Tensor,
        market_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid agent.

        Args:
            state: RL state tensor [batch_size, state_dim]
            market_data: Market data tensor [batch_size, seq_len, features] or None

        Returns:
            actions: Action probabilities [batch_size, action_dim]
            value: State value [batch_size, 1]
        """
        batch_size = state.shape[0]

        # Extract features from CNN+LSTM if market data provided
        if market_data is not None and self.cnn_lstm_model is not None:
            cnn_lstm_features = self.cnn_lstm_model(market_data)
        else:
            # Use zero features if no market data or no model
            cnn_lstm_features = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # Concatenate RL state with CNN+LSTM features
        combined_features = torch.cat([state, cnn_lstm_features], dim=1)

        # Fuse features
        fused_features = f.relu(self.feature_fusion(combined_features))

        # Get policy and value
        actions = self.policy_net(fused_features)
        value = self.value_net(fused_features)

        return actions, value

    def select_action(
        self,
        state: torch.Tensor,
        market_data: torch.Tensor | None = None,
        evaluate: bool = False,
    ) -> np.ndarray[np.int64, np.dtype[np.int64]]:
        """
        Select action using the hybrid agent.

        Args:
            state: Current state
            market_data: Market data for CNN+LSTM processing
            evaluate: Whether in evaluation mode (no exploration)

        Returns:
            action: Selected action
        """
        self.eval()

        with torch.no_grad():
            # Ensure tensors are on correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            state = state.to(self.device)

            if market_data is not None and not isinstance(market_data, torch.Tensor):
                market_data = torch.FloatTensor(market_data)
                market_data = market_data.to(self.device)

            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if market_data is not None and market_data.dim() == 2:
                market_data = market_data.unsqueeze(0)

            # Get action probabilities
            actions, _ = self.forward(state, market_data)

            if not evaluate and self.training_mode:
                # Add exploration noise
                if random.random() < self.epsilon:
                    # Random action
                    action: int = random.randint(0, self.action_dim - 1)
                else:
                    # Greedy action with some noise
                    action_probs = f.softmax(actions, dim=1)
                    action = int(torch.multinomial(action_probs, 1).item())
            else:
                # Greedy action
                action = int(torch.argmax(actions, dim=1).item())

        return np.array([action])

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Update the hybrid agent using a batch of experience.

        Args:
            batch: Dictionary containing:
                - states: [batch_size, state_dim]
                - market_data: [batch_size, seq_len, features] or None
                - actions: [batch_size, 1]
                - rewards: [batch_size, 1]
                - next_states: [batch_size, state_dim]
                - next_market_data: [batch_size, seq_len, features] or None
                - dones: [batch_size, 1]

        Returns:
            loss_dict: Dictionary containing loss values
        """
        self.train()

        # Extract batch data
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        market_data = batch.get("market_data")
        next_market_data = batch.get("next_market_data")

        if market_data is not None:
            market_data = market_data.to(self.device)
        if next_market_data is not None:
            next_market_data = next_market_data.to(self.device)

        # Current Q-values
        current_actions, current_values = self.forward(states, market_data)

        # Next Q-values
        with torch.no_grad():
            _, next_values = self.forward(next_states, next_market_data)

        # Compute losses
        policy_loss: torch.Tensor = self._compute_policy_loss(
            current_actions, actions, rewards, current_values, next_values, dones
        )
        value_loss: torch.Tensor = self._compute_value_loss(current_values, rewards, next_values, dones)

        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": (policy_loss + value_loss).item(),
        }

    def _compute_policy_loss(
        self,
        current_actions: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        current_values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy loss using advantage estimation."""
        # Compute advantage
        gamma = 0.99
        advantages = rewards + gamma * next_values * (1 - dones) - current_values

        # Policy gradient loss
        action_probs = f.softmax(current_actions, dim=1)
        selected_action_probs = action_probs.gather(1, actions)
        log_probs = torch.log(selected_action_probs + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()

        # Add entropy regularization for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        total_loss: torch.Tensor = policy_loss - 0.01 * entropy
        return total_loss

    def _compute_value_loss(
        self,
        current_values: torch.Tensor,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value loss (critic)."""
        gamma = 0.99
        target_values = rewards + gamma * next_values * (1 - dones)
        return f.mse_loss(current_values, target_values.detach())

    def save(self, path: str) -> None:
        """Save agent's state dicts."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "cnn_lstm_model_state_dict": (self.cnn_lstm_model.state_dict() if self.cnn_lstm_model else None),
            },
            path,
        )
        self.logger.info(f"Hybrid agent saved to {path}")

    def load(self, path: str) -> None:
        """Load the hybrid agent."""
        checkpoint = torch.load(path, map_location=self.device)  # nosec

        self.load_state_dict(checkpoint["model_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

    def set_training_mode(self, training: bool) -> None:
        """Set training mode."""
        self.training_mode = training
        if training:
            self.train()
        else:
            self.eval()

    def set_epsilon(self, epsilon: float) -> None:
        """Set exploration epsilon."""
        self.epsilon = epsilon


class EnsembleHybridAgent:
    """Ensemble of hybrid agents for improved robustness."""

    def __init__(self, num_agents: int = 3, **agent_kwargs: Any) -> None:
        self.agents = [HybridAgent(**agent_kwargs) for _ in range(num_agents)]
        self.num_agents = num_agents
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Ensemble of {num_agents} hybrid agents initialized.")

    def select_action(
        self,
        state: torch.Tensor,
        market_data: torch.Tensor | None = None,
        evaluate: bool = False,
    ) -> np.ndarray[np.int64, np.dtype[np.int64]]:
        """Select action using ensemble voting."""
        actions: list[int] = []

        for agent in self.agents:
            action = agent.select_action(state, market_data, evaluate)
            actions.append(int(action[0]))  # Extract scalar action

        # Majority voting
        action_counts = np.bincount(actions, minlength=3)  # Assuming 3 actions
        ensemble_action = np.argmax(action_counts)

        return np.array([ensemble_action])

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update all agents in the ensemble."""
        total_losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        for agent in self.agents:
            losses = agent.update(batch)
            for key in total_losses:
                total_losses[key] += losses[key]

        # Average losses
        for key in total_losses:
            if self.num_agents > 0:
                total_losses[key] /= self.num_agents
        return total_losses

    def save(self, path: str) -> None:
        """Save the ensemble."""
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent_{i}.pth"
            agent.save(agent_path)
        self.logger.info(f"Ensemble saved to {path}")

    def load(self, path: str) -> None:
        """Load the ensemble."""
        for i, agent in enumerate(self.agents):
            agent_path = f"{path}_agent_{i}.pth"
            agent.load(agent_path)
        self.logger.info(f"Ensemble loaded from {path}")
