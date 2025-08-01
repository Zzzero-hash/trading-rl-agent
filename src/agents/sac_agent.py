"""
SAC Agent Integration with CNN+LSTM Model

This module implements a SAC agent that uses the CNN+LSTM model as a feature extractor.
The implementation follows Ray RLlib conventions and integrates with the
custom CNN+LSTM architecture for processing financial time series data.
"""

from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from src.models.cnn_lstm import CNNLSTM


class CNNLSTMFeatureExtractor(TorchModelV2, nn.Module):
    """
    Feature extractor that uses the CNN+LSTM model for SAC.

    This class wraps the CNN+LSTM model to make it compatible with Ray RLlib's
    model interface. It processes multi-asset financial time series data
    and produces a fixed-size feature vector for the policy and value networks.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        time_steps: int = 30,
        assets: int = 100,
        features: int = 15,
        cnn_filters: Optional[list[int]] = None,
        lstm_units: Optional[list[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 64,
    ):
        """
        Initialize the CNN+LSTM feature extractor.

        Args:
            obs_space: The observation space of the environment
            action_space: The action space of the environment
            num_outputs: Number of outputs for the model
            model_config: Model configuration dictionary
            name: Name of the model
            time_steps: Number of time steps in the input sequence
            assets: Number of assets in the input
            features: Number of features per asset
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            output_dim: Dimension of the output features
        """
        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Create the CNN+LSTM model
        self.cnn_lstm = CNNLSTM(
            time_steps=time_steps,
            assets=assets,
            features=features,
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
        )

        # Store the output dimension
        self.output_dim = output_dim

    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """
        Forward pass through the feature extractor.

        Args:
            input_dict: Input dictionary containing observations
            state: Hidden state for recurrent models
            seq_lens: Sequence lengths for recurrent models

        Returns:
            Tuple of output tensor and hidden state
        """
        # Get observations from input dictionary
        obs = input_dict["obs"]

        # Process observations through CNN+LSTM model
        features = self.cnn_lstm(obs)

        return features, state


class CNNLSTMSACModel(TorchModelV2, nn.Module):
    """
    Custom SAC model that uses the CNN+LSTM model as a feature extractor.

    This model integrates the CNN+LSTM feature extractor with separate policy and
    Q-networks for SAC, following the Ray RLlib model interface.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        time_steps: int = 30,
        assets: int = 100,
        features: int = 15,
        cnn_filters: Optional[list[int]] = None,
        lstm_units: Optional[list[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 64,
        q_hiddens: Optional[list[int]] = None,
        policy_hiddens: Optional[list[int]] = None,
    ):
        """
        Initialize the CNN+LSTM SAC model.

        Args:
            obs_space: The observation space of the environment
            action_space: The action space of the environment
            num_outputs: Number of outputs for the model
            model_config: Model configuration dictionary
            name: Name of the model
            time_steps: Number of time steps in the input sequence
            assets: Number of assets in the input
            features: Number of features per asset
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            output_dim: Dimension of the output features
            q_hiddens: Hidden layers for Q-networks
            policy_hiddens: Hidden layers for policy network
        """
        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]
        if q_hiddens is None:
            q_hiddens = [256, 256]
        if policy_hiddens is None:
            policy_hiddens = [256, 256]

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Store parameters
        self.time_steps = time_steps
        self.assets = assets
        self.features = features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        # Create the feature extractor
        self.feature_extractor = CNNLSTMFeatureExtractor(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=output_dim,
            model_config=model_config,
            name=name + "_feature_extractor",
            time_steps=time_steps,
            assets=assets,
            features=features,
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
        )

        # Create policy network layers
        policy_layers = []
        prev_layer_size = output_dim
        for size in policy_hiddens:
            policy_layers.append(SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=nn.ReLU))
            prev_layer_size = size

        # Add final policy layer
        policy_layers.append(SlimFC(in_size=prev_layer_size, out_size=action_space.shape[0], activation_fn=None))
        self.policy_net = nn.Sequential(*policy_layers)

        # Create Q-network layers (two Q-networks for SAC)
        self.q_nets = nn.ModuleList()
        for _ in range(2):  # Two Q-networks for SAC
            q_layers = []
            # For Q-networks, we concatenate features with actions
            prev_layer_size = output_dim + action_space.shape[0]
            for size in q_hiddens:
                q_layers.append(SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=nn.ReLU))
                prev_layer_size = size
            # Add final Q-value layer
            q_layers.append(SlimFC(in_size=prev_layer_size, out_size=1, activation_fn=None))
            self.q_nets.append(nn.Sequential(*q_layers))

        # Create value network layers
        value_layers = []
        prev_layer_size = output_dim
        for size in q_hiddens:  # Using same size as Q-networks
            value_layers.append(SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=nn.ReLU))
            prev_layer_size = size
        # Add final value layer
        value_layers.append(SlimFC(in_size=prev_layer_size, out_size=1, activation_fn=None))
        self.value_net = nn.Sequential(*value_layers)

    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """
        Forward pass through the model.

        Args:
            input_dict: Input dictionary containing observations
            state: Hidden state for recurrent models
            seq_lens: Sequence lengths for recurrent models

        Returns:
            Tuple of output tensor and hidden state
        """
        # Extract features using the CNN+LSTM model
        features, _ = self.feature_extractor(input_dict, state, seq_lens)

        # Apply policy network
        policy_output = self.policy_net(features)

        # Return policy output and empty state (non-recurrent)
        return policy_output, []

    def get_q_values(
        self,
        features: TensorType,
        actions: TensorType,
    ) -> tuple[TensorType, TensorType]:
        """
        Get Q-values for state-action pairs.

        Args:
            features: Feature tensor from the feature extractor
            actions: Action tensor

        Returns:
            Tuple of Q-values from both Q-networks
        """
        # Concatenate features with actions for Q-networks
        q_input = torch.cat([features, actions], dim=-1)

        # Compute Q-values from both networks
        q1 = self.q_nets[0](q_input)
        q2 = self.q_nets[1](q_input)

        return q1, q2

    def get_value(self, features: TensorType) -> TensorType:
        """
        Get state value.

        Args:
            features: Feature tensor from the feature extractor

        Returns:
            State value
        """
        return self.value_net(features)

    def get_policy_outputs(self, features: TensorType) -> TensorType:
        """
        Get policy outputs.

        Args:
            features: Feature tensor from the feature extractor

        Returns:
            Policy outputs
        """
        return self.policy_net(features)


class SACAgent:
    """
    SAC Agent that uses the CNN+LSTM model for trading.

    This class provides a high-level interface for creating and using a SAC agent
    with the CNN+LSTM feature extractor for financial time series data.
    """

    def __init__(
        self,
        env: Union[str, gym.Env, type],
        time_steps: int = 30,
        assets: int = 100,
        features: int = 15,
        cnn_filters: Optional[list[int]] = None,
        lstm_units: Optional[list[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 64,
        learning_rate: float = 3e-4,
        tau: float = 0.005,
        target_entropy: Union[str, float] = "auto",
        q_hiddens: Optional[list[int]] = None,
        policy_hiddens: Optional[list[int]] = None,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        train_batch_size: int = 256,
        gamma: float = 0.99,
        n_step: int = 1,
        num_workers: int = 0,
        num_gpus: Union[int, float] = 0,
        framework: str = "torch",
        **kwargs,
    ):
        """
        Initialize the SAC agent.

        Args:
            env: The trading environment (name, class, or instance)
            time_steps: Number of time steps in the input sequence
            assets: Number of assets in the input
            features: Number of features per asset
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            output_dim: Dimension of the output features
            learning_rate: Learning rate for the optimizer
            tau: Target network update rate
            target_entropy: Target entropy for entropy regularization
            q_hiddens: Hidden layers for Q-networks
            policy_hiddens: Hidden layers for policy network
            buffer_size: Size of the replay buffer
            batch_size: Size of a batch sampled from replay buffer for training
            train_batch_size: Training batch size, if applicable
            gamma: Discount factor
            n_step: N-step learning
            num_workers: Number of rollout workers
            num_gpus: Number of GPUs to use
            framework: Deep learning framework to use
            **kwargs: Additional arguments to pass to the SAC config
        """
        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]
        if q_hiddens is None:
            q_hiddens = [256, 256]
        if policy_hiddens is None:
            policy_hiddens = [256, 256]

        # Create SAC configuration
        self.config = (
            SACConfig()
            .environment(env=env)
            .framework(framework)
            .rollouts(num_rollout_workers=num_workers)
            .resources(num_gpus=num_gpus)
            .training(
                lr=learning_rate,
                tau=tau,
                target_entropy=target_entropy,
                buffer_size=buffer_size,
                batch_size=batch_size,
                train_batch_size=train_batch_size,
                gamma=gamma,
                n_step=n_step,
                **kwargs,
            )
        )

        # Update model configuration with CNN+LSTM parameters
        model_config = {
            "custom_model": CNNLSTMSACModel,
            "custom_model_config": {
                "time_steps": time_steps,
                "assets": assets,
                "features": features,
                "cnn_filters": cnn_filters,
                "lstm_units": lstm_units,
                "dropout_rate": dropout_rate,
                "output_dim": output_dim,
                "q_hiddens": q_hiddens,
                "policy_hiddens": policy_hiddens,
            },
        }

        self.config = self.config.training(model=model_config)

        # Create the SAC algorithm
        self.algorithm = self.config.build()

    def train(self, num_iterations: int = 1) -> dict:
        """
        Train the SAC agent.

        Args:
            num_iterations: Number of training iterations

        Returns:
            Training results dictionary
        """
        results = []
        for i in range(num_iterations):
            result = self.algorithm.train()
            results.append(result)
            print(f"Training iteration {i+1}/{num_iterations}")
            print(f"Episode reward mean: {result['episode_reward_mean']}")

        return results[-1] if results else {}

    def compute_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute action for a given observation.

        Args:
            observation: Current observation
            **kwargs: Additional arguments to pass to the algorithm

        Returns:
            Computed action
        """
        return self.algorithm.compute_action(observation, **kwargs)

    def save(self, checkpoint_path: str) -> None:
        """
        Save the agent to a checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
        """
        self.algorithm.save(checkpoint_path)

    def restore(self, checkpoint_path: str) -> None:
        """
        Restore the agent from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint
        """
        self.algorithm.restore(checkpoint_path)

    def get_policy(self):
        """
        Get the policy of the agent.

        Returns:
            The policy
        """
        return self.algorithm.get_policy()


# Example usage
if __name__ == "__main__":
    # This is just an example of how to use the SACAgent
    # In practice, you would need to create a proper trading environment

    # Example parameters
    TIME_STEPS = 30
    ASSETS = 100
    FEATURES = 15

    # Note: This is just an example and won't run without a proper environment
    # env = YourTradingEnvironment()  # Replace with actual trading environment

    # Create the SAC agent
    # agent = SACAgent(
    #     env=env,
    #     time_steps=TIME_STEPS,
    #     assets=ASSETS,
    #     features=FEATURES,
    #     output_dim=64,
    #     learning_rate=3e-4,
    #     buffer_size=100000,
    #     batch_size=256,
    #     train_batch_size=256,
    #     gamma=0.99,
    #     tau=0.005,
    #     num_workers=0,
    #     num_gpus=0,
    #     framework="torch",
    # )

    print("SAC Agent with CNN+LSTM model implemented successfully!")
