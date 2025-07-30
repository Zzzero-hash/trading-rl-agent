"""
PPO Agent Integration with CNN+LSTM Model

This module implements a PPO agent that uses the CNN+LSTM model as a feature extractor.
The implementation follows Stable-Baselines3 conventions and integrates with the
custom CNN+LSTM architecture for processing financial time series data.
"""

from typing import Optional, Union

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from src.models.cnn_lstm import CNNLSTM


class CNNLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses the CNN+LSTM model as a feature extractor for PPO.

    This class wraps the CNN+LSTM model to make it compatible with Stable-Baselines3
    feature extractor interface. It processes multi-asset financial time series data
    and produces a fixed-size feature vector for the policy network.
    """

    def __init__(
        self,
        observation_space: gym.Space,
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
            observation_space: The observation space of the environment
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

        # Calculate the output dimension of the CNN+LSTM model
        # This should match the output_dim of the CNNLSTM model
        super().__init__(
            observation_space, features_dim=output_dim
        )

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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Input tensor of shape (batch_size, time_steps, assets, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.cnn_lstm(observations)


class CNNLSTMPolicy(ActorCriticPolicy):
    """
    Custom policy that uses the CNN+LSTM model as a feature extractor for PPO.

    This policy integrates the CNN+LSTM feature extractor with separate policy and
    value networks, following the Stable-Baselines3 policy interface.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        time_steps: int = 30,
        assets: int = 100,
        features: int = 15,
        cnn_filters: Optional[list[int]] = None,
        lstm_units: Optional[list[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 64,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CNNLSTMFeaturesExtractor,
        features_extractor_kwargs: Optional[dict] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the CNN+LSTM policy.

        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
            lr_schedule: Learning rate schedule
            time_steps: Number of time steps in the input sequence
            assets: Number of assets in the input
            features: Number of features per asset
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            output_dim: Dimension of the output features
            net_arch: Specification of the policy and value networks
            activation_fn: Activation function
            ortho_init: Whether to use orthogonal initialization
            use_sde: Whether to use generalized State Dependent Exploration
            log_std_init: Initial value for the log standard deviation
            full_std: Whether to use (n_features x n_actions) parameters for the std
            use_expln: Use expln activation instead of log for fixed std
            squash_output: Whether to squash the output using a tanh transformation
            features_extractor_class: Features extractor class
            features_extractor_kwargs: Keyword arguments to pass to the features extractor
            share_features_extractor: If True, the features extractor is shared between policy and value networks
            normalize_images: Whether to normalize images or not
            optimizer_class: The optimizer to use
            optimizer_kwargs: Additional keyword arguments to pass to the optimizer
        """
        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]

        # Set default features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # Add CNN+LSTM specific parameters to features extractor kwargs
        features_extractor_kwargs.update({
            "time_steps": time_steps,
            "assets": assets,
            "features": features,
            "cnn_filters": cnn_filters,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "output_dim": output_dim,
        })

        # Call parent constructor
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class PPOAgent:
    """
    PPO Agent that uses the CNN+LSTM model for trading.

    This class provides a high-level interface for creating and using a PPO agent
    with the CNN+LSTM feature extractor for financial time series data.
    """

    def __init__(
        self,
        env: gym.Env,
        time_steps: int = 30,
        assets: int = 100,
        features: int = 15,
        cnn_filters: Optional[list[int]] = None,
        lstm_units: Optional[list[int]] = None,
        dropout_rate: float = 0.2,
        output_dim: int = 64,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        Initialize the PPO agent.

        Args:
            env: The trading environment
            time_steps: Number of time steps in the input sequence
            assets: Number of assets in the input
            features: Number of features per asset
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            output_dim: Dimension of the output features
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            clip_range: Clipping parameter, it can be a function of the current progress
            clip_range_vf: Clipping parameter for the value function
            normalize_advantage: Whether to normalize or not the advantage
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: The maximum value for the gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            target_kl: Limit the KL divergence between updates
            tensorboard_log: The log location for tensorboard
            policy_kwargs: Additional arguments to be passed to the policy on creation
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
            seed: Seed for the pseudo random generators
            device: Device (cpu, cuda, ...) on which the code should be run
            _init_setup_model: Whether or not to build the network at the creation of the instance
        """
        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]

        # Set default policy kwargs
        if policy_kwargs is None:
            policy_kwargs = {}

        # Add CNN+LSTM specific parameters to policy kwargs
        policy_kwargs.update({
            "time_steps": time_steps,
            "assets": assets,
            "features": features,
            "cnn_filters": cnn_filters,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "output_dim": output_dim,
        })

        # Create the PPO model
        self.model = PPO(
            policy=CNNLSTMPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1,
              tb_log_name: str = "PPO", reset_num_timesteps: bool = True,
              progress_bar: bool = False):
        """
        Train the PPO agent.

        Args:
            total_timesteps: Number of training timesteps
            callback: Callback function or list of callback functions
            log_interval: The number of timesteps before logging
            tb_log_name: Name of the tensorboard log
            reset_num_timesteps: Whether to reset or not the number of timesteps
            progress_bar: Display a progress bar using tqdm and rich
        """
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Get action from the agent.

        Args:
            observation: Current observation
            state: Hidden state for recurrent policies
            episode_start: Whether the observation is the first of the episode
            deterministic: Whether to use deterministic actions or not

        Returns:
            action: Action predicted by the agent
            state: Hidden state for recurrent policies
        """
        return self.model.predict(observation, state, episode_start, deterministic)

    def save(self, path: str, exclude=None, include=None):
        """
        Save the model to a file.

        Args:
            path: Path to save the model
            exclude: List of parameters to exclude from the saved model
            include: List of parameters to include in the saved model
        """
        self.model.save(path, exclude, include)

    def load(self, path: str, env=None, device="auto", custom_objects=None,
             print_system_info=False, force_reset=True, **kwargs):
        """
        Load the model from a file.

        Args:
            path: Path to load the model from
            env: Environment to use for the loaded model
            device: Device to use for the loaded model
            custom_objects: Dictionary of objects to replace after loading
            print_system_info: Whether to print system info from the saved model
            force_reset: Whether to reset the environment attributes
            **kwargs: Additional keyword arguments

        Returns:
            PPOAgent: Loaded agent
        """
        self.model = PPO.load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs
        )
        return self


# Example usage
if __name__ == "__main__":
    # This is just an example of how to use the PPOAgent
    # In practice, you would need to create a proper trading environment

    # Example parameters
    TIME_STEPS = 30
    ASSETS = 100
    FEATURES = 15

    # Note: This is just an example and won't run without a proper environment
    # env = YourTradingEnvironment()  # Replace with actual trading environment

    # Create the PPO agent
    # agent = PPOAgent(
    #     env=env,
    #     time_steps=TIME_STEPS,
    #     assets=ASSETS,
    #     features=FEATURES,
    #     output_dim=64,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     verbose=1,
    # )

    print("PPO Agent with CNN+LSTM model implemented successfully!")
