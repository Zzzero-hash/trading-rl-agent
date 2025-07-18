"""
Ensemble Training Manager for Multi-Agent RL Trading Systems.

This module provides comprehensive training workflows for ensemble agents including:
- Multi-agent training coordination
- Ensemble-specific evaluation metrics
- Dynamic agent management
- Performance monitoring and diagnostics
- Integration with existing RL training pipeline
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .configs import EnsembleConfig, PPOConfig, SACConfig, TD3Config
from .policy_utils import EnsembleAgent

# Import Ray for initialization
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

# Import Ray RLlib algorithms with fallback handling
logger = logging.getLogger(__name__)

PPOTrainer: Any = None
SACTrainer: Any = None
TD3Trainer: Any = None

try:
    from ray.rllib.algorithms.ppo import PPO as _PPOTrainer

    PPOTrainer = _PPOTrainer
except ImportError:
    logger.warning("Ray RLlib PPO not available")

try:
    from ray.rllib.algorithms.sac import SAC as _SACTrainer

    SACTrainer = _SACTrainer
except ImportError:
    logger.warning("Ray RLlib SAC not available")

try:
    from ray.rllib.algorithms.td3 import TD3 as _TD3Trainer

    TD3Trainer = _TD3Trainer
except ImportError:
    logger.warning("Ray RLlib TD3 not available")

# Import Ray registry for environment registration
try:
    from ray.tune.registry import register_env

    RAY_REGISTRY_AVAILABLE = True
except ImportError:
    RAY_REGISTRY_AVAILABLE = False
    register_env = None


class EnsembleTrainer:
    """Comprehensive trainer for multi-agent ensembles."""

    def __init__(
        self,
        config: EnsembleConfig,
        env_creator: Callable[[], Any],
        save_dir: str = "outputs/ensemble",
        device: str = "auto",
    ):
        self.config = config
        self.env_creator = env_creator
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize ensemble
        self.ensemble: EnsembleAgent | None = None
        self.agents: dict[str, Any] = {}
        self.training_history: dict[str, list[float] | dict[str, list[float]]] = {
            "ensemble_reward": [],
            "agent_rewards": {},
            "diversity_score": [],
            "consensus_rate": [],
            "weight_stability": [],
            "performance_variance": [],
        }

        # Training state
        self.current_iteration = 0
        self.best_ensemble_reward = float("-inf")

        self.logger.info(f"Ensemble trainer initialized on {self.device}")

        # Initialize Ray if not already initialized
        if RAY_AVAILABLE:
            if not ray.is_initialized():
                self.logger.info("Initializing Ray...")
                # Force local cluster initialization with explicit parameters
                ray.init(
                    local_mode=False,
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    num_cpus=2,
                    object_store_memory=1000000000,  # 1GB
                )
            else:
                self.logger.info("Ray already initialized")
        else:
            self.logger.warning("Ray not available - some features may not work")

    def create_agents(self) -> None:
        """Create individual agents based on configuration."""
        self.logger.info("Creating ensemble agents...")

        for agent_name, agent_config in self.config.agents.items():
            if not agent_config.get("enabled", True):
                continue

            self.logger.info(f"Creating agent: {agent_name}")

            if agent_name == "sac":
                agent = self._create_sac_agent(agent_config)
            elif agent_name == "td3":
                agent = self._create_td3_agent(agent_config)
            elif agent_name == "ppo":
                agent = self._create_ppo_agent(agent_config)
            else:
                self.logger.warning(f"Unknown agent type: {agent_name}")
                continue

            if agent is not None:
                self.agents[agent_name] = agent

        # Create ensemble from individual agents
        if self.agents:
            # For the new Ray RLlib API, we need to create a simple wrapper
            # that can compute actions from the trained agents
            policies = {}
            for name, agent in self.agents.items():
                # Create a simple policy wrapper that uses the agent's new API
                class AgentPolicy:
                    def __init__(self, agent: Any, name: str) -> None:
                        self.agent = agent
                        self.name = name

                    def compute_single_action(self, obs: Any, **kwargs: Any) -> tuple[Any, list, dict]:
                        # Use the new RLModule API instead of deprecated compute_single_action
                        try:
                            # Try to get the RLModule and use forward_inference
                            module = self.agent.get_module()

                            # Handle different observation shapes and convert to tensor
                            if isinstance(obs, np.ndarray):
                                # Handle inhomogeneous shapes by flattening
                                if obs.dtype == object or obs.ndim > 1:
                                    # Convert to flat array, handling mixed types
                                    obs_flat = []
                                    for item in obs.flatten():
                                        if isinstance(item, (list, tuple, np.ndarray)):
                                            obs_flat.extend(np.array(item, dtype=np.float32).flatten())
                                        else:
                                            obs_flat.append(float(item))  # type: ignore[unreachable]
                                    obs_batch = np.array(obs_flat, dtype=np.float32).reshape(1, -1)
                                else:
                                    obs_batch = obs.reshape(1, -1)
                            else:
                                # Convert to numpy array and flatten
                                try:
                                    obs_batch = np.array(obs, dtype=np.float32).reshape(1, -1)
                                except (ValueError, TypeError):
                                    # Handle mixed types by converting to float
                                    obs_flat = []
                                    for item in obs:
                                        if isinstance(item, (list, tuple, np.ndarray)):
                                            obs_flat.extend(np.array(item, dtype=np.float32).flatten())
                                        else:
                                            obs_flat.append(float(item))  # type: ignore[unreachable]
                                    obs_batch = np.array(obs_flat, dtype=np.float32).reshape(1, -1)

                            # Convert to PyTorch tensor
                            obs_tensor = torch.FloatTensor(obs_batch)

                            result = module.forward_inference({"obs": obs_tensor})

                            # Handle different possible action formats
                            if isinstance(result, dict):
                                if "actions" in result:
                                    action = result["actions"][0]
                                elif "action" in result:
                                    action = result["action"][0]
                                elif "action_dist_inputs" in result:
                                    # For discrete actions, take argmax
                                    action = np.argmax(result["action_dist_inputs"][0])
                                else:
                                    # Try to extract action from any available key
                                    for value in result.values():
                                        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                                            action = value[0]
                                            break
                                    else:
                                        raise KeyError("No action found in result")
                            else:
                                # Direct action output
                                action = result[0] if hasattr(result, "__getitem__") else result

                            # Convert to numpy if it's a tensor
                            if hasattr(action, "cpu"):
                                action = action.cpu().numpy()

                            # Ensure action is a 1D numpy array with consistent shape
                            if isinstance(action, (list, tuple)):
                                action = np.array(action, dtype=np.float32)
                            elif not isinstance(action, np.ndarray):
                                action = np.array([action], dtype=np.float32)

                            # Flatten to 1D if needed
                            if action.ndim > 1:
                                action = action.flatten()

                            return action, [], {}
                        except Exception as e:
                            # Fallback to random action if module access fails
                            action = None  # Initialize action variable
                            try:
                                # Try to get action space from the environment
                                env = None
                                if hasattr(self.agent, "env_runner") and hasattr(self.agent.env_runner, "env"):
                                    env = self.agent.env_runner.env
                                elif hasattr(self.agent, "env"):
                                    env = self.agent.env
                                elif hasattr(self.agent, "get_policy"):
                                    # Try to get policy and then environment
                                    try:
                                        policy = self.agent.get_policy()
                                        if hasattr(policy, "observation_space"):
                                            # For continuous actions, use observation space to infer action space
                                            obs_dim = (
                                                policy.observation_space.shape[0]
                                                if hasattr(policy.observation_space, "shape")
                                                else 1
                                            )
                                            action = np.random.uniform(-1, 1, obs_dim)
                                        else:
                                            action = np.random.uniform(-1, 1, 1)
                                    except Exception:
                                        action = np.random.uniform(-1, 1, 1)
                                else:
                                    # Default action space for trading (single continuous action)
                                    action = np.random.uniform(-1, 1, 1)

                                if env and hasattr(env, "action_space"):
                                    action_space = env.action_space
                                    if hasattr(action_space, "shape"):
                                        action = np.random.uniform(-1, 1, action_space.shape)
                                    else:
                                        action = np.random.uniform(-1, 1, 1)
                            except Exception:
                                # Ultimate fallback
                                action = np.random.uniform(-1, 1, 1)

                            # Ensure action is a 1D numpy array
                            if isinstance(action, (list, tuple)):
                                action = np.array(action, dtype=np.float32)
                            elif not isinstance(action, np.ndarray):
                                action = np.array([action], dtype=np.float32)

                            # Flatten to 1D if needed
                            if action.ndim > 1:
                                action = action.flatten()

                            return action, [], {}

                    def get_uncertainty(self, obs: Any) -> float:
                        # Simple uncertainty estimation
                        return 0.1  # Fixed uncertainty for now

                policies[name] = AgentPolicy(agent, name)

            initial_weights = {name: 1.0 / len(policies) for name in policies}

            self.ensemble = EnsembleAgent(
                policies=policies,
                weights=initial_weights,
                ensemble_method=self.config.ensemble_method,
                diversity_penalty=self.config.diversity_penalty,
                performance_window=self.config.performance_window,
                min_weight=self.config.min_weight,
                risk_adjustment=self.config.risk_adjustment,
            )

            self.logger.info(f"Created ensemble with {len(policies)} agents")
        else:
            raise ValueError("No agents were created successfully")

    def _create_sac_agent(self, config: dict[str, Any]) -> Any:
        """Create SAC agent."""
        if SACTrainer is None:
            self.logger.error("SAC trainer not available - Ray RLlib SAC not installed")
            return None

        sac_config = SACConfig(**config.get("config", {}))

        # Register environment if not already registered
        if not RAY_REGISTRY_AVAILABLE:
            self.logger.error("Ray registry not available - cannot register environment")
            return None

        env_name = "EnsembleTradingEnv"

        def env_creator(config: dict[str, Any]) -> Any:
            return self.env_creator()

        register_env(env_name, env_creator)

        # Create SAC trainer configuration
        trainer_config = {
            "env": env_name,
            "framework": "torch",
            "model": {
                "fcnet_hiddens": sac_config.hidden_dims,
                "fcnet_activation": "relu",
            },
            "train_batch_size": sac_config.batch_size,
            "learning_rate": sac_config.learning_rate,
            "gamma": sac_config.gamma,
            "tau": sac_config.tau,
            "target_entropy": sac_config.target_entropy,
            "automatic_entropy_tuning": sac_config.automatic_entropy_tuning,
            "buffer_size": sac_config.buffer_capacity,
        }

        return SACTrainer(config=trainer_config)

    def _create_td3_agent(self, config: dict[str, Any]) -> Any:
        """Create TD3 agent."""
        if TD3Trainer is None:
            self.logger.error("TD3 trainer not available - Ray RLlib TD3 not installed")
            return None

        td3_config = TD3Config(**config.get("config", {}))

        # Use the same registered environment
        env_name = "EnsembleTradingEnv"

        trainer_config = {
            "env": env_name,
            "framework": "torch",
            "model": {
                "fcnet_hiddens": td3_config.hidden_dims,
                "fcnet_activation": "relu",
            },
            "train_batch_size": td3_config.batch_size,
            "learning_rate": td3_config.learning_rate,
            "gamma": td3_config.gamma,
            "tau": td3_config.tau,
            "policy_delay": td3_config.policy_delay,
            "target_noise": td3_config.target_noise,
            "noise_clip": td3_config.noise_clip,
            "exploration_noise": td3_config.exploration_noise,
        }

        return TD3Trainer(config=trainer_config)

    def _create_ppo_agent(self, config: dict[str, Any]) -> Any:
        """Create PPO agent."""
        if PPOTrainer is None:
            self.logger.error("PPO trainer not available - Ray RLlib PPO not installed")
            return None

        ppo_config = PPOConfig(**config.get("config", {}))

        # Use the same registered environment
        env_name = "EnsembleTradingEnv"

        trainer_config = {
            "env": env_name,
            "framework": "torch",
            "model": {
                "fcnet_hiddens": ppo_config.hidden_dims,
                "fcnet_activation": ppo_config.activation,
            },
            "train_batch_size": ppo_config.batch_size,
            "sgd_minibatch_size": ppo_config.minibatch_size,
            "num_sgd_iter": ppo_config.n_epochs,
            "learning_rate": ppo_config.learning_rate,
            "gamma": ppo_config.gamma,
            "lambda": ppo_config.gae_lambda,
            "clip_param": ppo_config.clip_ratio,
            "vf_clip_param": ppo_config.clip_vf_ratio,
            "vf_loss_coeff": ppo_config.vf_coef,
            "entropy_coeff": ppo_config.ent_coef,
            "target_kl": ppo_config.target_kl,
            "normalize_actions": True,
        }

        return PPOTrainer(config=trainer_config)

    def train_ensemble(
        self,
        total_iterations: int = 1000,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        early_stopping_patience: int = 50,
    ) -> dict[str, Any]:
        """Train the ensemble with comprehensive monitoring."""
        self.logger.info(f"Starting ensemble training for {total_iterations} iterations")

        best_reward = float("-inf")
        patience_counter = 0

        for iteration in range(total_iterations):
            self.current_iteration = iteration

            # Train individual agents
            agent_rewards = self._train_agents_step()

            # Update ensemble weights based on performance
            if self.ensemble:
                self.ensemble.update_weights(agent_rewards)

            # Evaluate ensemble
            if iteration % eval_frequency == 0:
                ensemble_metrics = self._evaluate_ensemble()

                # Update training history
                self._update_training_history(agent_rewards, ensemble_metrics)

                # Check for improvement
                current_reward = ensemble_metrics.get("ensemble_reward", 0.0)
                if current_reward > best_reward:
                    best_reward = current_reward
                    patience_counter = 0
                    self._save_ensemble("best")
                else:
                    patience_counter += 1

                # Log progress
                self._log_training_progress(iteration, agent_rewards, ensemble_metrics)

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at iteration {iteration}")
                    break

            # Save checkpoint
            if iteration % save_frequency == 0:
                self._save_ensemble(f"checkpoint_{iteration}")

        # Final evaluation and save
        final_metrics = self._evaluate_ensemble()
        self._save_ensemble("final")

        return {
            "training_history": self.training_history,
            "final_metrics": final_metrics,
            "best_reward": best_reward,
            "total_iterations": iteration + 1,
        }

    def _train_agents_step(self) -> dict[str, float]:
        """Train all agents for one step and return their rewards."""
        agent_rewards = {}

        for name, agent in self.agents.items():
            try:
                # Train agent for one step
                result = agent.train()

                # Extract reward from training result
                if hasattr(result, "episode_reward_mean"):
                    reward = result.episode_reward_mean
                elif hasattr(result, "episode_reward_max"):
                    reward = result.episode_reward_max
                else:
                    reward = 0.0

                agent_rewards[name] = reward

            except Exception as e:
                self.logger.warning(f"Error training agent {name}: {e}")
                agent_rewards[name] = 0.0

        return agent_rewards

    def _evaluate_ensemble(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate the ensemble and return comprehensive metrics."""
        if not self.ensemble:
            return {"ensemble_reward": 0.0}

        # Create evaluation environment
        env = self.env_creator()

        episode_rewards = []
        episode_lengths = []
        consensus_counts = 0
        total_decisions = 0

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                # Get ensemble action
                action = self.ensemble.select_action(obs)

                # Check for consensus (if multiple agents)
                if len(self.agents) > 1:
                    individual_actions = []
                    for agent in self.agents.values():
                        try:
                            # Try to use the new RLModule API first
                            if hasattr(agent, "get_module"):
                                module = agent.get_module()
                                if module is not None:
                                    try:
                                        # Use the new forward_inference API
                                        obs_batch = np.array([obs])
                                        # Convert to PyTorch tensor
                                        obs_tensor = torch.FloatTensor(obs_batch)
                                        action_batch = module.forward_inference({"obs": obs_tensor})

                                        # Handle different possible action formats
                                        if isinstance(action_batch, dict):
                                            if "actions" in action_batch:
                                                agent_action = action_batch["actions"][0]
                                            elif "action" in action_batch:
                                                agent_action = action_batch["action"][0]
                                            elif "action_dist_inputs" in action_batch:
                                                # For discrete actions, take argmax
                                                agent_action = np.argmax(action_batch["action_dist_inputs"][0])
                                            else:
                                                # Try to extract action from any available key
                                                for value in action_batch.values():
                                                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                                                        agent_action = value[0]
                                                        break
                                                else:
                                                    raise KeyError("No action found in action_batch")
                                        else:
                                            # Direct action output
                                            agent_action = (
                                                action_batch[0]
                                                if hasattr(action_batch, "__getitem__")
                                                else action_batch
                                            )

                                        # Ensure consistent action shape
                                        if isinstance(agent_action, (list, tuple)):
                                            agent_action = np.array(agent_action, dtype=np.float32)
                                        elif not isinstance(agent_action, np.ndarray):
                                            agent_action = np.array([agent_action], dtype=np.float32)

                                        # Flatten to 1D if needed
                                        if agent_action.ndim > 1:
                                            agent_action = agent_action.flatten()
                                    except Exception as module_error:
                                        self.logger.debug(
                                            f"RLModule API failed: {module_error}, falling back to compute_single_action"
                                        )
                                        agent_action, _, _ = agent.compute_single_action(obs)
                                else:
                                    # Fallback to compute_single_action if module not available
                                    agent_action, _, _ = agent.compute_single_action(obs)
                            else:
                                # Fallback to compute_single_action
                                agent_action, _, _ = agent.compute_single_action(obs)
                            individual_actions.append(agent_action)
                        except Exception as e:
                            self.logger.warning(f"Failed to get action from agent: {e}")
                            continue

                    # Check if actions are similar (consensus)
                    if len(individual_actions) > 1:
                        try:
                            # Ensure all actions have the same shape before stacking
                            max_dim = max(action.size for action in individual_actions)
                            normalized_actions = []
                            for action in individual_actions:
                                if action.size < max_dim:
                                    # Pad with zeros if needed
                                    padded_action = np.zeros(max_dim, dtype=np.float32)
                                    padded_action[: action.size] = action
                                    normalized_actions.append(padded_action)
                                else:
                                    normalized_actions.append(action)

                            action_array = np.array(normalized_actions)
                            std_action = np.std(action_array, axis=0)
                            if np.all(std_action < self.ensemble.consensus_threshold):
                                consensus_counts += 1
                            total_decisions += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to check consensus: {e}")
                            continue

                # Take action in environment
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    obs, reward, done, info = step_result
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Calculate metrics
        ensemble_reward = np.mean(episode_rewards)
        ensemble_std = np.std(episode_rewards)
        avg_episode_length = np.mean(episode_lengths)
        consensus_rate = consensus_counts / max(total_decisions, 1)

        # Get ensemble diagnostics
        diagnostics = self.ensemble.get_ensemble_diagnostics()

        return {
            "ensemble_reward": float(ensemble_reward),
            "ensemble_std": float(ensemble_std),
            "avg_episode_length": float(avg_episode_length),
            "consensus_rate": float(consensus_rate),
            "diversity_score": float(diagnostics["diversity_score"]),
            "weight_stability": float(diagnostics["weight_stability"]),
            "performance_variance": float(diagnostics["performance_variance"]),
        }

    def _update_training_history(self, agent_rewards: dict[str, float], ensemble_metrics: dict[str, float]) -> None:
        """Update training history with new metrics."""
        # Type-safe access to list fields
        ensemble_reward_list = self.training_history["ensemble_reward"]
        if isinstance(ensemble_reward_list, list):
            ensemble_reward_list.append(ensemble_metrics.get("ensemble_reward", 0.0))

        diversity_score_list = self.training_history["diversity_score"]
        if isinstance(diversity_score_list, list):
            diversity_score_list.append(ensemble_metrics.get("diversity_score", 0.0))

        consensus_rate_list = self.training_history["consensus_rate"]
        if isinstance(consensus_rate_list, list):
            consensus_rate_list.append(ensemble_metrics.get("consensus_rate", 0.0))

        weight_stability_list = self.training_history["weight_stability"]
        if isinstance(weight_stability_list, list):
            weight_stability_list.append(ensemble_metrics.get("weight_stability", 0.0))

        performance_variance_list = self.training_history["performance_variance"]
        if isinstance(performance_variance_list, list):
            performance_variance_list.append(ensemble_metrics.get("performance_variance", 0.0))

        # Update agent rewards
        agent_rewards_dict = self.training_history["agent_rewards"]
        if isinstance(agent_rewards_dict, dict):
            for agent_name, reward in agent_rewards.items():
                if agent_name not in agent_rewards_dict:
                    agent_rewards_dict[agent_name] = []
                agent_rewards_dict[agent_name].append(reward)

    def _log_training_progress(
        self, iteration: int, agent_rewards: dict[str, float], ensemble_metrics: dict[str, float]
    ) -> None:
        """Log training progress with detailed metrics."""
        self.logger.info(
            f"Iteration {iteration}: "
            f"Ensemble Reward: {ensemble_metrics.get('ensemble_reward', 0.0):.2f} ± "
            f"{ensemble_metrics.get('ensemble_std', 0.0):.2f}, "
            f"Diversity: {ensemble_metrics.get('diversity_score', 0.0):.3f}, "
            f"Consensus: {ensemble_metrics.get('consensus_rate', 0.0):.3f}"
        )

        # Log individual agent performance
        for agent_name, reward in agent_rewards.items():
            self.logger.debug(f"  {agent_name}: {reward:.2f}")

    def _save_ensemble(self, suffix: str) -> None:
        """Save ensemble and individual agents."""
        if not self.ensemble:
            return

        # Save ensemble
        ensemble_path = self.save_dir / f"ensemble_{suffix}.pkl"
        # Save ensemble state (placeholder - implement actual save logic)
        ensemble_state = {
            "weights": self.ensemble.weights,
            "ensemble_method": self.ensemble.ensemble_method,
            "agent_performances": self.ensemble.agent_performances,
        }
        torch.save(ensemble_state, str(ensemble_path))

        # Save individual agents
        for name, agent in self.agents.items():
            agent_path = self.save_dir / f"{name}_{suffix}.pkl"
            if hasattr(agent, "save"):
                # Use absolute path for Ray compatibility
                abs_path = str(agent_path.absolute())
                agent.save(abs_path)

        # Save training history
        history_path = self.save_dir / f"training_history_{suffix}.json"

        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Saved ensemble checkpoint: {suffix}")

    def load_ensemble(self, suffix: str) -> None:
        """Load ensemble and individual agents."""
        # Load ensemble
        ensemble_path = self.save_dir / f"ensemble_{suffix}.pkl"
        if ensemble_path.exists() and self.ensemble:
            # Load ensemble state (placeholder - implement actual load logic)
            ensemble_state = torch.load(str(ensemble_path))
            self.ensemble.weights = ensemble_state["weights"]
            self.ensemble.ensemble_method = ensemble_state["ensemble_method"]
            self.ensemble.agent_performances = ensemble_state["agent_performances"]

        # Load individual agents
        for name, agent in self.agents.items():
            agent_path = self.save_dir / f"{name}_{suffix}.pkl"
            if agent_path.exists() and hasattr(agent, "load"):
                agent.load(str(agent_path))

        # Load training history
        history_path = self.save_dir / f"training_history_{suffix}.json"
        if history_path.exists():
            with open(history_path) as f:
                self.training_history = json.load(f)

        self.logger.info(f"Loaded ensemble checkpoint: {suffix}")

    def get_ensemble_info(self) -> dict[str, Any]:
        """Get comprehensive information about the ensemble."""
        if not self.ensemble:
            return {"error": "Ensemble not initialized"}

        info = self.ensemble.get_agent_info()
        info.update(
            {
                "training_history": self.training_history,
                "current_iteration": self.current_iteration,
                "save_dir": str(self.save_dir),
                "device": str(self.device),
            }
        )

        return info

    def add_agent_dynamically(self, name: str, agent_type: str, config: dict[str, Any]) -> bool:
        """Dynamically add a new agent to the ensemble during training."""
        try:
            if agent_type == "sac":
                agent = self._create_sac_agent({"config": config})
            elif agent_type == "td3":
                agent = self._create_td3_agent({"config": config})
            elif agent_type == "ppo":
                agent = self._create_ppo_agent({"config": config})
            else:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return False

            if agent is None:
                return False

            self.agents[name] = agent

            if self.ensemble:
                # Create a policy wrapper for the new agent using the new RLlib API
                class AgentPolicy:
                    def __init__(self, agent: Any, name: str) -> None:
                        self.agent = agent
                        self.name = name

                    def compute_single_action(self, obs: Any, **kwargs: Any) -> tuple[Any, list, dict]:
                        # Use the new RLModule API instead of deprecated compute_single_action
                        try:
                            # Try to get the RLModule and use forward_inference
                            module = self.agent.get_module()

                            # Handle different observation shapes and convert to tensor
                            if isinstance(obs, np.ndarray):
                                # Handle inhomogeneous shapes by flattening
                                if obs.dtype == object or obs.ndim > 1:
                                    # Convert to flat array, handling mixed types
                                    obs_flat = []
                                    for item in obs.flatten():
                                        if isinstance(item, (list, tuple, np.ndarray)):
                                            obs_flat.extend(np.array(item, dtype=np.float32).flatten())
                                        else:
                                            obs_flat.append(float(item))  # type: ignore[unreachable]
                                    obs_batch = np.array(obs_flat, dtype=np.float32).reshape(1, -1)
                                else:
                                    obs_batch = obs.reshape(1, -1)
                            else:
                                # Convert to numpy array and flatten
                                try:
                                    obs_batch = np.array(obs, dtype=np.float32).reshape(1, -1)
                                except (ValueError, TypeError):
                                    # Handle mixed types by converting to float
                                    obs_flat = []
                                    for item in obs:
                                        if isinstance(item, (list, tuple, np.ndarray)):
                                            obs_flat.extend(np.array(item, dtype=np.float32).flatten())
                                        else:
                                            obs_flat.append(float(item))  # type: ignore[unreachable]
                                    obs_batch = np.array(obs_flat, dtype=np.float32).reshape(1, -1)

                            # Convert to PyTorch tensor
                            obs_tensor = torch.FloatTensor(obs_batch)

                            result = module.forward_inference({"obs": obs_tensor})

                            # Handle different possible action formats
                            if isinstance(result, dict):
                                if "actions" in result:
                                    action = result["actions"][0]
                                elif "action" in result:
                                    action = result["action"][0]
                                elif "action_dist_inputs" in result:
                                    # For discrete actions, take argmax
                                    action = np.argmax(result["action_dist_inputs"][0])
                                else:
                                    # Try to extract action from any available key
                                    for value in result.values():
                                        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                                            action = value[0]
                                            break
                                    else:
                                        raise KeyError("No action found in result")
                            else:
                                # Direct action output
                                action = result[0] if hasattr(result, "__getitem__") else result

                            # Convert to numpy if it's a tensor
                            if hasattr(action, "cpu"):
                                action = action.cpu().numpy()

                            # Ensure action is a 1D numpy array with consistent shape
                            if isinstance(action, (list, tuple)):
                                action = np.array(action, dtype=np.float32)
                            elif not isinstance(action, np.ndarray):
                                action = np.array([action], dtype=np.float32)

                            # Flatten to 1D if needed
                            if action.ndim > 1:
                                action = action.flatten()

                            return action, [], {}
                        except Exception as e:
                            # Fallback to random action if module access fails
                            action = np.random.uniform(-1, 1, 1)
                            return action, [], {}

                    def get_uncertainty(self, obs: Any) -> float:
                        # Simple uncertainty estimation
                        return 0.1  # Fixed uncertainty for now

                policy = AgentPolicy(agent, name)
                initial_weight = 1.0 / (len(self.ensemble.policy_map) + 1)
                self.ensemble.add_agent(name, policy, initial_weight)

            self.logger.info(f"Successfully added agent {name} of type {agent_type}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to add agent {name}: {e}")
            return False

    def remove_agent_dynamically(self, name: str) -> bool:
        """Dynamically remove an agent from the ensemble during training."""
        try:
            if name in self.agents:
                del self.agents[name]

            if self.ensemble:
                self.ensemble.remove_agent(name)

            self.logger.info(f"Successfully removed agent {name}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to remove agent {name}: {e}")
            return False
