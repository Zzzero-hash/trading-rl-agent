# Multi-Agent Ensemble System Guide

This guide explains how to use the advanced multi-agent ensemble system for RL trading, which combines multiple reinforcement learning agents (SAC, TD3, PPO) with sophisticated voting mechanisms, diversity measures, and dynamic weight management.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Voting Mechanisms](#voting-mechanisms)
7. [Training Workflows](#training-workflows)
8. [Evaluation and Diagnostics](#evaluation-and-diagnostics)
9. [Dynamic Agent Management](#dynamic-agent-management)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)

## Overview

The multi-agent ensemble system provides a robust framework for combining multiple RL agents to improve trading performance through:

- **Diverse Agent Strategies**: Different algorithms (SAC, TD3, PPO) with varying architectures
- **Advanced Voting Mechanisms**: Multiple ways to combine agent decisions
- **Dynamic Weight Management**: Automatic adjustment based on performance
- **Comprehensive Diagnostics**: Detailed analysis of ensemble behavior
- **Real-time Adaptation**: Dynamic agent addition/removal during training

## Key Features

### 🎯 Multiple Voting Mechanisms

- **Weighted Voting**: Performance-based weighted combination
- **Consensus Voting**: Agreement-based decision making
- **Diversity-Aware Voting**: Encourages agent diversity
- **Risk-Adjusted Voting**: Uncertainty-aware combination

### 📊 Advanced Metrics

- **Diversity Measures**: Action and policy diversity quantification
- **Consensus Analysis**: Agreement/disagreement patterns
- **Stability Metrics**: Performance consistency over time
- **Weight Entropy**: Distribution balance measurement

### 🔄 Dynamic Management

- **Real-time Weight Updates**: Based on recent performance
- **Agent Addition/Removal**: During training or inference
- **Performance Tracking**: Rolling window performance history
- **Automatic Rebalancing**: Maintains ensemble stability

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SAC Agent     │    │   TD3 Agent     │    │   PPO Agent     │
│                 │    │                 │    │                 │
│ • Actor Network │    │ • Actor Network │    │ • Policy Net    │
│ • Critic Net    │    │ • Critic Net    │    │ • Value Net     │
│ • Target Net    │    │ • Target Net    │    │ • GAE Buffer    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Ensemble Agent  │
                    │                 │
                    │ • Voting Logic  │
                    │ • Weight Mgmt   │
                    │ • Diagnostics   │
                    │ • Diversity     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Trading Env     │
                    │                 │
                    │ • Market Data   │
                    │ • Reward Func   │
                    │ • State Space   │
                    └─────────────────┘
```

## Quick Start

### Basic Ensemble Setup

```python
from trading_rl_agent.agents import (
    EnsembleAgent, EnsembleTrainer, EnsembleConfig
)

# Create ensemble configuration
config = EnsembleConfig(
    agents={
        "sac": {"enabled": True, "config": {"learning_rate": 3e-4}},
        "td3": {"enabled": True, "config": {"learning_rate": 3e-4}},
        "ppo": {"enabled": True, "config": {"learning_rate": 3e-4}},
    },
    ensemble_method="weighted_voting",
    diversity_penalty=0.1,
    performance_window=100,
)

# Create trainer
trainer = EnsembleTrainer(
    config=config,
    env_creator=your_env_creator,
    save_dir="outputs/ensemble"
)

# Create and train ensemble
trainer.create_agents()
results = trainer.train_ensemble(total_iterations=1000)
```

### Simple Evaluation

```python
from trading_rl_agent.agents import EnsembleEvaluator

# Create evaluator
evaluator = EnsembleEvaluator(trainer.ensemble)

# Evaluate ensemble
results = evaluator.evaluate_ensemble(
    env=your_env,
    num_episodes=100,
    include_diagnostics=True
)

# Generate report
report = evaluator.generate_evaluation_report(results)
print(report)
```

## Configuration

### EnsembleConfig Parameters

```python
@dataclass
class EnsembleConfig:
    agents: Dict[str, Dict[str, Any]]  # Agent configurations
    ensemble_method: str = "weighted_voting"  # Voting mechanism
    diversity_penalty: float = 0.1  # Diversity encouragement
    performance_window: int = 100  # Performance history length
    min_weight: float = 0.05  # Minimum agent weight
    risk_adjustment: bool = True  # Enable risk-aware voting
    consensus_threshold: float = 0.6  # Consensus threshold
```

### Agent-Specific Configurations

#### SAC Configuration

```python
SACConfig(
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    hidden_dims=[256, 256],
    automatic_entropy_tuning=True,
    target_entropy=-1.0,
)
```

#### TD3 Configuration

```python
TD3Config(
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    hidden_dims=[256, 256],
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
)
```

#### PPO Configuration

```python
PPOConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    batch_size=256,
    minibatch_size=64,
    n_epochs=10,
    hidden_dims=[256, 256],
    activation="tanh",
    vf_coef=0.5,
    ent_coef=0.01,
    target_kl=0.01,
)
```

## Voting Mechanisms

### 1. Weighted Voting

Combines agent actions based on their performance weights.

```python
ensemble = EnsembleAgent(
    policies=policies,
    weights={"sac": 0.4, "td3": 0.3, "ppo": 0.3},
    ensemble_method="weighted_voting"
)
```

**Advantages:**

- Simple and interpretable
- Performance-based weighting
- Stable combination

**Use Cases:**

- When agents have different performance levels
- Stable market conditions
- Clear performance differences

### 2. Consensus Voting

Uses agent agreement to make decisions.

```python
ensemble = EnsembleAgent(
    policies=policies,
    ensemble_method="consensus",
    consensus_threshold=0.6
)
```

**Advantages:**

- Reduces noise from individual agents
- More conservative decisions
- Higher confidence when agents agree

**Use Cases:**

- High-stakes decisions
- When agent reliability varies
- Conservative trading strategies

### 3. Diversity-Aware Voting

Encourages agent diversity while combining decisions.

```python
ensemble = EnsembleAgent(
    policies=policies,
    ensemble_method="diversity_aware",
    diversity_penalty=0.1
)
```

**Advantages:**

- Maintains agent diversity
- Reduces overfitting
- Better generalization

**Use Cases:**

- Complex market conditions
- When diversity is important
- Avoiding groupthink

### 4. Risk-Adjusted Voting

Considers agent uncertainty in decision making.

```python
ensemble = EnsembleAgent(
    policies=policies,
    ensemble_method="risk_adjusted"
)
```

**Advantages:**

- Uncertainty-aware decisions
- Better risk management
- Adaptive to market volatility

**Use Cases:**

- Volatile market conditions
- Risk-sensitive trading
- When uncertainty estimation is available

## Training Workflows

### Basic Training

```python
# Create trainer
trainer = EnsembleTrainer(config, env_creator, save_dir="outputs")

# Create agents
trainer.create_agents()

# Train ensemble
results = trainer.train_ensemble(
    total_iterations=1000,
    eval_frequency=50,
    save_frequency=100,
    early_stopping_patience=50
)
```

### Advanced Training with Monitoring

```python
# Custom training loop with monitoring
for iteration in range(total_iterations):
    # Train individual agents
    agent_rewards = trainer._train_agents_step()

    # Update ensemble weights
    if trainer.ensemble:
        trainer.ensemble.update_weights(agent_rewards)

    # Evaluate periodically
    if iteration % eval_frequency == 0:
        metrics = trainer._evaluate_ensemble()

        # Log metrics
        print(f"Iteration {iteration}: Reward = {metrics['ensemble_reward']:.3f}")

        # Save checkpoint
        if metrics['ensemble_reward'] > best_reward:
            trainer._save_ensemble("best")
            best_reward = metrics['ensemble_reward']
```

### Training with Dynamic Agent Management

```python
# Add new agent during training
success = trainer.add_agent_dynamically(
    "sac_v2", "sac",
    {"learning_rate": 1e-4, "hidden_dims": [128, 128]}
)

if success:
    print("New agent added successfully!")

    # Continue training with new agent
    trainer.train_ensemble(total_iterations=100)

# Remove underperforming agent
trainer.remove_agent_dynamically("sac_v2")
```

## Evaluation and Diagnostics

### Comprehensive Evaluation

```python
evaluator = EnsembleEvaluator(ensemble)

# Full evaluation
results = evaluator.evaluate_ensemble(
    env=env,
    num_episodes=100,
    include_diagnostics=True,
    save_results=True,
    results_path="evaluation_results.json"
)
```

### Performance Metrics

```python
# Performance metrics
performance = results["performance"]
print(f"Mean Reward: {performance['mean_reward']:.3f}")
print(f"Success Rate: {performance['success_rate']:.1%}")
print(f"Episode Length: {performance['mean_episode_length']:.1f}")

# Consensus metrics
consensus = results["consensus"]
print(f"Mean Consensus: {consensus['mean_consensus']:.3f}")
print(f"Consensus Stability: {consensus['consensus_stability']:.3f}")

# Diversity metrics
diversity = results["diversity"]
print(f"Action Diversity: {diversity['action_diversity']:.3f}")
print(f"Policy Diversity: {diversity['policy_diversity']:.3f}")

# Stability metrics
stability = results["stability"]
print(f"Reward Stability: {stability['reward_stability']:.3f}")
print(f"Overall Stability: {stability['overall_stability']:.3f}")
```

### Agent Comparison

```python
# Compare individual agents vs ensemble
comparison = evaluator.compare_agents(env, num_episodes=50)

for agent_name, results in comparison.items():
    print(f"{agent_name}:")
    print(f"  Mean Reward: {results['mean_reward']:.3f}")
    print(f"  Success Rate: {results['success_rate']:.1%}")
```

### Diagnostic Reports

```python
# Generate comprehensive report
report = evaluator.generate_evaluation_report(results)
print(report)

# Get evaluation summary
summary = evaluator.get_evaluation_summary()
print(f"Performance Trend: {summary['performance_trend']['reward_trend']}")
print(f"Diversity Trend: {summary['diversity_trend']['diversity_trend']}")
```

## Dynamic Agent Management

### Adding Agents Dynamically

```python
# Add new agent with different configuration
new_config = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "hidden_dims": [128, 128],
}

success = trainer.add_agent_dynamically("sac_v2", "sac", new_config)

if success:
    print("Agent added successfully!")
    # Ensemble automatically rebalances weights
```

### Removing Agents

```python
# Remove underperforming agent
success = trainer.remove_agent_dynamically("sac_v2")

if success:
    print("Agent removed successfully!")
    # Remaining agents' weights are renormalized
```

### Weight Management

```python
# Manual weight updates
performance_metrics = {
    "sac": 1.0,
    "td3": 0.8,
    "ppo": 0.6
}

ensemble.update_weights(performance_metrics)

# Check current weights
print(f"Current weights: {ensemble.weights}")

# Get weight analysis
info = ensemble.get_agent_info()
weight_analysis = info["diagnostics"]["weight_analysis"]
print(f"Weight entropy: {weight_analysis['weight_entropy']:.3f}")
```

## Advanced Usage

### Custom Voting Mechanisms

```python
class CustomEnsembleAgent(EnsembleAgent):
    def _custom_voting(self, obs):
        """Custom voting mechanism."""
        actions = {}
        for name, policy in self.policy_map.items():
            action, _, _ = policy.compute_single_action(obs)
            actions[name] = action

        # Custom combination logic
        # ... your custom logic here ...

        return combined_action

# Use custom voting
ensemble = CustomEnsembleAgent(policies, weights)
ensemble.ensemble_method = "custom"
```

### Ensemble of Ensembles

```python
# Create multiple ensembles
ensemble1 = EnsembleAgent(policies1, weights1, method="weighted_voting")
ensemble2 = EnsembleAgent(policies2, weights2, method="consensus")

# Combine ensembles
meta_ensemble = EnsembleAgent(
    policies={"ensemble1": ensemble1, "ensemble2": ensemble2},
    weights={"ensemble1": 0.6, "ensemble2": 0.4},
    ensemble_method="weighted_voting"
)
```

### Real-time Adaptation

```python
# Monitor ensemble performance
def adaptive_training(trainer, env, window_size=100):
    performance_history = deque(maxlen=window_size)

    for iteration in range(total_iterations):
        # Train step
        agent_rewards = trainer._train_agents_step()

        # Evaluate
        if iteration % eval_frequency == 0:
            metrics = trainer._evaluate_ensemble(env, num_episodes=10)
            performance_history.append(metrics['performance']['mean_reward'])

            # Adaptive actions based on performance
            if len(performance_history) >= window_size:
                recent_performance = np.mean(list(performance_history)[-window_size//2:])
                older_performance = np.mean(list(performance_history)[:window_size//2])

                if recent_performance < older_performance * 0.9:
                    # Performance declining - add diversity
                    trainer.add_agent_dynamically("diversity_agent", "ppo", {})
                elif recent_performance > older_performance * 1.1:
                    # Performance improving - optimize weights
                    trainer.ensemble.update_weights(agent_rewards)
```

## Troubleshooting

### Common Issues

#### 1. Low Ensemble Performance

**Symptoms:** Ensemble performs worse than individual agents

**Solutions:**

- Check agent diversity: `evaluator._calculate_diversity_metrics()`
- Adjust diversity penalty: `config.diversity_penalty = 0.2`
- Try different voting methods: `ensemble_method = "consensus"`
- Review agent configurations for compatibility

#### 2. Weight Instability

**Symptoms:** Agent weights fluctuate wildly

**Solutions:**

- Increase performance window: `config.performance_window = 200`
- Set minimum weight: `config.min_weight = 0.1`
- Use exponential moving average for weight updates
- Check for reward scaling issues

#### 3. Low Consensus

**Symptoms:** Agents rarely agree on actions

**Solutions:**

- Lower consensus threshold: `consensus_threshold = 0.4`
- Check state space normalization
- Review reward function consistency
- Consider agent architecture differences

#### 4. Memory Issues

**Symptoms:** Out of memory during training

**Solutions:**

- Reduce batch sizes in agent configs
- Use smaller network architectures
- Implement gradient checkpointing
- Use mixed precision training

### Performance Optimization

#### 1. Training Speed

```python
# Use mixed precision
trainer = EnsembleTrainer(
    config=config,
    env_creator=env_creator,
    device="cuda",  # Use GPU
    enable_amp=True  # Mixed precision
)

# Parallel evaluation
results = evaluator.evaluate_ensemble(
    env=env,
    num_episodes=100,
    parallel_episodes=4  # If supported
)
```

#### 2. Memory Efficiency

```python
# Smaller networks
config.agents["sac"]["config"]["hidden_dims"] = [128, 128]
config.agents["td3"]["config"]["hidden_dims"] = [128, 128]

# Smaller batch sizes
config.agents["sac"]["config"]["batch_size"] = 128
config.agents["td3"]["config"]["batch_size"] = 128
```

#### 3. Evaluation Efficiency

```python
# Quick evaluation for monitoring
quick_results = evaluator.evaluate_ensemble(
    env=env,
    num_episodes=10,  # Fewer episodes
    include_diagnostics=False  # Skip detailed diagnostics
)

# Full evaluation for final assessment
full_results = evaluator.evaluate_ensemble(
    env=env,
    num_episodes=1000,
    include_diagnostics=True
)
```

### Debugging Tools

#### 1. Ensemble Diagnostics

```python
# Get comprehensive diagnostics
info = ensemble.get_agent_info()
print(f"Agent performances: {info['agent_performances']}")
print(f"Ensemble diagnostics: {info['diagnostics']}")

# Check weight distribution
weight_entropy = ensemble._calculate_weight_entropy()
print(f"Weight entropy: {weight_entropy:.3f}")
```

#### 2. Action Analysis

```python
# Analyze agent actions
obs = env.reset()
actions = {}

for name, policy in ensemble.policy_map.items():
    action, _, _ = policy.compute_single_action(obs)
    actions[name] = action

print(f"Individual actions: {actions}")
ensemble_action = ensemble.select_action(obs)
print(f"Ensemble action: {ensemble_action}")
```

#### 3. Performance Tracking

```python
# Track performance over time
history = trainer.training_history
print(f"Reward history: {history['ensemble_reward']}")
print(f"Diversity history: {history['diversity_score']}")

# Plot trends
import matplotlib.pyplot as plt
plt.plot(history['ensemble_reward'])
plt.title('Ensemble Performance Over Time')
plt.show()
```

## Conclusion

The multi-agent ensemble system provides a powerful framework for combining multiple RL agents in trading applications. By leveraging different voting mechanisms, dynamic weight management, and comprehensive diagnostics, you can create robust trading systems that adapt to changing market conditions.

Key takeaways:

- Choose voting mechanism based on your trading strategy
- Monitor diversity and consensus metrics
- Use dynamic agent management for adaptation
- Regular evaluation and diagnostics are crucial
- Start simple and gradually add complexity

For more examples and advanced usage, see the `examples/ensemble_trading_example.py` file.
