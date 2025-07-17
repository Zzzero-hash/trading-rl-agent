# Advanced Policy Optimization for RL Agents

This document describes the advanced policy optimization techniques implemented in the trading RL agent framework. These methods provide state-of-the-art algorithms for training reinforcement learning agents with improved performance, stability, and risk management.

## Overview

The advanced policy optimization module includes:

1. **Proximal Policy Optimization (PPO) with Advanced Clipping**
2. **Trust Region Policy Optimization (TRPO)**
3. **Natural Policy Gradient Methods**
4. **Adaptive Learning Rate Scheduling**
5. **Multi-objective Optimization for Risk-adjusted Returns**

## Features

### 1. Advanced PPO

Enhanced PPO implementation with advanced features:

- **Adaptive Clipping**: Dynamic adjustment of clipping ratio based on performance
- **Trust Region Constraints**: KL divergence constraints for stable updates
- **Natural Gradient Integration**: Optional natural gradient updates
- **Multi-objective Optimization**: Risk-adjusted reward functions
- **Adaptive Learning Rate**: Cosine, linear, and exponential scheduling

```python
from trading_rl_agent.agents import AdvancedPPO, AdvancedPPOConfig
from trading_rl_agent.agents.advanced_trainer import PolicyNetwork, ValueNetwork

# Create networks
policy_net = PolicyNetwork(state_dim=50, action_dim=3)
value_net = ValueNetwork(state_dim=50)

# Configure Advanced PPO
config = AdvancedPPOConfig(
    adaptive_clip_ratio=True,
    use_trust_region=True,
    adaptive_lr=True,
    lr_schedule="cosine",
    risk_weight=0.1,
    return_weight=0.9,
)

# Create algorithm
ppo = AdvancedPPO(policy_net, value_net, config, device="cpu")
```

### 2. Trust Region Policy Optimization (TRPO)

TRPO implementation with trust region constraints:

- **Conjugate Gradient**: Efficient computation of natural gradient
- **Line Search**: Backtracking line search for step size
- **KL Divergence Constraints**: Maintains policy similarity
- **Stable Updates**: Prevents catastrophic policy changes

```python
from trading_rl_agent.agents import TRPO, TRPOConfig

config = TRPOConfig(
    max_kl_divergence=0.01,
    damping_coeff=0.1,
    max_backtrack_iter=10,
)

trpo = TRPO(policy_net, value_net, config, device="cpu")
```

### 3. Natural Policy Gradient

Natural policy gradient implementation:

- **Fisher Information Matrix**: Diagonal approximation for efficiency
- **Natural Gradient Updates**: Second-order optimization
- **Damping**: Prevents numerical instability
- **Conjugate Gradient**: Efficient matrix-vector products

```python
from trading_rl_agent.agents import NaturalPolicyGradient, NaturalPolicyGradientConfig

config = NaturalPolicyGradientConfig(
    damping_coeff=1e-3,
    max_cg_iter=10,
)

npg = NaturalPolicyGradient(policy_net, value_net, config, device="cpu")
```

### 4. Adaptive Learning Rate Scheduling

Intelligent learning rate scheduling:

- **Cosine Schedule**: Smooth decay with warmup
- **Linear Schedule**: Linear decay
- **Exponential Schedule**: Exponential decay
- **Performance-based Adaptation**: Adjust based on training progress

```python
from trading_rl_agent.agents import AdaptiveLearningRateScheduler

scheduler = AdaptiveLearningRateScheduler(
    optimizer,
    schedule_type="cosine",
    warmup_steps=1000,
    total_steps=1000000,
    min_lr_ratio=0.1,
)
```

### 5. Multi-objective Optimization

Multi-objective optimization for risk-adjusted returns:

- **Return Maximization**: Maximize cumulative returns
- **Risk Minimization**: Minimize volatility and drawdown
- **Sharpe Ratio**: Optimize risk-adjusted returns
- **Customizable Weights**: Balance different objectives

```python
from trading_rl_agent.agents import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    return_weight=0.8,
    risk_weight=0.1,
    sharpe_weight=0.1,
    max_drawdown_weight=0.0,
)
```

## Usage Examples

### Basic Training

```python
from trading_rl_agent.agents import AdvancedTrainer
from trading_rl_agent.agents.configs import AdvancedPPOConfig

# Create trainer
trainer = AdvancedTrainer(
    state_dim=50,
    action_dim=3,
    device="cpu",
    save_dir="outputs",
)

# Create environment
def env_creator():
    # Your environment creation logic
    return env

# Train with Advanced PPO
config = AdvancedPPOConfig()
results = trainer.train(
    "advanced_ppo",
    config,
    env_creator(),
    num_episodes=1000,
    eval_frequency=100,
)
```

### Multi-objective Training

```python
from trading_rl_agent.agents import MultiObjectiveTrainer
from trading_rl_agent.agents.configs import MultiObjectiveConfig

# Create multi-objective configuration
multi_obj_config = MultiObjectiveConfig(
    return_weight=0.7,
    risk_weight=0.2,
    sharpe_weight=0.1,
)

# Create trainer
trainer = MultiObjectiveTrainer(
    state_dim=50,
    action_dim=3,
    multi_obj_config=multi_obj_config,
    device="cpu",
)

# Train with multi-objective optimization
results = trainer.train(
    "advanced_ppo",
    config,
    env,
    num_episodes=1000,
)
```

### Algorithm Benchmarking

```python
from trading_rl_agent.agents import BenchmarkFramework, BenchmarkConfig
from trading_rl_agent.agents.configs import AdvancedPPOConfig, TRPOConfig

# Create benchmark configuration
config = BenchmarkConfig(
    state_dim=50,
    action_dim=3,
    num_episodes=500,
    num_runs=3,
    save_plots=True,
    save_data=True,
)

# Setup algorithms
config.algorithms = {
    "advanced_ppo": AdvancedPPOConfig(),
    "trpo": TRPOConfig(),
}

# Create framework
framework = BenchmarkFramework(config)

# Run benchmark
def env_creator():
    # Your environment creation logic
    return env

results = framework.run_benchmark(env_creator)
framework.print_summary()
```

## CLI Usage

The framework provides a command-line interface for easy usage:

### Training

```bash
# Train with Advanced PPO
python -m trading_rl_agent.agents.cli_advanced_optimization train \
    --algorithm advanced_ppo \
    --num-episodes 1000 \
    --learning-rate 3e-4 \
    --batch-size 256

# Train with TRPO
python -m trading_rl_agent.agents.cli_advanced_optimization train \
    --algorithm trpo \
    --num-episodes 1000
```

### Benchmarking

```bash
# Run benchmark comparison
python -m trading_rl_agent.agents.cli_advanced_optimization benchmark \
    --algorithms advanced_ppo trpo \
    --num-episodes 500 \
    --num-runs 3 \
    --save-plots \
    --save-data

# Quick benchmark
python -m trading_rl_agent.agents.cli_advanced_optimization quick-benchmark \
    --num-episodes 100 \
    --num-runs 2
```

### Multi-objective Training

```bash
# Train with multi-objective optimization
python -m trading_rl_agent.agents.cli_advanced_optimization multi-objective \
    --algorithm advanced_ppo \
    --return-weight 0.8 \
    --risk-weight 0.2 \
    --num-episodes 1000
```

## Configuration Options

### AdvancedPPOConfig

```python
@dataclass
class AdvancedPPOConfig:
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2

    # Advanced clipping
    adaptive_clip_ratio: bool = True
    clip_ratio_decay: float = 0.995
    min_clip_ratio: float = 0.05
    max_clip_ratio: float = 0.3

    # Trust region
    use_trust_region: bool = True
    max_kl_divergence: float = 0.01

    # Multi-objective
    risk_weight: float = 0.1
    return_weight: float = 0.9
    sharpe_weight: float = 0.0
    max_drawdown_weight: float = 0.0

    # Adaptive learning rate
    adaptive_lr: bool = True
    lr_schedule: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
```

### TRPOConfig

```python
@dataclass
class TRPOConfig:
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Trust region parameters
    max_kl_divergence: float = 0.01
    damping_coeff: float = 0.1
    max_backtrack_iter: int = 10
    backtrack_coeff: float = 0.8
    max_cg_iter: int = 10
    cg_tolerance: float = 1e-6
```

### MultiObjectiveConfig

```python
@dataclass
class MultiObjectiveConfig:
    # Objective weights
    return_weight: float = 0.9
    risk_weight: float = 0.1
    sharpe_weight: float = 0.0
    max_drawdown_weight: float = 0.0

    # Risk parameters
    var_alpha: float = 0.05
    max_position_size: float = 1.0
    max_leverage: float = 2.0

    # Performance tracking
    performance_window: int = 100
    rebalancing_frequency: int = 10
```

## Performance Comparison

The benchmarking framework provides comprehensive performance analysis:

### Metrics

- **Mean Reward**: Average episode reward
- **Convergence Speed**: Episodes to convergence
- **Training Time**: Computational efficiency
- **Stability**: Standard deviation of performance
- **Memory Usage**: Memory efficiency

### Visualization

The framework generates comprehensive plots:

1. **Learning Curves**: Episode rewards over time
2. **Performance Comparison**: Final performance comparison
3. **Training Time**: Computational efficiency comparison
4. **Convergence Analysis**: Convergence speed analysis

## Best Practices

### 1. Algorithm Selection

- **Advanced PPO**: Good balance of performance and stability
- **TRPO**: When stability is critical
- **Natural Policy Gradient**: For complex, high-dimensional problems

### 2. Hyperparameter Tuning

- Start with default configurations
- Use adaptive learning rates for better convergence
- Adjust multi-objective weights based on risk tolerance
- Monitor KL divergence for TRPO

### 3. Multi-objective Optimization

- Balance return and risk objectives
- Use Sharpe ratio for risk-adjusted optimization
- Consider drawdown constraints for risk management
- Adjust weights based on market conditions

### 4. Benchmarking

- Run multiple seeds for statistical significance
- Use consistent evaluation environments
- Monitor training stability
- Compare computational efficiency

## Integration with Existing Code

The advanced policy optimization methods integrate seamlessly with the existing trading RL agent framework:

```python
# Use with existing environments
from trading_rl_agent.envs.finrl_trading_env import TradingEnv

env = TradingEnv(env_config)
trainer = AdvancedTrainer(state_dim=env.observation_space.shape[0],
                         action_dim=env.action_space.shape[0])

# Use with existing risk management
from trading_rl_agent.risk.riskfolio import RiskfolioRiskManager

risk_manager = RiskfolioRiskManager(config)
# Integrate with multi-objective optimization
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/test_advanced_policy_optimization.py -v

# Run specific test categories
pytest tests/test_advanced_policy_optimization.py::TestAdvancedPPO -v
pytest tests/test_advanced_policy_optimization.py::TestTRPO -v
pytest tests/test_advanced_policy_optimization.py::TestMultiObjectiveOptimizer -v
```

## Performance Benchmarks

The framework includes built-in performance benchmarks:

```python
# Quick performance test
from trading_rl_agent.agents import run_quick_benchmark

results = run_quick_benchmark(
    algorithms=["advanced_ppo", "trpo"],
    num_episodes=100,
    num_runs=3,
)
```

## Future Enhancements

Planned improvements include:

1. **Distributed Training**: Multi-GPU and multi-node training
2. **Advanced Architectures**: Transformer-based policies
3. **Meta-Learning**: Few-shot adaptation to new environments
4. **Hierarchical RL**: Multi-level policy optimization
5. **Continuous Control**: Support for continuous action spaces

## Contributing

To contribute to the advanced policy optimization module:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for new algorithms
4. Include performance benchmarks
5. Ensure backward compatibility

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Schulman, J., et al. "Trust Region Policy Optimization." ICML (2015)
3. Kakade, S. "A Natural Policy Gradient." NIPS (2002)
4. Mnih, V., et al. "Asynchronous Methods for Deep Reinforcement Learning." ICML (2016)
