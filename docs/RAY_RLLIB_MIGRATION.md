# Ray RLlib TD3 → SAC Migration Guide

## Overview

**Important**: TD3 (Twin Delayed Deep Deterministic Policy Gradient) has been **completely removed** from Ray RLlib version 2.38.0+. This project has been updated to use **SAC (Soft Actor-Critic)** as the primary continuous control algorithm for Ray RLlib integration, leveraging **FinRL** environments for trading tasks. Production deployments now rely on **SAC** and **PPO** only.

## Migration Summary

### What Changed

- **Ray RLlib Integration**: The deprecated `optimize_td3_hyperparams()` utility has been removed. Use `optimize_sac_hyperparams()` instead.
- **Primary Algorithms**: SAC and PPO are the supported options for production workloads.
- **Custom Implementation**: TD3 remains available only for local experimentation.

### Why SAC?

SAC (Soft Actor-Critic) is an excellent replacement for TD3 because:

1. **Entropy Regularization**: Encourages exploration and prevents premature convergence
2. **Sample Efficiency**: Generally more sample-efficient than TD3
3. **Stability**: More robust to hyperparameter choices
4. **Ray RLlib Support**: Fully supported in Ray 2.38.0+ with active development

## Code Changes Required

### 1. Import Updates

```python
# OLD (TD3 - no longer works)
from ray.rllib.algorithms.td3 import TD3Config

# NEW (SAC - works in Ray 2.38.0+)
from ray.rllib.algorithms.sac import SACConfig
```

### 2. Algorithm Configuration

```python
# OLD (TD3)
config = TD3Config()
config.training(
    actor_lr=3e-4,
    critic_lr=3e-4,
    tau=0.005,
    gamma=0.99
)

# NEW (SAC)
config = SACConfig()
config.training(
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,  # SAC-specific: entropy coefficient learning rate
    tau=0.005,
    gamma=0.99,
    twin_q=True,    # SAC-specific: use twin Q-networks (similar to TD3)
    target_entropy="auto"  # SAC-specific: auto-tune entropy target
)
```

### 3. Hyperparameter Optimization

```python
from trading_rl_agent.optimization.rl_optimization import optimize_sac_hyperparams

results = optimize_sac_hyperparams(env_config, num_samples=20)
```

## Custom TD3 Implementation

While TD3 is no longer available in Ray RLlib, this project maintains a **custom TD3 implementation** for:

- Local development and testing
- Educational purposes
- Comparison with SAC performance
- Environments where Ray RLlib is not required

### Usage of Custom TD3

```python
from trading_rl_agent.agents.td3_agent import TD3Agent
from trading_rl_agent.agents.configs import TD3Config

# Create custom TD3 agent
config = TD3Config(
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    buffer_capacity=1000000
)

agent = TD3Agent(config, state_dim=10, action_dim=3)
```

## Configuration Files Updated

1. **`src/configs/model/td3_agent.yaml`**: Updated to use SAC configuration
2. **`src/optimization/rl_optimization.py`**: TD3 optimization utilities removed; use `optimize_sac_hyperparams()`
3. **Documentation**: All references updated to reflect SAC/PPO as production algorithms

## Testing

- **SAC Integration**: All Ray RLlib tests now use SAC
- **Custom TD3**: Local TD3 tests continue to pass
- **Hyperparameter Optimization**: Ray Tune optimization uses SAC

## Best Practices

### For New Development

1. **Use SAC** for Ray RLlib integration and distributed training
2. **Use custom TD3** only for local experiments or specific research needs
3. **Follow SAC hyperparameters** in the updated configuration files

### For Existing Code

1. **Replace TD3 imports** with SAC imports for Ray RLlib code
2. **Update configuration files** to use SAC-specific parameters
3. **Test thoroughly** as SAC may have different convergence characteristics

## Performance Considerations

| Algorithm        | Sample Efficiency | Stability | Ray RLlib Support | Use Case                  |
| ---------------- | ----------------- | --------- | ----------------- | ------------------------- |
| **SAC**          | High              | High      | ✅ Full Support   | **Production, Ray RLlib** |
| **TD3 (Custom)** | Medium            | High      | ❌ Not Available  | Local development only    |

## Migration Checklist

- [x] ✅ Update Ray Tune API calls (`tune.report` → `train.report`)
- [x] ✅ Add `metric` and `mode` parameters to `tune.run` calls
- [x] ✅ Replace TD3 imports with SAC imports in Ray RLlib code
- [x] ✅ Update hyperparameter optimization to use SAC
- [x] ✅ Update configuration files
- [x] ✅ Update documentation and README files
- [x] ✅ Test all Ray Tune/RLlib functionality
- [ ] 🔄 **Run comprehensive tests to validate changes**

## Need Help?

- **SAC Documentation**: [Ray RLlib SAC Guide](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac)
- **Ray Tune Documentation**: [Ray Tune User Guide](https://docs.ray.io/en/latest/tune/index.html)
- **Custom TD3 Code**: See `src/agents/td3_agent.py` for the maintained implementation

---

**Note**: This migration ensures compatibility with Ray 2.38.0+ while maintaining all the continuous control capabilities needed for trading applications.
