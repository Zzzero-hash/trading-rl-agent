# Examples

This section provides a practical example of using the `Trainer` class to train a trading agent.

## Basic Training Example

The following example demonstrates how to set up a `SystemConfig`, create a `Trainer`, and run the training process.

```python
from trading_rl_agent.agents.trainer import Trainer
from trading_rl_agent.core.config import SystemConfig, AgentConfig, ModelConfig, RiskConfig, DataConfig, InfrastructureConfig

# 1. Create a SystemConfig object
# This object holds all the configuration for the trading system.
# For this example, we will use the default settings.
system_config = SystemConfig(
    agent=AgentConfig(
        agent_type="sac",
        total_timesteps=1000,
    ),
    model=ModelConfig(
        cnn_filters=[32],
        lstm_units=64,
    ),
    risk=RiskConfig(
        max_position_size=0.1,
        max_drawdown=0.1,
    ),
    data=DataConfig(
        feature_window=20,
    ),
    infrastructure=InfrastructureConfig(
        gpu_enabled=False,
    )
)

# 2. Initialize the Trainer
# The Trainer class is the main entry point for training RL agents.
# It takes a SystemConfig object and a directory to save the output.
trainer = Trainer(system_cfg=system_config, save_dir="outputs/basic_example")

# 3. Run the training process
# The train() method will:
# - Load and preprocess the data
# - Set up the RL environment
# - Train the agent using the specified algorithm (e.g., SAC)
# - Save the trained agent and other artifacts to the save_dir
trainer.train()

print("Training complete! Check the 'outputs/basic_example' directory for results.")

```

---

For legal and safety notes see the [project disclaimer](disclaimer.md).
