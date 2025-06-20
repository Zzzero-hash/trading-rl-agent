# Getting Started

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- Docker (optional, for containerized deployment)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install development dependencies
pip install -r requirements-test.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest
```

### Docker Installation

```bash
# Build development image
docker build -f Dockerfile.dev -t trading-rl-agent:dev .

# Run with GPU support
docker run --gpus all -it trading-rl-agent:dev
```

## Quick Start Tutorial

### 1. Generate Sample Data

```python
from generate_sample_data import generate_sample_price_data

# Generate synthetic trading data
data = generate_sample_price_data(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2023-01-01",
    num_points=10000
)
data.to_csv("data/sample_data.csv", index=False)
```

### 2. Create Trading Environment

```python
from src.envs.trading_env import TradingEnv

env = TradingEnv(
    data_paths=["data/sample_data.csv"],
    initial_balance=10000,
    window_size=50,
    transaction_cost=0.001
)
```

### 3. Train an Agent

```python
from src.agents.sac_agent import SACAgent

# Initialize SAC agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=256,
    learning_rate=3e-4
)

# Train the agent
training_results = agent.train(
    env=env,
    episodes=1000,
    max_steps=1000,
    save_frequency=100
)
```

### 4. Evaluate Performance

```python
from evaluate_agent import evaluate_agent

# Evaluate trained agent
results = evaluate_agent(
    agent=agent,
    env=env,
    episodes=10
)

print(f"Average Return: {results['avg_return']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Configuration

The project uses YAML configuration files for different components:

### Environment Configuration

```yaml
# config/env_config.yaml
environment:
  initial_balance: 10000
  window_size: 50
  transaction_cost: 0.001
  max_position: 1.0
  normalize_observations: true
```

### Agent Configuration

```yaml
# config/sac_config.yaml
agent:
  hidden_dim: 256
  learning_rate: 0.0003
  batch_size: 256
  memory_size: 100000
  tau: 0.005
  gamma: 0.99
  alpha: 0.2
```

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand the system design
- Explore the [API Reference](api_reference.md) for detailed documentation
- Check out [Examples](examples.md) for more advanced use cases
- See [Contributing Guidelines](contributing.md) to contribute to the project
