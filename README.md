# Trading RL Agent

A reinforcement learning framework for algorithmic trading, providing customizable environments, agent implementations, and training pipelines.

## Table of Contents

- [Installation](#installation)
- [Running with Docker](#running-with-docker)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
# Clone repository
 git clone https://github.com/yourusername/trading-rl-agent.git
cd trading-rl-agent

# Set up a virtual environment and install dependencies
./setup_env.sh

# Build Docker image (CPU)
docker build -t trading-rl-agent .

# (Optional) Build GPU/ROCm image
 docker build -f Dockerfile.rocm -t trading-rl-agent:rocm .
```

## Running with Docker

```bash
# Interactive container shell
 docker run --rm -it \
   -v "$(pwd)/src:/app/src" \
   -v "$(pwd)/src/configs:/app/src/configs" \
   -v "$(pwd)/data:/app/src/data" \
   -w /app \
   trading-rl-agent bash

# Inside container use the CLI
 trade-agent \
   --env-config src/configs/env/trader_env.yaml \
   --model-config src/configs/model/cnn_lstm.yaml \
   --trainer-config src/configs/trainer/default.yaml \
   --train
```

## Project Structure

```text
trading-rl-agent/
├── Dockerfile                # Container setup
├── requirements.txt          # Python dependencies
├── setup.py                  # Package metadata
├── README.md                 # This file
├── src/                      # Source code
│   ├── main.py               # CLI entry-point
│   ├── agents/               # RL agents
│   │   ├── ddqn_agent.py     # Dueling Double DQN
│   │   ├── ppo_agent.py      # PPO
│   │   └── trainer.py        # Training orchestration
│   ├── envs/                 # Custom trading environments
│   ├── models/               # Neural network architectures
│   ├── configs/              # YAML configuration files
│   ├── data/                 # Data utilities and generators
│   └── utils/                # Metrics and helper functions
└── tests/                    # Unit and integration tests
```

## Configuration

All hyperparameters and environment settings are defined in YAML files under `src/configs/`:

- **env/**: Environment parameters (window size, transaction costs)
- **model/**: Model architecture and hyperparameters
- **trainer/**: Training parameters (learning rates, batch sizes)
- **ray/**: RLlib and Ray Tune configurations

## Usage

Train or evaluate an agent:

Running `./setup_env.sh` sets up the virtual environment, installs all
dependencies, and installs this package in editable mode.

```bash
# Start a local Ray cluster
ray start --head

# Using console script (after running `./setup_env.sh` which installs
# Ray[tune] and this package in editable mode)
trade-agent \
  --env-config src/configs/env/trader_env.yaml \
  --model-config src/configs/model/cnn_lstm.yaml \
  --trainer-config src/configs/trainer/default.yaml \
  --train

# Direct module invocation
env "PYTHONPATH=src" python -m trading_rl_agent.main \
  --env-config src/configs/env/trader_env.yaml \
  --model-config src/configs/model/cnn_lstm.yaml \
  --trainer-config src/configs/trainer/default.yaml \
  --eval
```

Train using the Ray RLlib configuration files:

```bash
trade-agent \
  --env-config src/configs/env/trader_env.yaml \
  --model-config src/configs/model/cnn_lstm.yaml \
  --trainer-config src/configs/ray/ppo_ray.yaml \
  --train
```

Run a Ray Tune hyperparameter search using the provided search space:

```bash
trade-agent \
  --env-config src/configs/ray/tune_search.yaml \
  --model-config src/configs/ray/tune_search.yaml \
  --trainer-config src/configs/ray/tune_search.yaml \
  --tune
```

```bash
ray stop
```

## Data Pipeline with Ray

`run_pipeline` now uses Ray to parallelize data ingestion. By default it
connects to a local Ray instance, or you can specify a cluster address with the
`RAY_ADDRESS` environment variable or `ray_address` field in the pipeline
configuration. Example:

```bash
export RAY_ADDRESS="ray://head-node:10001"
python -m src.data.pipeline --config src/configs/data/pipeline.yaml
```

## Distributed Training with Ray Cluster

The repository provides a small configuration file `ray_cluster_setup.yaml`
describing the addresses and resources of the Proxmox cluster. `train_rl.py`
will read this file if passed via `--cluster-config` and connect with
`ray.init(address="auto")` automatically. Example:

```bash
python -m src.train_rl --data data.csv --model-path model.pt \
  --cluster-config ray_cluster_setup.yaml
```

CPU-intensive tasks such as data ingestion run on the four CPU nodes while the
GPU nodes train the neural networks. Resource allocation is determined at run
time using `get_available_devices()`.

## Trading Environment

The `TradingEnv` module implements a Gym-compatible environment used for RL
training. Configure it with paths to CSV datasets and parameters like
`window_size`, `initial_balance` and `transaction_cost`. Example:

```python
from src.envs.trading_env import TradingEnv
env = TradingEnv({"dataset_paths": ["data.csv"], "window_size": 10})
obs, _ = env.reset()
```

## RLlib Training

Use the `Trainer` class to run training with Ray RLlib and Tune. Checkpoints and
logs are written to `save_dir` (default `outputs/`).

```python
from src.agents.trainer import Trainer

env_cfg = {"dataset_paths": ["data.csv"], "window_size": 10}
model_cfg = {}
trainer_cfg = {"algorithm": "ppo", "num_iterations": 10,
               "ray_config": {"framework": "torch"}}

trainer = Trainer(env_cfg, model_cfg, trainer_cfg)
trainer.train()
```

Specify `ray_address` in `trainer_cfg` or the `RAY_ADDRESS` environment variable
to connect to a remote Ray cluster.

## Deployment with Ray Serve

The module `src/serve_deployment.py` contains simple Ray Serve deployments for
the supervised predictor and the RL policy. They can be launched on any Ray
cluster:

```bash
ray start --head
python -m ray serve run src.serve_deployment:deployment_graph
```

Requests can then be sent via HTTP:

```bash
curl -X POST http://127.0.0.1:8000/predictor -d '{"features": [0.1, 0.2]}'
```

These deployments are stubs; in production they would load the latest model
checkpoints and could be integrated into a CI/CD pipeline for automated rollout.

## Testing

```pwsh
pytest --maxfail=1 -q
```

Run a specific test:

```pwsh
pytest tests/test_historical_live.py
```

## Contributing

Contributions are welcome! Please fork the repo, create a feature branch, add tests, and submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
