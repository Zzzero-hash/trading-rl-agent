from pathlib import Path
import sys
import tempfile

import pytest
import yaml

import trading_rl_agent.agents as _agents

# Ensure the package can be imported as ``agents`` when calling the CLI.
sys.modules.setdefault("agents", _agents)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield temp_path


@pytest.fixture
def sample_config_files(temp_dir):
    """Create sample configuration files for CLI tests."""
    env_config = {
        "dataset_paths": ["dummy_data.csv"],
        "window_size": 50,
        "initial_balance": 10000,
    }

    model_config = {
        "architecture": "ppo",
        "learning_rate": 0.001,
        "hidden_layers": [64, 64],
    }

    trainer_config = {
        "algorithm": "ppo",
        "num_iterations": 10,
        "ray_config": {"env": "TraderEnv"},
    }

    env_path = Path(temp_dir) / "env.yaml"
    model_path = Path(temp_dir) / "model.yaml"
    trainer_path = Path(temp_dir) / "trainer.yaml"

    with open(env_path, "w") as f:
        yaml.dump(env_config, f)
    with open(model_path, "w") as f:
        yaml.dump(model_config, f)
    with open(trainer_path, "w") as f:
        yaml.dump(trainer_config, f)

    return str(env_path), str(model_path), str(trainer_path)
