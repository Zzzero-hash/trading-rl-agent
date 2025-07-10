import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from trading_rl_agent.main import main

pytestmark = pytest.mark.integration


def test_main_loads_configs_correctly(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        call_args = mock_trainer_class.call_args
        env_cfg, model_cfg, trainer_cfg = call_args[0]
        assert isinstance(env_cfg, dict)
        assert isinstance(model_cfg, dict)
        assert isinstance(trainer_cfg, dict)
        assert env_cfg["window_size"] == 50
        assert model_cfg["learning_rate"] == 0.001
        assert trainer_cfg["algorithm"] == "ppo"


def test_main_invalid_config_file(temp_dir):
    invalid_path = Path(temp_dir) / "invalid.yaml"
    with Path(invalid_path).open(invalid_path, "w") as f:
        f.write("invalid: yaml: content: [")
    test_args = [
        "main.py",
        "--env-config",
        str(invalid_path),
        "--model-config",
        str(invalid_path),
        "--trainer-config",
        str(invalid_path),
    ]
    with (
        patch.object(sys, "argv", test_args),
        pytest.raises((yaml.YAMLError, FileNotFoundError)),
    ):
        main()


def test_main_missing_config_file():
    test_args = [
        "main.py",
        "--env-config",
        "nonexistent.yaml",
        "--model-config",
        "nonexistent.yaml",
        "--trainer-config",
        "nonexistent.yaml",
    ]
    with patch.object(sys, "argv", test_args), pytest.raises(FileNotFoundError):
        main()
