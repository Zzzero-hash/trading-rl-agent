import sys
from unittest.mock import Mock, patch

import pytest

from trading_rl_agent.main import main

pytestmark = pytest.mark.integration


def test_main_tune_mode(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--tune",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("agents.tune.run_tune") as mock_run_tune,
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        main()
        mock_run_tune.assert_called_once_with([env_path, model_path, trainer_path])
        mock_trainer_class.assert_not_called()


def test_main_tune_early_return(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--tune",
        "--train",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("agents.tune.run_tune") as mock_run_tune,
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_run_tune.assert_called_once()
        mock_trainer_class.assert_not_called()
        mock_trainer.train.assert_not_called()
