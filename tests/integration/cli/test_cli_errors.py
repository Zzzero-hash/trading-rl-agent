import sys
from unittest.mock import Mock, patch

import pytest

from src.main import main

pytestmark = pytest.mark.integration


def test_main_trainer_exception(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--train",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        with pytest.raises(Exception, match="Training failed"):
            main()


def test_main_tune_exception(sample_config_files):
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
    ):
        mock_run_tune.side_effect = Exception("Tuning failed")
        with pytest.raises(Exception, match="Tuning failed"):
            main()
