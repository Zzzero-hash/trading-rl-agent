import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.main import main

pytestmark = pytest.mark.integration


def test_main_train_only(sample_config_files):
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
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_trainer.train.assert_called_once()
        mock_trainer.evaluate.assert_not_called()
        mock_trainer.test.assert_not_called()


def test_main_train_with_custom_seed(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--seed",
        "999",
        "--train",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        call_args = mock_trainer_class.call_args
        assert call_args[1]["seed"] == 999


def test_main_eval_only(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--eval",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_trainer.evaluate.assert_called_once()
        mock_trainer.train.assert_not_called()
        mock_trainer.test.assert_not_called()


def test_main_test_only(sample_config_files, capsys):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--test",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_trainer.test.assert_called_once()
        mock_trainer.train.assert_not_called()
        mock_trainer.evaluate.assert_not_called()
        captured = capsys.readouterr()
        assert "Running tests..." in captured.out


def test_main_train_and_eval(sample_config_files):
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
        "--eval",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_trainer.train.assert_called_once()
        mock_trainer.evaluate.assert_called_once()


def test_main_all_modes(sample_config_files):
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
        "--eval",
        "--test",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        mock_trainer.train.assert_called_once()
        mock_trainer.evaluate.assert_called_once()
        mock_trainer.test.assert_called_once()


def test_main_no_mode_flags(sample_config_files):
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

        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_not_called()
        mock_trainer.evaluate.assert_not_called()
        mock_trainer.test.assert_not_called()


def test_main_custom_save_dir(sample_config_files, temp_dir):
    env_path, model_path, trainer_path = sample_config_files
    custom_save_dir = Path(temp_dir) / "custom_models"
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
        "--save-dir",
        str(custom_save_dir),
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("src.main.Trainer") as mock_trainer_class,
    ):
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        main()

        call_args = mock_trainer_class.call_args
        assert call_args[1]["save_dir"] == str(custom_save_dir)
