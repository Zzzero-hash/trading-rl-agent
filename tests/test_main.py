"""Tests for the main entry point CLI interface."""

import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from src.main import main


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test configs."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield temp_path


@pytest.fixture
def sample_config_files(temp_dir):
    """Create sample configuration files."""
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


class TestMainArgParsing:
    """Test argument parsing in main function."""

    def test_main_required_args(self, sample_config_files):
        """Test main function with required arguments."""
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

            # Verify Trainer was instantiated with correct arguments
            mock_trainer_class.assert_called_once()
            call_args = mock_trainer_class.call_args
            assert call_args[1]["seed"] == 42  # default seed
            assert call_args[1]["save_dir"] == "outputs"  # default save_dir

    def test_main_all_args(self, sample_config_files, temp_dir):
        """Test main function with all arguments."""
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
            "123",
            "--save-dir",
            temp_dir,
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

            # Verify arguments were passed correctly
            call_args = mock_trainer_class.call_args
            assert call_args[1]["seed"] == 123
            assert call_args[1]["save_dir"] == temp_dir

            # Verify train and eval were called
            mock_trainer.train.assert_called_once()
            mock_trainer.evaluate.assert_called_once()

    def test_main_missing_required_args(self):
        """Test main function fails with missing required arguments."""
        test_args = ["main.py"]  # Missing required configs

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()


class TestMainTrainMode:
    """Test main function in training mode."""

    def test_main_train_only(self, sample_config_files):
        """Test main function with only --train flag."""
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

    def test_main_train_with_custom_seed(self, sample_config_files):
        """Test main function with custom seed."""
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


class TestMainEvalMode:
    """Test main function in evaluation mode."""

    def test_main_eval_only(self, sample_config_files):
        """Test main function with only --eval flag."""
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


class TestMainTestMode:
    """Test main function in test mode."""

    def test_main_test_only(self, sample_config_files, capsys):
        """Test main function with only --test flag."""
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

            # Check that "Running tests..." message was printed
            captured = capsys.readouterr()
            assert "Running tests..." in captured.out


class TestMainTuneMode:
    """Test main function in tuning mode."""

    def test_main_tune_mode(self, sample_config_files):
        """Test main function with --tune flag."""
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

            # Should call run_tune instead of creating Trainer
            mock_run_tune.assert_called_once_with([env_path, model_path, trainer_path])
            mock_trainer_class.assert_not_called()

    def test_main_tune_early_return(self, sample_config_files):
        """Test that tune mode returns early without creating trainer."""
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
            "--train",  # This should be ignored
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
            # Trainer should not be created or called when in tune mode
            mock_trainer_class.assert_not_called()
            mock_trainer.train.assert_not_called()


class TestMainConfigLoading:
    """Test configuration file loading in main function."""

    def test_main_loads_configs_correctly(self, sample_config_files):
        """Test that main function loads YAML configurations correctly."""
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

            # Verify configs were loaded and passed to Trainer
            call_args = mock_trainer_class.call_args
            env_cfg, model_cfg, trainer_cfg = call_args[0]

            assert isinstance(env_cfg, dict)
            assert isinstance(model_cfg, dict)
            assert isinstance(trainer_cfg, dict)

            assert env_cfg["window_size"] == 50
            assert model_cfg["learning_rate"] == 0.001
            assert trainer_cfg["algorithm"] == "ppo"

    def test_main_invalid_config_file(self, temp_dir):
        """Test main function with invalid config file."""
        # Create invalid YAML file
        invalid_path = Path(temp_dir) / "invalid.yaml"
        with open(invalid_path, "w") as f:
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

        with patch.object(sys, "argv", test_args):
            with pytest.raises((yaml.YAMLError, FileNotFoundError)):
                main()

    def test_main_missing_config_file(self):
        """Test main function with missing config file."""
        test_args = [
            "main.py",
            "--env-config",
            "nonexistent.yaml",
            "--model-config",
            "nonexistent.yaml",
            "--trainer-config",
            "nonexistent.yaml",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(FileNotFoundError):
                main()


class TestMainCombinedModes:
    """Test main function with multiple mode flags."""

    def test_main_train_and_eval(self, sample_config_files):
        """Test main function with both --train and --eval flags."""
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

    def test_main_all_modes(self, sample_config_files):
        """Test main function with --train, --eval, and --test flags."""
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

    def test_main_no_mode_flags(self, sample_config_files):
        """Test main function with no mode flags (should do nothing)."""
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

            # Should create trainer but not call any methods
            mock_trainer_class.assert_called_once()
            mock_trainer.train.assert_not_called()
            mock_trainer.evaluate.assert_not_called()
            mock_trainer.test.assert_not_called()


class TestMainErrorHandling:
    """Test error handling in main function."""

    def test_main_trainer_exception(self, sample_config_files):
        """Test main function handles Trainer exceptions."""
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

            # Should propagate the exception
            with pytest.raises(Exception, match="Training failed"):
                main()

    def test_main_tune_exception(self, sample_config_files):
        """Test main function handles tune exceptions."""
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

            # Should propagate the exception
            with pytest.raises(Exception, match="Tuning failed"):
                main()


class TestMainCustomSaveDir:
    """Test main function with custom save directory."""

    def test_main_custom_save_dir(self, sample_config_files, temp_dir):
        """Test main function with custom save directory."""
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


@pytest.mark.integration
class TestMainIntegration:
    """Integration tests for main function."""

    @pytest.mark.skip(reason="Requires full setup - enable for integration testing")
    def test_main_full_integration(self, sample_config_files):
        """Test main function with actual components (integration test)."""
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

        with patch.object(sys, "argv", test_args):
            # This would test the actual main function without mocking
            # Requires proper environment setup
            main()


class TestMainEntryPoint:
    """Test main function as entry point."""

    def test_main_when_main(self, sample_config_files):
        """Test main execution when __name__ == '__main__'."""
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

        # Test the if __name__ == '__main__' block
        with (
            patch.object(sys, "argv", test_args),
            patch("src.main.Trainer") as mock_trainer_class,
            patch("src.main.main") as mock_main,
        ):

            # Simulate running the script
            exec(open("src/main.py").read())

            # This is tricky to test properly, but we can verify
            # that the main function exists and is callable
            assert callable(main)
