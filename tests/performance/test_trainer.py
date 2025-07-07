"""Tests for the Trainer class."""

import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from src.agents.trainer import Trainer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_configs():
    """Create sample configuration dictionaries."""
    env_cfg = {
        "dataset_paths": ["dummy_data.csv"],
        "window_size": 10,
        "initial_balance": 10000,
    }

    model_cfg = {
        "architecture": "ppo",
        "learning_rate": 0.001,
        "hidden_layers": [64, 64],
    }

    trainer_cfg = {
        "algorithm": "ppo",
        "num_iterations": 5,
        "ray_config": {"env": "TraderEnv", "framework": "torch"},
    }

    return env_cfg, model_cfg, trainer_cfg


@pytest.fixture
def config_files(temp_dir, sample_configs):
    """Create temporary config files."""
    env_cfg, model_cfg, trainer_cfg = sample_configs

    env_path = Path(temp_dir) / "env_config.yaml"
    model_path = Path(temp_dir) / "model_config.yaml"
    trainer_path = Path(temp_dir) / "trainer_config.yaml"

    with open(env_path, "w") as f:
        yaml.dump(env_cfg, f)
    with open(model_path, "w") as f:
        yaml.dump(model_cfg, f)
    with open(trainer_path, "w") as f:
        yaml.dump(trainer_cfg, f)

    return str(env_path), str(model_path), str(trainer_path)


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_init_with_dict_configs(self, sample_configs, temp_dir):
        """Test Trainer initialization with dictionary configs."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        with (
            patch("ray.init") as mock_ray_init,
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env") as mock_register,
        ):

            trainer = Trainer(
                env_cfg, model_cfg, trainer_cfg, seed=123, save_dir=temp_dir
            )

            assert trainer.env_cfg == env_cfg
            assert trainer.model_cfg == model_cfg
            assert trainer.trainer_cfg == trainer_cfg
            assert trainer.seed == 123
            assert trainer.save_dir == temp_dir
            assert trainer.algorithm == "ppo"
            assert trainer.num_iterations == 5

            mock_ray_init.assert_called_once()
            mock_register.assert_called_once()
            assert os.path.exists(temp_dir)

    def test_trainer_init_with_ray_address(self, sample_configs, temp_dir):
        """Test Trainer initialization with Ray address."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg["ray_address"] = "ray://localhost:10001"

        with (
            patch("ray.init") as mock_ray_init,
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert trainer.ray_address == "ray://localhost:10001"
            mock_ray_init.assert_called_once_with(address="ray://localhost:10001")

    def test_trainer_init_ray_already_initialized(self, sample_configs, temp_dir):
        """Test Trainer initialization when Ray is already initialized."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        with (
            patch("ray.init") as mock_ray_init,
            patch("ray.is_initialized", return_value=True),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            mock_ray_init.assert_not_called()

    def test_trainer_init_defaults(self, sample_configs, temp_dir):
        """Test Trainer initialization with default values."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        # Remove optional fields
        trainer_cfg.pop("algorithm", None)
        trainer_cfg.pop("num_iterations", None)
        trainer_cfg.pop("ray_config", None)

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert trainer.algorithm == "ppo"  # default
            assert trainer.num_iterations == 10  # default
            assert trainer.ray_config["env"] == "TraderEnv"  # default

    def test_trainer_init_with_total_episodes(self, sample_configs, temp_dir):
        """Test Trainer initialization with total_episodes instead of num_iterations."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg.pop("num_iterations")
        trainer_cfg["total_episodes"] = 15

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert trainer.num_iterations == 15

    def test_trainer_init_non_dict_config(self, temp_dir):
        """Test Trainer initialization with non-dict trainer config."""
        env_cfg = {"dataset_paths": ["dummy.csv"]}
        model_cfg = {"architecture": "ppo"}
        trainer_cfg = "string_config"  # Non-dict config

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert trainer.algorithm == "ppo"  # default
            assert trainer.num_iterations == 10  # default


class TestTrainerTraining:
    """Test Trainer training functionality."""

    def test_trainer_train_ppo(self, sample_configs, temp_dir):
        """Test training with PPO algorithm."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg["algorithm"] = "ppo"
        trainer_cfg["num_iterations"] = 2

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
            patch("ray.shutdown") as mock_shutdown,
            patch("src.agents.trainer.tune.Tuner") as mock_tuner_cls,
        ):

            mock_tuner = Mock()
            mock_tuner.fit.return_value = Mock()
            mock_tuner_cls.return_value = mock_tuner

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)
            trainer.train()

            mock_tuner_cls.assert_called_once()
            mock_tuner.fit.assert_called_once()
            mock_shutdown.assert_called_once()

    def test_trainer_train_dqn(self, sample_configs, temp_dir):
        """Test training with DQN algorithm."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg["algorithm"] = "dqn"
        trainer_cfg["num_iterations"] = 1

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
            patch("ray.shutdown"),
            patch("src.agents.trainer.tune.Tuner") as mock_tuner_cls,
        ):

            mock_tuner = Mock()
            mock_tuner.fit.return_value = Mock()
            mock_tuner_cls.return_value = mock_tuner

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)
            trainer.train()

            mock_tuner_cls.assert_called_once()
            mock_tuner.fit.assert_called_once()

    def test_trainer_train_saves_checkpoints(self, sample_configs, temp_dir):
        """Test that training saves checkpoints to correct directory."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg["num_iterations"] = 1

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
            patch("ray.shutdown"),
            patch("src.agents.trainer.tune.Tuner") as mock_tuner_cls,
        ):

            mock_tuner = Mock()
            mock_tuner.fit.return_value = Mock()
            mock_tuner_cls.return_value = mock_tuner

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)
            trainer.train()

            args, kwargs = mock_tuner_cls.call_args
            assert kwargs["run_config"].storage_path == f"file://{temp_dir}"


class TestTrainerEvaluation:
    """Test Trainer evaluation functionality."""

    def test_trainer_evaluate_exists(self, sample_configs, temp_dir):
        """Test that evaluate method exists and can be called."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            # Should have evaluate method
            assert hasattr(trainer, "evaluate")
            assert callable(trainer.evaluate)

            # Should be able to call it (even if it doesn't do anything yet)
            try:
                trainer.evaluate()
            except NotImplementedError:
                # This is acceptable - method exists but not implemented
                pass


class TestTrainerTesting:
    """Test Trainer test functionality."""

    def test_trainer_test_exists(self, sample_configs, temp_dir):
        """Test that test method exists and can be called."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            # Should have test method
            assert hasattr(trainer, "test")
            assert callable(trainer.test)

            # Should be able to call it (even if it doesn't do anything yet)
            try:
                trainer.test()
            except NotImplementedError:
                # This is acceptable - method exists but not implemented
                pass


class TestTrainerErrorHandling:
    """Test Trainer error handling."""

    def test_trainer_handles_ray_config_defaults(self, sample_configs, temp_dir):
        """Test that Trainer handles missing ray_config gracefully."""
        env_cfg, model_cfg, trainer_cfg = sample_configs
        trainer_cfg.pop("ray_config", None)

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert "env" in trainer.ray_config
            assert trainer.ray_config["env"] == "TraderEnv"
            assert "env_config" in trainer.ray_config
            assert trainer.ray_config["env_config"] == env_cfg

    def test_trainer_creates_save_directory(self, sample_configs):
        """Test that Trainer creates save directory if it doesn't exist."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "new_subdir", "models")

            with (
                patch("ray.init"),
                patch("ray.is_initialized", return_value=False),
                patch("src.agents.trainer.register_env"),
            ):

                trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=save_dir)

                assert os.path.exists(save_dir)


class TestTrainerIntegration:
    """Integration tests for Trainer."""

    def test_trainer_with_file_configs(self, config_files, temp_dir):
        """Test Trainer integration with actual config files."""
        env_path, model_path, trainer_path = config_files

        # This would be tested in main.py, but we can verify config loading
        with open(env_path) as f:
            env_cfg = yaml.safe_load(f)
        with open(model_path) as f:
            model_cfg = yaml.safe_load(f)
        with open(trainer_path) as f:
            trainer_cfg = yaml.safe_load(f)

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            assert trainer.env_cfg == env_cfg
            assert trainer.model_cfg == model_cfg
            assert trainer.trainer_cfg == trainer_cfg

    def test_trainer_ray_config_integration(self, sample_configs, temp_dir):
        """Test that ray_config is properly integrated."""
        env_cfg, model_cfg, trainer_cfg = sample_configs

        custom_ray_config = {"env": "CustomEnv", "framework": "torch", "num_workers": 2}
        trainer_cfg["ray_config"] = custom_ray_config

        with (
            patch("ray.init"),
            patch("ray.is_initialized", return_value=False),
            patch("src.agents.trainer.register_env"),
        ):

            trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=temp_dir)

            # Should preserve custom config but add env_config
            assert trainer.ray_config["env"] == "CustomEnv"
            assert trainer.ray_config["framework"] == "torch"
            assert trainer.ray_config["num_workers"] == 2
            assert trainer.ray_config["env_config"] == env_cfg
