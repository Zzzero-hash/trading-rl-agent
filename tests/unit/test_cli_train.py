"""
Unit tests for CLI train functions.

Tests the train CLI command functions and their logic.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import typer

from trading_rl_agent.cli_train import (
    train,
    resume,
    get_trainer,
)


class TestCLITrain:
    """Test CLI train functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================

    def test_get_trainer_rl_algorithms(self):
        """Test get_trainer with RL algorithms."""
        # Test PPO
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_ppo = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('ppo')
            assert trainer == mock_module.train_ppo

        # Test SAC
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_sac = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('sac')
            assert trainer == mock_module.train_sac

        # Test TD3
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_td3 = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('td3')
            assert trainer == mock_module.train_td3

    def test_get_trainer_supervised_algorithms(self):
        """Test get_trainer with supervised algorithms."""
        # Test DQN
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_dqn = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('dqn')
            assert trainer == mock_module.train_dqn

        # Test LSTM
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_lstm = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('lstm')
            assert trainer == mock_module.train_lstm

        # Test CNN+LSTM
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_cnn_lstm = Mock()
            mock_import.return_value = mock_module
            
            trainer = get_trainer('cnn_lstm')
            assert trainer == mock_module.train_cnn_lstm

    def test_get_trainer_invalid_algorithm(self):
        """Test get_trainer with invalid algorithm."""
        with pytest.raises(ImportError):
            get_trainer('invalid_algorithm')

    def test_get_trainer_import_error(self):
        """Test get_trainer when module import fails."""
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(ImportError):
                get_trainer('ppo')

    def test_get_trainer_attribute_error(self):
        """Test get_trainer when function doesn't exist in module."""
        with patch('trading_rl_agent.cli_train.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.train_ppo = None  # Function doesn't exist
            mock_import.return_value = mock_module
            
            with pytest.raises(ImportError):
                get_trainer('ppo')

    # ============================================================================
    # COMMAND FUNCTIONS
    # ============================================================================

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_success(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with successful training."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        train(
            config_file=None,
            epochs=50,
            lr=0.0005,
            devices='cuda',
            checkpoint_out=self.temp_path,
            log_interval=5
        )
        
        # Verify calls
        mock_console.print.assert_called()
        mock_trainer_func.assert_called_once()
        
        # Verify trainer was called with correct arguments
        call_args = mock_trainer_func.call_args[1]
        assert call_args['settings'] == mock_settings
        assert call_args['epochs'] == 50
        assert call_args['learning_rate'] == 0.0005
        assert call_args['devices'] == 'cuda'
        assert call_args['checkpoint_dir'] == str(self.temp_path)
        assert call_args['log_interval'] == 5

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_failure(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with failed training."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer that fails
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': False, 'error': 'Training failed'}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        with pytest.raises(typer.Exit):
            train(
                config_file=None,
                epochs=50,
                lr=0.0005,
                devices='cuda',
                checkpoint_out=self.temp_path,
                log_interval=5
            )
        
        # Verify error message was printed
        mock_console.print.assert_called()
        error_call = [call for call in mock_console.print.call_args_list 
                     if 'Training failed' in str(call)]
        assert len(error_call) > 0

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_exception(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command when trainer raises exception."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer that raises exception
        mock_trainer_func = Mock()
        mock_trainer_func.side_effect = Exception("Unexpected error")
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        with pytest.raises(typer.Exit):
            train(
                config_file=None,
                epochs=50,
                lr=0.0005,
                devices='cuda',
                checkpoint_out=self.temp_path,
                log_interval=5
            )
        
        # Verify error was handled
        mock_console.print.assert_called()

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_with_config_file(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with config file."""
        # Create config file
        config_file = self.temp_path / "config.yaml"
        config_file.touch()
        
        # Mock settings from config file
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_load_settings.return_value = mock_settings
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        train(
            config_file=config_file,
            epochs=None,
            lr=None,
            devices=None,
            checkpoint_out=None,
            log_interval=10
        )
        
        # Verify config was loaded
        mock_load_settings.assert_called_once_with(config_path=config_file)
        
        # Verify trainer was called with config values
        call_args = mock_trainer_func.call_args[1]
        assert call_args['epochs'] == 100
        assert call_args['learning_rate'] == 0.001
        assert call_args['devices'] == 'cpu'
        assert call_args['checkpoint_dir'] == str(self.temp_path)

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_resume_command_success(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test resume command with successful resume."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Create checkpoint file
        checkpoint_file = self.temp_path / "model.ckpt"
        checkpoint_file.touch()
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        resume(
            config_file=None,
            devices='cuda',
            log_interval=5
        )
        
        # Verify calls
        mock_console.print.assert_called()
        mock_trainer_func.assert_called_once()
        
        # Verify trainer was called with resume parameters
        call_args = mock_trainer_func.call_args[1]
        assert call_args['resume_from'] == str(checkpoint_file)
        assert call_args['devices'] == 'cuda'
        assert call_args['checkpoint_dir'] == str(self.temp_path)
        assert call_args['log_interval'] == 5

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    def test_resume_command_no_checkpoint(
        self, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test resume command when no checkpoint exists."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Test the command (no checkpoint files exist)
        with pytest.raises(typer.Exit):
            resume(
                config_file=None,
                devices='cuda',
                log_interval=5
            )
        
        # Verify error message was printed
        mock_console.print.assert_called()

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_resume_command_failure(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test resume command with failed resume."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Create checkpoint file
        checkpoint_file = self.temp_path / "model.ckpt"
        checkpoint_file.touch()
        
        # Mock trainer that fails
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': False, 'error': 'Resume failed'}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        with pytest.raises(typer.Exit):
            resume(
                config_file=None,
                devices='cuda',
                log_interval=5
            )
        
        # Verify error message was printed
        mock_console.print.assert_called()
        error_call = [call for call in mock_console.print.call_args_list 
                     if 'Resume failed' in str(call)]
        assert len(error_call) > 0

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_resume_command_exception(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test resume command when trainer raises exception."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Create checkpoint file
        checkpoint_file = self.temp_path / "model.ckpt"
        checkpoint_file.touch()
        
        # Mock trainer that raises exception
        mock_trainer_func = Mock()
        mock_trainer_func.side_effect = Exception("Unexpected error")
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command
        with pytest.raises(typer.Exit):
            resume(
                config_file=None,
                devices='cuda',
                log_interval=5
            )
        
        # Verify error was handled
        mock_console.print.assert_called()

    # ============================================================================
    # EDGE CASES
    # ============================================================================

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_with_none_values(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with None values (should use defaults)."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command with None values
        train(
            config_file=None,
            epochs=None,
            lr=None,
            devices=None,
            checkpoint_out=None,
            log_interval=10
        )
        
        # Verify trainer was called with default values
        call_args = mock_trainer_func.call_args[1]
        assert call_args['epochs'] == 100
        assert call_args['learning_rate'] == 0.001
        assert call_args['devices'] == 'cpu'
        assert call_args['checkpoint_dir'] == str(self.temp_path)

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_with_zero_values(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with zero values."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command with zero values
        train(
            config_file=None,
            epochs=0,
            lr=0.0,
            devices='cpu',
            checkpoint_out=self.temp_path,
            log_interval=0
        )
        
        # Verify trainer was called with zero values
        call_args = mock_trainer_func.call_args[1]
        assert call_args['epochs'] == 0
        assert call_args['learning_rate'] == 0.0
        assert call_args['log_interval'] == 0

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    @patch('trading_rl_agent.cli_train.get_trainer')
    def test_train_command_with_negative_values(
        self, mock_get_trainer, mock_get_settings, mock_load_settings, mock_console
    ):
        """Test train command with negative values."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.model.algorithm = 'ppo'
        mock_settings.model.epochs = 100
        mock_settings.model.learning_rate = 0.001
        mock_settings.model.device = 'cpu'
        mock_settings.model.checkpoint_dir = str(self.temp_path)
        mock_get_settings.return_value = mock_settings
        
        # Mock trainer
        mock_trainer_func = Mock()
        mock_trainer_func.return_value = {'success': True}
        mock_get_trainer.return_value = mock_trainer_func
        
        # Test the command with negative values
        train(
            config_file=None,
            epochs=-10,
            lr=-0.001,
            devices='cpu',
            checkpoint_out=self.temp_path,
            log_interval=-5
        )
        
        # Verify trainer was called with negative values
        call_args = mock_trainer_func.call_args[1]
        assert call_args['epochs'] == -10
        assert call_args['learning_rate'] == -0.001
        assert call_args['log_interval'] == -5

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    @patch('trading_rl_agent.cli_train.console')
    def test_train_command_invalid_config_file(self, mock_console):
        """Test train command with invalid config file."""
        with pytest.raises(typer.Exit):
            train(
                config_file=Path("/nonexistent/config.yaml"),
                epochs=50,
                lr=0.0005,
                devices='cuda',
                checkpoint_out=self.temp_path,
                log_interval=5
            )

    @patch('trading_rl_agent.cli_train.console')
    def test_resume_command_invalid_config_file(self, mock_console):
        """Test resume command with invalid config file."""
        with pytest.raises(typer.Exit):
            resume(
                config_file=Path("/nonexistent/config.yaml"),
                devices='cuda',
                log_interval=5
            )

    @patch('trading_rl_agent.cli_train.console')
    @patch('trading_rl_agent.cli_train.load_settings')
    @patch('trading_rl_agent.cli_train.get_settings')
    def test_resume_command_invalid_checkpoint_dir(self, mock_get_settings, mock_load_settings, mock_console):
        """Test resume command with invalid checkpoint directory."""
        # Mock settings with non-existent checkpoint dir
        mock_settings = Mock()
        mock_settings.model.checkpoint_dir = "/nonexistent/checkpoint/dir"
        mock_get_settings.return_value = mock_settings
        
        with pytest.raises(typer.Exit):
            resume(
                config_file=None,
                devices='cuda',
                log_interval=5
            )