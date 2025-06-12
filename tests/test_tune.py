"""Tests for Ray Tune utilities."""

import pytest
import tempfile
import yaml
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.agents.tune import _convert_value, _load_search_space, run_tune


class TestConvertValue:
    """Test the _convert_value function."""
    
    def test_convert_grid_search(self):
        """Test conversion of grid_search specification."""
        spec = {"grid_search": [0.1, 0.01, 0.001]}
        
        with patch('ray.tune.grid_search') as mock_grid_search:
            mock_grid_search.return_value = "grid_search_result"
            result = _convert_value(spec)
            
            mock_grid_search.assert_called_once_with([0.1, 0.01, 0.001])
            assert result == "grid_search_result"
    
    def test_convert_choice(self):
        """Test conversion of choice specification."""
        spec = {"choice": ["adam", "sgd", "rmsprop"]}
        
        with patch('ray.tune.choice') as mock_choice:
            mock_choice.return_value = "choice_result"
            result = _convert_value(spec)
            
            mock_choice.assert_called_once_with(["adam", "sgd", "rmsprop"])
            assert result == "choice_result"
    
    def test_convert_uniform(self):
        """Test conversion of uniform distribution specification."""
        spec = {"uniform": [0.0, 1.0]}
        
        with patch('ray.tune.uniform') as mock_uniform:
            mock_uniform.return_value = "uniform_result"
            result = _convert_value(spec)
            
            mock_uniform.assert_called_once_with(0.0, 1.0)
            assert result == "uniform_result"
    
    def test_convert_uniform_tuple(self):
        """Test conversion of uniform distribution with tuple."""
        spec = {"uniform": (0.001, 0.1)}
        
        with patch('ray.tune.uniform') as mock_uniform:
            mock_uniform.return_value = "uniform_result"
            result = _convert_value(spec)
            
            mock_uniform.assert_called_once_with(0.001, 0.1)
            assert result == "uniform_result"
    
    def test_convert_randint(self):
        """Test conversion of randint specification."""
        spec = {"randint": [1, 100]}
        
        with patch('ray.tune.randint') as mock_randint:
            mock_randint.return_value = "randint_result"
            result = _convert_value(spec)
            
            mock_randint.assert_called_once_with(1, 100)
            assert result == "randint_result"
    
    def test_convert_randint_tuple(self):
        """Test conversion of randint with tuple."""
        spec = {"randint": (10, 1000)}
        
        with patch('ray.tune.randint') as mock_randint:
            mock_randint.return_value = "randint_result"
            result = _convert_value(spec)
            
            mock_randint.assert_called_once_with(10, 1000)
            assert result == "randint_result"
    
    def test_convert_plain_value(self):
        """Test that plain values are returned unchanged."""
        values = [
            42,
            3.14,
            "string_value",
            [1, 2, 3],
            {"key": "value"}
        ]
        
        for value in values:
            result = _convert_value(value)
            assert result == value
    
    def test_convert_dict_without_tune_specs(self):
        """Test that dictionaries without tune specs are returned unchanged."""
        spec = {"learning_rate": 0.001, "batch_size": 32}
        result = _convert_value(spec)
        assert result == spec
    
    def test_convert_invalid_uniform(self):
        """Test handling of invalid uniform specification."""
        spec = {"uniform": "invalid"}
        result = _convert_value(spec)
        assert result == spec  # Should return unchanged
    
    def test_convert_invalid_randint(self):
        """Test handling of invalid randint specification."""
        spec = {"randint": {"invalid": "format"}}
        result = _convert_value(spec)
        assert result == spec  # Should return unchanged


class TestLoadSearchSpace:
    """Test the _load_search_space function."""
    
    def test_load_search_space_basic(self, tmp_path):
        """Test loading basic search space from YAML."""
        config = {
            "learning_rate": {"uniform": [0.001, 0.1]},
            "batch_size": {"choice": [16, 32, 64]},
            "num_layers": {"randint": [2, 10]}
        }
        
        config_path = tmp_path / "search_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('src.agents.tune._convert_value', side_effect=lambda x: f"converted_{x}"):
            result = _load_search_space(str(config_path))
            
            assert "learning_rate" in result
            assert "batch_size" in result
            assert "num_layers" in result
            assert result["learning_rate"] == "converted_{'uniform': [0.001, 0.1]}"
    
    def test_load_search_space_empty_file(self, tmp_path):
        """Test loading search space from empty YAML file."""
        config_path = tmp_path / "empty_config.yaml"
        with open(config_path, 'w') as f:
            f.write("")  # Empty file
        
        result = _load_search_space(str(config_path))
        assert result == {}
    
    def test_load_search_space_none_config(self, tmp_path):
        """Test loading search space when YAML contains None."""
        config_path = tmp_path / "none_config.yaml"
        with open(config_path, 'w') as f:
            f.write("null\n")  # YAML for None
        
        result = _load_search_space(str(config_path))
        assert result == {}
    
    def test_load_search_space_mixed_values(self, tmp_path):
        """Test loading search space with mixed value types."""
        config = {
            "tuned_param": {"grid_search": [1, 2, 3]},
            "fixed_param": 42,
            "nested_config": {
                "inner_param": {"choice": ["a", "b"]}
            }
        }
        
        config_path = tmp_path / "mixed_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock _convert_value to only convert tune specs
        def mock_convert(value):
            if isinstance(value, dict) and any(k in value for k in ["grid_search", "choice", "uniform", "randint"]):
                return f"tune_object_{value}"
            return value
        
        with patch('src.agents.tune._convert_value', side_effect=mock_convert):
            result = _load_search_space(str(config_path))
            
            assert result["tuned_param"] == "tune_object_{'grid_search': [1, 2, 3]}"
            assert result["fixed_param"] == 42
            assert result["nested_config"]["inner_param"] == "tune_object_{'choice': ['a', 'b']}"


class TestRunTune:
    """Test the run_tune function."""
    
    def test_run_tune_single_config(self, tmp_path):
        """Test run_tune with single configuration file."""
        config = {
            "algorithm": "PPO",
            "env_config": {"window_size": 50},
            "learning_rate": {"uniform": [0.001, 0.1]},
            "batch_size": {"choice": [16, 32, 64]}
        }
        
        config_path = tmp_path / "tune_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init') as mock_ray_init, \
             patch('ray.is_initialized', return_value=False) as mock_ray_initialized, \
             patch('ray.tune.run') as mock_tune_run, \
             patch('ray.shutdown') as mock_ray_shutdown, \
             patch('src.agents.tune.register_env') as mock_register_env, \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(str(config_path))
            
            # Verify Ray initialization and environment registration
            mock_ray_init.assert_called_once()
            mock_register_env.assert_called_once()
            
            # Verify tune.run was called
            mock_tune_run.assert_called_once()
            
            # Check the search space passed to tune.run
            call_args = mock_tune_run.call_args
            assert call_args is not None
    
    def test_run_tune_multiple_configs(self, tmp_path):
        """Test run_tune with multiple configuration files."""
        config1 = {
            "algorithm": "PPO",
            "learning_rate": {"uniform": [0.001, 0.1]}
        }
        
        config2 = {
            "env_config": {"window_size": 50},
            "batch_size": {"choice": [16, 32, 64]}
        }
        
        config_path1 = tmp_path / "config1.yaml"
        config_path2 = tmp_path / "config2.yaml"
        
        with open(config_path1, 'w') as f:
            yaml.dump(config1, f)
        with open(config_path2, 'w') as f:
            yaml.dump(config2, f)
        
        config_paths = [str(config_path1), str(config_path2)]
        
        with patch('ray.init') as mock_ray_init, \
             patch('ray.is_initialized', return_value=False) as mock_ray_is_initialized, \
             patch('ray.tune.run') as mock_tune_run, \
             patch('ray.shutdown') as mock_ray_shutdown, \
             patch('src.agents.tune.register_env') as mock_register_env, \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(config_paths)
            
            mock_ray_is_initialized.assert_called_once()
            mock_ray_init.assert_called_once()
            mock_register_env.assert_called_once()
            mock_tune_run.assert_called_once()
            mock_ray_shutdown.assert_called_once()
    
    def test_run_tune_string_input(self, tmp_path):
        """Test run_tune with string input (single config path)."""
        config = {
            "algorithm": "DQN",
            "learning_rate": 0.001
        }
        
        config_path = tmp_path / "single_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init'), \
             patch('ray.tune.run'), \
             patch('src.agents.tune.register_env'), \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            # Should handle string input by converting to list
            run_tune(str(config_path))
    
    def test_run_tune_algorithm_extraction(self, tmp_path):
        """Test that algorithm is properly extracted from search space."""
        config = {
            "algorithm": "PPO",
            "learning_rate": {"uniform": [0.001, 0.1]},
            "env_config": {"window_size": 50}
        }
        
        config_path = tmp_path / "algo_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env'), \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(str(config_path))
            
            # Verify that algorithm was removed from search space
            call_args = mock_tune_run.call_args
            search_space = call_args[1]['config']  # Assuming config is passed as kwarg
            assert "algorithm" not in search_space
    
    def test_run_tune_env_config_extraction(self, tmp_path):
        """Test that env_config is properly extracted from search space."""
        config = {
            "algorithm": "PPO",
            "learning_rate": {"uniform": [0.001, 0.1]},
            "env_config": {"window_size": 50, "initial_balance": 10000}
        }
        
        config_path = tmp_path / "env_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env'), \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(str(config_path))
            
            # Verify that env_config was removed from search space
            call_args = mock_tune_run.call_args
            search_space = call_args[1]['config']
            assert "env_config" not in search_space
    
    def test_run_tune_default_algorithm(self, tmp_path):
        """Test run_tune with default algorithm when not specified."""
        config = {
            "learning_rate": {"uniform": [0.001, 0.1]},
            "batch_size": 32
        }
        
        config_path = tmp_path / "no_algo_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env'), \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(str(config_path))
            
            # Should use default algorithm
            call_args = mock_tune_run.call_args
            # The algorithm should be used in the trainer configuration
            assert mock_tune_run.called
    
    def test_run_tune_default_env_config(self, tmp_path):
        """Test run_tune with default env_config when not specified."""
        config = {
            "algorithm": "PPO",
            "learning_rate": {"uniform": [0.001, 0.1]}
        }
        
        config_path = tmp_path / "no_env_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env'), \
             patch('src.agents.tune._convert_value', side_effect=lambda x: x):
            
            run_tune(str(config_path))
            
            # Should use default empty env_config
            assert mock_tune_run.called


class TestTuneIntegration:
    """Integration tests for tune functionality."""
    
    def test_tune_complete_workflow(self, tmp_path):
        """Test complete tune workflow with realistic configuration."""
        config = {
            "algorithm": "PPO",
            "env_config": {
                "dataset_paths": ["dummy_data.csv"],
                "window_size": {"choice": [10, 20, 50]},
                "initial_balance": 10000,
                "transaction_cost": {"uniform": [0.001, 0.01]}
            },
            "learning_rate": {"uniform": [0.0001, 0.01]},
            "gamma": {"uniform": [0.9, 0.999]},
            "num_iterations": {"randint": [10, 100]}
        }
        
        config_path = tmp_path / "complete_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('ray.init') as mock_ray_init, \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env') as mock_register_env:
            
            # Mock the tune conversion functions
            def mock_convert(value):
                if isinstance(value, dict):
                    if "choice" in value:
                        return f"tune.choice({value['choice']})"
                    elif "uniform" in value:
                        return f"tune.uniform({value['uniform'][0]}, {value['uniform'][1]})"
                    elif "randint" in value:
                        return f"tune.randint({value['randint'][0]}, {value['randint'][1]})"
                return value
            
            with patch('src.agents.tune._convert_value', side_effect=mock_convert):
                run_tune(str(config_path))
                
                # Verify all components were called
                mock_ray_init.assert_called_once()
                mock_register_env.assert_called_once()
                mock_tune_run.assert_called_once()
                
                # Verify search space structure
                call_args = mock_tune_run.call_args
                search_space = call_args[1]['config']
                
                # Algorithm and env_config should be extracted
                assert "algorithm" not in search_space
                assert "env_config" not in search_space
                
                # Tunable parameters should remain
                assert "learning_rate" in search_space
                assert "gamma" in search_space
                assert "num_iterations" in search_space


class TestTuneErrorHandling:
    """Test error handling in tune functionality."""
    
    def test_load_search_space_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            _load_search_space("nonexistent_file.yaml")
    
    def test_load_search_space_invalid_yaml(self, tmp_path):
        """Test handling of invalid YAML file."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            _load_search_space(str(config_path))
    
    def test_run_tune_empty_configs(self, tmp_path):
        """Test run_tune with empty configuration files."""
        config_path = tmp_path / "empty.yaml"
        with open(config_path, 'w') as f:
            f.write("")
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('src.agents.tune.register_env'):
            
            run_tune(str(config_path))
            
            # Should still call tune.run with empty search space
            mock_tune_run.assert_called_once()


@pytest.mark.integration
class TestTuneRealRay:
    """Integration tests that use real Ray components (skipped by default)."""
    
    @pytest.mark.skip(reason="Requires Ray cluster - enable for integration testing")
    def test_tune_real_ray_integration(self, tmp_path):
        """Test tune functionality with real Ray."""
        config = {
            "algorithm": "PPO",
            "learning_rate": {"uniform": [0.001, 0.01]},
            "env_config": {"dataset_paths": ["dummy.csv"]}
        }
        
        config_path = tmp_path / "real_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # This would test with real Ray initialization
        run_tune(str(config_path))
        
        # Verify Ray is initialized and cleanup
        import ray
        assert ray.is_initialized()
        ray.shutdown()
