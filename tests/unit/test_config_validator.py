"""Unit tests for configuration validator module."""


import pytest

from trade_agent.core.config_validator import ConfigValidationError, ConfigValidator
from trade_agent.core.unified_config import UnifiedConfig


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration for testing."""
        config_data = {
            "model": {
                "type": "cnn_lstm",
                "cnn_layers": [32, 64],
                "lstm_units": 128,
                "dropout": 0.2,
                "sequence_length": 30
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            },
            "data": {
                "symbols": ["AAPL", "MSFT"],
                "features": ["open", "high", "low", "close", "volume"]
            },
            "risk": {
                "max_position_size": 0.1,
                "max_drawdown": 0.2
            }
        }
        return UnifiedConfig(config_data)

    @pytest.fixture
    def invalid_config(self):
        """Create an invalid configuration for testing."""
        config_data = {
            "model": {
                "type": "invalid_model",
                "cnn_layers": [],  # Invalid empty layers
                "lstm_units": -10,  # Invalid negative units
                "dropout": 1.5,  # Invalid dropout rate
                "sequence_length": 0  # Invalid sequence length
            },
            "training": {
                "batch_size": 0,  # Invalid batch size
                "epochs": -5,  # Invalid epochs
                "learning_rate": 10.0  # Invalid learning rate
            },
            "data": {
                "symbols": [],  # Empty symbols list
                "features": []  # Empty features list
            },
            "risk": {
                "max_position_size": 2.0,  # Invalid > 1.0
                "max_drawdown": -0.1  # Invalid negative drawdown
            }
        }
        return UnifiedConfig(config_data)

    def test_init(self, valid_config):
        """Test ConfigValidator initialization."""
        validator = ConfigValidator(valid_config)
        assert validator.config == valid_config
        assert validator.errors == []
        assert validator.warnings == []

    def test_validate_all_valid_config(self, valid_config):
        """Test validation of valid configuration."""
        validator = ConfigValidator(valid_config)
        result = validator.validate_all()
        assert result is True
        assert len(validator.errors) == 0

    def test_validate_all_invalid_config(self, invalid_config):
        """Test validation of invalid configuration."""
        validator = ConfigValidator(invalid_config)
        result = validator.validate_all()
        assert result is False
        assert len(validator.errors) > 0

    def test_validate_model_valid(self, valid_config):
        """Test model validation with valid config."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_model()
        assert result is True

    def test_validate_model_invalid_type(self, valid_config):
        """Test model validation with invalid model type."""
        valid_config.config["model"]["type"] = "invalid_type"
        validator = ConfigValidator(valid_config)
        result = validator._validate_model()
        assert result is False
        assert any("model type" in error.lower() for error in validator.errors)

    def test_validate_training_params(self, valid_config):
        """Test training parameter validation."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_training()
        assert result is True

    def test_validate_training_invalid_batch_size(self, valid_config):
        """Test training validation with invalid batch size."""
        valid_config.config["training"]["batch_size"] = 0
        validator = ConfigValidator(valid_config)
        result = validator._validate_training()
        assert result is False
        assert any("batch_size" in error for error in validator.errors)

    def test_validate_data_config(self, valid_config):
        """Test data configuration validation."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_data()
        assert result is True

    def test_validate_data_empty_symbols(self, valid_config):
        """Test data validation with empty symbols."""
        valid_config.config["data"]["symbols"] = []
        validator = ConfigValidator(valid_config)
        result = validator._validate_data()
        assert result is False
        assert any("symbols" in error for error in validator.errors)

    def test_validate_risk_params(self, valid_config):
        """Test risk parameter validation."""
        validator = ConfigValidator(valid_config)
        result = validator._validate_risk()
        assert result is True

    def test_validate_risk_invalid_position_size(self, valid_config):
        """Test risk validation with invalid position size."""
        valid_config.config["risk"]["max_position_size"] = 1.5
        validator = ConfigValidator(valid_config)
        result = validator._validate_risk()
        assert result is False
        assert any("position_size" in error for error in validator.errors)

    def test_get_validation_report(self, invalid_config):
        """Test validation report generation."""
        validator = ConfigValidator(invalid_config)
        validator.validate_all()
        report = validator.get_validation_report()
        assert "errors" in report
        assert "warnings" in report
        assert len(report["errors"]) > 0

    def test_raise_on_errors(self, invalid_config):
        """Test raising exceptions on validation errors."""
        validator = ConfigValidator(invalid_config)
        validator.validate_all()
        with pytest.raises(ConfigValidationError):
            validator.raise_on_errors()

    def test_compatibility_check(self, valid_config):
        """Test compatibility between configuration sections."""
        validator = ConfigValidator(valid_config)
        result = validator._check_compatibility()
        assert result is True

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        incomplete_config = UnifiedConfig({"model": {"type": "cnn_lstm"}})
        validator = ConfigValidator(incomplete_config)
        result = validator.validate_all()
        assert result is False
        assert len(validator.errors) > 0

    @pytest.mark.parametrize("dropout_value,expected_valid", [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (-0.1, False),
        (1.1, False)
    ])
    def test_dropout_validation(self, valid_config, dropout_value, expected_valid):
        """Test dropout rate validation with various values."""
        valid_config.config["model"]["dropout"] = dropout_value
        validator = ConfigValidator(valid_config)
        result = validator._validate_model()
        assert result == expected_valid

    @pytest.mark.parametrize("learning_rate,expected_valid", [
        (0.001, True),
        (0.1, True),
        (1.0, False),  # Too high
        (0.0, False),  # Zero not allowed
        (-0.001, False)  # Negative not allowed
    ])
    def test_learning_rate_validation(self, valid_config, learning_rate, expected_valid):
        """Test learning rate validation with various values."""
        valid_config.config["training"]["learning_rate"] = learning_rate
        validator = ConfigValidator(valid_config)
        result = validator._validate_training()
        assert result == expected_valid
