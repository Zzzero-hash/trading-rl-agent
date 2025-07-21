import pytest

from trade_agent.agents.configs import EnsembleConfig
from trade_agent.agents.trainer import Trainer
from trade_agent.core.config import AgentConfig, SystemConfig
from trade_agent.core.exceptions import ConfigurationError


def test_ensemble_config_alias_and_validation():
    cfg = EnsembleConfig(agent_configs={"sac": {"enabled": True, "config": None}})
    assert cfg.agents == {"sac": {"enabled": True, "config": None}}


def test_ensemble_config_empty_agents():
    with pytest.raises(ValueError):
        EnsembleConfig(agents={})


def test_trainer_config_loading_and_validation():
    """Test trainer config loading and validation."""
    # Test that an invalid algorithm raises a ConfigurationError
    invalid_agent_config = AgentConfig(agent_type="invalid_algo")
    invalid_system_config = SystemConfig(agent=invalid_agent_config)
    with pytest.raises(ConfigurationError):
        Trainer(system_cfg=invalid_system_config)

    # Test that the default algorithm is 'sac'
    default_system_config = SystemConfig()
    trainer = Trainer(system_cfg=default_system_config)
    assert trainer.algorithm == "sac"
    assert isinstance(trainer.cfg.agent, AgentConfig)
