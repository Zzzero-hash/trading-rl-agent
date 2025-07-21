import sys
import types
from pathlib import Path

import pytest

from trade_agent.core.config import ConfigManager, SystemConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "src.envs.finrl_trading_env" not in sys.modules:
    sys.modules["src.envs.finrl_trading_env"] = types.SimpleNamespace(
        register_env=lambda: None,
    )

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"),
    ]
    sys.modules["trading_rl_agent"] = pkg

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")

    class DummySIA:
        def polarity_scores(self, _text):
            """
            Return a neutral sentiment score for the given text.

            Parameters:
                text (str): The input text to analyze.

            Returns:
                dict: A dictionary with a single key "compound" set to 0.0, indicating neutral sentiment.
            """
            return {"compound": 0.0}

    dummy.SentimentIntensityAnalyzer = DummySIA
    sys.modules["nltk.sentiment.vader"] = dummy
base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
for pkg in ["features", "portfolio", "risk"]:
    key = f"trading_rl_agent.{pkg}"
    if key not in sys.modules:
        mod = types.ModuleType(key)
        mod.__path__ = [str(base / pkg)]
        sys.modules[key] = mod

pytestmark = pytest.mark.unit


def test_default_config():
    """
    Test that the default configuration is loaded correctly and contains expected default values.
    """
    manager = ConfigManager()
    cfg = manager.load_config()
    assert isinstance(cfg, SystemConfig)
    assert cfg.environment == "development"
    assert cfg.data.cache_enabled is True


def test_load_update_save_config(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    manager = ConfigManager(config_path=config_path)

    # Create and save a default config
    default_config = manager.get_config()
    default_config.debug = True
    manager.save_config(default_config)

    # Load and verify
    loaded_config = manager.load_config()
    assert loaded_config.debug is True

    # Update and save
    loaded_config.debug = False
    manager.save_config(loaded_config)

    # Reload and verify update
    reloaded_config = manager.load_config()
    assert reloaded_config.debug is False
