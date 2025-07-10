import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: logging.getLogger(name),
    )
    sys.modules["structlog"] = stub

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
        def polarity_scores(self, text):
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
import pytest
import yaml

from trading_rl_agent.core.config import ConfigManager, SystemConfig

pytestmark = pytest.mark.unit


def test_default_config(tmp_path):
    """
    Test that the default configuration is loaded correctly and contains expected default values.
    """
    manager = ConfigManager()
    cfg = manager.load_config()
    assert isinstance(cfg, SystemConfig)
    assert cfg.environment == "development"
    assert cfg.data.cache_enabled is True


def test_load_update_save_config(tmp_path):
    """
    Test loading, updating, and saving a configuration using ConfigManager.

    Creates a YAML configuration file, loads it, verifies its contents, updates a value, saves the updated configuration, and checks that the saved file reflects the changes.
    """
    config_file = tmp_path / "config.yaml"
    data = {"environment": "production", "debug": True, "risk": {"max_drawdown": 0.2}}
    with Path(config_file).open(config_file, "w") as f:
        yaml.dump(data, f)

    manager = ConfigManager(config_file)
    cfg = manager.load_config()
    assert cfg.environment == "production"
    assert cfg.debug is True
    assert cfg.risk.max_drawdown == 0.2

    manager.update_config({"risk": {"max_leverage": 2.0}})
    assert manager.get_config().risk.max_leverage == 2.0

    out_file = tmp_path / "out.yaml"
    manager.save_config(manager.get_config(), out_file)
    saved = yaml.safe_load(out_file.read_text())
    assert saved["risk"]["max_leverage"] == 2.0
