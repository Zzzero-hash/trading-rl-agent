import sys
from pathlib import Path
import types
import logging

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

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent")]
    sys.modules["trading_rl_agent"] = pkg

base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
for pkg_name in ["features"]:
    key = f"trading_rl_agent.{pkg_name}"
    if key not in sys.modules:
        mod = types.ModuleType(key)
        mod.__path__ = [str(base / pkg_name)]
        sys.modules[key] = mod

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_rl_agent.features.pipeline import FeaturePipeline  # noqa: E402


class DummyTechnicalIndicators:
    def calculate_all_indicators(self, df):
        df = df.copy()
        df["dummy"] = 1
        return df

    def get_feature_names(self):
        return ["dummy"]


pytestmark = pytest.mark.unit


def test_feature_pipeline_basic():
    n = 30
    df = pd.DataFrame(
        {
            "open": np.arange(1, n + 1, dtype=float),
            "high": np.arange(1, n + 1, dtype=float) + 1,
            "low": np.arange(1, n + 1, dtype=float) - 1,
            "close": np.arange(1, n + 1, dtype=float) + 0.5,
            "volume": np.arange(100, 100 + n, dtype=float),
        }
    )
    ref = pd.DataFrame({"close": np.linspace(10, 10 + n - 1, n)})

    pipeline = FeaturePipeline(technical=DummyTechnicalIndicators())
    result = pipeline.transform(df, cross_df=ref)

    expected_cols = [
        "hl_spread",
        "close_open_diff",
        "volume_imbalance",
        f"corr_{pipeline.cross_asset.config.prefix}",
        pipeline.alternative.config.sentiment_column,
    ]

    for col in expected_cols:
        assert col in result.columns
        assert result[col].notnull().any()
