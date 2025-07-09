import sys
import types
import logging
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency stubs
# ---------------------------------------------------------------------------
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

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")

    class DummySIA:
        def polarity_scores(self, text):
            return {"compound": 0.0}

    dummy.SentimentIntensityAnalyzer = DummySIA
    sys.modules["nltk.sentiment.vader"] = dummy

# ---------------------------------------------------------------------------
# Add src to Python path so the package is importable without installation
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_rl_agent.data.synthetic import fetch_synthetic_data
from trading_rl_agent.features import FeaturePipeline
from trading_rl_agent import PPOAgent


OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PATH = OUTPUT_DIR / "sample_data.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "ppo_agent_checkpoint.zip"


def load_data() -> pd.DataFrame:
    """Load bundled sample dataset or generate synthetic data."""
    packaged = Path(__file__).resolve().parents[1] / "data" / "sample_data.csv"
    if packaged.exists():
        print(f"Loading bundled dataset from {packaged}")
        return pd.read_csv(packaged)
    print("Bundled sample dataset not found, generating synthetic data.")
    return fetch_synthetic_data(n_samples=120)


def main() -> None:
    df = load_data()
    pipeline = FeaturePipeline()
    df = pipeline.transform(df)
    df = df.select_dtypes(include=["number"]).dropna()
    df.to_csv(DATA_PATH, index=False)
    print(f"Prepared data saved to {DATA_PATH}")

    state_dim = df.shape[1]
    agent = PPOAgent(state_dim=state_dim, action_dim=1)
    agent.train(total_timesteps=100)
    agent.save(str(CHECKPOINT_PATH))
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
