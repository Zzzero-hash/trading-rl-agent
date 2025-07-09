import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.training.cnn_lstm import CNNLSTMTrainer, TrainingConfig


class DummySentimentAnalyzer:
    def get_symbol_sentiment(self, symbol, days_back=1):
        return 0.5


def test_add_sentiment_features_with_forex(monkeypatch):
    # Create dummy data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "open": np.arange(5),
            "high": np.arange(5),
            "low": np.arange(5),
            "close": np.arange(5),
            "volume": np.arange(5),
        }
    )
    symbols = ["AAPL"]
    forex_pairs = ["EURUSD"]
    trainer = CNNLSTMTrainer(TrainingConfig(include_sentiment=True))
    trainer.sentiment_analyzer = DummySentimentAnalyzer()
    # Patch get_forex_sentiment to return a fixed score
    import importlib

    forex_mod = importlib.import_module("src.data.forex_sentiment")
    monkeypatch.setattr(
        forex_mod,
        "get_forex_sentiment",
        lambda pair: [type("Obj", (), {"score": 0.7})()],
    )
    features = trainer._add_sentiment_features(df, symbols, forex_pairs)
    assert "sentiment_AAPL" in features.columns
    assert "forex_sentiment_EURUSD" in features.columns
    assert all(features["sentiment_AAPL"] == 0.5)
    assert all(features["forex_sentiment_EURUSD"] == 0.7)
