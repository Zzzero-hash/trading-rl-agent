import os
import sys
import types

import pandas as pd
import pytest

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
        get_logger=lambda name=None: None,
    )
    sys.modules["structlog"] = stub

import importlib.util
from pathlib import Path

base = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "trading_rl_agent"
    / "data"
    / "loaders"
)

spec_yf = importlib.util.spec_from_file_location(
    "load_yfinance", base / "yfinance_loader.py"
)
yf_mod = importlib.util.module_from_spec(spec_yf)
spec_yf.loader.exec_module(yf_mod)  # type: ignore
load_yfinance = yf_mod.load_yfinance

spec_av = importlib.util.spec_from_file_location(
    "load_alphavantage", base / "alphavantage_loader.py"
)
av_mod = importlib.util.module_from_spec(spec_av)
spec_av.loader.exec_module(av_mod)  # type: ignore
load_alphavantage = av_mod.load_alphavantage

spec_ccxt = importlib.util.spec_from_file_location("load_ccxt", base / "ccxt_loader.py")
ccxt_mod = importlib.util.module_from_spec(spec_ccxt)
spec_ccxt.loader.exec_module(ccxt_mod)  # type: ignore
load_ccxt = ccxt_mod.load_ccxt

pytestmark = [pytest.mark.integration, pytest.mark.network]


@pytest.mark.skipif(
    load_yfinance.__globals__.get("yf") is None, reason="yfinance not installed"
)
def test_yfinance_loader_live():
    df = load_yfinance("AAPL", start="2023-01-01", end="2023-01-05", interval="day")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(df.columns)


@pytest.mark.skipif(
    load_alphavantage.__globals__.get("TimeSeries") is None,
    reason="alpha_vantage not installed",
)
def test_alphavantage_loader_live(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
    df = load_alphavantage("IBM", start="2023-01-01", end="2023-01-10", interval="day")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(df.columns)


@pytest.mark.skipif(
    load_ccxt.__globals__.get("ccxt") is None, reason="ccxt not installed"
)
def test_ccxt_loader_live():
    try:
        df = load_ccxt(
            "BTC/USDT",
            start="2023-01-01",
            end="2023-01-03",
            interval="day",
            exchange="binance",
        )
    except Exception:
        pytest.skip("Network unavailable for ccxt test")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(df.columns)
