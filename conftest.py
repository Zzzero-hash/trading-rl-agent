# Add src directory to PYTHONPATH so that pytest can find the trading_rl_agent package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
