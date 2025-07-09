# Add src directory to PYTHONPATH so that pytest can find the trading_rl_agent package
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
