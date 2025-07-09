"""
Redirect package path to include src/trading_rl_agent.
"""

import sys
from pathlib import Path

# Insert the source package path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "trading_rl_agent"))
# Also adjust __path__ for package modules
__path__.insert(0, str(Path(__file__).parent.parent / "src" / "trading_rl_agent"))
