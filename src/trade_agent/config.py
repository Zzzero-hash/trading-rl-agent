"""
Configuration module for trade_agent package.
Re-exports configuration functions from their actual locations.
"""

import sys
from pathlib import Path

# Add the project root to the path to import from root config.py
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import get_settings, load_settings

from .core.logging import get_logger

__all__ = ["get_logger", "get_settings", "load_settings"]
