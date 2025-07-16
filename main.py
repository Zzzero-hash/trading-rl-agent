#!/usr/bin/env python3
"""
Main entry point for Trading RL Agent CLI.

This provides a clean interface to the unified Typer CLI.
"""

import sys
from pathlib import Path

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.cli import app

if __name__ == "__main__":
    app()
