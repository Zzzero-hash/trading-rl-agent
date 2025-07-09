#!/usr/bin/env python3
"""
Production entry point for Trading RL Agent System.
This provides a clean interface to the production CLI.
"""

from pathlib import Path
import sys

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cli import main as cli_main

if __name__ == "__main__":
    cli_main()
