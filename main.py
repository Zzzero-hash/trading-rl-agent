#!/usr/bin/env python3
"""
DEPRECATED: Main entry point for Trading RL Agent.

This script is deprecated. Please use 'trade-agent' command instead.
Usage: trade-agent [command] [options]

This script provides a simple command-line interface for the trading RL agent.
Usage: python main.py [command] [options]
"""

import sys
from pathlib import Path

# Add the src directory to Python path to resolve imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main() -> None:
    """Main entry point."""
    import warnings
    warnings.warn(
        "main.py is deprecated. Please use 'trade-agent' command instead.",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        # Import and run the CLI app
        from trade_agent.cli import app
        app()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed and the project structure is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
