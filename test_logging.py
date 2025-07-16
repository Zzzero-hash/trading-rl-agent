#!/usr/bin/env python3
"""
Test script to demonstrate the logging configuration with different verbose levels.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_rl_agent.logging_conf import get_logger, setup_logging_for_typer


def test_logging_levels() -> None:
    """Test different logging levels."""

    print("=== Testing Logging Configuration ===\n")

    # Test each verbose level
    for level in range(4):
        print(f"--- Verbose Level {level} ---")
        setup_logging_for_typer(level)

        # Test different loggers
        logger = get_logger("trading_rl_agent.test")
        logger.info("This is an info message")
        logger.debug("This is a debug message")
        logger.warning("This is a warning message")

        # Test external logger (should only work for level >= 2)
        if level >= 2:
            ext_logger = get_logger("torch")
            ext_logger.info("External library logging enabled")

        print()


if __name__ == "__main__":
    test_logging_levels()
