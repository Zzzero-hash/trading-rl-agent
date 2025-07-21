#!/usr/bin/env python3
"""
Minimal CLI for Trading RL Agent - Basic functionality without heavy dependencies.

This provides basic CLI commands that work with core dependencies only.
"""

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Initialize console for rich output
console = Console()

# Global state
verbose_count: int = 0

# Root app
app = typer.Typer(
    name="trade-agent",
    help="Production-grade trading RL agent with CNN+LSTM integration",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (use multiple times for more detail)",
        ),
    ] = 0,
) -> None:
    """
    Trading RL Agent - Production-grade live trading system.

    A hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning
    with deep RL optimization for algorithmic trading.
    """
    global verbose_count
    verbose_count = verbose

    # Setup logging
    try:
        from trade_agent.logging_conf import setup_logging_for_typer

        setup_logging_for_typer(verbose)
    except ImportError:
        # Fallback logging setup
        import logging

        log_level = logging.DEBUG if verbose > 0 else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


@app.command()
def version() -> None:
    """Show version information."""
    table = Table(title="Trading RL Agent")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    table.add_row("Trading RL Agent", "2.0.0")
    table.add_row("Python", sys.version.split()[0])

    console.print(table)


@app.command()
def info() -> None:
    """Show system information and configuration."""
    table = Table(title="System Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Create minimal settings for display
    table.add_row("Environment", "development")
    table.add_row("Debug Mode", "False")
    table.add_row("Data Source", "yfinance")
    table.add_row("Symbols", "AAPL, GOOGL, MSFT")
    table.add_row("Agent Type", "ppo")
    table.add_row("Risk Management", "Enabled")
    table.add_row("Execution Broker", "alpaca")
    table.add_row("Paper Trading", "True")

    console.print(table)


@app.command()
def show_help() -> None:
    """Show detailed help information."""
    console.print("[bold blue]Trading RL Agent - Available Commands[/bold blue]")
    console.print("")
    console.print("[bold green]Basic Commands:[/bold green]")
    console.print("  version    - Show version information")
    console.print("  info       - Show system information")
    console.print("  help       - Show this help message")
    console.print("")
    console.print("[bold yellow]Note:[/bold yellow] This is a minimal installation with core dependencies only.")
    console.print("For full functionality, install additional dependencies:")
    console.print("  pip install -r requirements-ml.txt    # For ML features")
    console.print("  pip install -r requirements-full.txt  # For all features")


if __name__ == "__main__":
    app()
