"""
Live Trading CLI - Command-line interface for live trading operations.

This module provides CLI commands for:
- Starting live trading sessions
- Stopping trading sessions
- Managing trading configuration
- API key validation and management
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core.logging import get_logger, setup_logging

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_settings, load_settings

# Initialize console for rich output
console = Console()
logger = get_logger(__name__)

# Global state
verbose_count: int = 0
_settings = None
_trading_engine = None


def get_config_manager() -> Any:
    """Get or create config manager instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_log_level() -> str:
    """Get log level based on verbose count."""
    if verbose_count >= 3:
        return "DEBUG"
    if verbose_count >= 2:
        return "INFO"
    if verbose_count >= 1:
        return "WARNING"
    return "ERROR"


def get_trading_engine() -> Any:
    """Get or create trading engine instance."""
    # Placeholder for trading engine initialization
    return None


# Main app
app = typer.Typer(
    name="trade",
    help="Live trading operations: start, stop, monitor trading sessions",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def main(
    config_file: Path | None = None,
    verbose: int | None = None,
    env_file: Path | None = None,
) -> None:
    """
    Live Trading CLI - Manage live trading sessions.

    Provides commands to start, stop, and monitor live trading sessions
    with support for paper trading and real money trading.
    """
    if config_file is None:
        config_file = typer.Option(None, "--config", "-c", help="Path to configuration file")
    if verbose is None:
        verbose = typer.Option(
            0, "--verbose", "-v", count=True, help="Increase verbosity (use multiple times for more detail)"
        )
    if env_file is None:
        env_file = typer.Option(None, "--env-file", help="Path to environment file (.env)")

    global verbose_count, _settings

    # Set global verbose count
    verbose_count = verbose or 0

    # Setup logging
    log_level = get_log_level()
    setup_logging(log_level=log_level)

    # Load environment file if provided
    if env_file:
        if not env_file.exists():
            console.print(f"[red]Environment file not found: {env_file}[/red]")
            raise typer.Exit(1)

        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            console.print(f"[green]Loaded environment from: {env_file}[/green]")
        except ImportError:
            console.print("[yellow]python-dotenv not installed, skipping .env file[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load environment file: {e}[/red]")
            raise typer.Exit(1) from e

    # Load configuration if provided
    if config_file:
        if not config_file.exists():
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
            raise typer.Exit(1)

        try:
            _settings = load_settings(config_path=config_file, env_file=env_file)
            console.print(f"[green]Loaded configuration from: {config_file}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load configuration: {e}[/red]")
            raise typer.Exit(1) from e


def validate_api_credentials(settings: Any, exchange: str) -> bool:
    """Validate API credentials for the specified exchange."""
    credentials = settings.get_api_credentials(exchange)

    if exchange.lower() == "alpaca":
        if not credentials.get("api_key"):
            console.print(
                "[red]Missing Alpaca API key. Set TRADING_RL_AGENT_ALPACA_API_KEY environment variable.[/red]"
            )
            return False
        if not credentials.get("secret_key"):
            console.print(
                "[red]Missing Alpaca secret key. Set TRADING_RL_AGENT_ALPACA_SECRET_KEY environment variable.[/red]"
            )
            return False
        return True
    if exchange.lower() == "yfinance":
        # yfinance doesn't require API keys
        return True
    console.print(f"[red]Unsupported exchange: {exchange}[/red]")
    return False


DEFAULT_PAPER = True
DEFAULT_LIVE = False
DEFAULT_EXCHANGE: str | None = None
DEFAULT_SYMBOL: str | None = None
DEFAULT_SYMBOLS: str | None = None
DEFAULT_SIZE: float | None = None
DEFAULT_STRATEGY: str | None = None
DEFAULT_MODEL_PATH: Path | None = None
DEFAULT_CNN_LSTM_PATH: Path | None = None
DEFAULT_MAX_POSITION_SIZE: float | None = None
DEFAULT_STOP_LOSS: float | None = None
DEFAULT_TAKE_PROFIT: float | None = None
DEFAULT_INITIAL_CAPITAL: float | None = None
DEFAULT_UPDATE_INTERVAL: int | None = None
DEFAULT_CONFIG_FILE: Path | None = None
DEFAULT_SESSION_ID: str | None = None
DEFAULT_ALL_SESSIONS = False
DEFAULT_FORCE = False
DEFAULT_DETAILED = False
DEFAULT_METRICS = "all"
DEFAULT_MONITOR_INTERVAL = 60


@app.command()
def start(
    paper: bool = DEFAULT_PAPER,
    live: bool = DEFAULT_LIVE,
    exchange: str | None = DEFAULT_EXCHANGE,
    symbol: str | None = DEFAULT_SYMBOL,
    symbols: str | None = DEFAULT_SYMBOLS,
    size: float | None = DEFAULT_SIZE,
    strategy: str | None = DEFAULT_STRATEGY,
    model_path: Path | None = DEFAULT_MODEL_PATH,
    cnn_lstm_path: Path | None = DEFAULT_CNN_LSTM_PATH,
    max_position_size: float | None = DEFAULT_MAX_POSITION_SIZE,
    stop_loss: float | None = DEFAULT_STOP_LOSS,
    take_profit: float | None = DEFAULT_TAKE_PROFIT,
    initial_capital: float | None = DEFAULT_INITIAL_CAPITAL,
    update_interval: int | None = DEFAULT_UPDATE_INTERVAL,
    config_file: Path | None = DEFAULT_CONFIG_FILE,
) -> None:
    """
    Start a live trading session.

    Creates and starts a new trading session with the specified configuration.
    Supports both paper trading (default) and live trading modes.
    """
    settings = get_config_manager()

    # Validate trading mode
    if paper and live:
        console.print("[red]Cannot specify both --paper and --live. Choose one.[/red]")
        raise typer.Exit(1)

    trading_mode = "paper" if paper else "live"

    # Validate exchange and API credentials
    if exchange is None:
        exchange = settings.execution.broker

    if exchange is None:
        console.print("[red]No exchange specified and no default broker configured.[/red]")
        raise typer.Exit(1)

    if not validate_api_credentials(settings, exchange):
        raise typer.Exit(1)

    # Parse symbols
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    elif symbol:
        symbol_list = [symbol]
    else:
        symbol_list = settings.data.symbols

    if not symbol_list:
        console.print("[red]No trading symbols specified. Use --symbol or --symbols.[/red]")
        raise typer.Exit(1)

    # Create trading configuration
    from .core.live_trading import TradingConfig

    trading_config = TradingConfig(
        symbols=symbol_list,
        data_source=exchange,
        update_interval=update_interval or settings.data.update_frequency,
        model_path=str(model_path) if model_path else None,
        cnn_lstm_path=str(cnn_lstm_path) if cnn_lstm_path else None,
        agent_type=settings.agent.agent_type,
        max_position_size=max_position_size or settings.risk.max_position_size,
        stop_loss_pct=stop_loss or settings.risk.stop_loss_pct,
        take_profit_pct=take_profit or settings.risk.take_profit_pct,
        initial_capital=initial_capital or 100000.0,
        max_drawdown=settings.risk.max_drawdown,
        slippage_pct=settings.execution.max_slippage,
        commission_pct=settings.execution.commission_rate,
    )

    # Display configuration
    console.print(
        Panel(
            f"[bold blue]Starting {trading_mode.upper()} Trading Session[/bold blue]\n"
            f"Exchange: {exchange}\n"
            f"Symbols: {', '.join(symbol_list)}\n"
            f"Strategy: {strategy or 'Default'}\n"
            f"Initial Capital: ${trading_config.initial_capital:,.2f}\n"
            f"Max Position Size: {trading_config.max_position_size:.1%}\n"
            f"Update Interval: {trading_config.update_interval}s",
            title="Trading Configuration",
        )
    )

    # Confirm for live trading
    if trading_mode == "live":
        console.print("[bold red]WARNING: This is LIVE trading with real money![/bold red]")
        confirm = typer.confirm("Are you sure you want to proceed with live trading?")
        if not confirm:
            console.print("[yellow]Live trading cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        # Get trading engine and create session
        engine = get_trading_engine()
        session = engine.create_session(trading_config)

        console.print(f"[green]Created trading session with ID: {id(session)}[/green]")

        # Start the session
        console.print("[blue]Starting trading session...[/blue]")
        asyncio.run(session.start())

    except Exception as e:
        logger.exception("Failed to start trading session")
        console.print(f"[red]Failed to start trading session: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def stop(
    session_id: str | None = DEFAULT_SESSION_ID,
    all_sessions: bool = DEFAULT_ALL_SESSIONS,
    force: bool = DEFAULT_FORCE,
) -> None:
    """
    Stop trading session(s).

    Stops one or more trading sessions. If no session ID is provided,
    stops all active sessions.
    """
    engine = get_trading_engine()

    if not engine.sessions:
        console.print("[yellow]No active trading sessions found.[/yellow]")
        return

    if all_sessions or session_id is None:
        # Stop all sessions
        if not force:
            confirm = typer.confirm(f"Stop all {len(engine.sessions)} trading sessions?")
            if not confirm:
                console.print("[yellow]Operation cancelled.[/yellow]")
                return

        console.print(f"[blue]Stopping {len(engine.sessions)} trading sessions...[/blue]")
        engine.stop_all_sessions()
        console.print("[green]All trading sessions stopped.[/green]")

    else:
        # Stop specific session
        # Note: This is a simplified implementation. In a real system,
        # you'd want to track session IDs properly
        console.print(f"[blue]Stopping session {session_id}...[/blue]")
        console.print("[yellow]Session-specific stopping not yet implemented.[/yellow]")


@app.command()
def status(
    session_id: str | None = DEFAULT_SESSION_ID,
    detailed: bool = DEFAULT_DETAILED,
) -> None:
    """
    Show trading session status.

    Displays current status of trading sessions including portfolio value,
    positions, and performance metrics.
    """
    engine = get_trading_engine()

    if not engine.sessions:
        console.print("[yellow]No active trading sessions found.[/yellow]")
        return

    # Create status table
    table = Table(title="Trading Session Status")
    table.add_column("Session ID", style="cyan")
    table.add_column("Symbols", style="green")
    table.add_column("Portfolio Value", style="yellow")
    table.add_column("Positions", style="blue")
    table.add_column("Status", style="magenta")

    for i, session in enumerate(engine.sessions):
        session_id_display = f"session_{i}"
        symbols = ", ".join(session.config.symbols)
        portfolio_value = f"${session.portfolio_value:,.2f}"
        positions = len(session.positions)
        status = "Active"  # Simplified status

        table.add_row(session_id_display, symbols, portfolio_value, str(positions), status)

    console.print(table)

    if detailed:
        # Show detailed information for each session
        for i, session in enumerate(engine.sessions):
            console.print(
                Panel(
                    f"[bold]Session {i}[/bold]\n"
                    f"Cash: ${session.cash:,.2f}\n"
                    f"Total Value: ${session.portfolio_value:,.2f}\n"
                    f"Positions: {len(session.positions)}\n"
                    f"Orders: {len(session.orders)}",
                    title=f"Session {i} Details",
                )
            )


@app.command()
async def monitor(
    session_id: str | None = DEFAULT_SESSION_ID,
    metrics: str = DEFAULT_METRICS,
    interval: int = DEFAULT_MONITOR_INTERVAL,
) -> None:
    """
    Monitor live trading session in real-time.

    Provides real-time monitoring of trading sessions with live updates
    on portfolio performance, risk metrics, and trading activity.
    """
    engine = get_trading_engine()

    if not engine.sessions:
        console.print("[yellow]No active trading sessions found.[/yellow]")
        return

    console.print(f"[blue]Starting real-time monitoring with {interval}s interval...[/blue]")
    console.print("[yellow]Press Ctrl+C to stop monitoring.[/yellow]")

    try:
        while True:
            # Clear console for real-time updates
            console.clear()

            # Display monitoring information
            for i, session in enumerate(engine.sessions):
                console.print(
                    Panel(
                        f"[bold]Session {i}[/bold]\n"
                        f"Portfolio Value: ${session.portfolio_value:,.2f}\n"
                        f"Cash: ${session.cash:,.2f}\n"
                        f"Positions: {len(session.positions)}\n"
                        f"Recent Orders: {len(session.orders)}",
                        title=f"Live Monitoring - Session {i}",
                    )
                )

            # Wait for next update
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped.[/yellow]")


@app.command()
def list_sessions() -> None:
    """List all active trading sessions."""
    engine = get_trading_engine()

    if not engine.sessions:
        console.print("[yellow]No active trading sessions found.[/yellow]")
        return

    table = Table(title="Active Trading Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Symbols", style="green")
    table.add_column("Initial Capital", style="yellow")
    table.add_column("Current Value", style="blue")
    table.add_column("P&L", style="magenta")

    for i, session in enumerate(engine.sessions):
        session_id = f"session_{i}"
        symbols = ", ".join(session.config.symbols)
        initial_capital = f"${session.config.initial_capital:,.2f}"
        current_value = f"${session.portfolio_value:,.2f}"

        pnl = session.portfolio_value - session.config.initial_capital
        pnl_str = f"${pnl:+,.2f}"
        pnl_color = "green" if pnl >= 0 else "red"

        table.add_row(session_id, symbols, initial_capital, current_value, pnl_str)

    console.print(table)


if __name__ == "__main__":
    app()
