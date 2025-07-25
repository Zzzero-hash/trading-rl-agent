"""
Trading CLI commands.

This module contains all live trading CLI commands including:
- Starting and stopping trading sessions
- Paper trading
- Live trading monitoring
- Risk management controls
"""

from pathlib import Path
from typing import Annotated

import typer

from .cli_main import (
    DEFAULT_MONITOR_INTERVAL,
    DEFAULT_MONITOR_METRICS,
    DEFAULT_PAPER_DURATION,
    DEFAULT_PAPER_SYMBOLS,
    DEFAULT_PAPER_TRADING,
    DEFAULT_TRADING_CAPITAL,
    console,
    logger,
)

# Trading operations sub-app
trade_app = typer.Typer(
    name="trade",
    help="Live trading operations: start, stop, monitor trading sessions",
    rich_markup_mode="rich",
)


@trade_app.command()
def start(
    model_path: Annotated[Path, typer.Argument(..., help="Path to trained trading model")],
    symbols: str = DEFAULT_PAPER_SYMBOLS,
    capital: float = DEFAULT_TRADING_CAPITAL,
    paper: bool = DEFAULT_PAPER_TRADING,
    exchange: str = typer.Option("alpaca", help="Trading exchange: alpaca, interactive_brokers"),
    risk_limit: float = typer.Option(0.02, help="Maximum portfolio risk per trade"),
    max_positions: int = typer.Option(10, help="Maximum number of open positions"),
    session_name: str | None = typer.Option(None, help="Custom session name"),
) -> None:
    """
    Start a live trading session.

    Launches a trading session using the specified model. Can run in paper trading
    mode for testing or live mode for real trading.

    Examples:
        trade-agent trade start models/hybrid_model.zip --paper
        trade-agent trade start models/model.zip --symbols "AAPL,GOOGL,MSFT" --capital 50000
        trade-agent trade start models/model.zip --exchange alpaca --risk-limit 0.01
    """
    trading_mode = "Paper Trading" if paper else "Live Trading"
    console.print(f"[bold blue]Starting {trading_mode} Session...[/bold blue]")

    try:
        from trade_agent.trading.session_manager import TradingSessionManager

        # Validate model path
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file does not exist: {model_path}[/bold red]")
            raise typer.Exit(1)

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Display configuration
        console.print("[yellow]Trading configuration:[/yellow]")
        console.print(f"  Model: {model_path}")
        console.print(f"  Mode: {trading_mode}")
        console.print(f"  Exchange: {exchange}")
        console.print(f"  Symbols: {symbol_list}")
        console.print(f"  Capital: ${capital:,.2f}")
        console.print(f"  Risk limit: {risk_limit*100:.1f}% per trade")
        console.print(f"  Max positions: {max_positions}")

        if session_name:
            console.print(f"  Session name: {session_name}")

        # Initialize session manager
        session_manager = TradingSessionManager(
            model_path=model_path,
            exchange=exchange,
            paper_trading=paper,
            capital=capital,
            risk_limit=risk_limit,
            max_positions=max_positions,
            session_name=session_name,
        )

        # Start trading session
        console.print("[green]Starting trading session...[/green]")

        if paper:
            console.print("[yellow]⚠️  PAPER TRADING MODE - No real money at risk[/yellow]")
        else:
            console.print("[red]⚠️  LIVE TRADING MODE - Real money at risk![/red]")
            confirm = typer.confirm("Are you sure you want to start live trading?")
            if not confirm:
                console.print("Trading session cancelled.")
                raise typer.Exit(0)

        result = session_manager.start_session(symbols=symbol_list)

        if result.get("success", False):
            session_id = result.get("session_id")
            console.print("[bold green]✓ Trading session started successfully![/bold green]")
            console.print(f"Session ID: {session_id}")
            console.print(f"\nTo monitor: trade-agent trade monitor --session-id {session_id}")
            console.print(f"To stop: trade-agent trade stop --session-id {session_id}")
        else:
            console.print("[bold red]✗ Failed to start trading session![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Trading session error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@trade_app.command()
def stop(
    session_id: Annotated[str, typer.Argument(..., help="Trading session ID to stop")],
    force: bool = typer.Option(False, help="Force stop without confirmation"),
) -> None:
    """
    Stop a running trading session.

    Gracefully stops a trading session, closing all open positions and
    saving session results.

    Examples:
        trade-agent trade stop session_12345
        trade-agent trade stop session_12345 --force
    """
    console.print(f"[bold blue]Stopping trading session: {session_id}[/bold blue]")

    try:
        from trade_agent.trading.session_manager import TradingSessionManager

        # Initialize session manager
        session_manager = TradingSessionManager()

        # Get session info first
        session_info = session_manager.get_session_info(session_id)

        if not session_info.get("exists", False):
            console.print(f"[bold red]Error: Session {session_id} not found![/bold red]")
            raise typer.Exit(1)

        # Display session info
        console.print("[yellow]Session information:[/yellow]")
        console.print(f"  Status: {session_info.get('status', 'Unknown')}")
        console.print(f"  Started: {session_info.get('start_time', 'Unknown')}")
        console.print(f"  Open positions: {session_info.get('open_positions', 0)}")
        console.print(f"  P&L: ${session_info.get('unrealized_pnl', 0):,.2f}")

        # Confirm stop action
        if not force:
            if session_info.get("open_positions", 0) > 0:
                console.print("[yellow]⚠️  This session has open positions that will be closed.[/yellow]")

            confirm = typer.confirm(f"Are you sure you want to stop session {session_id}?")
            if not confirm:
                console.print("Operation cancelled.")
                raise typer.Exit(0)

        # Stop session
        console.print("[green]Stopping trading session...[/green]")

        result = session_manager.stop_session(session_id)

        if result.get("success", False):
            console.print("[bold green]✓ Trading session stopped successfully![/bold green]")

            # Show final summary
            if "summary" in result:
                summary = result["summary"]
                console.print("\nSession Summary:")
                console.print(f"  Duration: {summary.get('duration', 'Unknown')}")
                console.print(f"  Total trades: {summary.get('total_trades', 0)}")
                console.print(f"  Final P&L: ${summary.get('final_pnl', 0):,.2f}")
                console.print(f"  Results saved to: {summary.get('results_file', 'Unknown')}")
        else:
            console.print("[bold red]✗ Failed to stop trading session![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Session stop error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@trade_app.command()
def monitor(
    session_id: str | None = typer.Option(None, help="Specific session ID to monitor"),
    all_sessions: bool = typer.Option(False, help="Monitor all active sessions"),
    interval: int = DEFAULT_MONITOR_INTERVAL,
    metrics: str = DEFAULT_MONITOR_METRICS,
    detailed: bool = typer.Option(False, help="Show detailed position information"),
) -> None:
    """
    Monitor active trading sessions.

    Displays real-time information about running trading sessions including
    P&L, positions, and performance metrics.

    Examples:
        trade-agent trade monitor --session-id session_12345
        trade-agent trade monitor --all-sessions
        trade-agent trade monitor --session-id session_12345 --detailed --interval 30
    """
    console.print("[bold blue]Monitoring trading sessions...[/bold blue]")

    try:
        from trade_agent.trading.monitor import TradingMonitor

        # Initialize monitor
        monitor = TradingMonitor(
            interval=interval,
            metrics=metrics,
            detailed=detailed
        )

        if session_id:
            console.print(f"Monitoring session: {session_id}")
            result = monitor.monitor_session(session_id)
        elif all_sessions:
            console.print("Monitoring all active sessions")
            result = monitor.monitor_all_sessions()
        else:
            # Interactive session selection
            console.print("Available sessions:")
            active_sessions = monitor.get_active_sessions()

            if not active_sessions:
                console.print("[yellow]No active trading sessions found.[/yellow]")
                return

            for i, session in enumerate(active_sessions, 1):
                console.print(f"  {i}. {session['id']} - {session['status']} - {session['symbols']}")

            choice = typer.prompt("Select session number")
            try:
                selected_session = active_sessions[int(choice) - 1]
                session_id = selected_session["id"]
                result = monitor.monitor_session(session_id)
            except (ValueError, IndexError):
                console.print("[bold red]Invalid selection![/bold red]")
                raise typer.Exit(1) from None

        if result.get("success", False):
            console.print("[bold green]✓ Monitoring started successfully![/bold green]")
            console.print("Press Ctrl+C to stop monitoring")
        else:
            console.print("[bold red]✗ Failed to start monitoring![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user.[/yellow]")
    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@trade_app.command()
def paper(
    model_path: Annotated[Path, typer.Argument(..., help="Path to trained trading model")],
    symbols: str = DEFAULT_PAPER_SYMBOLS,
    duration: str = DEFAULT_PAPER_DURATION,
    capital: float = DEFAULT_TRADING_CAPITAL,
) -> None:
    """
    Run a quick paper trading session.

    Starts a time-limited paper trading session for testing and evaluation.
    Useful for quick model validation before live trading.

    Examples:
        trade-agent trade paper models/model.zip
        trade-agent trade paper models/model.zip --symbols "TSLA,NVDA" --duration 2h
        trade-agent trade paper models/model.zip --capital 25000 --duration 30m
    """
    console.print(f"[bold blue]Starting paper trading session ({duration})...[/bold blue]")

    try:
        from trade_agent.trading.paper_trader import PaperTrader

        # Validate model path
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file does not exist: {model_path}[/bold red]")
            raise typer.Exit(1)

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Display configuration
        console.print("[yellow]Paper trading configuration:[/yellow]")
        console.print(f"  Model: {model_path}")
        console.print(f"  Symbols: {symbol_list}")
        console.print(f"  Duration: {duration}")
        console.print(f"  Capital: ${capital:,.2f}")

        # Initialize paper trader
        paper_trader = PaperTrader(
            model_path=model_path,
            symbols=symbol_list,
            capital=capital,
            duration=duration,
        )

        # Start paper trading
        console.print("[green]Starting paper trading...[/green]")
        console.print("[yellow]⚠️  PAPER TRADING MODE - No real money involved[/yellow]")

        result = paper_trader.run_session()

        if result.get("success", False):
            console.print("[bold green]✓ Paper trading session completed![/bold green]")

            # Show session summary
            if "summary" in result:
                summary = result["summary"]
                console.print("\nSession Summary:")
                console.print(f"  Duration: {summary.get('duration', 'Unknown')}")
                console.print(f"  Total trades: {summary.get('total_trades', 0)}")
                console.print(f"  Final P&L: ${summary.get('final_pnl', 0):,.2f}")
                console.print(f"  Return: {summary.get('return_pct', 0):.2%}")
                console.print(f"  Win rate: {summary.get('win_rate', 0):.2%}")
                console.print(f"  Results saved to: {summary.get('results_file', 'Unknown')}")
        else:
            console.print("[bold red]✗ Paper trading session failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Paper trading error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@trade_app.command()
def status(
    session_id: str | None = typer.Option(None, help="Specific session ID to check"),
) -> None:
    """
    Show status of trading sessions.

    Displays current status and summary information for trading sessions.

    Examples:
        trade-agent trade status
        trade-agent trade status --session-id session_12345
    """
    console.print("[bold blue]Trading Session Status[/bold blue]")

    try:
        from trade_agent.trading.session_manager import TradingSessionManager

        # Initialize session manager
        session_manager = TradingSessionManager()

        if session_id:
            # Show specific session status
            result = session_manager.get_session_status(session_id)

            if result.get("success", False):
                status_info = result["status"]
                console.print(f"\nSession: {session_id}")
                console.print(f"  Status: {status_info.get('status', 'Unknown')}")
                console.print(f"  Started: {status_info.get('start_time', 'Unknown')}")
                console.print(f"  Runtime: {status_info.get('runtime', 'Unknown')}")
                console.print(f"  Symbols: {status_info.get('symbols', [])}")
                console.print(f"  Open positions: {status_info.get('open_positions', 0)}")
                console.print(f"  Total trades: {status_info.get('total_trades', 0)}")
                console.print(f"  P&L: ${status_info.get('unrealized_pnl', 0):,.2f}")
            else:
                console.print(f"[bold red]Session {session_id} not found![/bold red]")
        else:
            # Show all sessions status
            result = session_manager.get_all_sessions_status()

            if result.get("success", False):
                sessions = result["sessions"]

                if not sessions:
                    console.print("[yellow]No trading sessions found.[/yellow]")
                    return

                # Display sessions table
                from rich.table import Table

                table = Table(title="Trading Sessions", show_header=True)
                table.add_column("Session ID", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Started", style="blue")
                table.add_column("Symbols", style="yellow")
                table.add_column("Positions", style="magenta")
                table.add_column("P&L", style="red")

                for session in sessions:
                    table.add_row(
                        session.get("id", "Unknown"),
                        session.get("status", "Unknown"),
                        session.get("start_time", "Unknown"),
                        ", ".join(session.get("symbols", [])),
                        str(session.get("open_positions", 0)),
                        f"${session.get('unrealized_pnl', 0):,.2f}"
                    )

                console.print(table)
            else:
                console.print("[bold red]Failed to retrieve session status![/bold red]")
                if "error" in result:
                    console.print(f"Error: {result['error']}")

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Status check error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e
