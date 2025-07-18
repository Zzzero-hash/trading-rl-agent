"""
Console utilities for Trading RL Agent CLI.

Provides helper functions for formatted table output with various styling options.
"""

from typing import Any

from rich.console import Console
from rich.table import Table

# Initialize console for rich output
console = Console()


def print_table(
    rows: list[list[Any]],
    headers: list[str] | None = None,
    title: str | None = None,
    style: str = "ascii",
    no_style: bool = False,
) -> None:
    """
    Print a formatted table with auto-width columns and configurable styling.

    Args:
        rows: List of rows, where each row is a list of values
        headers: Optional list of column headers
        title: Optional table title
        style: Table style - "ascii", "simple", "grid", "minimal"
        no_style: If True, output as TSV (tab-separated values)

    Examples:
        # Basic usage
        print_table([[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]],
                   headers=["ID", "Symbol", "Price"])

        # Single row (falls back to print)
        print_table([[1, "AAPL", 150.25]], headers=["ID", "Symbol", "Price"])

        # TSV output
        print_table([[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]],
                   headers=["ID", "Symbol", "Price"], no_style=True)
    """
    if not rows:
        console.print("[yellow]No data to display[/yellow]")
        return

    # Single row fallback
    if len(rows) == 1:
        if headers:
            for header, value in zip(headers, rows[0]):
                console.print(f"{header}: {value}")
        else:
            console.print(rows[0])
        return

    # TSV output when no_style is True
    if no_style:
        if headers:
            print("\t".join(str(h) for h in headers))
        for row in rows:
            print("\t".join(str(cell) for cell in row))
        return

    # Rich table with styling
    table = Table(title=title, show_header=True)

    # Determine column widths automatically
    all_data = rows.copy()
    if headers:
        all_data.insert(0, headers)

    # Calculate max width for each column
    col_widths = []
    for col_idx in range(len(all_data[0])):
        max_width = 0
        for row in all_data:
            cell_str = str(row[col_idx])
            # Account for potential rich formatting
            if hasattr(cell_str, "__rich__"):
                # For rich objects, estimate width
                max_width = max(max_width, len(str(cell_str)))
            else:
                max_width = max(max_width, len(cell_str))
        col_widths.append(max_width)

    # Add columns with calculated widths
    if headers:
        for i, header in enumerate(headers):
            # Ensure minimum width for headers, especially for metrics table
            min_width = len(str(header)) + 2
            # For metrics table, allow wider columns to prevent truncation
            max_width = 80 if "strategy" in str(header).lower() else 50
            table.add_column(
                header,
                width=max(min_width, min(col_widths[i] + 2, max_width)),  # Add padding, cap at max_width
                style="cyan",
                no_wrap=True if "strategy" in str(header).lower() else False,  # Prevent wrapping for strategy column
            )
    else:
        for i in range(len(rows[0])):
            # Ensure minimum width for "Col X" headers
            min_width = len(f"Col {i + 1}") + 2
            table.add_column(f"Col {i + 1}", width=max(min_width, min(col_widths[i] + 2, 50)), style="cyan")

    # Add rows
    for row in rows:
        # Convert all cells to strings and handle None values
        formatted_row = []
        for cell in row:
            if cell is None:
                formatted_row.append("-")
            elif isinstance(cell, (int, float)):
                # Format numbers nicely
                if isinstance(cell, float):
                    formatted_row.append(f"{cell:.2f}")
                else:
                    formatted_row.append(str(cell))
            else:
                formatted_row.append(str(cell))
        table.add_row(*formatted_row)

    # Apply style
    if style == "minimal":
        table.border_style = "dim"
        table.show_edge = False
    elif style == "simple":
        table.border_style = "white"
    elif style == "grid":
        table.border_style = "bright_white"
    else:  # ascii
        table.border_style = "white"

    console.print(table)


def print_metrics_table(results: list[dict], title: str = "Results Summary") -> None:
    """
    Print a metrics table specifically formatted for trading results.

    Args:
        results: List of dictionaries containing metrics
        title: Table title
    """
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Extract headers from the first result
    headers = list(results[0].keys())

    # Convert to rows
    rows = []
    for result in results:
        row = []
        for header in headers:
            value = result.get(header, "-")
            # Format percentages and numbers
            if isinstance(value, float):
                if "return" in header.lower() or "rate" in header.lower():
                    row.append(f"{value:.2%}")
                elif "ratio" in header.lower():
                    # Ratios like Sharpe ratio should be decimal, not percentage
                    row.append(f"{value:.2f}")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        rows.append(row)

    print_table(rows, headers, title=title)


def print_status_table(status_data: dict, title: str = "Status") -> None:
    """
    Print a status table for system information.

    Args:
        status_data: Dictionary of status information
        title: Table title
    """
    if not status_data:
        console.print("[yellow]No status data to display[/yellow]")
        return

    headers = ["Setting", "Value"]
    rows = [[key, value] for key, value in status_data.items()]

    print_table(rows, headers, title=title)


def print_error_summary(errors: list[str], title: str = "Errors") -> None:
    """
    Print an error summary table.

    Args:
        errors: List of error messages
        title: Table title
    """
    if not errors:
        return

    headers = ["Error"]
    rows = [[error] for error in errors]

    # Use red styling for errors
    table = Table(title=title, title_style="red")
    table.add_column("Error", style="red")

    for error in errors:
        table.add_row(error)

    console.print(table)


# Example usage functions for CLI commands
def example_backtest_usage() -> None:
    """Example of how to use print_table in backtest commands."""
    # Simulate backtest results
    results = [
        {"Strategy": "PPO", "Period": "2023-01-01 to 2023-06-01", "CAGR": 0.15, "Sharpe": 1.25, "Max Drawdown": -0.08},
        {"Strategy": "SAC", "Period": "2023-01-01 to 2023-06-01", "CAGR": 0.18, "Sharpe": 1.45, "Max Drawdown": -0.06},
    ]

    print("=== Backtest Results ===")
    print_metrics_table(results, "Backtest Summary")

    print("\n=== TSV Output ===")
    rows = [
        [r["Strategy"], r["Period"], f"{r['CAGR']:.2%}", f"{r['Sharpe']:.2f}", f"{r['Max Drawdown']:.2%}"]
        for r in results
    ]
    headers = ["Strategy", "Period", "CAGR", "Sharpe", "Max Drawdown"]
    print_table(rows, headers, no_style=True)


def example_train_usage() -> None:
    """Example of how to use print_table in train commands."""
    # Simulate training metrics
    epochs = [10, 20, 30, 40, 50]
    losses = [0.85, 0.72, 0.65, 0.58, 0.52]
    accuracies = [0.65, 0.72, 0.78, 0.82, 0.85]

    rows = [[epoch, f"{loss:.3f}", f"{acc:.2%}"] for epoch, loss, acc in zip(epochs, losses, accuracies)]
    headers = ["Epoch", "Loss", "Accuracy"]

    print("=== Training Progress ===")
    print_table(rows, headers, title="Training Metrics")

    # Single row example
    print("\n=== Final Results ===")
    final_row = [50, 0.52, 0.85]
    print_table([final_row], headers, title="Final Training Results")


if __name__ == "__main__":
    # Demo the functionality
    print("=== Console Helper Demo ===\n")

    print("1. Basic table:")
    print_table([[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]], headers=["ID", "Symbol", "Price"])

    print("\n2. Single row (fallback to print):")
    print_table([[1, "AAPL", 150.25]], headers=["ID", "Symbol", "Price"])

    print("\n3. TSV output:")
    print_table([[1, "AAPL", 150.25], [2, "GOOGL", 2750.50]], headers=["ID", "Symbol", "Price"], no_style=True)

    print("\n4. Backtest example:")
    example_backtest_usage()

    print("\n5. Training example:")
    example_train_usage()
