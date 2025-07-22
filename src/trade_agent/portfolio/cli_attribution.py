"""
CLI interface for performance attribution analysis.

Provides command-line access to all attribution analysis capabilities including:
- Factor attribution analysis
- Brinson attribution
- Risk-adjusted analysis
- Interactive dashboards
- Automated reporting
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .attribution import AttributionConfig
from .attribution_integration import (
    AttributionIntegration,
    AutomatedAttributionWorkflow,
    PortfolioManager,
)

console = Console()

# Global demo portfolio manager instance for consistent state across CLI commands
_demo_portfolio_manager_instance: PortfolioManager | None = None


@click.group()
def attribution() -> None:
    """Performance Attribution Analysis CLI."""


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="attribution_output",
    help="Output directory",
)
@click.option("--start-date", "-s", type=click.DateTime(), help="Analysis start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", type=click.DateTime(), help="Analysis end date (YYYY-MM-DD)")
@click.option("--symbols", "-t", multiple=True, help="Symbols to include in analysis")
@click.option("--risk-free-rate", "-r", type=float, default=0.02, help="Risk-free rate")
@click.option("--confidence-level", type=float, default=0.95, help="Confidence level for VaR")
@click.option("--use-plotly/--no-plotly", default=True, help="Use Plotly for interactive charts")
def analyze(
    config_file: str | None,
    output_dir: str,
    start_date: datetime | None,
    end_date: datetime | None,
    symbols: tuple,
    risk_free_rate: float,
    confidence_level: float,
    use_plotly: bool,
) -> None:
    """Run comprehensive performance attribution analysis."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing attribution analysis...", total=None)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Load configuration
        if config_file:
            progress.update(task, description="Loading configuration...")
            config = _load_config(config_file)
        else:
            config = AttributionConfig(
                risk_free_rate=risk_free_rate,
                confidence_level=confidence_level,
                use_plotly=use_plotly,
            )

            # Initialize portfolio manager (simulated for demo)
        portfolio_manager = _create_demo_portfolio_manager()

        # Initialize attribution integration
        progress.update(task, description="Setting up attribution integration...")
        integration = AttributionIntegration(portfolio_manager, config)

        # Convert symbols tuple to list
        symbols_list = list(symbols) if symbols else None

        # Run attribution analysis
        progress.update(task, description="Running attribution analysis...")
        results = integration.run_attribution_analysis(start_date=start_date, end_date=end_date, symbols=symbols_list)

        # Generate report
        progress.update(task, description="Generating attribution report...")
        report_path = output_path / "attribution_report.txt"
        integration.generate_attribution_report(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list,
            output_path=str(report_path),
        )

        # Export data
        progress.update(task, description="Exporting attribution data...")
        data_path = output_path / "attribution_data.xlsx"
        integration.export_attribution_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list,
            output_path=str(data_path),
        )

        # Create dashboard
        progress.update(task, description="Creating attribution dashboard...")
        dashboard = integration.create_attribution_dashboard(
            start_date=start_date, end_date=end_date, symbols=symbols_list
        )

        # Save dashboard
        dashboard_path = output_path / "attribution_dashboard.html"
        if hasattr(dashboard, "write_html"):
            dashboard.write_html(str(dashboard_path))

        progress.update(task, description="Analysis completed!", completed=True)

    # Display results summary
    _display_analysis_summary(results, output_path)


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--start-date", "-s", type=click.DateTime(), help="Analysis start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", type=click.DateTime(), help="Analysis end date (YYYY-MM-DD)")
@click.option("--symbols", "-t", multiple=True, help="Symbols to include in analysis")
def factor_analysis(
    config_file: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    symbols: tuple,
) -> None:
    """Analyze factor contributions to portfolio performance."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)

    # Run factor analysis
    symbols_list = list(symbols) if symbols else None
    factor_results = integration.analyze_factor_contributions(
        start_date=start_date, end_date=end_date, symbols=symbols_list
    )

    # Display results
    _display_factor_analysis(factor_results)


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--start-date", "-s", type=click.DateTime(), help="Analysis start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", type=click.DateTime(), help="Analysis end date (YYYY-MM-DD)")
@click.option("--symbols", "-t", multiple=True, help="Symbols to include in analysis")
def sector_attribution(
    config_file: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    symbols: tuple,
) -> None:
    """Analyze sector-level attribution using Brinson methodology."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)

    # Run sector attribution
    symbols_list = list(symbols) if symbols else None
    sector_results = integration.analyze_sector_attribution(
        start_date=start_date, end_date=end_date, symbols=symbols_list
    )

    # Display results
    _display_sector_attribution(sector_results)


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--start-date", "-s", type=click.DateTime(), help="Analysis start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", type=click.DateTime(), help="Analysis end date (YYYY-MM-DD)")
@click.option("--symbols", "-t", multiple=True, help="Symbols to include in analysis")
def risk_analysis(
    config_file: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    symbols: tuple,
) -> None:
    """Analyze risk-adjusted performance metrics."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)

    # Run risk analysis
    symbols_list = list(symbols) if symbols else None
    risk_results = integration.analyze_risk_adjusted_performance(
        start_date=start_date, end_date=end_date, symbols=symbols_list
    )

    # Display results
    _display_risk_analysis(risk_results)


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option(
    "--frequency",
    "-f",
    type=click.Choice(["daily", "weekly", "monthly"]),
    default="monthly",
    help="Analysis frequency",
)
@click.option("--auto-reports/--no-auto-reports", default=True, help="Auto-generate reports")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="attribution_reports",
    help="Report output directory",
)
def setup_automation(config_file: str | None, frequency: str, auto_reports: bool, output_dir: str) -> None:
    """Set up automated attribution analysis workflow."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    workflow = AutomatedAttributionWorkflow(portfolio_manager, config)

    # Configure workflow
    workflow.analysis_frequency = frequency
    workflow.auto_generate_reports = auto_reports
    workflow.report_output_dir = output_dir

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Save workflow configuration
    config_data = {
        "analysis_frequency": frequency,
        "auto_generate_reports": auto_reports,
        "report_output_dir": output_dir,
        "created_at": datetime.now().isoformat(),
    }

    config_path = Path(output_dir) / "workflow_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    console.print("[green]✓[/green] Automated attribution workflow configured")
    console.print(f"  Frequency: {frequency}")
    console.print(f"  Auto-reports: {auto_reports}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Configuration saved to: {config_path}")


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="attribution_reports",
    help="Output directory",
)
def run_automated(config_file: str | None, output_dir: str) -> None:
    """Run automated attribution analysis workflow."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    workflow = AutomatedAttributionWorkflow(portfolio_manager, config)

    # Configure workflow
    workflow.report_output_dir = output_dir

    # Run automated analysis
    console.print("Running automated attribution analysis...")
    results = workflow.run_scheduled_analysis()

    if results:
        console.print("[green]✓[/green] Automated analysis completed successfully")
        console.print(f"Results saved to: {output_dir}")
    else:
        console.print("[yellow]⚠[/yellow] No analysis was performed (not due yet)")


@attribution.command()
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--start-date", "-s", type=click.DateTime(), help="Analysis start date (YYYY-MM-DD)")
@click.option("--end-date", "-e", type=click.DateTime(), help="Analysis end date (YYYY-MM-DD)")
@click.option("--symbols", "-t", multiple=True, help="Symbols to include in analysis")
@click.option("--output-path", "-o", type=click.Path(), help="Output file path")
def export_data(
    config_file: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    symbols: tuple,
    output_path: str | None,
) -> None:
    """Export attribution data to various formats."""

    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)

    # Export data
    symbols_list = list(symbols) if symbols else None
    output_file = output_path or "attribution_data.xlsx"

    integration.export_attribution_data(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols_list,
        output_path=output_file,
    )

    console.print(f"[green]✓[/green] Data exported to: {output_file}")


def _load_config(config_file: str) -> AttributionConfig:
    """Load attribution configuration from file."""
    import yaml

    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    return AttributionConfig(**config_data)


def _create_demo_portfolio_manager() -> PortfolioManager:
    """Create a demo portfolio manager for testing with singleton pattern."""
    global _demo_portfolio_manager_instance

    # Return existing instance if already created
    if _demo_portfolio_manager_instance is not None:
        return _demo_portfolio_manager_instance

    import numpy as np

    # Create mock configuration
    config = type("Config", (), {"benchmark_symbol": "SPY"})()

    # Create mock performance history
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    portfolio_values = 1000000 * (1 + np.random.normal(0.0005, 0.015, len(dates))).cumprod()
    benchmark_values = 1000000 * (1 + np.random.normal(0.0004, 0.012, len(dates))).cumprod()

    performance_history = pd.DataFrame(
        {
            "timestamp": dates,
            "portfolio_value": portfolio_values,
            "benchmark_value": benchmark_values,
        }
    )

    portfolio_manager = PortfolioManager(config)
    portfolio_manager.performance_history = performance_history

    # Store the instance for reuse
    _demo_portfolio_manager_instance = portfolio_manager

    return portfolio_manager


def _reset_demo_portfolio_manager() -> None:
    """Reset the demo portfolio manager instance (useful for testing)."""
    global _demo_portfolio_manager_instance
    _demo_portfolio_manager_instance = None


def _display_analysis_summary(results: dict[str, Any], output_path: Path) -> None:
    """Display summary of attribution analysis results."""
    console.print("\n[bold]Attribution Analysis Summary[/bold]")
    console.print("=" * 50)

    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Display factor attribution
    factor_attrib = results.get("factor_attribution", {})
    if factor_attrib:
        console.print("\n[bold]Factor Attribution:[/bold]")
        for factor, contrib in factor_attrib.items():
            console.print(f"  {factor:15}: {contrib:8.4f}")

    # Display risk metrics
    risk_metrics = results.get("risk_analysis", {})
    if risk_metrics:
        console.print("\n[bold]Risk Metrics:[/bold]")
        console.print(f"  Sharpe Ratio:     {risk_metrics.get('portfolio_sharpe', 0):.4f}")
        console.print(f"  Volatility:       {risk_metrics.get('portfolio_volatility', 0):.4f}")
        console.print(f"  Max Drawdown:     {risk_metrics.get('max_drawdown', 0):.4f}")

    console.print(f"\n[green]✓[/green] Analysis completed! Check output directory: {output_path}")


def _display_factor_analysis(factor_results: dict[str, Any]) -> None:
    """Display factor analysis results."""
    console.print("\n[bold]Factor Analysis Results[/bold]")
    console.print("=" * 40)

    factor_contributions = factor_results.get("factor_contributions", {})
    if factor_contributions:
        console.print("\nFactor Contributions:")
        for factor, contrib in factor_contributions.items():
            console.print(f"  {factor:15}: {contrib:8.4f}")
    else:
        console.print("[yellow]No factor contributions available[/yellow]")


def _display_sector_attribution(sector_results: dict[str, Any]) -> None:
    """Display sector attribution results."""
    console.print("\n[bold]Sector Attribution Results[/bold]")
    console.print("=" * 40)

    sector_summary = sector_results.get("sector_summary", {})
    if sector_summary:
        console.print("\nSector Effects:")
        for effect, value in sector_summary.items():
            console.print(f"  {effect:15}: {value:8.4f}")
    else:
        console.print("[yellow]No sector attribution available[/yellow]")


def _display_risk_analysis(risk_results: dict[str, Any]) -> None:
    """Display risk analysis results."""
    console.print("\n[bold]Risk Analysis Results[/bold]")
    console.print("=" * 40)

    if risk_results:
        console.print("\nRisk Metrics:")
        console.print(f"  Sharpe Ratio:     {risk_results.get('information_ratio', 0):.4f}")
        console.print(f"  Volatility:       {risk_results.get('portfolio_risk', {}).get('volatility', 0):.4f}")
        console.print(f"  Max Drawdown:     {risk_results.get('portfolio_risk', {}).get('max_drawdown', 0):.4f}")
    else:
        console.print("[yellow]No risk metrics available[/yellow]")


if __name__ == "__main__":
    attribution()
