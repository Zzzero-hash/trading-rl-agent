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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.logging import get_logger
from .attribution import AttributionConfig, PerformanceAttributor
from .attribution_integration import AttributionIntegration, AutomatedAttributionWorkflow
from .manager import PortfolioManager, PortfolioConfig

logger = get_logger(__name__)
console = Console()


@click.group()
def attribution():
    """Performance Attribution Analysis CLI."""
    pass


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--output-dir', '-o', type=click.Path(), default='attribution_output', help='Output directory')
@click.option('--start-date', '-s', type=click.DateTime(), help='Analysis start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(), help='Analysis end date (YYYY-MM-DD)')
@click.option('--symbols', '-t', multiple=True, help='Symbols to include in analysis')
@click.option('--risk-free-rate', '-r', type=float, default=0.02, help='Risk-free rate')
@click.option('--confidence-level', type=float, default=0.95, help='Confidence level for VaR')
@click.option('--use-plotly/--no-plotly', default=True, help='Use Plotly for interactive charts')
def analyze(
    config_file: Optional[str],
    output_dir: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    symbols: tuple,
    risk_free_rate: float,
    confidence_level: float,
    use_plotly: bool
):
    """Run comprehensive performance attribution analysis."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
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
                use_plotly=use_plotly
            )
        
        # Initialize portfolio manager (simulated for demo)
+@click.option('--portfolio-source', type=click.Choice(['demo', 'file', 'api']), default='demo', help='Portfolio data source')
+@click.option('--portfolio-file', type=click.Path(exists=True), help='Portfolio data file (if source is file)')
 def analyze(
     config_file: Optional[str],
     output_dir: str,
     start_date: Optional[datetime],
     end_date: Optional[datetime],
     symbols: tuple,
     risk_free_rate: float,
     confidence_level: float,
-    use_plotly: bool
+    use_plotly: bool,
+    portfolio_source: str,
+    portfolio_file: Optional[str]
 ):
        
        # Initialize attribution integration
        progress.update(task, description="Setting up attribution integration...")
        integration = AttributionIntegration(portfolio_manager, config)
        
        # Convert symbols tuple to list
        symbols_list = list(symbols) if symbols else None
        
        # Run attribution analysis
        progress.update(task, description="Running attribution analysis...")
        results = integration.run_attribution_analysis(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list
        )
        
        # Generate report
        progress.update(task, description="Generating attribution report...")
        report_path = output_path / "attribution_report.txt"
        report_content = integration.generate_attribution_report(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list,
            output_path=str(report_path)
        )
        
        # Export data
        progress.update(task, description="Exporting attribution data...")
        data_path = output_path / "attribution_data.xlsx"
        integration.export_attribution_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list,
            output_path=str(data_path)
        )
        
        # Create dashboard
        progress.update(task, description="Creating attribution dashboard...")
        dashboard = integration.create_attribution_dashboard(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols_list
        )
        
        # Save dashboard
        dashboard_path = output_path / "attribution_dashboard.html"
        if hasattr(dashboard, 'write_html'):
            dashboard.write_html(str(dashboard_path))
        
        progress.update(task, description="Analysis completed!", completed=True)
    
    # Display results summary
    _display_analysis_summary(results, output_path)


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--start-date', '-s', type=click.DateTime(), help='Analysis start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(), help='Analysis end date (YYYY-MM-DD)')
@click.option('--symbols', '-t', multiple=True, help='Symbols to include in analysis')
def factor_analysis(
    config_file: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    symbols: tuple
):
    """Analyze factor contributions to portfolio performance."""
    
    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)
    
    # Run factor analysis
    symbols_list = list(symbols) if symbols else None
    factor_results = integration.analyze_factor_contributions(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols_list
    )
    
    # Display results
    _display_factor_analysis(factor_results)


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--start-date', '-s', type=click.DateTime(), help='Analysis start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(), help='Analysis end date (YYYY-MM-DD)')
@click.option('--symbols', '-t', multiple=True, help='Symbols to include in analysis')
def sector_attribution(
    config_file: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    symbols: tuple
):
    """Analyze sector-level attribution using Brinson methodology."""
    
    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)
    
    # Run sector attribution
    symbols_list = list(symbols) if symbols else None
    sector_results = integration.analyze_sector_attribution(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols_list
    )
    
    # Display results
    _display_sector_attribution(sector_results)


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--start-date', '-s', type=click.DateTime(), help='Analysis start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(), help='Analysis end date (YYYY-MM-DD)')
@click.option('--symbols', '-t', multiple=True, help='Symbols to include in analysis')
def risk_analysis(
    config_file: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    symbols: tuple
):
    """Analyze risk-adjusted performance metrics."""
    
    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    integration = AttributionIntegration(portfolio_manager, config)
    
    # Run risk analysis
    symbols_list = list(symbols) if symbols else None
    risk_results = integration.analyze_risk_adjusted_performance(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols_list
    )
    
    # Display results
    _display_risk_analysis(risk_results)


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--frequency', '-f', type=click.Choice(['daily', 'weekly', 'monthly']), default='monthly', help='Analysis frequency')
@click.option('--auto-reports/--no-auto-reports', default=True, help='Auto-generate reports')
@click.option('--output-dir', '-o', type=click.Path(), default='attribution_reports', help='Report output directory')
def setup_automation(
    config_file: Optional[str],
    frequency: str,
    auto_reports: bool,
    output_dir: str
):
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
        'analysis_frequency': frequency,
        'auto_generate_reports': auto_reports,
        'report_output_dir': output_dir,
        'created_at': datetime.now().isoformat()
    }
    
    config_path = Path(output_dir) / 'workflow_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    console.print(f"[green]✓[/green] Automated attribution workflow configured")
    console.print(f"  Frequency: {frequency}")
    console.print(f"  Auto-reports: {auto_reports}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Configuration saved to: {config_path}")


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--output-dir', '-o', type=click.Path(), default='attribution_reports', help='Report output directory')
def run_automated(
    config_file: Optional[str],
    output_dir: str
):
    """Run automated attribution analysis."""
    
    # Initialize components
    portfolio_manager = _create_demo_portfolio_manager()
    config = _load_config(config_file) if config_file else AttributionConfig()
    workflow = AutomatedAttributionWorkflow(portfolio_manager, config)
    workflow.report_output_dir = output_dir
    
    # Run scheduled analysis
    results = workflow.run_scheduled_analysis()
    
    if results:
        console.print("[green]✓[/green] Automated attribution analysis completed")
        _display_analysis_summary(results, Path(output_dir))
    else:
        console.print("[yellow]⚠[/yellow] No analysis performed (not due yet)")


@attribution.command()
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--start-date', '-s', type=click.DateTime(), help='Analysis start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(), help='Analysis end date (YYYY-MM-DD)')
@click.option('--symbols', '-t', multiple=True, help='Symbols to include in analysis')
@click.option('--output-path', '-o', type=click.Path(), help='Output file path')
def export_data(
    config_file: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    symbols: tuple,
    output_path: Optional[str]
):
    """Export attribution data to Excel file."""
    
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
        output_path=output_file
    )
    
    console.print(f"[green]✓[/green] Attribution data exported to {output_file}")


def _load_config(config_file: str) -> AttributionConfig:
    """Load configuration from file."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "The 'yaml' library is required to load configuration files but is not installed. "
            "Please install it by running 'pip install PyYAML'."
        )
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return AttributionConfig(**config_data)


def _create_demo_portfolio_manager() -> PortfolioManager:
    """Create a demo portfolio manager for testing."""
    # This is a simplified implementation for demonstration
    # In practice, you would load actual portfolio data
    
    config = PortfolioConfig()
    portfolio_manager = PortfolioManager(initial_capital=1000000, config=config)
    
    # Add some demo performance history
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    for date in dates:
        portfolio_manager.performance_history.append({
            'timestamp': date,
            'total_value': 1000000 + (date - dates[0]).days * 100,  # Simple growth
            'total_return': 1000000 + (date - dates[0]).days * 100,
            'cash': 100000,
            'equity_value': 900000 + (date - dates[0]).days * 100
        })
    
    return portfolio_manager


def _display_analysis_summary(results: dict, output_path: Path):
    """Display analysis summary."""
    console.print("\n[bold blue]Performance Attribution Analysis Summary[/bold blue]")
    console.print("=" * 60)
    
    # Factor attribution
    if 'factor_attribution' in results:
        console.print("\n[bold]Factor Attribution:[/bold]")
        factor_attrib = results['factor_attribution']
        for factor, contrib in factor_attrib.items():
            console.print(f"  {factor}: {contrib:.4f}")
    
    # Risk metrics
    if 'risk_adjusted' in results and 'portfolio_risk' in results['risk_adjusted']:
        console.print("\n[bold]Risk Metrics:[/bold]")
        portfolio_risk = results['risk_adjusted']['portfolio_risk']
        console.print(f"  Volatility: {portfolio_risk.get('volatility', 0):.4f}")
        console.print(f"  Max Drawdown: {portfolio_risk.get('max_drawdown', 0):.4f}")
        console.print(f"  VaR: {portfolio_risk.get('var', 0):.4f}")
    
    # Information ratio
    if 'risk_adjusted' in results and 'information_ratio' in results['risk_adjusted']:
        console.print(f"\n[bold]Information Ratio:[/bold] {results['risk_adjusted']['information_ratio']:.4f}")
    
    # Output files
    console.print(f"\n[bold]Output Files:[/bold]")
    console.print(f"  Report: {output_path / 'attribution_report.txt'}")
    console.print(f"  Data: {output_path / 'attribution_data.xlsx'}")
    console.print(f"  Dashboard: {output_path / 'attribution_dashboard.html'}")


def _display_factor_analysis(factor_results: dict):
    """Display factor analysis results."""
    console.print("\n[bold blue]Factor Analysis Results[/bold blue]")
    console.print("=" * 40)
    
    # Factor contributions
    if 'factor_contributions' in factor_results:
        console.print("\n[bold]Factor Contributions:[/bold]")
        for factor, contrib in factor_results['factor_contributions'].items():
            console.print(f"  {factor}: {contrib:.4f}")
    
    # Model quality
    if 'model_quality' in factor_results and factor_results['model_quality'] is not None:
        r_squared = factor_results['model_quality']
        console.print(f"\n[bold]Model Quality:[/bold]")
        console.print(f"  Average R-squared: {r_squared.mean():.4f}")
        console.print(f"  Min R-squared: {r_squared.min():.4f}")
        console.print(f"  Max R-squared: {r_squared.max():.4f}")


def _display_sector_attribution(sector_results: dict):
    """Display sector attribution results."""
    console.print("\n[bold blue]Sector Attribution Results[/bold blue]")
    console.print("=" * 40)
    
    # Sector summary
    if 'sector_summary' in sector_results:
        console.print("\n[bold]Sector Attribution Summary:[/bold]")
        for component, value in sector_results['sector_summary'].items():
            console.print(f"  {component}: {value:.4f}")


def _display_risk_analysis(risk_results: dict):
    """Display risk analysis results."""
    console.print("\n[bold blue]Risk Analysis Results[/bold blue]")
    console.print("=" * 40)
    
    # Portfolio risk
    if 'portfolio_risk' in risk_results:
        console.print("\n[bold]Portfolio Risk Metrics:[/bold]")
        portfolio_risk = risk_results['portfolio_risk']
        for metric, value in portfolio_risk.items():
            console.print(f"  {metric}: {value:.4f}")
    
    # Information ratio
    if 'information_ratio' in risk_results:
        console.print(f"\n[bold]Information Ratio:[/bold] {risk_results['information_ratio']:.4f}")
    
    # Factor risk contributions
    if 'factor_risk_contributions' in risk_results:
        console.print("\n[bold]Factor Risk Contributions:[/bold]")
        for factor, contrib in risk_results['factor_risk_contributions'].items():
            console.print(f"  {factor}: {contrib['contribution']:.4f} (corr: {contrib['correlation']:.4f})")


if __name__ == '__main__':
    attribution()