"""
Backtesting CLI commands.

This module contains all backtesting and evaluation CLI commands including:
- Strategy backtesting
- Model evaluation
- Performance analysis
- Comparison reports
"""

from pathlib import Path
from typing import Annotated

import typer

from .cli_main import (
    DEFAULT_BACKTEST_OUTPUT,
    DEFAULT_COMMISSION,
    DEFAULT_COMPARISON_OUTPUT,
    DEFAULT_EVALUATION_OUTPUT,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_REPORT_FORMAT,
    DEFAULT_REPORTS_OUTPUT,
    DEFAULT_SLIPPAGE,
    console,
    logger,
)

# Backtesting operations sub-app
backtest_app = typer.Typer(
    name="backtest",
    help="Backtesting operations: strategy evaluation, performance analysis",
    rich_markup_mode="rich",
)


@backtest_app.command()
def evaluate(
    model_path: Annotated[Path, typer.Argument(..., help="Path to trained model file")],
    data_path: Annotated[Path, typer.Argument(..., help="Path to test dataset")],
    output_dir: Path = DEFAULT_BACKTEST_OUTPUT,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    start_date: str | None = typer.Option(None, help="Backtest start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, help="Backtest end date (YYYY-MM-DD)"),
    benchmark: str = typer.Option("SPY", help="Benchmark symbol for comparison"),
) -> None:
    """
    Evaluate a trained model using backtesting.

    Runs a comprehensive backtest of a trained model against historical data,
    calculating performance metrics and generating detailed reports.

    Examples:
        trade-agent backtest evaluate models/hybrid_model.zip data/test_data.csv
        trade-agent backtest evaluate models/model.zip data/test.csv --initial-capital 100000
        trade-agent backtest evaluate models/model.zip data/test.csv --start-date 2023-01-01
    """
    console.print("[bold blue]Running model backtest evaluation...[/bold blue]")

    try:
        from trade_agent.backtesting.evaluator import ModelEvaluator

        # Validate paths
        if not model_path.exists():
            console.print(f"[bold red]Error: Model file does not exist: {model_path}[/bold red]")
            raise typer.Exit(1)

        if not data_path.exists():
            console.print(f"[bold red]Error: Data file does not exist: {data_path}[/bold red]")
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Backtest configuration:[/yellow]")
        console.print(f"  Model: {model_path}")
        console.print(f"  Data: {data_path}")
        console.print(f"  Initial capital: ${initial_capital:,.2f}")
        console.print(f"  Commission: {commission*100:.3f}%")
        console.print(f"  Slippage: {slippage*100:.4f}%")
        console.print(f"  Benchmark: {benchmark}")
        console.print(f"  Output: {output_dir}")

        if start_date:
            console.print(f"  Start date: {start_date}")
        if end_date:
            console.print(f"  End date: {end_date}")

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=model_path,
            data_path=data_path,
            output_dir=output_dir,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            benchmark=benchmark,
        )

        # Run backtest
        console.print("[green]Starting backtest evaluation...[/green]")

        result = evaluator.evaluate(
            start_date=start_date,
            end_date=end_date
        )

        if result.get("success", False):
            console.print("[bold green]✓ Backtest completed successfully![/bold green]")
            console.print(f"Results saved to: {output_dir}")

            # Show key metrics
            if "metrics" in result:
                metrics = result["metrics"]
                console.print("\nKey Performance Metrics:")
                console.print(f"  Total Return: {metrics.get('total_return', 'N/A'):.2%}")
                console.print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                console.print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
                console.print(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
                console.print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")

                # Benchmark comparison
                if "benchmark_return" in metrics:
                    console.print(f"  Benchmark Return: {metrics['benchmark_return']:.2%}")
                    alpha = metrics.get("total_return", 0) - metrics.get("benchmark_return", 0)
                    console.print(f"  Alpha: {alpha:.2%}")
        else:
            console.print("[bold red]✗ Backtest failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Backtest evaluation error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@backtest_app.command()
def compare(
    models_dir: Annotated[Path, typer.Argument(..., help="Directory containing multiple model files")],
    data_path: Annotated[Path, typer.Argument(..., help="Path to test dataset")],
    output_dir: Path = DEFAULT_COMPARISON_OUTPUT,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission: float = DEFAULT_COMMISSION,
    benchmark: str = typer.Option("SPY", help="Benchmark symbol for comparison"),
) -> None:
    """
    Compare multiple models through backtesting.

    Runs backtests on multiple models and generates a comparative analysis
    report showing relative performance metrics.

    Examples:
        trade-agent backtest compare models/ data/test_data.csv
        trade-agent backtest compare models/ data/test.csv --benchmark QQQ
    """
    console.print("[bold blue]Running model comparison backtest...[/bold blue]")

    try:
        from trade_agent.backtesting.comparator import ModelComparator

        # Validate paths
        if not models_dir.exists() or not models_dir.is_dir():
            console.print(f"[bold red]Error: Models directory does not exist: {models_dir}[/bold red]")
            raise typer.Exit(1)

        if not data_path.exists():
            console.print(f"[bold red]Error: Data file does not exist: {data_path}[/bold red]")
            raise typer.Exit(1)

        # Find model files
        model_files = list(models_dir.glob("*.zip")) + list(models_dir.glob("*.pkl"))
        if not model_files:
            console.print(f"[bold red]Error: No model files found in {models_dir}[/bold red]")
            raise typer.Exit(1)

        console.print(f"Found {len(model_files)} models to compare:")
        for model_file in model_files:
            console.print(f"  - {model_file.name}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize comparator
        comparator = ModelComparator(
            models_dir=models_dir,
            data_path=data_path,
            output_dir=output_dir,
            initial_capital=initial_capital,
            commission=commission,
            benchmark=benchmark,
        )

        # Run comparison
        console.print("[green]Starting model comparison...[/green]")

        result = comparator.compare_models()

        if result.get("success", False):
            console.print("[bold green]✓ Model comparison completed![/bold green]")
            console.print(f"Results saved to: {output_dir}")

            # Show comparison summary
            if "rankings" in result:
                rankings = result["rankings"]
                console.print("\nModel Rankings (by Sharpe Ratio):")
                for i, (model_name, metrics) in enumerate(rankings, 1):
                    console.print(f"  {i}. {model_name}: {metrics['sharpe_ratio']:.3f}")
        else:
            console.print("[bold red]✗ Model comparison failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@backtest_app.command()
def report(
    results_path: Annotated[Path, typer.Argument(..., help="Path to backtest results")],
    output_dir: Path = DEFAULT_REPORTS_OUTPUT,
    format: str = DEFAULT_REPORT_FORMAT,
    include_plots: bool = typer.Option(True, help="Include performance plots in report"),
) -> None:
    """
    Generate detailed backtest reports.

    Creates comprehensive reports from backtest results including performance
    metrics, charts, and analysis.

    Examples:
        trade-agent backtest report backtest_results/
        trade-agent backtest report results.json --format pdf
        trade-agent backtest report results/ --format html --include-plots
    """
    console.print("[bold blue]Generating backtest report...[/bold blue]")

    try:
        from trade_agent.backtesting.reporter import BacktestReporter

        # Validate paths
        if not results_path.exists():
            console.print(f"[bold red]Error: Results path does not exist: {results_path}[/bold red]")
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Report configuration:[/yellow]")
        console.print(f"  Results: {results_path}")
        console.print(f"  Format: {format}")
        console.print(f"  Include plots: {include_plots}")
        console.print(f"  Output: {output_dir}")

        # Initialize reporter
        reporter = BacktestReporter(
            results_path=results_path,
            output_dir=output_dir,
            format=format,
            include_plots=include_plots,
        )

        # Generate report
        console.print("[green]Generating report...[/green]")

        result = reporter.generate_report()

        if result.get("success", False):
            console.print("[bold green]✓ Report generated successfully![/bold green]")
            console.print(f"Report saved to: {result.get('output_file', output_dir)}")
        else:
            console.print("[bold red]✗ Report generation failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@backtest_app.command()
def analyze(
    results_path: Annotated[Path, typer.Argument(..., help="Path to backtest results")],
    analysis_type: str = typer.Option("full", help="Analysis type: full, risk, performance, trades"),
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT,
) -> None:
    """
    Perform detailed analysis of backtest results.

    Conducts various types of analysis on backtest results including risk analysis,
    performance attribution, and trade analysis.

    Examples:
        trade-agent backtest analyze backtest_results/
        trade-agent backtest analyze results.json --analysis-type risk
        trade-agent backtest analyze results/ --analysis-type trades
    """
    console.print(f"[bold blue]Running {analysis_type} analysis...[/bold blue]")

    try:
        from trade_agent.backtesting.analyzer import BacktestAnalyzer

        # Validate paths
        if not results_path.exists():
            console.print(f"[bold red]Error: Results path does not exist: {results_path}[/bold red]")
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzer
        analyzer = BacktestAnalyzer(
            results_path=results_path,
            output_dir=output_dir
        )

        # Run analysis
        console.print("[green]Running analysis...[/green]")

        result = analyzer.analyze(analysis_type=analysis_type)

        if result.get("success", False):
            console.print(f"[bold green]✓ {analysis_type.title()} analysis completed![/bold green]")
            console.print(f"Results saved to: {output_dir}")

            # Show key findings
            if "summary" in result:
                summary = result["summary"]
                console.print("\nKey Findings:")
                for finding in summary.get("key_findings", []):
                    console.print(f"  • {finding}")
        else:
            console.print(f"[bold red]✗ {analysis_type.title()} analysis failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e
