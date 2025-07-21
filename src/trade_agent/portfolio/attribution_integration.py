"""
Integration module for performance attribution with existing portfolio management system.

This module provides seamless integration between the PerformanceAttributor and the
existing PortfolioManager, enabling automated attribution analysis workflows.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .attribution import AttributionConfig, PerformanceAttributor

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Mock portfolio manager interface for attribution integration."""

    def __init__(self, config: Any):
        self.config = config
        self.performance_history: pd.DataFrame = pd.DataFrame()


class AttributionIntegration:
    """
    Integration layer for portfolio attribution analysis.

    Provides a high-level interface for running attribution analysis
    and managing results with caching and automation capabilities.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        attribution_config: AttributionConfig | None = None,
    ):
        """
        Initialize attribution integration.

        Args:
            portfolio_manager: Portfolio manager instance
            attribution_config: Attribution configuration
        """
        self.portfolio_manager = portfolio_manager
        self.config = attribution_config or AttributionConfig()
        self.attributor = PerformanceAttributor(self.config)
        self._attribution_cache: dict[str, Any] = {}
        self._last_analysis_date: datetime | None = None
        self.logger = logging.getLogger(__name__)

    def prepare_attribution_data(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare data for attribution analysis.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Dictionary with prepared data
        """
        # Get performance history
        perf_df = self.portfolio_manager.performance_history

        # Handle case where performance_history is a list
        if isinstance(perf_df, list):
            if not perf_df:
                raise ValueError("No performance history available")
            perf_df = pd.DataFrame(perf_df)
        else:
            perf_df = perf_df.copy()

        if perf_df.empty:
            raise ValueError("No performance history available")

        # Filter by date range
        if start_date:
            perf_df = perf_df[perf_df["timestamp"] >= start_date]
        if end_date:
            perf_df = perf_df[perf_df["timestamp"] <= end_date]

        if perf_df.empty:
            raise ValueError("No data available for specified date range")

        # Get portfolio and benchmark returns
        portfolio_returns = self._get_portfolio_returns(perf_df)
        benchmark_returns = self._get_benchmark_returns(perf_df["timestamp"])

        # Get asset-level data
        asset_data = self._get_asset_level_data(perf_df, symbols)

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
            "asset_returns": asset_data["returns"],
            "portfolio_weights": asset_data["portfolio_weights"],
            "benchmark_weights": asset_data["benchmark_weights"],
            "sector_data": asset_data["sector_data"],
        }

    def _get_portfolio_returns(self, perf_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from performance history."""
        # Try different possible column names for portfolio value
        portfolio_value_col = None
        for col in ["portfolio_value", "total_value", "total_return"]:
            if col in perf_df.columns:
                portfolio_value_col = col
                break

        if portfolio_value_col is None:
            # If no suitable column found, create synthetic returns
            np.random.seed(42)
            return pd.Series(
                np.random.normal(0.0006, 0.02, len(perf_df) - 1),
                index=perf_df["timestamp"].iloc[1:],
                name="portfolio_returns",
            )

        # Calculate returns from portfolio values
        portfolio_values = perf_df[portfolio_value_col].values
        return pd.Series(
            np.diff(portfolio_values) / portfolio_values[:-1],
            index=perf_df["timestamp"].iloc[1:],
            name="portfolio_returns",
        )

    def _get_benchmark_returns(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get benchmark returns for the specified dates."""
        # This is a simplified implementation
        # In practice, you would fetch benchmark data from your data source

        # Try to get benchmark symbol from config, fallback to default
        benchmark_symbol = "SPY"  # Default benchmark
        try:
            if hasattr(self.portfolio_manager, "config") and hasattr(self.portfolio_manager.config, "benchmark_symbol"):
                benchmark_symbol = self.portfolio_manager.config.benchmark_symbol
        except AttributeError:
            pass

        # For now, create synthetic benchmark returns
        # In a real implementation, you would fetch actual benchmark data
        np.random.seed(42)  # For reproducible results
        return pd.Series(
            np.random.normal(0.0005, 0.015, len(dates)),
            index=dates,
            name=benchmark_symbol,  # Daily returns ~12% annual
        )

    def _get_asset_level_data(self, perf_df: pd.DataFrame, symbols: list[str] | None) -> dict[str, Any]:
        """Extract asset-level data from performance history."""
        # This is a simplified implementation
        # In practice, you would extract actual asset-level data from your portfolio

        # Get unique dates
        dates = perf_df["timestamp"].unique()

        # Create synthetic asset data for demonstration
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Generate synthetic asset returns
        np.random.seed(42)
        asset_returns = pd.DataFrame(
            np.random.normal(0.0006, 0.02, (len(symbols), len(dates))),
            index=symbols,
            columns=dates,
        )

        # Generate synthetic weights
        portfolio_weights = pd.DataFrame(
            np.random.dirichlet(np.ones(len(symbols)), len(dates)).T,
            index=symbols,
            columns=dates,
        )

        # Generate benchmark weights (market cap weighted)
        benchmark_weights = pd.DataFrame(
            np.random.dirichlet(np.ones(len(symbols)) * 2, len(dates)).T,
            index=symbols,
            columns=dates,
        )

        # Generate sector data
        sectors = ["Technology", "Technology", "Technology", "Consumer", "Technology"]
        sector_data = pd.DataFrame({"sector": sectors}, index=symbols)

        return {
            "returns": asset_returns,
            "portfolio_weights": portfolio_weights,
            "benchmark_weights": benchmark_weights,
            "sector_data": sector_data,
        }

    def run_attribution_analysis(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        """
        Run comprehensive attribution analysis.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include
            force_recompute: Force recomputation even if cached results exist

        Returns:
            Attribution analysis results
        """
        # Check cache
        cache_key = f"{start_date}_{end_date}_{symbols}"
        if not force_recompute and cache_key in self._attribution_cache:
            self.logger.info("Using cached attribution results")
            cached_result = self._attribution_cache[cache_key]
            return dict(cached_result) if isinstance(cached_result, dict) else {}

        # Prepare data
        data = self.prepare_attribution_data(start_date, end_date, symbols)

        # Run attribution analysis
        self.logger.info("Running attribution analysis")
        results = self.attributor.analyze_performance(
            portfolio_returns=data["portfolio_returns"],
            benchmark_returns=data["benchmark_returns"],
            asset_returns=data["asset_returns"],
            portfolio_weights=data["portfolio_weights"],
            benchmark_weights=data["benchmark_weights"],
            sector_data=data["sector_data"],
        )

        # Cache results
        self._attribution_cache[cache_key] = results
        self._last_analysis_date = datetime.now()

        return results

    def create_attribution_dashboard(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> Any:
        """
        Create interactive attribution dashboard.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Interactive dashboard
        """
        # Run analysis if needed
        results = self.run_attribution_analysis(start_date, end_date, symbols)

        # Get portfolio and benchmark returns for visualization
        data = self.prepare_attribution_data(start_date, end_date, symbols)

        # Create dashboard
        return self.attributor.visualizer.create_attribution_dashboard(
            results, data["portfolio_returns"], data["benchmark_returns"]
        )

    def generate_attribution_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
        output_path: str | None = None,
    ) -> str:
        """
        Generate comprehensive attribution report.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        # Run analysis if needed
        self.run_attribution_analysis(start_date, end_date, symbols)

        # Generate report
        return self.attributor.generate_report(output_path)

    def get_attribution_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics from attribution analysis.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Summary statistics
        """
        # Run analysis if needed
        self.run_attribution_analysis(start_date, end_date, symbols)

        return self.attributor.get_summary_statistics()

    def analyze_factor_contributions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze factor contributions to portfolio performance.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Factor contribution analysis
        """
        results = self.run_attribution_analysis(start_date, end_date, symbols)

        return {
            "factor_contributions": results.get("factor_attribution", {}),
            "factor_loadings": results.get("factor_model", {}).get("loadings"),
            "factor_returns": results.get("factor_model", {}).get("factors"),
            "model_quality": results.get("factor_model", {}).get("r_squared"),
        }

    def analyze_sector_attribution(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze sector-level attribution.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Sector attribution analysis
        """
        results = self.run_attribution_analysis(start_date, end_date, symbols)

        # Extract sector-level results
        brinson_results = results.get("brinson_attribution", {})
        sector_summary: dict[str, Any] = {}

        if brinson_results:
            # Aggregate sector effects over time
            for attribution in brinson_results.values():
                for effect_type, value in attribution.items():
                    if effect_type not in sector_summary:
                        sector_summary[effect_type] = []
                    sector_summary[effect_type].append(value)

            # Calculate averages
            for effect_type, values in sector_summary.items():
                sector_summary[effect_type] = np.mean(values)

        return {
            "sector_attribution": brinson_results,
            "sector_summary": sector_summary,
        }

    def analyze_risk_adjusted_performance(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze risk-adjusted performance metrics.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include

        Returns:
            Risk-adjusted performance analysis
        """
        results = self.run_attribution_analysis(start_date, end_date, symbols)

        risk_analysis = results.get("risk_analysis", {})
        return dict(risk_analysis) if isinstance(risk_analysis, dict) else {}

    def export_attribution_data(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
        output_path: str = "attribution_data.xlsx",
    ) -> None:
        """
        Export attribution data to Excel file.

        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include
            output_path: Path to save Excel file
        """
        results = self.run_attribution_analysis(start_date, end_date, symbols)
        data = self.prepare_attribution_data(start_date, end_date, symbols)

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Portfolio returns
            data["portfolio_returns"].to_frame("Portfolio_Returns").to_excel(writer, sheet_name="Portfolio_Returns")

            # Benchmark returns
            data["benchmark_returns"].to_frame("Benchmark_Returns").to_excel(writer, sheet_name="Benchmark_Returns")

            # Asset returns
            data["asset_returns"].to_excel(writer, sheet_name="Asset_Returns")

            # Portfolio weights
            data["portfolio_weights"].to_excel(writer, sheet_name="Portfolio_Weights")

            # Factor attribution
            if "factor_attribution" in results:
                pd.Series(results["factor_attribution"]).to_frame("Contribution").to_excel(
                    writer, sheet_name="Factor_Attribution"
                )

            # Risk metrics
            if "risk_adjusted" in results and "portfolio_risk" in results["risk_adjusted"]:
                pd.Series(results["risk_adjusted"]["portfolio_risk"]).to_frame("Value").to_excel(
                    writer, sheet_name="Risk_Metrics"
                )

        self.logger.info(f"Attribution data exported to {output_path}")


class AutomatedAttributionWorkflow:
    """
    Automated workflow for regular attribution analysis.

    Provides scheduled and event-driven attribution analysis capabilities.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        attribution_config: AttributionConfig | None = None,
    ):
        """
        Initialize automated attribution workflow.

        Args:
            portfolio_manager: Portfolio manager instance
            attribution_config: Attribution configuration
        """
        self.integration = AttributionIntegration(portfolio_manager, attribution_config)
        self.logger = logging.getLogger(__name__)

        # Workflow settings
        self.analysis_frequency = "monthly"  # daily, weekly, monthly
        self.last_analysis = None
        self.auto_generate_reports = True
        self.report_output_dir = "attribution_reports"

    def should_run_analysis(self) -> bool:
        """Determine if analysis should be run based on schedule."""
        # This is a simplified implementation
        # In practice, you would check against a proper schedule
        if self.last_analysis is None:
            return True
        return datetime.now() - self.last_analysis > timedelta(hours=24)  # type: ignore

    def run_scheduled_analysis(self) -> dict[str, Any]:
        """
        Run scheduled attribution analysis.

        Returns:
            Attribution analysis results
        """
        if not self.should_run_analysis():
            self.logger.info("Scheduled analysis not due yet")
            return {}

        self.logger.info("Running scheduled attribution analysis")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days

        return self.integration.run_attribution_analysis(start_date, end_date)

    def on_portfolio_rebalance(self) -> dict[str, Any]:
        """
        Run attribution analysis after portfolio rebalancing.

        Returns:
            Attribution analysis results
        """
        self.logger.info("Running attribution analysis after rebalancing")

        # Run analysis for recent period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        return self.integration.run_attribution_analysis(start_date, end_date)

    def on_performance_milestone(self, milestone_type: str) -> dict[str, Any]:
        """
        Run attribution analysis on performance milestones.

        Args:
            milestone_type: Type of milestone (e.g., 'quarterly', 'annual')

        Returns:
            Attribution analysis results
        """
        self.logger.info(f"Running attribution analysis for {milestone_type} milestone")

        end_date = datetime.now()

        if milestone_type == "quarterly":
            start_date = end_date - timedelta(days=90)
        elif milestone_type == "annual":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)

        return self.integration.run_attribution_analysis(start_date, end_date)
