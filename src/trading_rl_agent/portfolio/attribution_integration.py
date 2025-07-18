"""
Integration module for performance attribution with existing portfolio management system.

This module provides seamless integration between the PerformanceAttributor and the
existing PortfolioManager, enabling automated attribution analysis workflows.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from .attribution import AttributionConfig, PerformanceAttributor
from .manager import PortfolioManager

logger = get_logger(__name__)


class AttributionIntegration:
    """
    Integration layer between PortfolioManager and PerformanceAttributor.
    
    Provides automated workflows for attribution analysis and seamless
    integration with existing portfolio management capabilities.
    """
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        attribution_config: Optional[AttributionConfig] = None
    ):
        """
        Initialize attribution integration.
        
        Args:
            portfolio_manager: Existing portfolio manager instance
            attribution_config: Configuration for attribution analysis
        """
        self.portfolio_manager = portfolio_manager
        self.attributor = PerformanceAttributor(attribution_config)
        self.logger = get_logger(self.__class__.__name__)
        
        # Cache for attribution results
        self._attribution_cache: Dict[str, Any] = {}
        self._last_analysis_date: Optional[datetime] = None
        
    def prepare_attribution_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for attribution analysis from portfolio manager.
        
        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            symbols: List of symbols to include (None for all)
            
        Returns:
            Dictionary containing prepared data for attribution
        """
        self.logger.info("Preparing attribution data from portfolio manager")
        
        # Get portfolio performance history
        performance_history = self.portfolio_manager.performance_history
        
        if not performance_history:
            raise ValueError("No performance history available in portfolio manager")
            
        # Convert to DataFrame
        perf_df = pd.DataFrame(performance_history)
        
        # Filter by date range
        if start_date:
            perf_df = perf_df[perf_df['timestamp'] >= start_date]
        if end_date:
            perf_df = perf_df[perf_df['timestamp'] <= end_date]
            
        if perf_df.empty:
            raise ValueError("No data available for specified date range")
            
        # Extract portfolio returns
        portfolio_returns = perf_df.set_index('timestamp')['total_return'].pct_change().dropna()
        
        # Get benchmark returns (if available)
        benchmark_returns = self._get_benchmark_returns(portfolio_returns.index)
        
        # Get asset-level data
        asset_data = self._get_asset_level_data(perf_df, symbols)
        
        return {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'asset_returns': asset_data['returns'],
            'portfolio_weights': asset_data['portfolio_weights'],
            'benchmark_weights': asset_data['benchmark_weights'],
            'sector_data': asset_data['sector_data']
        }
        
    def _get_benchmark_returns(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get benchmark returns for the specified dates."""
        # This is a simplified implementation
        # In practice, you would fetch benchmark data from your data source
        benchmark_symbol = self.portfolio_manager.config.benchmark_symbol
        
        # For now, create synthetic benchmark returns
        # In a real implementation, you would fetch actual benchmark data
        if random_seed is not None:
            np.random.seed(random_seed)  # For reproducible results
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(dates)),  # Daily returns ~12% annual
            index=dates,
            name=benchmark_symbol
        )
        
        return benchmark_returns
        
    def _get_asset_level_data(
        self,
        perf_df: pd.DataFrame,
        symbols: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Extract asset-level data from performance history."""
        # This is a simplified implementation
        # In practice, you would extract actual asset-level data from your portfolio
        
        # Get unique dates
        dates = perf_df['timestamp'].unique()
        
        # Create synthetic asset data for demonstration
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            
        # Generate synthetic asset returns
        np.random.seed(42)
        asset_returns = pd.DataFrame(
            np.random.normal(0.0006, 0.02, (len(symbols), len(dates))),
            index=symbols,
            columns=dates
        )
        
        # Generate synthetic weights
        portfolio_weights = pd.DataFrame(
            np.random.dirichlet(np.ones(len(symbols)), len(dates)).T,
            index=symbols,
            columns=dates
        )
        
        # Generate benchmark weights (market cap weighted)
        benchmark_weights = pd.DataFrame(
            np.random.dirichlet(np.ones(len(symbols)) * 2, len(dates)).T,
            index=symbols,
            columns=dates
        )
        
        # Generate sector data
        sectors = ['Technology', 'Technology', 'Technology', 'Consumer', 'Technology']
        sector_data = pd.DataFrame({
            'sector': sectors
        }, index=symbols)
        
        return {
            'returns': asset_returns,
            'portfolio_weights': portfolio_weights,
            'benchmark_weights': benchmark_weights,
            'sector_data': sector_data
        }
        
    def run_attribution_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
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
            return self._attribution_cache[cache_key]
            
        # Prepare data
        data = self.prepare_attribution_data(start_date, end_date, symbols)
        
        # Run attribution analysis
        self.logger.info("Running attribution analysis")
        results = self.attributor.analyze_performance(
            portfolio_returns=data['portfolio_returns'],
            benchmark_returns=data['benchmark_returns'],
            asset_returns=data['asset_returns'],
            portfolio_weights=data['portfolio_weights'],
            benchmark_weights=data['benchmark_weights'],
            sector_data=data['sector_data']
        )
        
        # Cache results
        self._attribution_cache[cache_key] = results
        self._last_analysis_date = datetime.now()
        
        return results
        
    def create_attribution_dashboard(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
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
            results, data['portfolio_returns'], data['benchmark_returns']
        )
        
    def generate_attribution_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        output_path: Optional[str] = None
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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        
        factor_analysis = {
            'factor_contributions': results.get('factor_attribution', {}),
            'factor_loadings': results.get('factor_model', {}).get('loadings'),
            'factor_returns': results.get('factor_model', {}).get('factors'),
            'model_quality': results.get('factor_model', {}).get('r_squared')
        }
        
        return factor_analysis
        
    def analyze_sector_attribution(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        
        brinson_results = results.get('brinson_attribution', {})
        
        # Aggregate sector attribution over time
        if brinson_results:
            sector_summary = {}
            for date, attribution in brinson_results.items():
                for component, value in attribution.items():
                    if component not in sector_summary:
                        sector_summary[component] = []
                    sector_summary[component].append(value)
                    
            # Calculate averages
            sector_summary = {
                component: np.mean(values) for component, values in sector_summary.items()
            }
        else:
            sector_summary = {}
            
        return {
            'sector_attribution': brinson_results,
            'sector_summary': sector_summary
        }
        
    def analyze_risk_adjusted_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        
        return results.get('risk_adjusted', {})
        
    def export_attribution_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        output_path: str = "attribution_data.xlsx"
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
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Portfolio returns
            data['portfolio_returns'].to_frame('Portfolio_Returns').to_excel(
                writer, sheet_name='Portfolio_Returns'
            )
            
            # Benchmark returns
            data['benchmark_returns'].to_frame('Benchmark_Returns').to_excel(
                writer, sheet_name='Benchmark_Returns'
            )
            
            # Asset returns
            data['asset_returns'].to_excel(writer, sheet_name='Asset_Returns')
            
            # Portfolio weights
            data['portfolio_weights'].to_excel(writer, sheet_name='Portfolio_Weights')
            
            # Factor attribution
            if 'factor_attribution' in results:
                pd.Series(results['factor_attribution']).to_frame('Contribution').to_excel(
                    writer, sheet_name='Factor_Attribution'
                )
            
            # Risk metrics
            if 'risk_adjusted' in results and 'portfolio_risk' in results['risk_adjusted']:
                pd.Series(results['risk_adjusted']['portfolio_risk']).to_frame('Value').to_excel(
                    writer, sheet_name='Risk_Metrics'
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
        attribution_config: Optional[AttributionConfig] = None
    ):
        """
        Initialize automated attribution workflow.
        
        Args:
            portfolio_manager: Portfolio manager instance
            attribution_config: Attribution configuration
        """
        self.integration = AttributionIntegration(portfolio_manager, attribution_config)
        self.logger = get_logger(self.__class__.__name__)
        
        # Workflow settings
        self.analysis_frequency = "monthly"  # daily, weekly, monthly
        self.last_analysis = None
        self.auto_generate_reports = True
        self.report_output_dir = "attribution_reports"
        
    def should_run_analysis(self) -> bool:
        """Determine if attribution analysis should be run."""
        if self.last_analysis is None:
            return True
            
        now = datetime.now()
        
        if self.analysis_frequency == "daily":
            return (now - self.last_analysis).days >= 1
        elif self.analysis_frequency == "weekly":
            return (now - self.last_analysis).days >= 7
        elif self.analysis_frequency == "monthly":
            return (now - self.last_analysis).days >= 30
            
        return False
        
    def run_scheduled_analysis(self) -> Dict[str, Any]:
        """
        Run scheduled attribution analysis.
        
        Returns:
            Attribution analysis results
        """
        if not self.should_run_analysis():
            self.logger.info("Scheduled analysis not due yet")
            return {}
            
        self.logger.info("Running scheduled attribution analysis")
        
        # Calculate analysis period
        end_date = datetime.now()
        if self.analysis_frequency == "daily":
            start_date = end_date - timedelta(days=30)
        elif self.analysis_frequency == "weekly":
            start_date = end_date - timedelta(days=90)
        else:  # monthly
            start_date = end_date - timedelta(days=365)
            
        # Run analysis
        results = self.integration.run_attribution_analysis(
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate report if enabled
        if self.auto_generate_reports:
            import os
            try:
                # Ensure directory exists
                os.makedirs(self.report_output_dir, exist_ok=True)

                report_filename = f"attribution_report_{end_date.strftime('%Y%m%d')}.txt"
                report_path = f"{self.report_output_dir}/{report_filename}"

                self.integration.generate_attribution_report(
                    start_date=start_date,
                    end_date=end_date,
                    output_path=report_path
                )
            except Exception as e:
                self.logger.error(f"Failed to generate report: {e}")
                # Continue without failing the analysis
            
        self.last_analysis = end_date
        
        return results
        
    def on_portfolio_rebalance(self) -> Dict[str, Any]:
        """
        Run attribution analysis after portfolio rebalancing.
        
        Returns:
            Attribution analysis results
        """
        self.logger.info("Running attribution analysis after rebalancing")
        
        # Run analysis for recent period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        return self.integration.run_attribution_analysis(
            start_date=start_date,
            end_date=end_date
        )
        
    def on_performance_milestone(self, milestone_type: str) -> Dict[str, Any]:
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
            
        return self.integration.run_attribution_analysis(
            start_date=start_date,
            end_date=end_date
        )