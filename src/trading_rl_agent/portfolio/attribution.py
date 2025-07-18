"""
Comprehensive Performance Attribution Analysis System.

This module provides sophisticated performance attribution capabilities including:
- Systematic vs idiosyncratic return decomposition
- Factor attribution analysis
- Brinson attribution for sector/asset allocation
- Risk-adjusted attribution analysis
- Interactive attribution dashboards
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttributionConfig:
    """Configuration for performance attribution analysis."""
    
    # Factor model settings
    risk_free_rate: float = 0.02  # Annual risk-free rate
    confidence_level: float = 0.95  # For VaR calculations
    lookback_period: int = 252  # Days for rolling calculations
    
    # Brinson attribution settings
    sector_column: str = "sector"
    asset_class_column: str = "asset_class"
    
    # Visualization settings
    use_plotly: bool = True
    figure_size: Tuple[int, int] = (12, 8)
    
    # Factor model parameters
    min_observations: int = 60  # Minimum observations for factor model
    max_factors: int = 10  # Maximum number of factors to extract


class FactorModel:
    """Implements factor model for systematic return decomposition."""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        self.factors: Optional[pd.DataFrame] = None
        self.factor_loadings: Optional[pd.DataFrame] = None
        self.residuals: Optional[pd.DataFrame] = None
        self.r_squared: Optional[pd.Series] = None
        
    def fit(self, returns: pd.DataFrame, market_returns: pd.Series) -> None:
        """
        Fit factor model to returns data.
        
        Args:
            returns: Asset returns (assets x time)
            market_returns: Market returns series
        """
        # Standardize returns
        returns_std = returns.sub(returns.mean(axis=1), axis=0).div(returns.std(axis=1), axis=0)
        
        # Extract factors using PCA
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(self.config.max_factors, returns.shape[0] - 1))
        factors = pca.fit_transform(returns_std.T)
        
        # Create factor DataFrame
        self.factors = pd.DataFrame(
            factors,
            index=returns.columns,
            columns=[f"Factor_{i+1}" for i in range(factors.shape[1])]
        )
        
        # Add market factor
        self.factors['Market'] = market_returns
        
        # Calculate factor loadings
        self._calculate_loadings(returns)
        
    def _calculate_loadings(self, returns: pd.DataFrame) -> None:
        """Calculate factor loadings using OLS regression."""
        loadings = {}
        residuals = {}
        r_squared = {}
        
        for asset in returns.index:
            # Regress asset returns on factors
            X = self.factors.values
            y = returns.loc[asset].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if mask.sum() < self.config.min_observations:
                continue
                
            X_clean = X[mask]
            y_clean = y[mask]
            
            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X_clean)), X_clean])
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X_with_const, y_clean, rcond=None)[0]
                y_pred = X_with_const @ beta
                residuals[asset] = y_clean - y_pred
                
                # Calculate R-squared
                ss_res = np.sum(residuals[asset] ** 2)
                ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
                r_squared[asset] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                loadings[asset] = beta[1:]  # Exclude constant
                
            except np.linalg.LinAlgError:
                logger.warning(f"Could not calculate loadings for {asset}")
                continue
        
        self.factor_loadings = pd.DataFrame(loadings).T
        self.factor_loadings.columns = self.factors.columns
        self.residuals = pd.DataFrame(residuals).T
        self.r_squared = pd.Series(r_squared)
        
    def decompose_returns(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Decompose returns into systematic and idiosyncratic components.
        
        Returns:
            Dictionary with 'systematic' and 'idiosyncratic' components
        """
        if self.factor_loadings is None or self.factors is None:
            raise ValueError("Factor model must be fitted first")
            
        systematic = pd.DataFrame(index=returns.index, columns=returns.columns)
        idiosyncratic = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.index:
            if asset in self.factor_loadings.index:
                # Systematic component: factor loadings * factor returns
                asset_loadings = self.factor_loadings.loc[asset]
                systematic.loc[asset] = (self.factors * asset_loadings).sum(axis=1)
                
                # Idiosyncratic component: residuals
                if asset in self.residuals.index:
                    idiosyncratic.loc[asset] = self.residuals.loc[asset]
                else:
                    idiosyncratic.loc[asset] = returns.loc[asset] - systematic.loc[asset]
        
        return {
            'systematic': systematic,
            'idiosyncratic': idiosyncratic
        }


class BrinsonAttributor:
    """Implements Brinson attribution for sector/asset allocation analysis."""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        
    def calculate_attribution(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        returns: pd.Series,
        grouping_column: str
    ) -> Dict[str, float]:
        """
        Calculate Brinson attribution components.
        
        Args:
            portfolio_weights: Portfolio weights by group
            benchmark_weights: Benchmark weights by group
            returns: Returns by group
            grouping_column: Column name for grouping (sector, asset_class, etc.)
            
        Returns:
            Dictionary with allocation, selection, and interaction effects
        """
        # Ensure all series have the same index
        common_index = portfolio_weights.index.intersection(benchmark_weights.index).intersection(returns.index)
        
        if len(common_index) == 0:
            raise ValueError("No common groups found between portfolio, benchmark, and returns")
            
        portfolio_weights = portfolio_weights[common_index]
        benchmark_weights = benchmark_weights[common_index]
        returns = returns[common_index]
        
        # Calculate attribution components
        allocation_effect = (portfolio_weights - benchmark_weights) * returns
        selection_effect = benchmark_weights * (returns - returns.mean())
        interaction_effect = (portfolio_weights - benchmark_weights) * (returns - returns.mean())
        
        return {
            'allocation': allocation_effect.sum(),
            'selection': selection_effect.sum(),
            'interaction': interaction_effect.sum(),
            'total': allocation_effect.sum() + selection_effect.sum() + interaction_effect.sum()
        }
        
    def calculate_sector_attribution(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate sector-level attribution.
        
        Args:
            portfolio_data: DataFrame with portfolio weights and sector info
            benchmark_data: DataFrame with benchmark weights and sector info
            returns_data: DataFrame with returns and sector info
            
        Returns:
            Dictionary with sector attribution results
        """
        results = {}
        
        for date in returns_data.index:
            if date in portfolio_data.index and date in benchmark_data.index:
                portfolio_weights = portfolio_data.loc[date].groupby(self.config.sector_column)['weight'].sum()
                benchmark_weights = benchmark_data.loc[date].groupby(self.config.sector_column)['weight'].sum()
                returns = returns_data.loc[date].groupby(self.config.sector_column)['return'].mean()
                
                results[date] = self.calculate_attribution(
                    portfolio_weights, benchmark_weights, returns, self.config.sector_column
                )
                
        return results


class RiskAdjustedAttributor:
    """Implements risk-adjusted attribution analysis."""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if len(returns) < 2:
            return {}
            
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_hist = np.percentile(returns, (1 - self.config.confidence_level) * 100)
        cvar_hist = returns[returns <= var_hist].mean()
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'volatility': volatility,
            'var': var_hist,
            'cvar': cvar_hist,
            'downside_volatility': downside_vol,
            'max_drawdown': max_drawdown,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
    def calculate_risk_adjusted_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate risk-adjusted attribution metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            factor_returns: Factor return series
            
        Returns:
            Dictionary with risk-adjusted attribution results
        """
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        # Risk metrics for portfolio and benchmark
        portfolio_risk = self.calculate_risk_metrics(portfolio_returns)
        benchmark_risk = self.calculate_risk_metrics(benchmark_returns)
        excess_risk = self.calculate_risk_metrics(excess_returns)
        
        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Factor risk contributions
        factor_risk_contrib = {}
        if not factor_returns.empty:
            for factor in factor_returns.columns:
                factor_ret = factor_returns[factor]
                correlation = portfolio_returns.corr(factor_ret)
                factor_risk_contrib[factor] = {
                    'correlation': correlation,
                    'contribution': correlation * portfolio_risk['volatility'] * factor_ret.std() * np.sqrt(252)
                }
        
        return {
            'portfolio_risk': portfolio_risk,
            'benchmark_risk': benchmark_risk,
            'excess_risk': excess_risk,
            'information_ratio': information_ratio,
            'factor_risk_contributions': factor_risk_contrib
        }


class AttributionVisualizer:
    """Creates interactive attribution dashboards and visualizations."""
    
    def __init__(self, config: AttributionConfig):
        self.config = config
        self.use_plotly = config.use_plotly and PLOTLY_AVAILABLE
        
    def create_attribution_dashboard(
        self,
        attribution_results: Dict[str, Any],
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Any:
        """Create comprehensive attribution dashboard."""
        if self.use_plotly:
            return self._create_plotly_dashboard(attribution_results, portfolio_returns, benchmark_returns)
        else:
            return self._create_matplotlib_dashboard(attribution_results, portfolio_returns, benchmark_returns)
            
    def _create_plotly_dashboard(
        self,
        attribution_results: Dict[str, Any],
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> go.Figure:
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Return Attribution',
                'Risk Metrics', 'Factor Contributions',
                'Sector Attribution', 'Risk-Adjusted Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Cumulative returns
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(x=cumulative_portfolio.index, y=cumulative_portfolio.values,
                      name='Portfolio', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark.values,
                      name='Benchmark', line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. Return attribution
        if 'factor_attribution' in attribution_results:
            factor_contrib = attribution_results['factor_attribution']
            fig.add_trace(
                go.Bar(x=list(factor_contrib.keys()), y=list(factor_contrib.values()),
                      name='Factor Contributions'),
                row=1, col=2
            )
        
        # 3. Risk metrics
        if 'risk_metrics' in attribution_results:
            risk_metrics = attribution_results['risk_metrics']
            fig.add_trace(
                go.Bar(x=list(risk_metrics.keys()), y=list(risk_metrics.values()),
                      name='Risk Metrics'),
                row=2, col=1
            )
        
        # 4. Factor loadings
        if 'factor_loadings' in attribution_results:
            loadings = attribution_results['factor_loadings']
            fig.add_trace(
                go.Scatter(x=loadings.index, y=loadings.values,
                          mode='markers', name='Factor Loadings'),
                row=2, col=2
            )
        
        # 5. Sector attribution
        if 'sector_attribution' in attribution_results:
            sector_attrib = attribution_results['sector_attribution']
            fig.add_trace(
                go.Bar(x=list(sector_attrib.keys()), y=list(sector_attrib.values()),
                      name='Sector Attribution'),
                row=3, col=1
            )
        
        # 6. Risk-adjusted metrics
        if 'risk_adjusted' in attribution_results:
            risk_adj = attribution_results['risk_adjusted']
            fig.add_trace(
                go.Bar(x=list(risk_adj.keys()), y=list(risk_adj.values()),
                      name='Risk-Adjusted Metrics'),
                row=3, col=2
            )
        
        fig.update_layout(
            title='Performance Attribution Dashboard',
            height=1200,
            showlegend=True
        )
        
        return fig
        
    def _create_matplotlib_dashboard(
        self,
        attribution_results: Dict[str, Any],
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> plt.Figure:
        """Create matplotlib dashboard."""
        fig, axes = plt.subplots(3, 2, figsize=self.config.figure_size)
        fig.suptitle('Performance Attribution Dashboard', fontsize=16)
        
        # 1. Cumulative returns
        cumulative_portfolio = (1 + portfolio_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()
        
        axes[0, 0].plot(cumulative_portfolio.index, cumulative_portfolio.values, label='Portfolio')
        axes[0, 0].plot(cumulative_benchmark.index, cumulative_benchmark.values, label='Benchmark')
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].legend()
        
        # 2. Return attribution
        if 'factor_attribution' in attribution_results:
            factor_contrib = attribution_results['factor_attribution']
            axes[0, 1].bar(factor_contrib.keys(), factor_contrib.values())
            axes[0, 1].set_title('Factor Attribution')
        
        # 3. Risk metrics
        if 'risk_metrics' in attribution_results:
            risk_metrics = attribution_results['risk_metrics']
            axes[1, 0].bar(risk_metrics.keys(), risk_metrics.values())
            axes[1, 0].set_title('Risk Metrics')
        
        # 4. Factor loadings
        if 'factor_loadings' in attribution_results:
            loadings = attribution_results['factor_loadings']
            axes[1, 1].scatter(range(len(loadings)), loadings.values)
            axes[1, 1].set_title('Factor Loadings')
        
        # 5. Sector attribution
        if 'sector_attribution' in attribution_results:
            sector_attrib = attribution_results['sector_attribution']
            axes[2, 0].bar(sector_attrib.keys(), sector_attrib.values())
            axes[2, 0].set_title('Sector Attribution')
        
        # 6. Risk-adjusted metrics
        if 'risk_adjusted' in attribution_results:
            risk_adj = attribution_results['risk_adjusted']
            axes[2, 1].bar(risk_adj.keys(), risk_adj.values())
            axes[2, 1].set_title('Risk-Adjusted Metrics')
        
        plt.tight_layout()
        return fig


class PerformanceAttributor:
    """
    Comprehensive Performance Attribution Analysis System.
    
    This class provides a unified interface for all attribution analysis capabilities:
    - Systematic vs idiosyncratic return decomposition
    - Factor attribution analysis
    - Brinson attribution for sector/asset allocation
    - Risk-adjusted attribution analysis
    - Interactive attribution dashboards
    """
    
    def __init__(self, config: Optional[AttributionConfig] = None):
        """
        Initialize the performance attributor.
        
        Args:
            config: Configuration for attribution analysis
        """
        self.config = config or AttributionConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.factor_model = FactorModel(self.config)
        self.brinson_attributor = BrinsonAttributor(self.config)
        self.risk_attributor = RiskAdjustedAttributor(self.config)
        self.visualizer = AttributionVisualizer(self.config)
        
        # Storage for results
        self.attribution_results: Dict[str, Any] = {}
        
    def analyze_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        asset_returns: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            asset_returns: Asset return matrix (assets x time)
            portfolio_weights: Portfolio weights matrix (assets x time)
            benchmark_weights: Benchmark weights matrix (assets x time)
            sector_data: Optional sector classification data
            
        Returns:
            Dictionary containing all attribution analysis results
        """
        self.logger.info("Starting comprehensive performance attribution analysis")
        
        # 1. Factor model analysis
        self.logger.info("Fitting factor model...")
        self.factor_model.fit(asset_returns, benchmark_returns)
        
        # 2. Return decomposition
        self.logger.info("Decomposing returns...")
        decomposition = self.factor_model.decompose_returns(asset_returns)
        
        # 3. Factor attribution
        self.logger.info("Calculating factor attribution...")
        factor_attribution = self._calculate_factor_attribution(
            portfolio_weights, self.factor_model.factor_loadings, self.factor_model.factors
        )
        
        # 4. Brinson attribution
        self.logger.info("Calculating Brinson attribution...")
        brinson_results = self._calculate_brinson_attribution(
            portfolio_weights, benchmark_weights, asset_returns, sector_data
        )
        
        # 5. Risk-adjusted attribution
        self.logger.info("Calculating risk-adjusted attribution...")
        risk_adjusted = self.risk_attributor.calculate_risk_adjusted_attribution(
            portfolio_returns, benchmark_returns, self.factor_model.factors
        )
        
        # 6. Compile results
        self.attribution_results = {
            'decomposition': decomposition,
            'factor_attribution': factor_attribution,
            'brinson_attribution': brinson_results,
            'risk_adjusted': risk_adjusted,
            'factor_model': {
                'loadings': self.factor_model.factor_loadings,
                'factors': self.factor_model.factors,
                'r_squared': self.factor_model.r_squared
            }
        }
        
        self.logger.info("Performance attribution analysis completed")
        return self.attribution_results
        
    def _calculate_factor_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        factor_loadings: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate factor contribution to portfolio performance."""
        factor_contrib = {}
        
        for factor in factor_returns.columns:
            # Calculate factor contribution for each period
            factor_contrib_period = []
            
            for date in portfolio_weights.index:
                if date in factor_returns.index:
                    weights = portfolio_weights.loc[date]
                    loadings = factor_loadings.loc[weights.index]
                    factor_ret = factor_returns.loc[date, factor]
                    
                    # Factor contribution = weight * loading * factor_return
                    contrib = (weights * loadings[factor] * factor_ret).sum()
                    factor_contrib_period.append(contrib)
            
            # Average contribution over time
            factor_contrib[factor] = np.mean(factor_contrib_period) if factor_contrib_period else 0
            
        return factor_contrib
        
    def _calculate_brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        sector_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate Brinson attribution if sector data is available."""
        if sector_data is None:
            return {}
            
        # Merge sector data with weights and returns
        portfolio_with_sector = portfolio_weights.join(sector_data[self.config.sector_column])
        benchmark_with_sector = benchmark_weights.join(sector_data[self.config.sector_column])
        returns_with_sector = asset_returns.join(sector_data[self.config.sector_column])
        
        # Calculate sector-level attribution
        sector_results = {}
        
        for date in portfolio_weights.index:
            if date in benchmark_weights.index and date in asset_returns.index:
                try:
                    portfolio_weights_sector = portfolio_with_sector.loc[date].groupby(self.config.sector_column)['weight'].sum()
                    benchmark_weights_sector = benchmark_with_sector.loc[date].groupby(self.config.sector_column)['weight'].sum()
                    returns_sector = returns_with_sector.loc[date].groupby(self.config.sector_column)['return'].mean()
                    
                    attribution = self.brinson_attributor.calculate_attribution(
                        portfolio_weights_sector, benchmark_weights_sector, returns_sector, self.config.sector_column
                    )
                    sector_results[date] = attribution
                except Exception as e:
                    self.logger.warning(f"Could not calculate Brinson attribution for {date}: {e}")
                    
        return sector_results
        
    def create_dashboard(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Any:
        """Create interactive attribution dashboard."""
        if not self.attribution_results:
            raise ValueError("No attribution results available. Run analyze_performance() first.")
        
        return self.visualizer.create_attribution_dashboard(
            self.attribution_results, portfolio_returns, benchmark_returns
        )
        
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive attribution report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report content as string
        """
        if not self.attribution_results:
            raise ValueError("No attribution results available. Run analyze_performance() first.")
            
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE ATTRIBUTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Factor attribution summary
        if 'factor_attribution' in self.attribution_results:
            report.append("FACTOR ATTRIBUTION SUMMARY")
            report.append("-" * 40)
            factor_attrib = self.attribution_results['factor_attribution']
            for factor, contrib in factor_attrib.items():
                report.append(f"{factor}: {contrib:.4f}")
            report.append("")
        
        # Risk-adjusted metrics
        if 'risk_adjusted' in self.attribution_results:
            report.append("RISK-ADJUSTED METRICS")
            report.append("-" * 40)
            risk_adj = self.attribution_results['risk_adjusted']
            if 'information_ratio' in risk_adj:
                report.append(f"Information Ratio: {risk_adj['information_ratio']:.4f}")
            if 'portfolio_risk' in risk_adj:
                portfolio_risk = risk_adj['portfolio_risk']
                report.append(f"Portfolio Volatility: {portfolio_risk.get('volatility', 0):.4f}")
                report.append(f"Maximum Drawdown: {portfolio_risk.get('max_drawdown', 0):.4f}")
            report.append("")
        
        # Factor model quality
        if 'factor_model' in self.attribution_results:
            report.append("FACTOR MODEL QUALITY")
            report.append("-" * 40)
            r_squared = self.attribution_results['factor_model']['r_squared']
            if r_squared is not None:
                report.append(f"Average R-squared: {r_squared.mean():.4f}")
                report.append(f"Min R-squared: {r_squared.min():.4f}")
                report.append(f"Max R-squared: {r_squared.max():.4f}")
            report.append("")
        
        report_content = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Report saved to {output_path}")
            
        return report_content
        
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from attribution analysis."""
        if not self.attribution_results:
            return {}
            
        summary = {}
        
        # Factor attribution summary
        if 'factor_attribution' in self.attribution_results:
            factor_attrib = self.attribution_results['factor_attribution']
            summary['total_factor_contribution'] = sum(factor_attrib.values())
            summary['top_factor'] = max(factor_attrib.items(), key=lambda x: abs(x[1])) if factor_attrib else None
            
        # Risk metrics summary
        if 'risk_adjusted' in self.attribution_results:
            risk_adj = self.attribution_results['risk_adjusted']
            summary['information_ratio'] = risk_adj.get('information_ratio', 0)
            if 'portfolio_risk' in risk_adj:
                summary['portfolio_volatility'] = risk_adj['portfolio_risk'].get('volatility', 0)
                summary['max_drawdown'] = risk_adj['portfolio_risk'].get('max_drawdown', 0)
                
        return summary