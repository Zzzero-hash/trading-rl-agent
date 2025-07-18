#!/usr/bin/env python3
"""
Standalone Performance Attribution Demo

This script demonstrates the comprehensive performance attribution analysis system
without depending on the full trading_rl_agent package structure.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# ATTRIBUTION CLASSES (Copied from attribution.py)
# =============================================================================

@dataclass
class AttributionConfig:
    """Configuration for performance attribution analysis."""
    factor_model_type: str = "pca"  # 'pca' or 'ols'
    n_factors: int = 3
    confidence_level: float = 0.95
    visualization_backend: str = "plotly"  # 'plotly' or 'matplotlib'
    risk_free_rate: float = 0.02  # Annual risk-free rate
    lookback_period: int = 252  # Days for rolling calculations

class FactorModel:
    """Factor model for decomposing returns into systematic and idiosyncratic components."""

    def __init__(self, config: AttributionConfig):
        self.config = config
        self.pca: PCA | None = None
        self.factor_loadings: pd.DataFrame | None = None
        self.factor_returns: pd.DataFrame | None = None
        self.fitted: bool = False

    def fit(self, returns: pd.DataFrame, market_returns: pd.Series) -> None:
        """Fit the factor model using PCA or OLS."""
        if self.config.factor_model_type == "pca":
            self._fit_pca(returns)
        else:
            self._fit_ols(returns, market_returns)

    def _fit_pca(self, returns: pd.DataFrame) -> None:
        """Fit PCA factor model."""
        # Standardize returns
        returns_std = (returns - returns.mean()) / returns.std()

        # Fit PCA
        self.pca = PCA(n_components=self.config.n_factors)
        self.pca.fit(returns_std)

        # Get factor loadings and returns
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,
            index=returns.columns,
            columns=[f"factor_{i+1}" for i in range(self.config.n_factors)]
        )

        # Transform to get factor returns
        factor_scores = self.pca.transform(returns_std)
        self.factor_returns = pd.DataFrame(
            factor_scores,
            index=returns.index,
            columns=[f"factor_{i+1}" for i in range(self.config.n_factors)]
        )

        self.fitted = True

    def _fit_ols(self, returns: pd.DataFrame, market_returns: pd.Series) -> None:
        """Fit OLS factor model."""
        # Simple market model
        self.factor_loadings = pd.DataFrame(index=returns.columns, columns=["market"])
        self.factor_returns = pd.DataFrame(index=returns.index, columns=["market"])

        for asset in returns.columns:
            # Fit linear regression
            model = LinearRegression()
            model.fit(market_returns.values.reshape(-1, 1), returns[asset].values)  # type: ignore

            # Store beta (factor loading)
            self.factor_loadings.loc[asset, "market"] = model.coef_[0]

        # Factor returns are market returns
        self.factor_returns["market"] = market_returns

        self.fitted = True

    def decompose_returns(self, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Decompose returns into systematic and idiosyncratic components."""
        if not self.fitted:
            raise ValueError("Factor model must be fitted before decomposition")

        # Calculate systematic returns
        systematic_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        for asset in returns.columns:
            systematic_returns[asset] = (
                self.factor_loadings.loc[asset] * self.factor_returns  # type: ignore
            ).sum(axis=1)

        # Calculate idiosyncratic returns
        idiosyncratic_returns = returns - systematic_returns

        return {
            "systematic": systematic_returns,
            "idiosyncratic": idiosyncratic_returns,
            "total": returns
        }

    def analyze_factor_contributions(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   factor_returns: pd.DataFrame) -> dict[str, float]:
        """Analyze factor contributions to portfolio performance."""
        if not self.fitted:
            raise ValueError("Factor model must be fitted before analysis")

        # Calculate factor contributions
        factor_contributions = {}
        for factor in factor_returns.columns:
            if factor in self.factor_returns.columns:  # type: ignore
                # Calculate portfolio's exposure to this factor
                portfolio_exposure = self.factor_loadings[factor].mean()  # type: ignore
                factor_contribution = portfolio_exposure * factor_returns[factor].mean()
                factor_contributions[f"{factor}_contribution"] = factor_contribution

        # Calculate total factor contribution
        total_factor_contribution = sum(factor_contributions.values())

        # Calculate idiosyncratic contribution
        idiosyncratic_contribution = portfolio_returns.mean() - total_factor_contribution

        return {
            **factor_contributions,
            "total_factor_contribution": total_factor_contribution,
            "idiosyncratic_contribution": idiosyncratic_contribution
        }

class BrinsonAttributor:
    """Brinson attribution for sector/asset allocation analysis."""

    def __init__(self, config: AttributionConfig):
        self.config = config

    def calculate_attribution(self, portfolio_weights: pd.Series,
                            benchmark_weights: pd.Series,
                            returns: pd.Series,
                            grouping_column: str) -> dict[str, float]:
        """Calculate Brinson attribution for a single period."""
        # Group by sector/asset class
        portfolio_grouped = portfolio_weights.groupby(grouping_column).sum()
        benchmark_grouped = benchmark_weights.groupby(grouping_column).sum()
        returns_grouped = returns.groupby(grouping_column).mean()

        # Calculate effects
        allocation_effect = ((portfolio_grouped - benchmark_grouped) *
                           (returns_grouped - returns.mean())).sum()

        selection_effect = (benchmark_grouped *
                          (returns_grouped - returns.mean())).sum()

        interaction_effect = ((portfolio_grouped - benchmark_grouped) *
                            (returns_grouped - returns.mean())).sum()

        total_attribution = allocation_effect + selection_effect + interaction_effect

        return {
            "allocation_effect": allocation_effect,
            "selection_effect": selection_effect,
            "interaction_effect": interaction_effect,
            "total_attribution": total_attribution
        }

    def calculate_sector_attribution(self, portfolio_weights: pd.DataFrame,
                                   benchmark_weights: pd.DataFrame,
                                   returns: pd.DataFrame,
                                   sector_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate sector attribution over time."""
        # Create sector mapping
        symbol_to_sector = sector_data["sector"].to_dict()

        # Calculate attribution for each period
        attribution_results = []
        sector_effects = {}

        # Get unique sectors
        sectors = sector_data["sector"].unique()

        for date in portfolio_weights.index:
            if date in returns.index:
                # Create portfolio weights with sector information
                portfolio_series = portfolio_weights.loc[date]
                benchmark_series = benchmark_weights.loc[date]
                returns_series = returns.loc[date]

                # Add sector information to series
                portfolio_with_sectors = pd.Series(index=portfolio_series.index)
                benchmark_with_sectors = pd.Series(index=benchmark_series.index)
                returns_with_sectors = pd.Series(index=returns_series.index)

                for symbol in portfolio_series.index:
                    if symbol in symbol_to_sector:
                        sector = symbol_to_sector[symbol]
                        portfolio_with_sectors[symbol] = portfolio_series[symbol]
                        benchmark_with_sectors[symbol] = benchmark_series[symbol]
                        returns_with_sectors[symbol] = returns_series[symbol]

                # Group by sector
                portfolio_grouped = portfolio_with_sectors.groupby([symbol_to_sector.get(s, "Unknown") for s in portfolio_with_sectors.index]).sum()
                benchmark_grouped = benchmark_with_sectors.groupby([symbol_to_sector.get(s, "Unknown") for s in benchmark_with_sectors.index]).sum()
                returns_grouped = returns_with_sectors.groupby([symbol_to_sector.get(s, "Unknown") for s in returns_with_sectors.index]).mean()

                # Calculate effects
                allocation_effect = ((portfolio_grouped - benchmark_grouped) *
                                   (returns_grouped - returns_series.mean())).sum()

                selection_effect = (benchmark_grouped *
                                  (returns_grouped - returns_series.mean())).sum()

                interaction_effect = ((portfolio_grouped - benchmark_grouped) *
                                    (returns_grouped - returns_series.mean())).sum()

                total_attribution = allocation_effect + selection_effect + interaction_effect

                attribution_results.append({
                    "allocation_effect": allocation_effect,
                    "selection_effect": selection_effect,
                    "interaction_effect": interaction_effect,
                    "total_attribution": total_attribution
                })

        # Aggregate results
        if attribution_results:
            total_allocation = sum(r["allocation_effect"] for r in attribution_results)
            total_selection = sum(r["selection_effect"] for r in attribution_results)
            total_interaction = sum(r["interaction_effect"] for r in attribution_results)
            total_attribution = sum(r["total_attribution"] for r in attribution_results)

            # Calculate sector-level effects (simplified)
            for sector in sectors:
                sector_effects[sector] = {
                    "allocation": total_allocation / len(sectors),
                    "selection": total_selection / len(sectors),
                    "interaction": total_interaction / len(sectors)
                }

        return {
            "allocation_effect": total_allocation if attribution_results else 0.0,
            "selection_effect": total_selection if attribution_results else 0.0,
            "interaction_effect": total_interaction if attribution_results else 0.0,
            "total_attribution": total_attribution if attribution_results else 0.0,
            "sector_effects": sector_effects
        }

class RiskAdjustedAttributor:
    """Risk-adjusted attribution analysis."""

    def __init__(self, config: AttributionConfig):
        self.config = config

    def calculate_risk_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if len(returns) < 2:
            return {}

        # Ensure returns is numeric
        returns_numeric = pd.to_numeric(returns, errors="coerce").dropna()
        if len(returns_numeric) < 2:
            return {}

        # Basic statistics
        mean_return = returns_numeric.mean()
        volatility = returns_numeric.std()

        # Value at Risk (VaR)
        var_95 = np.percentile(returns_numeric, 5)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns_numeric[returns_numeric <= var_95].mean()

        # Maximum drawdown
        cumulative_returns: pd.Series = (1 + returns_numeric).cumprod()
        running_max = cumulative_returns.expanding().max()  # type: ignore
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Higher moments
        skewness = stats.skew(returns_numeric)
        kurtosis = stats.kurtosis(returns_numeric)

        # Risk-adjusted return
        sharpe_ratio = (mean_return - self.config.risk_free_rate / 252) / volatility if volatility > 0 else 0

        return {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "sharpe_ratio": sharpe_ratio
        }

    def calculate_risk_adjusted_attribution(self, portfolio_returns: pd.Series,
                                          benchmark_returns: pd.Series,
                                          factor_returns: pd.DataFrame) -> dict[str, Any]:
        """Calculate risk-adjusted attribution metrics."""
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns

        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Factor risk contributions
        factor_risk_contributions = {}
        for factor in factor_returns.columns:
            factor_vol = factor_returns[factor].std()
            factor_risk_contributions[factor] = factor_vol

        return {
            "information_ratio": information_ratio,
            "factor_risk_contributions": factor_risk_contributions
        }

class AttributionVisualizer:
    """Visualization for attribution analysis."""

    def __init__(self, config: AttributionConfig):
        self.config = config

    def create_attribution_dashboard(self, attribution_results: dict[str, Any],
                                   portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> go.Figure:
        """Create interactive attribution dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Cumulative Returns", "Factor Contributions",
                          "Risk Metrics", "Sector Attribution",
                          "Rolling Performance", "Drawdown"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Cumulative returns
        cumulative_portfolio: pd.Series = (1 + portfolio_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()  # type: ignore

        fig.add_trace(
            go.Scatter(x=cumulative_portfolio.index, y=cumulative_portfolio.values,
                      name="Portfolio", line=dict(color="blue")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark.values,
                      name="Benchmark", line=dict(color="red")),
            row=1, col=1
        )

        # Factor contributions
        if "factor_analysis" in attribution_results:
            factors = ["market", "size", "value"]
            contributions = []
            for factor in factors:
                key = f"{factor}_contribution"
                if key in attribution_results["factor_analysis"]:
                    contributions.append(attribution_results["factor_analysis"][key])
                else:
                    contributions.append(0.0)

            fig.add_trace(
                go.Bar(x=factors, y=contributions, name="Factor Contributions"),
                row=1, col=2
            )

        # Risk metrics comparison
        if "risk_analysis" in attribution_results:
            portfolio_risk = attribution_results["risk_analysis"]["portfolio_risk"]
            benchmark_risk = attribution_results["risk_analysis"]["benchmark_risk"]

            metrics = ["volatility", "sharpe_ratio"]
            portfolio_values = [portfolio_risk.get(m, 0) for m in metrics]
            benchmark_values = [benchmark_risk.get(m, 0) for m in metrics]

            fig.add_trace(
                go.Bar(x=metrics, y=portfolio_values, name="Portfolio"),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=metrics, y=benchmark_values, name="Benchmark"),
                row=2, col=1
            )

        # Sector attribution
        if "brinson_attribution" in attribution_results:
            sector_effects = attribution_results["brinson_attribution"]["sector_effects"]
            sectors = list(sector_effects.keys())
            allocation_effects = [sector_effects[s]["allocation"] for s in sectors]

            fig.add_trace(
                go.Bar(x=sectors, y=allocation_effects, name="Allocation Effect"),
                row=2, col=2
            )

        # Rolling performance
        rolling_window = 30
        rolling_portfolio = portfolio_returns.rolling(rolling_window).mean()
        rolling_benchmark = benchmark_returns.rolling(rolling_window).mean()

        fig.add_trace(
            go.Scatter(x=rolling_portfolio.index, y=rolling_portfolio.values,
                      name="Portfolio (30d)", line=dict(color="blue")),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_benchmark.index, y=rolling_benchmark.values,
                      name="Benchmark (30d)", line=dict(color="red")),
            row=3, col=1
        )

        # Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name="Drawdown", line=dict(color="red"), fill="tonexty"),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title="Performance Attribution Dashboard",
            height=900,
            showlegend=True
        )

        return fig

class PerformanceAttributor:
    """Main performance attribution orchestrator."""

    def __init__(self, config: AttributionConfig | None = None):
        self.config = config or AttributionConfig()
        self.factor_model = FactorModel(self.config)
        self.brinson_attributor = BrinsonAttributor(self.config)
        self.risk_attributor = RiskAdjustedAttributor(self.config)
        self.visualizer = AttributionVisualizer(self.config)
        self.results: dict[str, Any] = {}

    def analyze_performance(self, portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          asset_returns: pd.DataFrame,
                          portfolio_weights: pd.DataFrame,
                          benchmark_weights: pd.DataFrame,
                          sector_data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Run comprehensive performance attribution analysis."""
        # Fit factor model
        self.factor_model.fit(asset_returns, benchmark_returns)

        # Factor analysis
        factor_results = self.factor_model.analyze_factor_contributions(
            portfolio_returns, benchmark_returns, asset_returns
        )

        # Brinson attribution
        if sector_data is not None:
            brinson_results = self.brinson_attributor.calculate_sector_attribution(
                portfolio_weights, benchmark_weights, asset_returns, sector_data
            )
        else:
            brinson_results = {}

        # Risk-adjusted analysis
        portfolio_risk = self.risk_attributor.calculate_risk_metrics(portfolio_returns)
        benchmark_risk = self.risk_attributor.calculate_risk_metrics(benchmark_returns)
        risk_attribution = self.risk_attributor.calculate_risk_adjusted_attribution(
            portfolio_returns, benchmark_returns, asset_returns
        )

        # Store results
        self.results = {
            "factor_analysis": factor_results,
            "brinson_attribution": brinson_results,
            "risk_analysis": {
                "portfolio_risk": portfolio_risk,
                "benchmark_risk": benchmark_risk,
                "risk_attribution": risk_attribution
            }
        }

        return self.results

    def generate_report(self, output_path: str | None = None) -> str:
        """Generate comprehensive attribution report."""
        if not self.results:
            return "No analysis results available. Run analyze_performance() first."

        report = []
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Factor analysis
        if "factor_analysis" in self.results:
            report.append("FACTOR ANALYSIS")
            report.append("-" * 20)
            for key, value in self.results["factor_analysis"].items():
                report.append(f"{key}: {value:.4f}")
            report.append("")

        # Brinson attribution
        if "brinson_attribution" in self.results:
            report.append("BRINSON ATTRIBUTION")
            report.append("-" * 20)
            for key, value in self.results["brinson_attribution"].items():
                if key != "sector_effects":
                    report.append(f"{key}: {value:.4f}")
            report.append("")

        # Risk analysis
        if "risk_analysis" in self.results:
            report.append("RISK ANALYSIS")
            report.append("-" * 20)
            portfolio_risk = self.results["risk_analysis"]["portfolio_risk"]
            benchmark_risk = self.results["risk_analysis"]["benchmark_risk"]

            report.append("Portfolio Risk Metrics:")
            for key, value in portfolio_risk.items():
                report.append(f"  {key}: {value:.4f}")

            report.append("\nBenchmark Risk Metrics:")
            for key, value in benchmark_risk.items():
                report.append(f"  {key}: {value:.4f}")

        return "\n".join(report)

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary of key attribution results."""
        if not self.results:
            return {}

        summary = {}

        # Factor contributions
        if "factor_analysis" in self.results:
            summary["total_factor_contribution"] = self.results["factor_analysis"].get("total_factor_contribution", 0.0)
            summary["idiosyncratic_contribution"] = self.results["factor_analysis"].get("idiosyncratic_contribution", 0.0)

        # Brinson attribution
        if "brinson_attribution" in self.results:
            summary["total_attribution"] = self.results["brinson_attribution"].get("total_attribution", 0.0)

        # Risk metrics
        if "risk_analysis" in self.results:
            portfolio_risk = self.results["risk_analysis"]["portfolio_risk"]
            summary["portfolio_volatility"] = portfolio_risk.get("volatility", 0.0)
            summary["portfolio_sharpe"] = portfolio_risk.get("sharpe_ratio", 0.0)

        return summary

# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def create_sample_data() -> dict[str, Any]:
    """Create realistic sample data for attribution analysis."""
    print("Creating sample data...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Create asset symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    sectors = ["Technology", "Technology", "Technology", "Consumer", "Technology", "Technology", "Technology", "Consumer"]

    # Generate realistic asset returns with some correlation
    n_assets = len(symbols)
    n_days = len(dates)

    # Create factor returns (market, size, value factors)
    market_factor = np.random.normal(0.0005, 0.015, n_days)  # Daily market returns
    size_factor = np.random.normal(0.0002, 0.008, n_days)    # Size factor
    value_factor = np.random.normal(0.0001, 0.006, n_days)   # Value factor

    # Create asset-specific factor loadings
    factor_loadings = np.random.uniform(0.5, 1.5, (n_assets, 3))

    # Generate asset returns with factor exposure and idiosyncratic component
    asset_returns = pd.DataFrame(index=dates, columns=symbols)

    for i, symbol in enumerate(symbols):
        # Systematic component
        systematic = (factor_loadings[i, 0] * market_factor +
                     factor_loadings[i, 1] * size_factor +
                     factor_loadings[i, 2] * value_factor)

        # Idiosyncratic component
        idiosyncratic = np.random.normal(0.0001, 0.02, n_days)

        # Total returns
        asset_returns[symbol] = systematic + idiosyncratic

    # Create portfolio weights (time-varying)
    portfolio_weights = pd.DataFrame(index=dates, columns=symbols)
    for date in dates:
        # Random weights that sum to 1
        weights = np.random.dirichlet(np.ones(n_assets))
        portfolio_weights.loc[date] = weights  # type: ignore

    # Create benchmark weights (market cap weighted)
    benchmark_weights = pd.DataFrame(index=dates, columns=symbols)
    market_caps = [2.5, 1.8, 2.2, 1.5, 0.8, 1.2, 0.9, 0.3]  # Trillions
    market_caps_normalized = [cap / sum(market_caps) for cap in market_caps]
    for date in dates:
        benchmark_weights.loc[date] = market_caps_normalized  # type: ignore

    # Calculate portfolio and benchmark returns
    portfolio_returns = (portfolio_weights * asset_returns).sum(axis=1)
    benchmark_returns = (benchmark_weights * asset_returns).sum(axis=1)

    # Create sector data
    sector_data = pd.DataFrame({
        "symbol": symbols,
        "sector": sectors
    }).set_index("symbol")

    return {
        "asset_returns": asset_returns,
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "sector_data": sector_data,
        "dates": dates,
        "symbols": symbols
    }

def demo_factor_analysis(attributor: PerformanceAttributor, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate factor analysis capabilities."""
    print("\n" + "="*60)
    print("FACTOR ANALYSIS DEMONSTRATION")
    print("="*60)

    # Run factor analysis
    factor_results = attributor.factor_model.analyze_factor_contributions(
        data["portfolio_returns"],
        data["benchmark_returns"],
        data["asset_returns"]
    )

    print("Factor Analysis Results:")
    for key, value in factor_results.items():
        print(f"  {key}: {value:.4f}")

    return factor_results

def demo_brinson_attribution(attributor: PerformanceAttributor, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate Brinson attribution analysis."""
    print("\n" + "="*60)
    print("BRINSON ATTRIBUTION DEMONSTRATION")
    print("="*60)

    # Run Brinson attribution
    brinson_results = attributor.brinson_attributor.calculate_sector_attribution(
        data["portfolio_weights"],
        data["benchmark_weights"],
        data["asset_returns"],
        data["sector_data"]
    )

    print("Brinson Attribution Results:")
    for key, value in brinson_results.items():
        if key != "sector_effects":
            print(f"  {key}: {value:.4f}")

    print("\nSector-Level Attribution:")
    for sector, effects in brinson_results["sector_effects"].items():
        print(f"  {sector}:")
        for effect, value in effects.items():
            print(f"    {effect}: {value:.4f}")

    return brinson_results

def demo_risk_analysis(attributor: PerformanceAttributor, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate risk-adjusted analysis."""
    print("\n" + "="*60)
    print("RISK-ADJUSTED ANALYSIS DEMONSTRATION")
    print("="*60)

    # Calculate risk metrics
    portfolio_risk = attributor.risk_attributor.calculate_risk_metrics(data["portfolio_returns"])
    benchmark_risk = attributor.risk_attributor.calculate_risk_metrics(data["benchmark_returns"])

    print("Portfolio Risk Metrics:")
    for metric, value in portfolio_risk.items():
        print(f"  {metric}: {value:.4f}")

    print("\nBenchmark Risk Metrics:")
    for metric, value in benchmark_risk.items():
        print(f"  {metric}: {value:.4f}")

    # Calculate risk-adjusted attribution
    risk_attribution = attributor.risk_attributor.calculate_risk_adjusted_attribution(
        data["portfolio_returns"],
        data["benchmark_returns"],
        data["asset_returns"]
    )

    print("\nRisk-Adjusted Attribution:")
    print(f"Information Ratio: {risk_attribution['information_ratio']:.4f}")
    print("Factor Risk Contributions:")
    for factor, contribution in risk_attribution["factor_risk_contributions"].items():
        print(f"  {factor}: {contribution:.4f}")

    return {
        "portfolio_risk": portfolio_risk,
        "benchmark_risk": benchmark_risk,
        "risk_attribution": risk_attribution
    }

def demo_visualization(attributor: PerformanceAttributor, data: dict[str, Any]) -> dict[str, Any]:
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("VISUALIZATION DEMONSTRATION")
    print("="*60)

    # Create comprehensive attribution results for visualization
    attribution_results = {
        "factor_analysis": demo_factor_analysis(attributor, data),
        "brinson_attribution": demo_brinson_attribution(attributor, data),
        "risk_analysis": demo_risk_analysis(attributor, data)
    }

    # Create dashboard
    print("Creating attribution dashboard...")
    dashboard = attributor.visualizer.create_attribution_dashboard(
        attribution_results,
        data["portfolio_returns"],
        data["benchmark_returns"]
    )

    # Save dashboard as HTML
    dashboard_path = "attribution_dashboard.html"
    dashboard.write_html(dashboard_path)
    print(f"Dashboard saved to: {dashboard_path}")

    # Create additional plots
    print("Creating additional visualizations...")

    # Cumulative returns plot
    plt.figure(figsize=(12, 6))
    cumulative_portfolio = (1 + data["portfolio_returns"]).cumprod()
    cumulative_benchmark = (1 + data["benchmark_returns"]).cumprod()

    plt.plot(cumulative_portfolio.index, cumulative_portfolio.values, label="Portfolio", linewidth=2)
    plt.plot(cumulative_benchmark.index, cumulative_benchmark.values, label="Benchmark", linewidth=2)
    plt.title("Cumulative Returns: Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_returns.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Factor contributions plot
    plt.figure(figsize=(10, 6))
    factors = ["market", "size", "value"]
    contributions = [attribution_results["factor_analysis"].get(f"{f}_contribution", 0.0) for f in factors]

    plt.bar(factors, contributions, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Factor Contributions to Portfolio Performance")
    plt.ylabel("Contribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("factor_contributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Visualizations saved:")
    print("  - attribution_dashboard.html (interactive)")
    print("  - cumulative_returns.png")
    print("  - factor_contributions.png")

    return attribution_results

def demo_reporting(attributor: PerformanceAttributor, data: dict[str, Any]) -> str:
    """Demonstrate automated reporting capabilities."""
    print("\n" + "="*60)
    print("AUTOMATED REPORTING DEMONSTRATION")
    print("="*60)

    # Generate comprehensive report
    print("Generating comprehensive attribution report...")
    report = attributor.generate_report()

    # Save report to file
    report_path = "attribution_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print(report[:500] + "..." if len(report) > 500 else report)

    # Get summary statistics
    summary = attributor.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return report

def main() -> int:
    """Run the complete attribution analysis demonstration."""
    print("PERFORMANCE ATTRIBUTION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the comprehensive performance attribution system")
    print("with factor analysis, Brinson attribution, risk-adjusted metrics,")
    print("visualization, automation, and reporting capabilities.")
    print()

    try:
        # Create sample data
        data = create_sample_data()
        print(f"✓ Generated {len(data['dates'])} days of data for {len(data['symbols'])} assets")

        # Initialize attribution system
        print("\nInitializing attribution system...")
        config = AttributionConfig(
            factor_model_type="pca",
            n_factors=3,
            confidence_level=0.95,
            visualization_backend="plotly"
        )

        attributor = PerformanceAttributor(config)
        print("✓ Attribution system initialized")

        # Run comprehensive analysis
        print("\nRunning comprehensive attribution analysis...")
        results = attributor.analyze_performance(
            data["portfolio_returns"],
            data["benchmark_returns"],
            data["asset_returns"],
            data["portfolio_weights"],
            data["benchmark_weights"],
            data["sector_data"]
        )
        print("✓ Comprehensive analysis completed")

        # Demonstrate individual components
        demo_visualization(attributor, data)
        demo_reporting(attributor, data)

        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The performance attribution system has been demonstrated with:")
        print("✓ Factor analysis and decomposition")
        print("✓ Brinson attribution for sector allocation")
        print("✓ Risk-adjusted performance metrics")
        print("✓ Interactive visualizations")
        print("✓ Comprehensive reporting")
        print()
        print("Generated files:")
        print("  - attribution_dashboard.html (interactive dashboard)")
        print("  - cumulative_returns.png (performance chart)")
        print("  - factor_contributions.png (factor analysis)")
        print("  - attribution_report.txt (detailed report)")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
