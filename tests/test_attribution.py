"""
Comprehensive tests for performance attribution analysis system.

Tests all components including:
- Factor model analysis
- Brinson attribution
- Risk-adjusted analysis
- Integration with portfolio manager
- CLI functionality
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.trading_rl_agent.portfolio.attribution import (
    AttributionConfig,
    AttributionVisualizer,
    BrinsonAttributor,
    FactorModel,
    PerformanceAttributor,
    RiskAdjustedAttributor,
)
from src.trading_rl_agent.portfolio.attribution_integration import (
    AttributionIntegration,
    AutomatedAttributionWorkflow,
)
from src.trading_rl_agent.portfolio.manager import PortfolioConfig, PortfolioManager


@pytest.fixture
def sample_data(random_seed=42):
    """Create sample data for testing."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Generate sample returns
    np.random.seed(random_seed)
    asset_returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(symbols), len(dates))), index=symbols, columns=dates
    )

    # Generate sample weights
    portfolio_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(len(symbols)), len(dates)).T, index=symbols, columns=dates
    )

    benchmark_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(len(symbols)) * 2, len(dates)).T, index=symbols, columns=dates
    )

    # Calculate portfolio and benchmark returns
    portfolio_returns = pd.Series(index=dates, dtype=float)
    benchmark_returns = pd.Series(index=dates, dtype=float)

    for date in dates:
        portfolio_returns[date] = (portfolio_weights.loc[:, date] * asset_returns.loc[:, date]).sum()
        benchmark_returns[date] = (benchmark_weights.loc[:, date] * asset_returns.loc[:, date]).sum()

    # Create sector data
    sector_data = pd.DataFrame(
        {"sector": ["Technology", "Technology", "Technology", "Consumer", "Technology"]}, index=symbols
    )

    return {
        "asset_returns": asset_returns,
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "sector_data": sector_data,
        "dates": dates,
        "symbols": symbols,
    }


@pytest.fixture
def attribution_config():
    """Create attribution configuration for testing."""
    return AttributionConfig(
        risk_free_rate=0.02,
        confidence_level=0.95,
        lookback_period=252,
        use_plotly=False,  # Use matplotlib for testing
    )


class TestAttributionConfig:
    """Test attribution configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AttributionConfig()
        assert config.risk_free_rate == 0.02
        assert config.confidence_level == 0.95
        assert config.lookback_period == 252
        assert config.use_plotly is True
        assert config.max_factors == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AttributionConfig(
            risk_free_rate=0.03, confidence_level=0.99, lookback_period=500, use_plotly=False, max_factors=5
        )
        assert config.risk_free_rate == 0.03
        assert config.confidence_level == 0.99
        assert config.lookback_period == 500
        assert config.use_plotly is False
        assert config.max_factors == 5


class TestFactorModel:
    """Test factor model functionality."""

    def test_factor_model_initialization(self, attribution_config):
        """Test factor model initialization."""
        model = FactorModel(attribution_config)
        assert model.config == attribution_config
        assert model.factors is None
        assert model.factor_loadings is None
        assert model.residuals is None
        assert model.r_squared is None

    def test_factor_model_fit(self, attribution_config, sample_data):
        """Test factor model fitting."""
        model = FactorModel(attribution_config)

        # Fit the model
        model.fit(sample_data["asset_returns"], sample_data["benchmark_returns"])

        # Check that factors were created
        assert model.factors is not None
        assert model.factor_loadings is not None
        assert model.residuals is not None
        assert model.r_squared is not None

        # Check dimensions
        assert model.factors.shape[0] == len(sample_data["dates"])
        assert "Market" in model.factors.columns

    def test_return_decomposition(self, attribution_config, sample_data):
        """Test return decomposition."""
        model = FactorModel(attribution_config)
        model.fit(sample_data["asset_returns"], sample_data["benchmark_returns"])

        decomposition = model.decompose_returns(sample_data["asset_returns"])

        assert "systematic" in decomposition
        assert "idiosyncratic" in decomposition
        assert decomposition["systematic"].shape == sample_data["asset_returns"].shape
        assert decomposition["idiosyncratic"].shape == sample_data["asset_returns"].shape

    def test_factor_model_with_insufficient_data(self, attribution_config):
        """Test factor model with insufficient data."""
        model = FactorModel(attribution_config)

        # Create data with insufficient observations
        small_returns = pd.DataFrame(
            np.random.normal(0, 0.01, (3, 10)), index=["A", "B", "C"], columns=pd.date_range("2023-01-01", periods=10)
        )
        small_benchmark = pd.Series(np.random.normal(0, 0.01, 10), index=pd.date_range("2023-01-01", periods=10))

        # Should handle insufficient data gracefully
        model.fit(small_returns, small_benchmark)
        assert model.factors is not None


class TestBrinsonAttributor:
    """Test Brinson attribution functionality."""

    def test_brinson_attributor_initialization(self, attribution_config):
        """Test Brinson attributor initialization."""
        attributor = BrinsonAttributor(attribution_config)
        assert attributor.config == attribution_config

    def test_calculate_attribution(self, attribution_config):
        """Test Brinson attribution calculation."""
        attributor = BrinsonAttributor(attribution_config)

        # Create test data
        portfolio_weights = pd.Series({"Tech": 0.6, "Consumer": 0.4})
        benchmark_weights = pd.Series({"Tech": 0.5, "Consumer": 0.5})
        returns = pd.Series({"Tech": 0.1, "Consumer": 0.05})

        benchmark_returns = pd.Series({"Tech": 0.08, "Consumer": 0.06})
        result = attributor.calculate_attribution(portfolio_weights, benchmark_weights, returns, benchmark_returns)

        assert "allocation" in result
        assert "selection" in result
        assert "interaction" in result
        assert "total" in result

        # Check that components sum to total
        total = result["allocation"] + result["selection"] + result["interaction"]
        assert abs(total - result["total"]) < 1e-10

    def test_calculate_attribution_no_common_groups(self, attribution_config):
        """Test attribution with no common groups."""
        attributor = BrinsonAttributor(attribution_config)

        portfolio_weights = pd.Series({"Tech": 0.6, "Consumer": 0.4})
        benchmark_weights = pd.Series({"Finance": 0.5, "Energy": 0.5})
        returns = pd.Series({"Tech": 0.1, "Consumer": 0.05})

        benchmark_returns = pd.Series({"Finance": 0.08, "Energy": 0.06})
        with pytest.raises(ValueError, match="No common groups found"):
            attributor.calculate_attribution(portfolio_weights, benchmark_weights, returns, benchmark_returns)


class TestRiskAdjustedAttributor:
    """Test risk-adjusted attribution functionality."""

    def test_risk_adjusted_attributor_initialization(self, attribution_config):
        """Test risk-adjusted attributor initialization."""
        attributor = RiskAdjustedAttributor(attribution_config)
        assert attributor.config == attribution_config

    def test_calculate_risk_metrics(self, attribution_config):
        """Test risk metrics calculation."""
        attributor = RiskAdjustedAttributor(attribution_config)

        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        metrics = attributor.calculate_risk_metrics(returns)

        assert "volatility" in metrics
        assert "var" in metrics
        assert "cvar" in metrics
        assert "max_drawdown" in metrics
        assert "skewness" in metrics
        assert "kurtosis" in metrics

        # Check that volatility is positive
        assert metrics["volatility"] > 0

    def test_calculate_risk_metrics_insufficient_data(self, attribution_config):
        """Test risk metrics with insufficient data."""
        attributor = RiskAdjustedAttributor(attribution_config)

        # Single return value
        returns = pd.Series([0.01])

        metrics = attributor.calculate_risk_metrics(returns)
        assert metrics == {}

    def test_risk_adjusted_attribution(self, attribution_config, sample_data):
        """Test risk-adjusted attribution calculation."""
        attributor = RiskAdjustedAttributor(attribution_config)

        # Create factor returns
        factor_returns = pd.DataFrame(
            {
                "Factor_1": np.random.normal(0, 0.01, len(sample_data["dates"])),
                "Factor_2": np.random.normal(0, 0.01, len(sample_data["dates"])),
            },
            index=sample_data["dates"],
        )

        result = attributor.calculate_risk_adjusted_attribution(
            sample_data["portfolio_returns"], sample_data["benchmark_returns"], factor_returns
        )

        assert "portfolio_risk" in result
        assert "benchmark_risk" in result
        assert "excess_risk" in result
        assert "information_ratio" in result
        assert "factor_risk_contributions" in result


class TestAttributionVisualizer:
    """Test attribution visualization functionality."""

    def test_visualizer_initialization(self, attribution_config):
        """Test visualizer initialization."""
        visualizer = AttributionVisualizer(attribution_config)
        assert visualizer.config == attribution_config

    def test_create_matplotlib_dashboard(self, attribution_config, sample_data):
        """Test matplotlib dashboard creation."""
        visualizer = AttributionVisualizer(attribution_config)

        # Create mock attribution results
        attribution_results = {
            "factor_attribution": {"Factor_1": 0.01, "Factor_2": 0.005},
            "risk_metrics": {"volatility": 0.15, "max_drawdown": -0.1},
            "factor_loadings": pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"]),
            "sector_attribution": {"Tech": 0.02, "Consumer": 0.01},
            "risk_adjusted": {"information_ratio": 0.8, "sharpe_ratio": 1.2},
        }

        fig = visualizer.create_attribution_dashboard(
            attribution_results, sample_data["portfolio_returns"], sample_data["benchmark_returns"]
        )

        assert fig is not None
        assert hasattr(fig, "savefig")


class TestPerformanceAttributor:
    """Test main performance attributor functionality."""

    def test_attributor_initialization(self, attribution_config):
        """Test performance attributor initialization."""
        attributor = PerformanceAttributor(attribution_config)
        assert attributor.config == attribution_config
        assert attributor.attribution_results == {}

    def test_analyze_performance(self, attribution_config, sample_data):
        """Test comprehensive performance analysis."""
        attributor = PerformanceAttributor(attribution_config)

        results = attributor.analyze_performance(
            portfolio_returns=sample_data["portfolio_returns"],
            benchmark_returns=sample_data["benchmark_returns"],
            asset_returns=sample_data["asset_returns"],
            portfolio_weights=sample_data["portfolio_weights"],
            benchmark_weights=sample_data["benchmark_weights"],
            sector_data=sample_data["sector_data"],
        )

        assert "decomposition" in results
        assert "factor_attribution" in results
        assert "brinson_attribution" in results
        assert "risk_adjusted" in results
        assert "factor_model" in results

        # Check that results are stored
        assert attributor.attribution_results == results

    def test_generate_report(self, attribution_config, sample_data):
        """Test report generation."""
        attributor = PerformanceAttributor(attribution_config)

        # Run analysis first
        attributor.analyze_performance(
            portfolio_returns=sample_data["portfolio_returns"],
            benchmark_returns=sample_data["benchmark_returns"],
            asset_returns=sample_data["asset_returns"],
            portfolio_weights=sample_data["portfolio_weights"],
            benchmark_weights=sample_data["benchmark_weights"],
            sector_data=sample_data["sector_data"],
        )

        # Generate report
        report = attributor.generate_report()

        assert isinstance(report, str)
        assert "PERFORMANCE ATTRIBUTION ANALYSIS REPORT" in report

    def test_get_summary_statistics(self, attribution_config, sample_data):
        """Test summary statistics generation."""
        attributor = PerformanceAttributor(attribution_config)

        # Run analysis first
        attributor.analyze_performance(
            portfolio_returns=sample_data["portfolio_returns"],
            benchmark_returns=sample_data["benchmark_returns"],
            asset_returns=sample_data["asset_returns"],
            portfolio_weights=sample_data["portfolio_weights"],
            benchmark_weights=sample_data["benchmark_weights"],
            sector_data=sample_data["sector_data"],
        )

        # Get summary
        summary = attributor.get_summary_statistics()

        assert isinstance(summary, dict)
        assert "total_factor_contribution" in summary or summary == {}


class TestAttributionIntegration:
    """Test attribution integration with portfolio manager."""

    def test_integration_initialization(self, attribution_config):
        """Test integration initialization."""
        # Create mock portfolio manager
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.performance_history = pd.DataFrame()

        integration = AttributionIntegration(portfolio_manager, attribution_config)

        assert integration.portfolio_manager == portfolio_manager
        assert integration.attributor is not None
        assert integration._attribution_cache == {}

    def test_prepare_attribution_data_no_history(self, attribution_config):
        """Test data preparation with no performance history."""
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.performance_history = pd.DataFrame()

        integration = AttributionIntegration(portfolio_manager, attribution_config)

        with pytest.raises(ValueError, match="No performance history available"):
            integration.prepare_attribution_data()

    def test_prepare_attribution_data_with_history(self, attribution_config):
        """Test data preparation with performance history."""
        portfolio_manager = Mock(spec=PortfolioManager)

        # Create mock performance history
        dates = pd.date_range("2023-01-01", periods=10)
        portfolio_manager.performance_history = pd.DataFrame(
            [
                {
                    "timestamp": date,
                    "total_return": 1000000 + i * 1000,
                    "total_value": 1000000 + i * 1000,
                    "cash": 100000,
                    "equity_value": 900000 + i * 1000,
                }
                for i, date in enumerate(dates)
            ]
        )

        integration = AttributionIntegration(portfolio_manager, attribution_config)

        data = integration.prepare_attribution_data()

        assert "portfolio_returns" in data
        assert "benchmark_returns" in data
        assert "asset_returns" in data
        assert "portfolio_weights" in data
        assert "benchmark_weights" in data
        assert "sector_data" in data


class TestAutomatedAttributionWorkflow:
    """Test automated attribution workflow."""

    def test_workflow_initialization(self, attribution_config):
        """Test workflow initialization."""
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.performance_history = pd.DataFrame()

        workflow = AutomatedAttributionWorkflow(portfolio_manager, attribution_config)

        assert workflow.integration is not None
        assert workflow.analysis_frequency == "monthly"
        assert workflow.auto_generate_reports is True

    def test_should_run_analysis(self, attribution_config):
        """Test analysis scheduling logic."""
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.performance_history = pd.DataFrame()

        workflow = AutomatedAttributionWorkflow(portfolio_manager, attribution_config)

        # Should run if no previous analysis
        assert workflow.should_run_analysis() is True

        # Should not run if recent analysis
        workflow.last_analysis = datetime.now()
        assert workflow.should_run_analysis() is False

        # Should run if enough time has passed
        workflow.last_analysis = datetime.now() - timedelta(days=35)
        assert workflow.should_run_analysis() is True


class TestIntegrationWithRealData:
    """Test integration with realistic data scenarios."""

    def test_end_to_end_analysis(self, attribution_config, sample_data):
        """Test complete end-to-end attribution analysis."""
        # Create portfolio manager with sample data
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.config = PortfolioConfig()

        # Add performance history
        portfolio_manager.performance_history = [
            {
                "timestamp": date,
                "total_return": sample_data["portfolio_returns"][
                    :date
                ].sum(),  # or multiply by initial capital for absolute return
                "total_value": 1000000 * (1 + sample_data["portfolio_returns"][:date].sum()),
                "cash": 100000,
                "equity_value": 900000 * (1 + sample_data["portfolio_returns"][:date].sum()),
            }
            for date in sample_data["dates"]
        ]

        # Create integration
        integration = AttributionIntegration(portfolio_manager, attribution_config)

        # Run analysis
        results = integration.run_attribution_analysis(
            start_date=sample_data["dates"][0], end_date=sample_data["dates"][-1]
        )

        # Verify results
        assert results is not None
        assert "decomposition" in results
        assert "factor_attribution" in results
        assert "risk_adjusted" in results

    def test_error_handling(self, attribution_config):
        """Test error handling in attribution analysis."""
        attributor = PerformanceAttributor(attribution_config)

        # Test with invalid data
        with pytest.raises(ValueError):
            attributor.analyze_performance(
                portfolio_returns=pd.Series(),
                benchmark_returns=pd.Series(),
                asset_returns=pd.DataFrame(),
                portfolio_weights=pd.DataFrame(),
                benchmark_weights=pd.DataFrame(),
                sector_data=None,
            )


if __name__ == "__main__":
    pytest.main([__file__])
