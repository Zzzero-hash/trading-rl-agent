"""
Comprehensive tests for Monte Carlo VaR implementation.

Tests cover:
- Historical simulation with bootstrapping
- Parametric VaR with multiple distributions
- Monte Carlo simulation with correlations
- Stress testing scenarios
- Parallel processing capabilities
- VaR backtesting and validation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from trading_rl_agent.risk.monte_carlo_var import (
    MonteCarloVaR,
    MonteCarloVaRConfig,
    VaRResult
)
from trading_rl_agent.risk.parallel_var import (
    ParallelVaRCalculator,
    ParallelVaRConfig
)


@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    
    # Generate correlated returns
    n_days = 500
    n_assets = 5
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Market factor
    market_factor = np.random.normal(0, 0.02, n_days)
    
    # Asset returns
    asset_returns = {}
    for i in range(n_assets):
        asset_name = f"ASSET_{i+1:02d}"
        beta = np.random.uniform(0.5, 1.5)
        idiosyncratic = np.random.normal(0, 0.015, n_days)
        returns = beta * market_factor + idiosyncratic
        asset_returns[asset_name] = returns
    
    return pd.DataFrame(asset_returns, index=dates)


@pytest.fixture
def sample_weights():
    """Sample portfolio weights."""
    return {
        "ASSET_01": 0.3,
        "ASSET_02": 0.25,
        "ASSET_03": 0.2,
        "ASSET_04": 0.15,
        "ASSET_05": 0.1
    }


@pytest.fixture
def var_config():
    """Sample VaR configuration."""
    return MonteCarloVaRConfig(
        n_simulations=1000,
        confidence_level=0.05,
        time_horizon=1,
        lookback_period=252
    )


class TestMonteCarloVaRConfig:
    """Test MonteCarloVaRConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MonteCarloVaRConfig()
        
        assert config.n_simulations == 10000
        assert config.confidence_level == 0.05
        assert config.time_horizon == 1
        assert config.lookback_period == 252
        assert config.distribution_type == "normal"
        assert config.bootstrap_samples == 1000
        assert config.block_size == 5
        assert config.correlation_method == "ledoit_wolf"
        assert config.use_parallel is True
        assert config.n_workers == 4
        assert config.backtest_window == 252
        assert config.var_breaches_threshold == 0.05
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MonteCarloVaRConfig(
            n_simulations=5000,
            confidence_level=0.01,
            time_horizon=5,
            distribution_type="t"
        )
        
        assert config.n_simulations == 5000
        assert config.confidence_level == 0.01
        assert config.time_horizon == 5
        assert config.distribution_type == "t"
    
    def test_stress_scenarios_initialization(self):
        """Test stress scenarios initialization."""
        config = MonteCarloVaRConfig()
        
        assert "market_crash" in config.stress_scenarios
        assert "flash_crash" in config.stress_scenarios
        assert "liquidity_crisis" in config.stress_scenarios
        
        # Check scenario parameters
        market_crash = config.stress_scenarios["market_crash"]
        assert "volatility_multiplier" in market_crash
        assert "correlation_increase" in market_crash
        assert "mean_shift" in market_crash
    
    def test_invalid_config(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            MonteCarloVaRConfig(confidence_level=1.5)  # Invalid confidence level
        
        with pytest.raises(ValueError):
            MonteCarloVaRConfig(n_simulations=-1)  # Invalid simulation count
        
        with pytest.raises(ValueError):
            MonteCarloVaRConfig(lookback_period=0)  # Invalid lookback period


class TestMonteCarloVaR:
    """Test MonteCarloVaR class."""
    
    def test_initialization(self, var_config):
        """Test VaR calculator initialization."""
        var_calc = MonteCarloVaR(var_config)
        
        assert var_calc.config == var_config
        assert var_calc._returns_data is None
        assert var_calc._covariance_matrix is None
        assert var_calc._correlation_matrix is None
        assert len(var_calc._var_history) == 0
        assert len(var_calc._backtest_results) == 0
    
    def test_update_data(self, var_config, sample_returns_data):
        """Test data update functionality."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        assert var_calc._returns_data is not None
        assert len(var_calc._returns_data) == len(sample_returns_data)
        assert var_calc._covariance_matrix is not None
        assert var_calc._correlation_matrix is not None
        
        # Check covariance matrix dimensions
        assert var_calc._covariance_matrix.shape == (5, 5)
        assert var_calc._correlation_matrix.shape == (5, 5)
    
    def test_historical_simulation_var(self, var_config, sample_returns_data, sample_weights):
        """Test historical simulation VaR calculation."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Test without bootstrapping
        result = var_calc.historical_simulation_var(sample_weights, use_bootstrap=False)
        
        assert isinstance(result, VaRResult)
        assert result.method == "historical_simulation"
        assert result.distribution == "empirical"
        assert result.confidence_level == var_config.confidence_level
        assert result.time_horizon == var_config.time_horizon
        assert result.var_value > 0
        assert result.cvar_value > 0
        assert result.calculation_time > 0
        assert result.simulation_count == len(sample_returns_data)
        
        # Test with bootstrapping
        result_bootstrap = var_calc.historical_simulation_var(sample_weights, use_bootstrap=True)
        
        assert isinstance(result_bootstrap, VaRResult)
        assert result_bootstrap.additional_metrics["bootstrap_used"] is True
    
    def test_parametric_var_normal(self, var_config, sample_returns_data, sample_weights):
        """Test parametric VaR with normal distribution."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        result = var_calc.parametric_var(sample_weights, distribution="normal")
        
        assert isinstance(result, VaRResult)
        assert result.method == "parametric"
        assert result.distribution == "normal"
        assert result.var_value > 0
        assert result.cvar_value > 0
        assert "portfolio_mean" in result.additional_metrics
        assert "portfolio_std" in result.additional_metrics
    
    def test_parametric_var_t_distribution(self, var_config, sample_returns_data, sample_weights):
        """Test parametric VaR with t-distribution."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        result = var_calc.parametric_var(sample_weights, distribution="t")
        
        assert isinstance(result, VaRResult)
        assert result.method == "parametric"
        assert result.distribution == "t"
        assert result.var_value > 0
        assert result.cvar_value > 0
    
    def test_parametric_var_laplace(self, var_config, sample_returns_data, sample_weights):
        """Test parametric VaR with Laplace distribution."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        result = var_calc.parametric_var(sample_weights, distribution="laplace")
        
        assert isinstance(result, VaRResult)
        assert result.method == "parametric"
        assert result.distribution == "laplace"
        assert result.var_value > 0
        assert result.cvar_value > 0
    
    def test_parametric_var_logistic(self, var_config, sample_returns_data, sample_weights):
        """Test parametric VaR with logistic distribution."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        result = var_calc.parametric_var(sample_weights, distribution="logistic")
        
        assert isinstance(result, VaRResult)
        assert result.method == "parametric"
        assert result.distribution == "logistic"
        assert result.var_value > 0
        assert result.cvar_value > 0
    
    def test_parametric_var_invalid_distribution(self, var_config, sample_returns_data, sample_weights):
        """Test parametric VaR with invalid distribution."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        with pytest.raises(ValueError):
            var_calc.parametric_var(sample_weights, distribution="invalid")
    
    def test_monte_carlo_var(self, var_config, sample_returns_data, sample_weights):
        """Test Monte Carlo VaR calculation."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Test with correlation
        result = var_calc.monte_carlo_var(sample_weights, use_correlation=True)
        
        assert isinstance(result, VaRResult)
        assert result.method == "monte_carlo"
        assert result.distribution == "simulated"
        assert result.var_value > 0
        assert result.cvar_value > 0
        assert result.simulation_count == var_config.n_simulations
        assert result.additional_metrics["correlation_used"] is True
        
        # Test without correlation
        result_no_corr = var_calc.monte_carlo_var(sample_weights, use_correlation=False)
        
        assert result_no_corr.additional_metrics["correlation_used"] is False
    
    def test_stress_test_var(self, var_config, sample_returns_data, sample_weights):
        """Test stress testing functionality."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Test all scenarios
        stress_results = var_calc.stress_test_var(sample_weights)
        
        assert isinstance(stress_results, dict)
        assert len(stress_results) > 0
        
        for scenario_name, result in stress_results.items():
            assert isinstance(result, VaRResult)
            assert result.method.startswith("stress_test_")
            assert result.var_value > 0
            assert result.cvar_value > 0
        
        # Test specific scenario
        specific_results = var_calc.stress_test_var(sample_weights, scenario="market_crash")
        assert "market_crash" in specific_results
    
    def test_backtest_var(self, var_config, sample_returns_data, sample_weights):
        """Test VaR backtesting functionality."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Test Monte Carlo backtesting
        backtest_results = var_calc.backtest_var(sample_weights, method="monte_carlo")
        
        assert isinstance(backtest_results, dict)
        assert "method" in backtest_results
        assert "total_predictions" in backtest_results
        assert "breach_count" in backtest_results
        assert "breach_rate" in backtest_results
        assert "expected_breach_rate" in backtest_results
        assert "kupiec_statistic" in backtest_results
        assert "kupiec_pvalue" in backtest_results
        assert "christoffersen_statistic" in backtest_results
        assert "christoffersen_pvalue" in backtest_results
        
        assert backtest_results["method"] == "monte_carlo"
        assert backtest_results["total_predictions"] > 0
        assert 0 <= backtest_results["breach_rate"] <= 1
        assert backtest_results["expected_breach_rate"] == var_config.confidence_level
    
    def test_get_var_summary(self, var_config, sample_returns_data, sample_weights):
        """Test VaR summary functionality."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Calculate some VaR values first
        var_calc.historical_simulation_var(sample_weights)
        var_calc.parametric_var(sample_weights)
        var_calc.monte_carlo_var(sample_weights)
        
        summary = var_calc.get_var_summary()
        
        assert isinstance(summary, dict)
        assert summary["total_calculations"] == 3
        assert len(summary["methods_used"]) > 0
        assert len(summary["distributions_used"]) > 0
        assert summary["latest_var"] is not None
        assert summary["average_calculation_time"] > 0
    
    def test_clear_history(self, var_config, sample_returns_data, sample_weights):
        """Test history clearing functionality."""
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Calculate some VaR values
        var_calc.historical_simulation_var(sample_weights)
        var_calc.backtest_var(sample_weights)
        
        assert len(var_calc._var_history) > 0
        assert len(var_calc._backtest_results) > 0
        
        var_calc.clear_history()
        
        assert len(var_calc._var_history) == 0
        assert len(var_calc._backtest_results) == 0


class TestParallelVaRCalculator:
    """Test ParallelVaRCalculator class."""
    
    def test_initialization(self):
        """Test parallel VaR calculator initialization."""
        config = ParallelVaRConfig()
        parallel_calc = ParallelVaRCalculator(config)
        
        assert parallel_calc.config == config
        assert parallel_calc._process_executor is None
        assert parallel_calc._thread_executor is None
        assert len(parallel_calc._results) == 0
    
    def test_context_manager(self):
        """Test context manager functionality."""
        config = ParallelVaRConfig(use_multiprocessing=False)  # Use threading only for testing
        
        with ParallelVaRCalculator(config) as parallel_calc:
            assert parallel_calc._thread_executor is not None
            assert parallel_calc._process_executor is None
        
        # After context exit, executors should be cleaned up
        assert parallel_calc._thread_executor is None
        assert parallel_calc._process_executor is None
    
    def test_parallel_monte_carlo_var(self, var_config, sample_returns_data, sample_weights):
        """Test parallel Monte Carlo VaR calculation."""
        parallel_config = ParallelVaRConfig(
            use_multiprocessing=False,  # Use threading for testing
            n_threads=2
        )
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            result = parallel_calc.parallel_monte_carlo_var(
                var_config, sample_returns_data, sample_weights
            )
            
            assert isinstance(result, VaRResult)
            assert result.method == "parallel_monte_carlo"
            assert result.var_value > 0
            assert result.cvar_value > 0
            assert result.simulation_count == var_config.n_simulations
            assert "n_chunks" in result.additional_metrics
    
    def test_parallel_stress_test(self, var_config, sample_returns_data, sample_weights):
        """Test parallel stress testing."""
        parallel_config = ParallelVaRConfig(
            use_multiprocessing=False,
            n_threads=2
        )
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            stress_results = parallel_calc.parallel_stress_test(
                var_config, sample_returns_data, sample_weights
            )
            
            assert isinstance(stress_results, dict)
            assert len(stress_results) > 0
            
            for scenario_name, result in stress_results.items():
                assert isinstance(result, VaRResult)
                assert result.var_value > 0
    
    def test_parallel_method_comparison(self, var_config, sample_returns_data, sample_weights):
        """Test parallel method comparison."""
        parallel_config = ParallelVaRConfig(
            use_multiprocessing=False,
            n_threads=2
        )
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            method_results = parallel_calc.parallel_method_comparison(
                var_config, sample_returns_data, sample_weights
            )
            
            assert isinstance(method_results, dict)
            assert len(method_results) > 0
            
            for method_name, result in method_results.items():
                assert isinstance(result, VaRResult)
                assert result.var_value > 0
    
    def test_get_performance_metrics(self, var_config, sample_returns_data, sample_weights):
        """Test performance metrics functionality."""
        parallel_config = ParallelVaRConfig(use_multiprocessing=False)
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            # Run some calculations first
            parallel_calc.parallel_monte_carlo_var(var_config, sample_returns_data, sample_weights)
            
            metrics = parallel_calc.get_performance_metrics()
            
            assert isinstance(metrics, dict)
            assert metrics["total_calculations"] > 0
            assert metrics["total_simulations"] > 0
            assert "parallel_config" in metrics
    
    def test_clear_results(self, var_config, sample_returns_data, sample_weights):
        """Test results clearing functionality."""
        parallel_config = ParallelVaRConfig(use_multiprocessing=False)
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            # Run some calculations
            parallel_calc.parallel_monte_carlo_var(var_config, sample_returns_data, sample_weights)
            
            assert len(parallel_calc._results) > 0
            
            parallel_calc.clear_results()
            
            assert len(parallel_calc._results) == 0


class TestVaRResult:
    """Test VaRResult dataclass."""
    
    def test_var_result_creation(self):
        """Test VaRResult creation."""
        result = VaRResult(
            var_value=0.02,
            cvar_value=0.03,
            confidence_level=0.05,
            time_horizon=1,
            method="test_method",
            distribution="test_dist",
            simulation_count=1000,
            calculation_time=1.5,
            additional_metrics={"test": "value"},
            timestamp=datetime.now()
        )
        
        assert result.var_value == 0.02
        assert result.cvar_value == 0.03
        assert result.confidence_level == 0.05
        assert result.time_horizon == 1
        assert result.method == "test_method"
        assert result.distribution == "test_dist"
        assert result.simulation_count == 1000
        assert result.calculation_time == 1.5
        assert result.additional_metrics["test"] == "value"
        assert isinstance(result.timestamp, datetime)


class TestIntegration:
    """Integration tests for Monte Carlo VaR system."""
    
    def test_full_workflow(self, var_config, sample_returns_data, sample_weights):
        """Test complete VaR calculation workflow."""
        # Initialize VaR calculator
        var_calc = MonteCarloVaR(var_config)
        var_calc.update_data(sample_returns_data)
        
        # Calculate VaR using different methods
        results = {}
        
        # Historical simulation
        results["historical"] = var_calc.historical_simulation_var(sample_weights)
        
        # Parametric VaR
        results["parametric"] = var_calc.parametric_var(sample_weights)
        
        # Monte Carlo VaR
        results["monte_carlo"] = var_calc.monte_carlo_var(sample_weights)
        
        # Stress testing
        stress_results = var_calc.stress_test_var(sample_weights)
        
        # Backtesting
        backtest_results = var_calc.backtest_var(sample_weights)
        
        # Verify all results
        for method, result in results.items():
            assert isinstance(result, VaRResult)
            assert result.var_value > 0
            assert result.cvar_value > 0
        
        assert len(stress_results) > 0
        assert isinstance(backtest_results, dict)
        
        # Get summary
        summary = var_calc.get_var_summary()
        assert summary["total_calculations"] == 3
    
    def test_parallel_integration(self, var_config, sample_returns_data, sample_weights):
        """Test parallel processing integration."""
        parallel_config = ParallelVaRConfig(
            use_multiprocessing=False,
            n_threads=2
        )
        
        with ParallelVaRCalculator(parallel_config) as parallel_calc:
            # Parallel Monte Carlo VaR
            parallel_result = parallel_calc.parallel_monte_carlo_var(
                var_config, sample_returns_data, sample_weights
            )
            
            # Parallel stress testing
            stress_results = parallel_calc.parallel_stress_test(
                var_config, sample_returns_data, sample_weights
            )
            
            # Parallel method comparison
            method_results = parallel_calc.parallel_method_comparison(
                var_config, sample_returns_data, sample_weights
            )
            
            # Verify results
            assert isinstance(parallel_result, VaRResult)
            assert len(stress_results) > 0
            assert len(method_results) > 0


if __name__ == "__main__":
    pytest.main([__file__])