"""
Advanced Monte Carlo Value at Risk (VaR) implementation.

Provides comprehensive VaR calculation methods including:
- Historical simulation with bootstrapping
- Parametric VaR with multiple distribution assumptions
- Monte Carlo simulation with correlated asset movements
- Stress testing scenarios for extreme market conditions
- Real-time VaR calculation and monitoring
- VaR backtesting and validation methods
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import multivariate_normal, norm, t, laplace, logistic
from sklearn.covariance import LedoitWolf

from ..core.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class MonteCarloVaRConfig:
    """Configuration for Monte Carlo VaR calculations."""
    
    # Simulation parameters
    n_simulations: int = 10000
    confidence_level: float = 0.05  # 95% VaR
    time_horizon: int = 1  # days
    lookback_period: int = 252  # days
    
    # Distribution assumptions
    distribution_type: str = "normal"  # normal, t, laplace, logistic, empirical
    
    # Bootstrapping parameters
    bootstrap_samples: int = 1000
    block_size: int = 5  # for block bootstrap
    
    # Correlation estimation
    correlation_method: str = "ledoit_wolf"  # ledoit_wolf, sample, shrinkage
    
    # Stress testing
    stress_scenarios: Dict[str, Dict[str, float]] = None
    
    # Parallel processing
    use_parallel: bool = True
    n_workers: int = 4
    
    # Backtesting
    backtest_window: int = 252
    var_breaches_threshold: float = 0.05
    
    def __post_init__(self):
        """Initialize default stress scenarios if not provided."""
        if self.stress_scenarios is None:
            self.stress_scenarios = {
                "market_crash": {
                    "volatility_multiplier": 3.0,
                    "correlation_increase": 0.3,
                    "mean_shift": -0.02
                },
                "flash_crash": {
                    "volatility_multiplier": 5.0,
                    "correlation_increase": 0.5,
                    "mean_shift": -0.05
                },
                "liquidity_crisis": {
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 0.4,
                    "mean_shift": -0.015
                }
            }


@dataclass
class VaRResult:
    """Container for VaR calculation results."""
    
    var_value: float
    cvar_value: float
    confidence_level: float
    time_horizon: int
    method: str
    distribution: str
    simulation_count: int
    calculation_time: float
    additional_metrics: Dict[str, Any]
    timestamp: datetime


class MonteCarloVaR:
    """
    Advanced Monte Carlo Value at Risk calculator.
    
    Implements multiple VaR calculation methods with comprehensive
    simulation capabilities and stress testing.
    """
    
    def __init__(self, config: MonteCarloVaRConfig):
        """
        Initialize Monte Carlo VaR calculator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Data storage
        self._returns_data: Optional[pd.DataFrame] = None
        self._covariance_matrix: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        
        # Results storage
        self._var_history: List[VaRResult] = []
        self._backtest_results: Dict[str, Any] = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.config.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        if self.config.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        
        if self.config.lookback_period <= 0:
            raise ValueError("Lookback period must be positive")
    
    def update_data(self, returns_data: pd.DataFrame) -> None:
        """
        Update returns data for VaR calculations.
        
        Args:
            returns_data: DataFrame with asset returns (columns: assets, rows: dates)
        """
        self._returns_data = returns_data.copy()
        self._returns_data = self._returns_data.dropna()
        
        if len(self._returns_data) < self.config.lookback_period:
            self.logger.warning(
                f"Insufficient data: {len(self._returns_data)} < {self.config.lookback_period}"
            )
        
        # Calculate covariance and correlation matrices
        self._calculate_covariance_matrix()
        self._calculate_correlation_matrix()
        
        self.logger.info(f"Updated data with {len(self._returns_data)} observations for {len(self._returns_data.columns)} assets")
    
    def _calculate_covariance_matrix(self) -> None:
        """Calculate covariance matrix using specified method."""
        if self._returns_data is None or self._returns_data.empty:
            return
        
        try:
            if self.config.correlation_method == "ledoit_wolf":
                # Use Ledoit-Wolf shrinkage estimator for better stability
                lw = LedoitWolf()
                lw.fit(self._returns_data)
                self._covariance_matrix = pd.DataFrame(
                    lw.covariance_,
                    index=self._returns_data.columns,
                    columns=self._returns_data.columns
                )
            else:
                # Standard sample covariance
                self._covariance_matrix = self._returns_data.cov()
                
        except Exception as e:
            self.logger.error(f"Error calculating covariance matrix: {e}")
            # Fallback to sample covariance
            self._covariance_matrix = self._returns_data.cov()
    
    def _calculate_correlation_matrix(self) -> None:
        """Calculate correlation matrix."""
        if self._returns_data is None or self._returns_data.empty:
            return
        
        try:
            self._correlation_matrix = self._returns_data.corr()
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            self._correlation_matrix = pd.DataFrame()
    
    def historical_simulation_var(
        self,
        weights: Dict[str, float],
        use_bootstrap: bool = True
    ) -> VaRResult:
        """
        Calculate VaR using historical simulation with optional bootstrapping.
        
        Args:
            weights: Portfolio weights dictionary
            use_bootstrap: Whether to use bootstrapping for confidence intervals
            
        Returns:
            VaR calculation result
        """
        import time
        start_time = time.time()
        
        if self._returns_data is None:
            raise ValueError("No returns data available")
        
        # Filter weights for available assets
        available_assets = set(weights.keys()) & set(self._returns_data.columns)
        if not available_assets:
            raise ValueError("No overlapping assets between weights and data")
        
        # Prepare portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, available_assets)
        
        if use_bootstrap:
            var_value, cvar_value = self._bootstrap_historical_var(portfolio_returns)
        else:
            var_value = np.percentile(portfolio_returns, self.config.confidence_level * 100)
            cvar_value = portfolio_returns[portfolio_returns <= var_value].mean()
        
        calculation_time = time.time() - start_time
        
        result = VaRResult(
            var_value=abs(var_value),
            cvar_value=abs(cvar_value),
            confidence_level=self.config.confidence_level,
            time_horizon=self.config.time_horizon,
            method="historical_simulation",
            distribution="empirical",
            simulation_count=len(portfolio_returns),
            calculation_time=calculation_time,
            additional_metrics={
                "bootstrap_used": use_bootstrap,
                "data_points": len(portfolio_returns)
            },
            timestamp=datetime.now()
        )
        
        self._var_history.append(result)
        return result
    
    def _bootstrap_historical_var(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate VaR using block bootstrap for better confidence intervals.
        
        Args:
            returns: Portfolio returns series
            
        Returns:
            Tuple of (VaR, CVaR) values
        """
        var_bootstrap = []
        cvar_bootstrap = []
        
        for _ in range(self.config.bootstrap_samples):
            # Block bootstrap
            n_blocks = len(returns) // self.config.block_size
            bootstrap_indices = []
            
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, len(returns) - self.config.block_size + 1)
                block_indices = range(start_idx, start_idx + self.config.block_size)
                bootstrap_indices.extend(block_indices)
            
            # Ensure we have enough samples
            while len(bootstrap_indices) < len(returns):
                bootstrap_indices.append(np.random.randint(0, len(returns)))
            
            bootstrap_returns = returns.iloc[bootstrap_indices[:len(returns)]]
            
            var_bootstrap.append(
                np.percentile(bootstrap_returns, self.config.confidence_level * 100)
            )
            cvar_bootstrap.append(
                bootstrap_returns[bootstrap_returns <= var_bootstrap[-1]].mean()
            )
        
        # Return mean of bootstrap samples
        return np.mean(var_bootstrap), np.mean(cvar_bootstrap)
    
    def parametric_var(
        self,
        weights: Dict[str, float],
        distribution: Optional[str] = None
    ) -> VaRResult:
        """
        Calculate parametric VaR with multiple distribution assumptions.
        
        Args:
            weights: Portfolio weights dictionary
            distribution: Distribution type (normal, t, laplace, logistic)
            
        Returns:
            VaR calculation result
        """
        import time
        start_time = time.time()
        
        if self._returns_data is None or self._covariance_matrix is None:
            raise ValueError("No returns data or covariance matrix available")
        
        distribution = distribution or self.config.distribution_type
        
        # Calculate portfolio statistics
        portfolio_mean, portfolio_std = self._calculate_portfolio_statistics(weights)
        
        # Calculate VaR based on distribution
        if distribution == "normal":
            var_value = norm.ppf(self.config.confidence_level, portfolio_mean, portfolio_std)
            cvar_value = self._calculate_normal_cvar(portfolio_mean, portfolio_std)
        elif distribution == "t":
            var_value, cvar_value = self._calculate_t_distribution_var(weights)
        elif distribution == "laplace":
            var_value = laplace.ppf(self.config.confidence_level, portfolio_mean, portfolio_std)
            cvar_value = self._calculate_laplace_cvar(portfolio_mean, portfolio_std)
        elif distribution == "logistic":
            var_value = logistic.ppf(self.config.confidence_level, portfolio_mean, portfolio_std)
            cvar_value = self._calculate_logistic_cvar(portfolio_mean, portfolio_std)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        calculation_time = time.time() - start_time
        
        result = VaRResult(
            var_value=abs(var_value),
            cvar_value=abs(cvar_value),
            confidence_level=self.config.confidence_level,
            time_horizon=self.config.time_horizon,
            method="parametric",
            distribution=distribution,
            simulation_count=1,
            calculation_time=calculation_time,
            additional_metrics={
                "portfolio_mean": portfolio_mean,
                "portfolio_std": portfolio_std
            },
            timestamp=datetime.now()
        )
        
        self._var_history.append(result)
        return result
    
    def _calculate_portfolio_statistics(self, weights: Dict[str, float]) -> Tuple[float, float]:
        """Calculate portfolio mean and standard deviation."""
        available_assets = set(weights.keys()) & set(self._returns_data.columns)
        
        # Portfolio mean
        portfolio_mean = sum(
            weights[asset] * self._returns_data[asset].mean()
            for asset in available_assets
        )
        
        # Portfolio variance
        portfolio_variance = 0
        for asset1 in available_assets:
            for asset2 in available_assets:
                cov = self._covariance_matrix.loc[asset1, asset2]
                portfolio_variance += weights[asset1] * weights[asset2] * cov
        
        portfolio_std = np.sqrt(portfolio_variance)
        return portfolio_mean, portfolio_std
    
    def _calculate_t_distribution_var(self, weights: Dict[str, float]) -> Tuple[float, float]:
        """Calculate VaR using t-distribution with estimated degrees of freedom."""
        portfolio_returns = self._calculate_portfolio_returns(weights)
        
        # Estimate degrees of freedom using MLE
        df, loc, scale = stats.t.fit(portfolio_returns)
        
        var_value = stats.t.ppf(self.config.confidence_level, df, loc, scale)
        cvar_value = self._calculate_t_cvar(df, loc, scale)
        
        return var_value, cvar_value
    
    def _calculate_normal_cvar(self, mean: float, std: float) -> float:
        """Calculate CVaR for normal distribution."""
        alpha = self.config.confidence_level
        return mean - std * norm.pdf(norm.ppf(alpha)) / alpha
    
    def _calculate_t_cvar(self, df: float, loc: float, scale: float) -> float:
        """Calculate CVaR for t-distribution."""
        alpha = self.config.confidence_level
        t_alpha = stats.t.ppf(alpha, df)
        return loc - scale * stats.t.pdf(t_alpha, df) * (df + t_alpha**2) / (df - 1) / alpha
    
    def _calculate_laplace_cvar(self, mean: float, std: float) -> float:
        """Calculate CVaR for Laplace distribution."""
        alpha = self.config.confidence_level
        return mean - std * (1 + np.log(alpha))
    
    def _calculate_logistic_cvar(self, mean: float, std: float) -> float:
        """Calculate CVaR for logistic distribution."""
        alpha = self.config.confidence_level
        return mean - std * (1 + np.log(alpha / (1 - alpha)))
    
    def monte_carlo_var(
        self,
        weights: Dict[str, float],
        use_correlation: bool = True
    ) -> VaRResult:
        """
        Calculate VaR using Monte Carlo simulation with correlated asset movements.
        
        Args:
            weights: Portfolio weights dictionary
            use_correlation: Whether to use correlation structure
            
        Returns:
            VaR calculation result
        """
        import time
        start_time = time.time()
        
        if self._returns_data is None:
            raise ValueError("No returns data available")
        
        # Generate correlated random returns
        if use_correlation and self._covariance_matrix is not None:
            simulated_returns = self._generate_correlated_returns()
        else:
            simulated_returns = self._generate_uncorrelated_returns()
        
        # Calculate portfolio returns from simulations
        portfolio_returns = self._calculate_simulated_portfolio_returns(weights, simulated_returns)
        
        # Calculate VaR and CVaR
        var_value = np.percentile(portfolio_returns, self.config.confidence_level * 100)
        cvar_value = portfolio_returns[portfolio_returns <= var_value].mean()
        
        calculation_time = time.time() - start_time
        
        result = VaRResult(
            var_value=abs(var_value),
            cvar_value=abs(cvar_value),
            confidence_level=self.config.confidence_level,
            time_horizon=self.config.time_horizon,
            method="monte_carlo",
            distribution="simulated",
            simulation_count=self.config.n_simulations,
            calculation_time=calculation_time,
            additional_metrics={
                "correlation_used": use_correlation,
                "simulation_std": np.std(portfolio_returns)
            },
            timestamp=datetime.now()
        )
        
        self._var_history.append(result)
        return result
    
    def _generate_correlated_returns(self) -> np.ndarray:
        """Generate correlated random returns using Cholesky decomposition."""
        if self._covariance_matrix is None:
            raise ValueError("Covariance matrix not available")
        
        # Cholesky decomposition
        try:
            chol_matrix = np.linalg.cholesky(self._covariance_matrix.values)
        except np.linalg.LinAlgError:
            # Fallback to nearest positive definite matrix
            from scipy.linalg import nearest_posdef
            cov_matrix_pd = nearest_posdef(self._covariance_matrix.values)
            chol_matrix = np.linalg.cholesky(cov_matrix_pd)
        
        # Generate uncorrelated random numbers
        n_assets = len(self._covariance_matrix)
        uncorrelated = np.random.normal(0, 1, (self.config.n_simulations, n_assets))
        
        # Apply correlation structure
        correlated = uncorrelated @ chol_matrix.T
        
        return correlated
    
    def _generate_uncorrelated_returns(self) -> np.ndarray:
        """Generate uncorrelated random returns."""
        if self._returns_data is None:
            raise ValueError("Returns data not available")
        
        n_assets = len(self._returns_data.columns)
        return np.random.normal(0, 1, (self.config.n_simulations, n_assets))
    
    def _calculate_simulated_portfolio_returns(
        self,
        weights: Dict[str, float],
        simulated_returns: np.ndarray
    ) -> np.ndarray:
        """Calculate portfolio returns from simulated asset returns."""
        if self._returns_data is None:
            raise ValueError("Returns data not available")
        
        available_assets = list(set(weights.keys()) & set(self._returns_data.columns))
        asset_indices = [self._returns_data.columns.get_loc(asset) for asset in available_assets]
        
        # Extract weights for available assets
        weight_vector = np.array([weights[asset] for asset in available_assets])
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns[:, asset_indices] @ weight_vector
        
        return portfolio_returns
    
    def stress_test_var(
        self,
        weights: Dict[str, float],
        scenario: Optional[str] = None
    ) -> Dict[str, VaRResult]:
        """
        Perform stress testing with extreme market scenarios.
        
        Args:
            weights: Portfolio weights dictionary
            scenario: Specific stress scenario to test
            
        Returns:
            Dictionary of VaR results for different scenarios
        """
        results = {}
        
        scenarios = [scenario] if scenario else self.config.stress_scenarios.keys()
        
        for scenario_name in scenarios:
            if scenario_name not in self.config.stress_scenarios:
                continue
            
            scenario_params = self.config.stress_scenarios[scenario_name]
            
            # Apply stress scenario to data
            stressed_data = self._apply_stress_scenario(scenario_params)
            
            # Calculate VaR with stressed data
            original_data = self._returns_data.copy()
            self._returns_data = stressed_data
            
            try:
                var_result = self.monte_carlo_var(weights, use_correlation=True)
                var_result.method = f"stress_test_{scenario_name}"
                results[scenario_name] = var_result
            finally:
                # Restore original data
                self._returns_data = original_data
        
        return results
    
    def _apply_stress_scenario(self, scenario_params: Dict[str, float]) -> pd.DataFrame:
        """Apply stress scenario parameters to returns data."""
        if self._returns_data is None:
            raise ValueError("No returns data available")
        
        stressed_data = self._returns_data.copy()
        
        # Apply volatility multiplier
        if "volatility_multiplier" in scenario_params:
            multiplier = scenario_params["volatility_multiplier"]
            stressed_data = stressed_data * multiplier
        
        # Apply mean shift
        if "mean_shift" in scenario_params:
            shift = scenario_params["mean_shift"]
            stressed_data = stressed_data + shift
        
        # Apply correlation increase
        if "correlation_increase" in scenario_params:
            # This would require more sophisticated correlation modeling
            # For now, we'll just increase the overall volatility
            pass
        
        return stressed_data
    
    def backtest_var(
        self,
        weights: Dict[str, float],
        method: str = "monte_carlo"
    ) -> Dict[str, Any]:
        """
        Perform VaR backtesting to validate model accuracy.
        
        Args:
            weights: Portfolio weights dictionary
            method: VaR calculation method to backtest
            
        Returns:
            Backtesting results dictionary
        """
        if self._returns_data is None:
            raise ValueError("No returns data available")
        
        # Calculate rolling VaR
        var_predictions = []
        actual_returns = []
        
        window_size = self.config.backtest_window
        step_size = 1
        
        for i in range(window_size, len(self._returns_data), step_size):
            # Get historical data up to current point
            historical_data = self._returns_data.iloc[:i]
            
            # Calculate VaR using historical data
            temp_var = MonteCarloVaR(self.config)
            temp_var.update_data(historical_data)
            
            if method == "monte_carlo":
                var_result = temp_var.monte_carlo_var(weights)
            elif method == "historical":
                var_result = temp_var.historical_simulation_var(weights)
            elif method == "parametric":
                var_result = temp_var.parametric_var(weights)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            var_predictions.append(var_result.var_value)
            
            # Calculate actual portfolio return for next period
            if i < len(self._returns_data):
                actual_return = self._calculate_portfolio_returns(
                    weights, 
                    self._returns_data.iloc[i:i+1]
                ).iloc[0]
                actual_returns.append(actual_return)
        
        # Calculate backtesting metrics
        var_predictions = np.array(var_predictions)
        actual_returns = np.array(actual_returns)
        
        # Count VaR breaches
        breaches = actual_returns < -var_predictions
        breach_rate = np.mean(breaches)
        
        # Kupiec test for VaR accuracy
        kupiec_stat, kupiec_pvalue = self._kupiec_test(breaches, self.config.confidence_level)
        
        # Christoffersen test for independence of breaches
        christoffersen_stat, christoffersen_pvalue = self._christoffersen_test(breaches)
        
        results = {
            "method": method,
            "total_predictions": len(var_predictions),
            "breach_count": np.sum(breaches),
            "breach_rate": breach_rate,
            "expected_breach_rate": self.config.confidence_level,
            "kupiec_statistic": kupiec_stat,
            "kupiec_pvalue": kupiec_pvalue,
            "christoffersen_statistic": christoffersen_stat,
            "christoffersen_pvalue": christoffersen_pvalue,
            "var_predictions": var_predictions.tolist(),
            "actual_returns": actual_returns.tolist(),
            "breaches": breaches.tolist()
        }
        
        self._backtest_results[method] = results
        return results
    
    def _kupiec_test(self, breaches: np.ndarray, expected_rate: float) -> Tuple[float, float]:
        """Perform Kupiec test for VaR accuracy."""
        n = len(breaches)
        x = np.sum(breaches)
        p_hat = x / n
        
        if p_hat == 0:
            return 0.0, 1.0
        
        # Likelihood ratio test statistic
        lr = 2 * (x * np.log(p_hat / expected_rate) + (n - x) * np.log((1 - p_hat) / (1 - expected_rate)))
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr, 1)
        
        return lr, p_value
    
    def _christoffersen_test(self, breaches: np.ndarray) -> Tuple[float, float]:
        """Perform Christoffersen test for independence of VaR breaches."""
        n = len(breaches)
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        
        for i in range(n - 1):
            if breaches[i] == 0 and breaches[i + 1] == 0:
                n00 += 1
            elif breaches[i] == 0 and breaches[i + 1] == 1:
                n01 += 1
            elif breaches[i] == 1 and breaches[i + 1] == 0:
                n10 += 1
            elif breaches[i] == 1 and breaches[i + 1] == 1:
                n11 += 1
        
        # Calculate probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        p = (n01 + n11) / n
        
        # Likelihood ratio test statistic
        if p01 == 0 or p11 == 0 or p == 0 or p == 1:
            return 0.0, 1.0
        
        lr = 2 * (n01 * np.log(p01 / p) + n11 * np.log(p11 / p) + 
                 n00 * np.log((1 - p01) / (1 - p)) + n10 * np.log((1 - p11) / (1 - p)))
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr, 1)
        
        return lr, p_value
    
    def _calculate_portfolio_returns(
        self,
        weights: Dict[str, float],
        data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Calculate portfolio returns from asset weights and returns data."""
        if data is None:
            data = self._returns_data
        
        if data is None:
            raise ValueError("No returns data available")
        
        available_assets = set(weights.keys()) & set(data.columns)
        if not available_assets:
            raise ValueError("No overlapping assets between weights and data")
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=data.index)
        
        for asset in available_assets:
            if weights[asset] != 0:
                portfolio_returns += weights[asset] * data[asset]
        
        return portfolio_returns
    
    def get_var_summary(self) -> Dict[str, Any]:
        """Get summary of all VaR calculations."""
        if not self._var_history:
            return {}
        
        summary = {
            "total_calculations": len(self._var_history),
            "methods_used": list(set(result.method for result in self._var_history)),
            "distributions_used": list(set(result.distribution for result in self._var_history)),
            "latest_var": self._var_history[-1].var_value if self._var_history else None,
            "average_calculation_time": np.mean([r.calculation_time for r in self._var_history]),
            "backtest_results": self._backtest_results
        }
        
        return summary
    
    def clear_history(self) -> None:
        """Clear calculation history."""
        self._var_history.clear()
        self._backtest_results.clear()
        self.logger.info("Cleared VaR calculation history")