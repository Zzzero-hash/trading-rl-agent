"""
Statistical Tests for Trading Model Evaluation

This module provides comprehensive statistical testing capabilities for
evaluating trading model performance and data characteristics.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


class StatisticalTests:
    """
    Comprehensive statistical testing for trading model evaluation.

    Provides tests for:
    - Normality testing
    - Stationarity testing
    - Autocorrelation testing
    - Volatility clustering
    - Structural breaks
    - Model residuals analysis
    """

    def __init__(self) -> None:
        """Initialize the statistical tests."""

    def test_normality(self, data: np.ndarray, test_type: str = "shapiro") -> dict[str, float]:
        """
        Test for normality of data.

        Args:
            data: Data to test
            test_type: Type of normality test ("shapiro", "jarque_bera", "anderson")

        Returns:
            Dictionary with test results
        """
        if len(data) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "is_normal": False}

        if test_type == "shapiro":
            statistic, p_value = stats.shapiro(data)
        elif test_type == "jarque_bera":
            statistic, p_value = stats.jarque_bera(data)
        elif test_type == "anderson":
            result = stats.anderson(data)
            statistic = result.statistic
            p_value = result.significance_level[2]  # 5% significance level
        else:
            raise ValueError(f"Unsupported normality test: {test_type}")

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05,
        }

    def test_stationarity(
        self,
        data: np.ndarray,
        test_type: str = "adf",
        significance_level: float = 0.05,
    ) -> dict[str, float]:
        """
        Test for stationarity of time series data.

        Args:
            data: Time series data to test
            test_type: Type of stationarity test ("adf", "kpss", "pp")
            significance_level: Significance level for the test

        Returns:
            Dictionary with test results
        """
        if len(data) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "is_stationary": False}

        if test_type == "adf":
            # Augmented Dickey-Fuller test
            result = adfuller(data, autolag="AIC")
            statistic = result[0]
            p_value = result[1]
            critical_values = result[4]

            is_stationary = p_value < significance_level

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_stationary": is_stationary,
                "critical_values": critical_values,
            }

        if test_type == "kpss":
            # KPSS test
            result = kpss(data, regression="c")
            statistic = result[0]
            p_value = result[1]
            critical_values = result[3]

            # KPSS null hypothesis is stationarity, so we invert the result
            is_stationary = p_value > significance_level

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_stationary": is_stationary,
                "critical_values": critical_values,
            }

        raise ValueError(f"Unsupported stationarity test: {test_type}")

    def test_autocorrelation(
        self,
        data: np.ndarray,
        lags: int = 10,
        test_type: str = "ljung_box",
    ) -> dict[str, float]:
        """
        Test for autocorrelation in time series data.

        Args:
            data: Time series data to test
            lags: Number of lags to test
            test_type: Type of autocorrelation test ("ljung_box", "durbin_watson")

        Returns:
            Dictionary with test results
        """
        if len(data) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "has_autocorrelation": False}

        if test_type == "ljung_box":
            # Ljung-Box test
            result = acorr_ljungbox(data, lags=lags, return_df=True)
            statistic = result["lb_stat"].iloc[-1]
            p_value = result["lb_pvalue"].iloc[-1]

            has_autocorrelation = p_value < 0.05

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "has_autocorrelation": has_autocorrelation,
                "lags": lags,
            }

        if test_type == "durbin_watson":
            # Durbin-Watson test
            statistic = self._calculate_durbin_watson(data)

            # Durbin-Watson critical values (approximate)
            n = len(data)
            if n > 100:
                # For large samples, DW ≈ 2(1-r) where r is the first-order autocorrelation
                r = np.corrcoef(data[:-1], data[1:])[0, 1]
                dw_approx = 2 * (1 - r)
                statistic = dw_approx

            # Approximate p-value based on DW statistic
            if statistic < 1.5:
                p_value = 0.01  # Strong positive autocorrelation
            elif statistic > 2.5:
                p_value = 0.01  # Strong negative autocorrelation
            else:
                p_value = 0.5  # No significant autocorrelation

            has_autocorrelation = p_value < 0.05

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "has_autocorrelation": has_autocorrelation,
            }

        raise ValueError(f"Unsupported autocorrelation test: {test_type}")

    def test_volatility_clustering(
        self,
        returns: np.ndarray,
        test_type: str = "arch",
        lags: int = 5,
    ) -> dict[str, float]:
        """
        Test for volatility clustering (ARCH effects).

        Args:
            returns: Return series to test
            test_type: Type of volatility clustering test ("arch", "garch")
            lags: Number of lags to test

        Returns:
            Dictionary with test results
        """
        if len(returns) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "has_clustering": False}

        if test_type == "arch":
            # Engle's ARCH test
            squared_returns = returns**2

            # Create lagged variables
            X = np.ones((len(returns) - lags, lags + 1))
            for i in range(lags):
                X[:, i + 1] = squared_returns[i : -(lags - i)]

            y = squared_returns[lags:]

            # Perform regression
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta

                # Calculate test statistic
                r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2)
                statistic = len(y) * r_squared

                # Chi-square test with lags degrees of freedom
                p_value = 1 - stats.chi2.cdf(statistic, lags)

                has_clustering = p_value < 0.05

                return {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "has_clustering": has_clustering,
                    "lags": lags,
                }

            except np.linalg.LinAlgError:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "has_clustering": False,
                    "lags": lags,
                }

        else:
            raise ValueError(f"Unsupported volatility clustering test: {test_type}")

    def test_structural_breaks(
        self,
        data: np.ndarray,
        test_type: str = "chow",
        break_point: int | None = None,
    ) -> dict[str, float]:
        """
        Test for structural breaks in time series data.

        Args:
            data: Time series data to test
            test_type: Type of structural break test ("chow", "cusum")
            break_point: Point at which to test for break (if None, will be estimated)

        Returns:
            Dictionary with test results
        """
        if len(data) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "has_break": False}

        if test_type == "chow":
            # Chow test for structural break
            if break_point is None:
                break_point = len(data) // 2

            if break_point < 10 or break_point > len(data) - 10:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "has_break": False,
                    "break_point": break_point,
                }

            # Split data
            data_1 = data[:break_point]
            data_2 = data[break_point:]

            # Calculate statistics
            n1, n2 = len(data_1), len(data_2)
            n = n1 + n2

            # Calculate RSS for full sample
            mean_full = np.mean(data)
            rss_full = np.sum((data - mean_full) ** 2)

            # Calculate RSS for subsamples
            mean_1 = np.mean(data_1)
            mean_2 = np.mean(data_2)
            rss_1 = np.sum((data_1 - mean_1) ** 2)
            rss_2 = np.sum((data_2 - mean_2) ** 2)
            rss_restricted = rss_1 + rss_2

            # Calculate Chow statistic
            k = 1  # Number of parameters (mean)
            statistic = ((rss_restricted - rss_full) / k) / (rss_full / (n - 2 * k))

            # F-test with k and n-2k degrees of freedom
            p_value = 1 - stats.f.cdf(statistic, k, n - 2 * k)

            has_break = p_value < 0.05

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "has_break": has_break,
                "break_point": break_point,
            }

        if test_type == "cusum":
            # CUSUM test for structural breaks
            cumulative_sum = np.cumsum(data - np.mean(data))
            max_cusum = np.max(np.abs(cumulative_sum))

            # Critical value approximation
            critical_value = 1.36 * np.sqrt(len(data))

            has_break = max_cusum > critical_value
            p_value = 0.05 if has_break else 0.5  # Approximate

            return {
                "statistic": float(max_cusum),
                "p_value": float(p_value),
                "has_break": has_break,
                "critical_value": float(critical_value),
            }

        raise ValueError(f"Unsupported structural break test: {test_type}")

    def test_model_residuals(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Comprehensive test of model residuals.

        Args:
            residuals: Model residuals
            fitted_values: Fitted values from the model

        Returns:
            Dictionary with all residual test results
        """
        if len(residuals) == 0:
            return {
                "normality": {"statistic": 0.0, "p_value": 1.0, "is_normal": False},
                "autocorrelation": {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "has_autocorrelation": False,
                },
                "heteroscedasticity": {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "is_heteroscedastic": False,
                },
            }

        results = {}

        # Test for normality
        results["normality"] = self.test_normality(residuals)

        # Test for autocorrelation
        results["autocorrelation"] = self.test_autocorrelation(residuals)

        # Test for heteroscedasticity
        results["heteroscedasticity"] = self.test_heteroscedasticity(residuals, fitted_values)

        return results

    def test_heteroscedasticity(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray | None = None,
        test_type: str = "breusch_pagan",
    ) -> dict[str, float]:
        """
        Test for heteroscedasticity in residuals.

        Args:
            residuals: Model residuals
            fitted_values: Fitted values from the model
            test_type: Type of heteroscedasticity test ("breusch_pagan", "white")

        Returns:
            Dictionary with test results
        """
        if len(residuals) == 0:
            return {"statistic": 0.0, "p_value": 1.0, "is_heteroscedastic": False}

        if test_type == "breusch_pagan":
            # Breusch-Pagan test
            if fitted_values is None:
                fitted_values = np.arange(len(residuals))

            # Square the residuals
            squared_residuals = residuals**2

            # Regress squared residuals on fitted values
            X = np.column_stack([np.ones(len(fitted_values)), fitted_values])

            try:
                beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
                predicted_squared = X @ beta

                # Calculate test statistic
                r_squared = 1 - np.sum((squared_residuals - predicted_squared) ** 2) / np.sum(
                    (squared_residuals - np.mean(squared_residuals)) ** 2,
                )
                statistic = len(residuals) * r_squared

                # Chi-square test with 1 degree of freedom
                p_value = 1 - stats.chi2.cdf(statistic, 1)

                is_heteroscedastic = p_value < 0.05

                return {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_heteroscedastic": is_heteroscedastic,
                }

            except np.linalg.LinAlgError:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "is_heteroscedastic": False,
                }

        else:
            raise ValueError(f"Unsupported heteroscedasticity test: {test_type}")

    def _calculate_durbin_watson(self, data: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic."""
        if len(data) < 2:
            return 2.0

        residuals = data - np.mean(data)
        diff_residuals = np.diff(residuals)

        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
        return float(dw_stat)

    def run_comprehensive_tests(self, data: np.ndarray) -> dict[str, dict[str, float]]:
        """
        Run a comprehensive battery of statistical tests.

        Args:
            data: Data to test

        Returns:
            Dictionary with all test results
        """
        results = {}

        # Basic distribution tests
        results["normality"] = self.test_normality(data)

        # Time series tests
        results["stationarity"] = self.test_stationarity(data)
        results["autocorrelation"] = self.test_autocorrelation(data)

        # Volatility tests (if data looks like returns)
        if np.std(data) > 0:
            results["volatility_clustering"] = self.test_volatility_clustering(data)

        # Structural break tests
        results["structural_breaks"] = self.test_structural_breaks(data)

        return results
