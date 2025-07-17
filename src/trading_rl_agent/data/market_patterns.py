"""
Advanced synthetic market data generator with realistic patterns.

This module provides sophisticated synthetic data generation capabilities including:
- Trend patterns (uptrends, downtrends, sideways markets)
- Reversal patterns (head and shoulders, double tops/bottoms)
- Volatility clustering and regime changes
- Market microstructure effects
- Multi-asset correlated scenarios
"""

from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CALM = "calm"
    REVERSAL = "reversal"


class PatternType(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"


class MarketPatternGenerator:
    """
    Advanced synthetic market data generator with realistic patterns.
    """

    def __init__(self, base_price: float = 100.0, base_volatility: float = 0.02, seed: int | None = None):
        self.base_price = base_price
        self.base_volatility = base_volatility
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        if not STATSMODELS_AVAILABLE:
            print("Warning: statsmodels/arch not available. Using simplified models.")

    def generate_trend_pattern(
        self,
        n_periods: int,
        trend_type: str = "uptrend",
        trend_strength: float = 0.001,
        volatility: float = 0.02,
        regime_changes: list[dict] | None = None,
    ) -> pd.DataFrame:
        """Generate trend patterns (uptrend, downtrend, sideways)."""
        if trend_type == "uptrend":
            drift = trend_strength
        elif trend_type == "downtrend":
            drift = -trend_strength
        else:
            drift = 0.0
        returns = np.random.normal(drift, volatility, n_periods)
        if regime_changes:
            returns = self._apply_regime_changes(returns, regime_changes)
        prices = self.base_price * np.exp(np.cumsum(returns))
        df = self._generate_ohlcv_from_prices(prices, volatility)
        df["pattern_type"] = trend_type
        df["trend_strength"] = trend_strength
        return df

    def generate_reversal_pattern(
        self, pattern_type: str, n_periods: int = 252, pattern_intensity: float = 0.5, base_volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate reversal patterns (head and shoulders, double tops/bottoms)."""
        if pattern_type == PatternType.HEAD_AND_SHOULDERS.value:
            return self._generate_head_and_shoulders(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS.value:
            df = self._generate_head_and_shoulders(n_periods, pattern_intensity, base_volatility)
            for col in ["open", "high", "low", "close"]:
                df[col] = 2 * self.base_price - df[col]
            return df
        if pattern_type == PatternType.DOUBLE_TOP.value:
            return self._generate_double_top(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.DOUBLE_BOTTOM.value:
            df = self._generate_double_top(n_periods, pattern_intensity, base_volatility)
            for col in ["open", "high", "low", "close"]:
                df[col] = 2 * self.base_price - df[col]
            return df
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    def generate_volatility_clustering(
        self, n_periods: int, volatility_regimes: list[dict], regime_durations: list[int], base_price: float = 100.0
    ) -> pd.DataFrame:
        """Simulate volatility clustering and regime changes using GARCH or simplified models."""
        returns = []
        current_period = 0
        for regime, duration in zip(volatility_regimes, regime_durations):
            if current_period >= n_periods:
                break
            if STATSMODELS_AVAILABLE:
                omega = regime.get("omega", 0.0001)
                alpha = regime.get("alpha", 0.1)
                beta = regime.get("beta", 0.8)
                mu = regime.get("mu", 0.0)
                model = arch_model(
                    np.zeros(min(duration, n_periods - current_period)),
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    q=1,
                    dist="normal",
                )
                sim = model.simulate([mu, omega, alpha, beta], min(duration, n_periods - current_period))
                regime_returns = sim["data"]
            else:
                regime_vol = regime.get("volatility", 0.02)
                regime_drift = regime.get("drift", 0.0)
                regime_returns = np.random.normal(regime_drift, regime_vol, min(duration, n_periods - current_period))
            returns.extend(regime_returns)
            current_period += duration
        prices = base_price * np.exp(np.cumsum(returns[:n_periods]))
        df = self._generate_ohlcv_from_prices(prices, self.base_volatility)
        df["volatility_regime"] = self._assign_volatility_regimes(n_periods, volatility_regimes, regime_durations)
        return df

    def generate_microstructure_effects(
        self,
        base_data: pd.DataFrame,
        bid_ask_spread: float = 0.001,
        order_book_depth: int = 10,
        tick_size: float = 0.01,
        market_impact: float = 0.0001,
    ) -> pd.DataFrame:
        """Add market microstructure effects (bid-ask spread, order book, tick size, market impact)."""
        df = base_data.copy()
        df["bid"] = df["close"] * (1 - bid_ask_spread / 2)
        df["ask"] = df["close"] * (1 + bid_ask_spread / 2)
        for col in ["open", "high", "low", "close"]:
            df[col] = np.round(df[col] / tick_size) * tick_size
        volume_impact = df["volume"] * market_impact
        df["close"] += volume_impact * np.random.normal(0, 1, len(df))
        df["order_book_depth"] = order_book_depth
        df["spread"] = df["ask"] - df["bid"]
        return df

    def generate_correlated_assets(
        self,
        n_assets: int,
        n_periods: int,
        correlation_matrix: np.ndarray | None = None,
        base_prices: list[float] | None = None,
        volatilities: list[float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate correlated multi-asset scenarios."""
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_assets)
        if base_prices is None:
            base_prices = [self.base_price] * n_assets
        if volatilities is None:
            volatilities = [self.base_volatility] * n_assets
        uncorrelated_returns = np.random.normal(0, 1, (n_periods, n_assets))
        cholesky = np.linalg.cholesky(correlation_matrix)
        correlated_returns = uncorrelated_returns @ cholesky.T
        for i in range(n_assets):
            correlated_returns[:, i] *= volatilities[i]
        assets_data = {}
        for i in range(n_assets):
            prices = base_prices[i] * np.exp(np.cumsum(correlated_returns[:, i]))
            df = self._generate_ohlcv_from_prices(prices, volatilities[i])
            df["symbol"] = f"ASSET_{i}"
            assets_data[f"ASSET_{i}"] = df
        return assets_data

    def detect_market_regime(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect market regime from price data using rolling statistics."""
        df = data.copy()
        df["returns"] = df["close"].pct_change()
        df["rolling_mean"] = df["returns"].rolling(window).mean()
        df["rolling_std"] = df["returns"].rolling(window).std()
        df["rolling_skew"] = df["returns"].rolling(window).skew()
        df["regime"] = self._classify_regime(df["rolling_mean"], df["rolling_std"], df["rolling_skew"])
        return df

    def validate_pattern_quality(self, data: pd.DataFrame, pattern_type: str) -> dict:
        """Validate the quality of generated patterns."""
        validation_results = {
            "pattern_type": pattern_type,
            "data_quality": self._check_data_quality(data),
            "pattern_quality": {},
            "statistical_tests": {},
        }
        if pattern_type in [PatternType.UPTREND.value, PatternType.DOWNTREND.value, PatternType.SIDEWAYS.value]:
            validation_results["pattern_quality"] = self._validate_trend_pattern(data, pattern_type)
        elif "head_and_shoulders" in pattern_type or "double" in pattern_type:
            validation_results["pattern_quality"] = self._validate_reversal_pattern(data, pattern_type)
        validation_results["statistical_tests"] = self._run_statistical_tests(data)
        return validation_results

    # --- Internal helpers ---

    def _generate_ohlcv_from_prices(self, prices: np.ndarray, volatility: float) -> pd.DataFrame:
        n_periods = len(prices)
        start_date = datetime.now() - timedelta(days=n_periods)
        timestamps = [start_date + timedelta(days=i) for i in range(n_periods)]
        opens = prices * np.exp(np.random.normal(0, volatility * 0.5, n_periods))
        highs = np.maximum(opens, prices) * np.exp(np.abs(np.random.normal(0, volatility * 0.3, n_periods)))
        lows = np.minimum(opens, prices) * np.exp(-np.abs(np.random.normal(0, volatility * 0.3, n_periods)))
        price_moves = np.abs(prices - opens) / prices
        volumes = np.random.normal(10000, 2000, n_periods) * (1 + 5 * price_moves)
        volumes = np.maximum(volumes, 1000).astype(int)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )

    def _generate_head_and_shoulders(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        left_shoulder = int(n_periods * 0.2)
        head = int(n_periods * 0.5)
        right_shoulder = int(n_periods * 0.8)
        trend = np.linspace(0, -0.1 * intensity, n_periods)
        pattern = np.zeros(n_periods)
        pattern[:left_shoulder] = np.sin(np.linspace(0, np.pi, left_shoulder)) * 0.05 * intensity
        pattern[left_shoulder:head] = np.sin(np.linspace(0, np.pi, head - left_shoulder)) * 0.08 * intensity
        pattern[head:right_shoulder] = np.sin(np.linspace(0, np.pi, right_shoulder - head)) * 0.05 * intensity
        prices = self.base_price * np.exp(trend + pattern)
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)
        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_double_top(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        peak1 = int(n_periods * 0.3)
        peak2 = int(n_periods * 0.7)
        pattern = np.zeros(n_periods)
        pattern[:peak1] = np.sin(np.linspace(0, np.pi, peak1)) * 0.06 * intensity
        pattern[peak1:peak2] = np.sin(np.linspace(0, np.pi, peak2 - peak1)) * 0.06 * intensity
        prices = self.base_price * np.exp(pattern)
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)
        return self._generate_ohlcv_from_prices(prices, volatility)

    def _apply_regime_changes(self, returns: np.ndarray, regime_changes: list[dict]) -> np.ndarray:
        modified_returns = returns.copy()
        for change in regime_changes:
            period = change["period"]
            if period < len(returns):
                new_volatility = change.get("new_volatility", 1.0)
                new_drift = change.get("new_drift", 0.0)
                modified_returns[period:] = modified_returns[period:] * new_volatility + new_drift
        return modified_returns

    def _assign_volatility_regimes(
        self, n_periods: int, volatility_regimes: list[dict], regime_durations: list[int]
    ) -> list[str]:
        regimes = []
        current_period = 0
        for i, (regime, duration) in enumerate(zip(volatility_regimes, regime_durations)):
            if current_period >= n_periods:
                break
            regime_label = regime.get("label", f"regime_{i}")
            periods_in_regime = min(duration, n_periods - current_period)
            regimes.extend([regime_label] * periods_in_regime)
            current_period += duration
        while len(regimes) < n_periods:
            regimes.append("default")
        return regimes[:n_periods]

    def _classify_regime(self, rolling_mean: pd.Series, rolling_std: pd.Series, rolling_skew: pd.Series) -> pd.Series:
        regime = pd.Series(index=rolling_mean.index, dtype="object")
        mean_threshold = 0.001
        std_threshold = 0.02
        skew_threshold = 0.5
        for i in rolling_mean.index:
            if pd.isna(rolling_mean[i]):
                regime[i] = MarketRegime.SIDEWAYS.value
                continue
            mean_val = rolling_mean[i]
            std_val = rolling_std[i]
            skew_val = rolling_skew[i]
            if std_val > std_threshold:
                regime[i] = MarketRegime.VOLATILE.value
            elif mean_val > mean_threshold:
                regime[i] = MarketRegime.TRENDING_UP.value
            elif mean_val < -mean_threshold:
                regime[i] = MarketRegime.TRENDING_DOWN.value
            elif abs(skew_val) > skew_threshold:
                regime[i] = MarketRegime.REVERSAL.value
            else:
                regime[i] = MarketRegime.SIDEWAYS.value
        return regime

    def _check_data_quality(self, data: pd.DataFrame) -> dict:
        return {
            "missing_values": data.isnull().sum().to_dict(),
            "high_ge_low": (data["high"] >= data["low"]).all(),
            "high_ge_open": (data["high"] >= data["open"]).all(),
            "high_ge_close": (data["high"] >= data["close"]).all(),
            "low_le_open": (data["low"] <= data["open"]).all(),
            "low_le_close": (data["low"] <= data["close"]).all(),
            "volume_positive": (data["volume"] > 0).all(),
            "price_positive": (data[["open", "high", "low", "close"]] > 0).all().all(),
        }

    def _validate_trend_pattern(self, data: pd.DataFrame, pattern_type: str) -> dict:
        returns = data["close"].pct_change().dropna()
        return {
            "linear_trend": float(np.polyfit(range(len(returns)), returns, 1)[0]),
            "trend_consistency": float(np.sum(np.diff(np.sign(returns)) == 0) / len(returns)),
            "trend_strength": float(abs(returns.mean()) / returns.std()),
        }

    def _validate_reversal_pattern(self, data: pd.DataFrame, pattern_type: str) -> dict:
        prices = data["close"].values
        return {
            "peak_count": int(self._count_peaks(prices)),
            "trough_count": int(self._count_troughs(prices)),
            "pattern_symmetry": float(self._calculate_pattern_symmetry(prices)),
        }

    def _run_statistical_tests(self, data: pd.DataFrame) -> dict:
        returns = data["close"].pct_change().dropna()
        return {
            "stationarity": self._test_stationarity(returns),
            "normality": self._test_normality(returns),
            "autocorrelation": self._test_autocorrelation(returns),
        }

    def _count_peaks(self, prices: np.ndarray) -> int:
        return int(np.sum((prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])))

    def _count_troughs(self, prices: np.ndarray) -> int:
        return int(np.sum((prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])))

    def _calculate_pattern_symmetry(self, prices: np.ndarray) -> float:
        mid = len(prices) // 2
        first_half = prices[:mid]
        second_half = prices[mid : 2 * mid][::-1]
        if len(first_half) != len(second_half):
            return 0.0
        correlation = np.corrcoef(first_half, second_half)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _test_stationarity(self, returns: pd.Series) -> dict:
        if not STATSMODELS_AVAILABLE:
            return {"test_statistic": None, "p_value": None, "is_stationary": False}
        try:
            adf_result = adfuller(returns)
            return {
                "test_statistic": adf_result[0],
                "p_value": adf_result[1],
                "is_stationary": adf_result[1] < 0.05,
            }
        except Exception:
            return {"test_statistic": None, "p_value": None, "is_stationary": False}

    def _test_normality(self, returns: pd.Series) -> dict:
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            return {
                "test_statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > 0.05,
            }
        except Exception:
            return {"test_statistic": None, "p_value": None, "is_normal": False}

    def _test_autocorrelation(self, returns: pd.Series) -> dict:
        try:
            lagged_returns = returns.shift(1).dropna()
            returns_aligned = returns.iloc[1:]
            correlation = np.corrcoef(returns_aligned, lagged_returns)[0, 1]
            return {
                "lag1_correlation": correlation,
                "has_autocorrelation": abs(correlation) > 0.1,
            }
        except Exception:
            return {"lag1_correlation": None, "has_autocorrelation": False}
