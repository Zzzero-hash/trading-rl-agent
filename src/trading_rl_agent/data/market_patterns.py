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
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    CHANNEL = "channel"


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

    def generate_arima_trend(
        self,
        n_periods: int,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        trend_strength: float = 0.001,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Generate trend patterns using ARIMA models for more realistic price movements."""
        if not STATSMODELS_AVAILABLE:
            df = self.generate_trend_pattern(n_periods, "uptrend", trend_strength, volatility)
            df["pattern_type"] = "arima_trend"
            df["arima_order"] = str(order)
            df["trend_strength"] = trend_strength
            return df

        # Generate ARIMA process
        np.random.seed(self.seed if self.seed is not None else np.random.randint(0, 10000))

        if seasonal_order:
            model = SARIMAX(np.zeros(n_periods), order=order, seasonal_order=seasonal_order, trend="c")
        else:
            model = ARIMA(np.zeros(n_periods), order=order, trend="c")

        # Simulate ARIMA process
        arima_returns = model.simulate(
            params=[trend_strength, 0.1, 0.1, 0.8] + ([0.1, 0.1, 0.8, 12] if seasonal_order else []),
            nsimulations=n_periods,
        )

        # Add volatility clustering
        volatility_process = np.exp(np.random.normal(0, volatility, n_periods))
        returns = arima_returns * volatility_process

        prices = self.base_price * np.exp(np.cumsum(returns))
        df = self._generate_ohlcv_from_prices(prices, volatility)
        df["pattern_type"] = "arima_trend"
        df["arima_order"] = str(order)
        df["trend_strength"] = trend_strength
        return df

    def generate_triangle_pattern(
        self,
        pattern_type: str,
        n_periods: int = 100,
        pattern_intensity: float = 0.5,
        base_volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Generate triangle patterns (ascending, descending, symmetrical)."""
        if pattern_type == PatternType.ASCENDING_TRIANGLE.value:
            return self._generate_ascending_triangle(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.DESCENDING_TRIANGLE.value:
            return self._generate_descending_triangle(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.SYMMETRICAL_TRIANGLE.value:
            return self._generate_symmetrical_triangle(n_periods, pattern_intensity, base_volatility)
        raise ValueError(f"Unknown triangle pattern type: {pattern_type}")

    def generate_continuation_pattern(
        self,
        pattern_type: str,
        n_periods: int = 50,
        pattern_intensity: float = 0.5,
        base_volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Generate continuation patterns (flags, pennants, wedges)."""
        if pattern_type == PatternType.FLAG.value:
            return self._generate_flag_pattern(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.PENNANT.value:
            return self._generate_pennant_pattern(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.WEDGE.value:
            return self._generate_wedge_pattern(n_periods, pattern_intensity, base_volatility)
        if pattern_type == PatternType.CHANNEL.value:
            return self._generate_channel_pattern(n_periods, pattern_intensity, base_volatility)
        raise ValueError(f"Unknown continuation pattern type: {pattern_type}")

    def generate_enhanced_microstructure(
        self,
        base_data: pd.DataFrame,
        bid_ask_spread: float = 0.001,
        order_book_depth: int = 10,
        tick_size: float = 0.01,
        market_impact: float = 0.0001,
        liquidity_profile: str = "normal",
        trading_hours: dict | None = None,
    ) -> pd.DataFrame:
        """Enhanced market microstructure effects with realistic order book dynamics."""
        df = base_data.copy()

        # Generate bid-ask spreads with time-varying characteristics
        base_spread = bid_ask_spread
        if liquidity_profile == "high":
            spread_multiplier = 0.5
        elif liquidity_profile == "low":
            spread_multiplier = 2.0
        else:
            spread_multiplier = 1.0

        # Add intraday spread patterns
        if trading_hours:
            hour_of_day = pd.to_datetime(df["timestamp"]).dt.hour
            intraday_spread = np.where(
                (hour_of_day >= trading_hours.get("open", 9)) & (hour_of_day <= trading_hours.get("close", 16)),
                1.0,  # Normal hours
                1.5,  # After hours
            )
        else:
            intraday_spread = np.ones(len(df))

        # Generate realistic spreads
        spreads = base_spread * spread_multiplier * intraday_spread * np.exp(np.random.normal(0, 0.1, len(df)))

        df["bid"] = df["close"] * (1 - spreads / 2)
        df["ask"] = df["close"] * (1 + spreads / 2)

        # Add order book levels
        for i in range(1, order_book_depth + 1):
            bid_level = f"bid_level_{i}"
            ask_level = f"ask_level_{i}"
            bid_volume = f"bid_volume_{i}"
            ask_volume = f"ask_volume_{i}"

            # Price levels with exponential decay
            price_decay = np.exp(-i * 0.1)
            volume_decay = np.exp(-i * 0.3)

            df[bid_level] = df["bid"] * (1 - spreads * i * price_decay)
            df[ask_level] = df["ask"] * (1 + spreads * i * price_decay)

            # Volume at each level
            base_volume = df["volume"] * volume_decay
            df[bid_volume] = np.random.poisson(base_volume * 0.8)
            df[ask_volume] = np.random.poisson(base_volume * 0.8)

        # Market impact based on volume
        volume_impact = df["volume"] * market_impact
        df["close"] += volume_impact * np.random.normal(0, 1, len(df))

        # Apply tick size constraints after all modifications
        price_cols = (
            ["open", "high", "low", "close"]
            + [f"bid_level_{i}" for i in range(1, order_book_depth + 1)]
            + [f"ask_level_{i}" for i in range(1, order_book_depth + 1)]
        )
        for col in price_cols:
            if col in df.columns:
                # Ensure prices are aligned to tick size by rounding to nearest tick
                # Use a more precise method to avoid floating point issues
                df[col] = np.round(df[col] / tick_size) * tick_size
                # Ensure exact alignment by handling floating point precision
                df[col] = np.where(np.abs(df[col] % tick_size) < 1e-10, df[col] - (df[col] % tick_size), df[col])

        df["spread"] = df["ask"] - df["bid"]
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        df["liquidity_profile"] = liquidity_profile

        return df

    def detect_enhanced_regime(
        self, data: pd.DataFrame, window: int = 20, method: str = "rolling_stats"
    ) -> pd.DataFrame:
        """Enhanced market regime detection with multiple methods."""
        df = data.copy()
        df["returns"] = df["close"].pct_change()

        if method == "rolling_stats":
            return self._detect_regime_rolling_stats(df, window)
        if method == "markov_switching":
            return self._detect_regime_markov_switching(df, window)
        if method == "volatility_regime":
            return self._detect_regime_volatility(df, window)
        raise ValueError(f"Unknown regime detection method: {method}")

    def validate_pattern_quality(self, data: pd.DataFrame, pattern_type: str) -> dict:
        """Validate the quality of generated patterns."""
        validation_results = {
            "pattern_type": pattern_type,
            "data_quality": self._check_data_quality(data),
            "pattern_quality": {},
            "statistical_tests": {},
            "pattern_specific_tests": {},
        }

        if pattern_type in [PatternType.UPTREND.value, PatternType.DOWNTREND.value, PatternType.SIDEWAYS.value]:
            validation_results["pattern_quality"] = self._validate_trend_pattern(data, pattern_type)
        elif "head_and_shoulders" in pattern_type or "double" in pattern_type:
            validation_results["pattern_quality"] = self._validate_reversal_pattern(data, pattern_type)
        elif "triangle" in pattern_type:
            validation_results["pattern_quality"] = self._validate_triangle_pattern(data, pattern_type)
        elif pattern_type in [
            PatternType.FLAG.value,
            PatternType.PENNANT.value,
            PatternType.WEDGE.value,
            PatternType.CHANNEL.value,
        ]:
            validation_results["pattern_quality"] = self._validate_continuation_pattern(data, pattern_type)

        validation_results["statistical_tests"] = self._run_enhanced_statistical_tests(data)
        validation_results["pattern_specific_tests"] = self._run_pattern_specific_tests(data, pattern_type)

        return validation_results

    # --- Internal helpers ---

    def _generate_ohlcv_from_prices(self, prices: np.ndarray, volatility: float) -> pd.DataFrame:
        n_periods = len(prices)
        # Use fixed start date for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
        start_date = datetime(2023, 1, 1) - timedelta(days=n_periods)
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

    # --- New pattern generation methods ---

    def _generate_ascending_triangle(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate ascending triangle pattern."""
        # Upper resistance line (flat)
        resistance = np.ones(n_periods) * 0.05 * intensity

        # Lower support line (ascending)
        support = np.linspace(0, 0.04 * intensity, n_periods)

        # Generate prices within the triangle
        prices = self.base_price * np.exp((resistance + support) / 2)

        # Add noise and convergence
        noise = np.random.normal(0, volatility, n_periods)
        convergence = np.exp(-np.linspace(0, 2, n_periods))  # Pattern converges
        prices *= np.exp(noise * convergence)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_descending_triangle(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate descending triangle pattern."""
        # Upper resistance line (descending)
        resistance = np.linspace(0.05 * intensity, 0.01 * intensity, n_periods)

        # Lower support line (flat)
        support = np.ones(n_periods) * 0.01 * intensity

        # Generate prices within the triangle
        prices = self.base_price * np.exp((resistance + support) / 2)

        # Add noise and convergence
        noise = np.random.normal(0, volatility, n_periods)
        convergence = np.exp(-np.linspace(0, 2, n_periods))
        prices *= np.exp(noise * convergence)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_symmetrical_triangle(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate symmetrical triangle pattern."""
        # Upper resistance line (descending)
        resistance = np.linspace(0.05 * intensity, 0.01 * intensity, n_periods)

        # Lower support line (ascending)
        support = np.linspace(0.01 * intensity, 0.05 * intensity, n_periods)

        # Generate prices within the triangle
        prices = self.base_price * np.exp((resistance + support) / 2)

        # Add noise and convergence
        noise = np.random.normal(0, volatility, n_periods)
        convergence = np.exp(-np.linspace(0, 2, n_periods))
        prices *= np.exp(noise * convergence)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_flag_pattern(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate flag pattern (small rectangle after strong move)."""
        # Strong initial move
        initial_move = np.linspace(0, 0.1 * intensity, int(n_periods * 0.3))

        # Flag consolidation (small rectangle)
        flag_periods = n_periods - len(initial_move)
        flag_high = 0.1 * intensity + 0.02 * intensity
        flag_low = 0.1 * intensity - 0.02 * intensity
        flag_prices = np.random.uniform(flag_low, flag_high, flag_periods)

        # Combine initial move and flag
        prices = self.base_price * np.exp(np.concatenate([initial_move, flag_prices]))

        # Add noise
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_pennant_pattern(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate pennant pattern (small triangle after strong move)."""
        # Strong initial move
        initial_move = np.linspace(0, 0.1 * intensity, int(n_periods * 0.3))

        # Pennant consolidation (small triangle)
        pennant_periods = n_periods - len(initial_move)
        pennant_high = 0.1 * intensity + np.linspace(0.02 * intensity, 0, pennant_periods)
        pennant_low = 0.1 * intensity - np.linspace(0.02 * intensity, 0, pennant_periods)
        pennant_prices = (pennant_high + pennant_low) / 2

        # Combine initial move and pennant
        prices = self.base_price * np.exp(np.concatenate([initial_move, pennant_prices]))

        # Add noise
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_wedge_pattern(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate wedge pattern."""
        # Rising wedge (both lines slope up, but resistance slopes more)
        resistance = np.linspace(0, 0.08 * intensity, n_periods)
        support = np.linspace(0, 0.04 * intensity, n_periods)

        # Generate prices within the wedge
        prices = self.base_price * np.exp((resistance + support) / 2)

        # Add noise
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)

        return self._generate_ohlcv_from_prices(prices, volatility)

    def _generate_channel_pattern(self, n_periods: int, intensity: float, volatility: float) -> pd.DataFrame:
        """Generate channel pattern (parallel lines)."""
        # Parallel channel lines
        upper_channel = np.linspace(0, 0.06 * intensity, n_periods)
        lower_channel = np.linspace(0, 0.02 * intensity, n_periods)

        # Generate prices within the channel
        prices = self.base_price * np.exp((upper_channel + lower_channel) / 2)

        # Add noise
        noise = np.random.normal(0, volatility, n_periods)
        prices *= np.exp(noise)

        return self._generate_ohlcv_from_prices(prices, volatility)

    # --- Enhanced regime detection methods ---

    def _detect_regime_rolling_stats(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Detect regime using rolling statistics."""
        df["rolling_mean"] = df["returns"].rolling(window).mean()
        df["rolling_std"] = df["returns"].rolling(window).std()
        df["rolling_skew"] = df["returns"].rolling(window).skew()
        df["rolling_kurt"] = df["returns"].rolling(window).kurt()
        df["regime"] = self._classify_regime(df["rolling_mean"], df["rolling_std"], df["rolling_skew"])
        return df

    def _detect_regime_markov_switching(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Detect regime using Markov switching model (simplified)."""
        if not STATSMODELS_AVAILABLE:
            return self._detect_regime_rolling_stats(df, window)

        try:
            # Simplified Markov switching using volatility regimes
            volatility = df["returns"].rolling(window).std()
            high_vol_threshold = volatility.quantile(0.75)
            low_vol_threshold = volatility.quantile(0.25)

            df["regime"] = "sideways"
            df.loc[volatility > high_vol_threshold, "regime"] = "volatile"
            df.loc[volatility < low_vol_threshold, "regime"] = "calm"

            # Add trend detection
            trend = df["returns"].rolling(window).mean()
            df.loc[(trend > 0.001) & (df["regime"] == "sideways"), "regime"] = "trending_up"
            df.loc[(trend < -0.001) & (df["regime"] == "sideways"), "regime"] = "trending_down"

        except Exception:
            df["regime"] = "sideways"

        return df

    def _detect_regime_volatility(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Detect regime based on volatility clustering."""
        volatility = df["returns"].rolling(window).std()

        # Define volatility regimes
        vol_quantiles = volatility.quantile([0.25, 0.5, 0.75])

        df["regime"] = "medium_vol"
        df.loc[volatility < vol_quantiles[0.25], "regime"] = "low_vol"
        df.loc[volatility > vol_quantiles[0.75], "regime"] = "high_vol"

        return df

    # --- Enhanced validation methods ---

    def _validate_triangle_pattern(self, data: pd.DataFrame, pattern_type: str) -> dict:
        """Validate triangle pattern quality."""
        prices = data["close"].values
        highs = data["high"].values
        lows = data["low"].values

        # Calculate trend lines
        mid_point = len(prices) // 2
        first_half_highs = highs[:mid_point]
        second_half_highs = highs[mid_point:]
        first_half_lows = lows[:mid_point]
        second_half_lows = lows[mid_point:]

        # Fit trend lines
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]

        return {
            "high_trend_slope": float(high_trend),
            "low_trend_slope": float(low_trend),
            "convergence_quality": float(abs(high_trend - low_trend)),
            "pattern_symmetry": float(self._calculate_pattern_symmetry(prices)),
        }

    def _validate_continuation_pattern(self, data: pd.DataFrame, pattern_type: str) -> dict:
        """Validate continuation pattern quality."""
        prices = data["close"].values

        # Calculate consolidation metrics
        price_range = (prices.max() - prices.min()) / prices.mean()
        price_volatility = np.std(prices) / np.mean(prices)

        return {
            "consolidation_quality": float(price_range),
            "volatility_compression": float(price_volatility),
            "pattern_duration": len(prices),
        }

    def _run_enhanced_statistical_tests(self, data: pd.DataFrame) -> dict:
        """Run enhanced statistical tests."""
        returns = data["close"].pct_change().dropna()

        basic_tests = self._run_statistical_tests(data)

        return {
            **basic_tests,
            "seasonality": self._test_seasonality(returns),
            "heteroskedasticity": self._test_heteroskedasticity(returns),
            "long_memory": self._test_long_memory(returns),
        }

    def _run_pattern_specific_tests(self, data: pd.DataFrame, pattern_type: str) -> dict:
        """Run pattern-specific validation tests."""
        returns = data["close"].pct_change().dropna()
        prices = data["close"].values

        return {
            "price_momentum": float(np.mean(returns)),
            "price_acceleration": float(np.mean(np.diff(returns))),
            "peak_to_trough_ratio": float(self._calculate_peak_to_trough_ratio(prices)),
            "pattern_completeness": float(self._calculate_pattern_completeness(prices, pattern_type)),
        }

    def _test_seasonality(self, returns: pd.Series) -> dict:
        """Test for seasonality in returns."""
        if not STATSMODELS_AVAILABLE or len(returns) < 50:
            return {"has_seasonality": False, "seasonal_strength": 0.0}

        try:
            # Simple seasonality test using autocorrelation
            seasonal_lags = [5, 10, 20]  # Weekly, bi-weekly, monthly
            seasonal_correlations = []

            for lag in seasonal_lags:
                if lag < len(returns):
                    correlation = returns.autocorr(lag=lag)
                    seasonal_correlations.append(abs(correlation) if not pd.isna(correlation) else 0.0)

            seasonal_strength = np.mean(seasonal_correlations) if seasonal_correlations else 0.0

            return {
                "has_seasonality": seasonal_strength > 0.1,
                "seasonal_strength": float(seasonal_strength),
            }
        except Exception:
            return {"has_seasonality": False, "seasonal_strength": 0.0}

    def _test_heteroskedasticity(self, returns: pd.Series) -> dict:
        """Test for heteroskedasticity (ARCH effects)."""
        if not STATSMODELS_AVAILABLE or len(returns) < 30:
            return {"has_heteroskedasticity": False, "arch_effect_strength": 0.0}

        try:
            # Engle's ARCH test
            squared_returns = returns**2
            lagged_squared = squared_returns.shift(1).dropna()
            squared_aligned = squared_returns.iloc[1:]

            if len(lagged_squared) > 0:
                correlation = np.corrcoef(squared_aligned, lagged_squared)[0, 1]
                arch_strength = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                arch_strength = 0.0

            return {
                "has_heteroskedasticity": arch_strength > 0.1,
                "arch_effect_strength": float(arch_strength),
            }
        except Exception:
            return {"has_heteroskedasticity": False, "arch_effect_strength": 0.0}

    def _test_long_memory(self, returns: pd.Series) -> dict:
        """Test for long memory effects."""
        if len(returns) < 50:
            return {"has_long_memory": False, "hurst_exponent": 0.5}

        try:
            # Simplified Hurst exponent calculation
            lags = range(2, min(20, len(returns) // 4))
            tau = []

            for lag in lags:
                # Calculate variance of differences
                diff = returns.diff(lag).dropna()
                tau.append(np.sqrt(np.var(diff)))

            if len(tau) > 1:
                # Fit log-log plot
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = reg[0] * 2.0
            else:
                hurst = 0.5

            return {
                "has_long_memory": abs(hurst - 0.5) > 0.1,
                "hurst_exponent": float(hurst),
            }
        except Exception:
            return {"has_long_memory": False, "hurst_exponent": 0.5}

    def _calculate_peak_to_trough_ratio(self, prices: np.ndarray) -> float:
        """Calculate peak to trough ratio."""
        peaks = prices[1:-1][(prices[1:-1] > prices[:-2]) & (prices[1:-1] > prices[2:])]
        troughs = prices[1:-1][(prices[1:-1] < prices[:-2]) & (prices[1:-1] < prices[2:])]

        if len(peaks) > 0 and len(troughs) > 0:
            return float(np.mean(peaks) / np.mean(troughs))
        return 1.0

    def _calculate_pattern_completeness(self, prices: np.ndarray, pattern_type: str) -> float:
        """Calculate pattern completeness score."""
        if "triangle" in pattern_type:
            # Check if pattern converges
            first_half_vol = np.std(prices[: len(prices) // 2])
            second_half_vol = np.std(prices[len(prices) // 2 :])
            completeness = float(second_half_vol / first_half_vol) if first_half_vol > 0 else 1.0
            return min(completeness, 1.0)  # Cap at 1.0
        if "flag" in pattern_type or "pennant" in pattern_type:
            # Check consolidation quality
            consolidation_periods = len(prices) // 3
            consolidation_vol = np.std(prices[-consolidation_periods:])
            overall_vol = np.std(prices)
            completeness = float(consolidation_vol / overall_vol) if overall_vol > 0 else 1.0
            return min(completeness, 1.0)  # Cap at 1.0
        return 1.0
