#!/usr/bin/env python3
"""
Enhanced Market Pattern Generator Demo

This script demonstrates the advanced synthetic market data generation capabilities
including trend patterns, reversal patterns, volatility clustering, market microstructure,
and multi-asset correlated scenarios.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trade_agent.data.market_patterns import (
    STATSMODELS_AVAILABLE,
    MarketPatternGenerator,
)

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def plot_pattern_comparison(generator: MarketPatternGenerator, n_periods: int = 100) -> None:
    """Plot comparison of different pattern types."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Market Pattern Comparison", fontsize=16)

    patterns = [
        ("uptrend", "Uptrend"),
        ("downtrend", "Downtrend"),
        ("sideways", "Sideways"),
        ("head_and_shoulders", "Head & Shoulders"),
        ("double_top", "Double Top"),
        ("ascending_triangle", "Ascending Triangle"),
        ("flag", "Flag"),
        ("pennant", "Pennant"),
        ("channel", "Channel"),
    ]

    for idx, (pattern_type, title) in enumerate(patterns):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        if pattern_type in ["uptrend", "downtrend", "sideways"]:
            df = generator.generate_trend_pattern(
                n_periods=n_periods,
                trend_type=pattern_type,
                trend_strength=0.002,
                volatility=0.02,
            )
        elif pattern_type in ["head_and_shoulders", "double_top"]:
            df = generator.generate_reversal_pattern(
                pattern_type=pattern_type,
                n_periods=n_periods,
                pattern_intensity=0.8,
                base_volatility=0.02,
            )
        elif pattern_type in [
            "ascending_triangle",
            "descending_triangle",
            "symmetrical_triangle",
        ]:
            df = generator.generate_triangle_pattern(
                pattern_type=pattern_type,
                n_periods=n_periods,
                pattern_intensity=0.8,
                base_volatility=0.02,
            )
        else:
            df = generator.generate_continuation_pattern(
                pattern_type=pattern_type,
                n_periods=n_periods,
                pattern_intensity=0.8,
                base_volatility=0.02,
            )

        ax.plot(df.index, df["close"], linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

    plt.tight_layout()
    plt.savefig("pattern_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_arima_trends(generator: MarketPatternGenerator) -> None:
    """Demonstrate ARIMA-based trend generation."""
    if not STATSMODELS_AVAILABLE:
        print("statsmodels not available, skipping ARIMA demo")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ARIMA Trend Generation", fontsize=16)

    # Different ARIMA orders
    orders = [(1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1)]
    titles = ["ARIMA(1,1,1)", "ARIMA(2,1,2)", "ARIMA(1,1,0)", "ARIMA(0,1,1)"]

    for idx, (order, title) in enumerate(zip(orders, titles, strict=False)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        df = generator.generate_arima_trend(n_periods=100, order=order, trend_strength=0.001, volatility=0.02)

        ax.plot(df.index, df["close"], linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

    plt.tight_layout()
    plt.savefig("arima_trends.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_volatility_clustering(generator: MarketPatternGenerator) -> None:
    """Demonstrate volatility clustering and regime changes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Volatility Clustering and Regime Changes", fontsize=16)

    # Define volatility regimes
    volatility_regimes = [
        {"volatility": 0.01, "drift": 0.001, "label": "low_vol"},
        {"volatility": 0.05, "drift": 0.002, "label": "high_vol"},
        {"volatility": 0.02, "drift": -0.001, "label": "medium_vol"},
        {"volatility": 0.08, "drift": 0.003, "label": "crisis_vol"},
    ]
    regime_durations = [30, 20, 25, 15]

    # Generate data with volatility clustering
    df = generator.generate_volatility_clustering(
        n_periods=90,
        volatility_regimes=volatility_regimes,
        regime_durations=regime_durations,
        base_price=100.0,
    )

    # Plot price series
    axes[0].plot(df.index, df["close"], linewidth=2)
    axes[0].set_title("Price Series with Volatility Clustering")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Price")

    # Plot volatility regimes
    for regime in volatility_regimes:
        regime_data = df[df["volatility_regime"] == regime["label"]]
        if len(regime_data) > 0:
            axes[1].scatter(
                regime_data.index,
                regime_data["close"],
                label=regime["label"],
                alpha=0.7,
            )

    axes[1].set_title("Volatility Regimes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.savefig("volatility_clustering.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_microstructure_effects(generator: MarketPatternGenerator) -> None:
    """Demonstrate market microstructure effects."""
    # Generate base data
    base_data = generator.generate_trend_pattern(
        n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.02
    )

    # Add microstructure effects
    df = generator.generate_enhanced_microstructure(
        base_data=base_data,
        bid_ask_spread=0.002,
        order_book_depth=5,
        tick_size=0.01,
        market_impact=0.0001,
        liquidity_profile="normal",
        trading_hours={"open": 9, "close": 16},
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Market Microstructure Effects", fontsize=16)

    # Plot bid-ask spread
    axes[0, 0].plot(df.index, df["bid"], label="Bid", alpha=0.7)
    axes[0, 0].plot(df.index, df["ask"], label="Ask", alpha=0.7)
    axes[0, 0].plot(df.index, df["close"], label="Close", linewidth=2)
    axes[0, 0].set_title("Bid-Ask Spread")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot spread over time
    axes[0, 1].plot(df.index, df["spread"])
    axes[0, 1].set_title("Spread Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel("Spread")

    # Plot order book depth (first level)
    axes[1, 0].plot(df.index, df["bid_level_1"], label="Bid Level 1")
    axes[1, 0].plot(df.index, df["ask_level_1"], label="Ask Level 1")
    axes[1, 0].set_title("Order Book Level 1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot volume at different levels
    volume_cols = [col for col in df.columns if "volume" in col]
    for col in volume_cols[:3]:  # Plot first 3 levels
        axes[1, 1].plot(df.index, df[col], label=col)
    axes[1, 1].set_title("Volume at Different Levels")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("microstructure_effects.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_correlated_assets(generator: MarketPatternGenerator) -> None:
    """Demonstrate correlated multi-asset generation."""
    # Define correlation matrix
    correlation_matrix = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]])

    # Generate correlated assets
    assets_data = generator.generate_correlated_assets(
        n_assets=3,
        n_periods=100,
        correlation_matrix=correlation_matrix,
        base_prices=[100, 50, 200],
        volatilities=[0.02, 0.03, 0.025],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Correlated Multi-Asset Generation", fontsize=16)

    # Plot individual asset prices
    colors = ["blue", "red", "green"]
    for idx, (symbol, df) in enumerate(assets_data.items()):
        axes[0, 0].plot(df.index, df["close"], label=symbol, color=colors[idx], linewidth=2)

    axes[0, 0].set_title("Asset Prices")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Price")

    # Plot returns
    for idx, (symbol, df) in enumerate(assets_data.items()):
        returns = df["close"].pct_change()
        axes[0, 1].plot(df.index[1:], returns[1:], label=symbol, color=colors[idx], alpha=0.7)

    axes[0, 1].set_title("Asset Returns")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel("Returns")

    # Plot correlation heatmap
    returns_data = pd.DataFrame()
    for symbol, df in assets_data.items():
        returns_data[symbol] = df["close"].pct_change()

    correlation_actual = returns_data.corr()
    sns.heatmap(correlation_actual, annot=True, cmap="coolwarm", center=0, ax=axes[1, 0])
    axes[1, 0].set_title("Actual Correlation Matrix")

    # Plot scatter plot of returns
    asset_names = list(assets_data.keys())
    axes[1, 1].scatter(
        returns_data[asset_names[0]],
        returns_data[asset_names[1]],
        alpha=0.6,
        label=f"{asset_names[0]} vs {asset_names[1]}",
    )
    axes[1, 1].set_xlabel(f"{asset_names[0]} Returns")
    axes[1, 1].set_ylabel(f"{asset_names[1]} Returns")
    axes[1, 1].set_title("Return Correlation Scatter")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("correlated_assets.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_regime_detection(generator: MarketPatternGenerator) -> None:
    """Demonstrate enhanced regime detection."""
    # Generate data with regime changes
    regime_changes = [
        {"period": 20, "new_volatility": 2.0, "new_drift": 0.002},
        {"period": 50, "new_volatility": 0.5, "new_drift": -0.001},
        {"period": 80, "new_volatility": 3.0, "new_drift": 0.003},
    ]

    df = generator.generate_trend_pattern(
        n_periods=100,
        trend_type="uptrend",
        trend_strength=0.001,
        volatility=0.02,
        regime_changes=regime_changes,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle("Enhanced Regime Detection", fontsize=16)

    # Plot price series
    axes[0].plot(df.index, df["close"], linewidth=2)
    axes[0].set_title("Price Series")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Price")

    # Detect regimes using different methods
    methods = ["rolling_stats", "markov_switching", "volatility_regime"]
    method_names = ["Rolling Statistics", "Markov Switching", "Volatility Regime"]

    for idx, (method, name) in enumerate(zip(methods, method_names, strict=False)):
        df_regime = generator.detect_enhanced_regime(df, window=10, method=method)

        # Plot regime classification
        regime_colors = {
            "trending_up": "green",
            "trending_down": "red",
            "sideways": "gray",
            "volatile": "orange",
            "calm": "blue",
            "low_vol": "lightblue",
            "medium_vol": "yellow",
            "high_vol": "red",
        }

        for regime in df_regime["regime"].unique():
            if pd.notna(regime):
                regime_data = df_regime[df_regime["regime"] == regime]
                color = regime_colors.get(regime, "black")
                axes[idx + 1].scatter(
                    regime_data.index,
                    regime_data["close"],
                    label=regime,
                    color=color,
                    alpha=0.7,
                )

        axes[idx + 1].set_title(f"Regime Detection: {name}")
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)
        axes[idx + 1].set_ylabel("Price")

    plt.tight_layout()
    plt.savefig("regime_detection.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_validation(generator: MarketPatternGenerator) -> None:
    """Demonstrate pattern validation capabilities."""
    # Generate different patterns and validate them
    patterns_to_test = [
        ("uptrend", generator.generate_trend_pattern(50, "uptrend", 0.002, 0.02)),
        (
            "head_and_shoulders",
            generator.generate_reversal_pattern("head_and_shoulders", 50, 0.8, 0.02),
        ),
        (
            "ascending_triangle",
            generator.generate_triangle_pattern("ascending_triangle", 50, 0.8, 0.02),
        ),
        ("flag", generator.generate_continuation_pattern("flag", 50, 0.8, 0.02)),
    ]

    print("\n" + "=" * 60)
    print("PATTERN VALIDATION RESULTS")
    print("=" * 60)

    for pattern_type, data in patterns_to_test:
        print(f"\n{pattern_type.upper()} PATTERN:")
        print("-" * 40)

        validation = generator.validate_pattern_quality(data, pattern_type)

        # Data quality
        print("Data Quality:")
        for key, value in validation["data_quality"].items():
            print(f"  {key}: {value}")

        # Pattern quality
        print("\nPattern Quality:")
        for key, value in validation["pattern_quality"].items():
            print(f"  {key}: {value:.4f}")

        # Statistical tests
        print("\nStatistical Tests:")
        for test_name, test_results in validation["statistical_tests"].items():
            print(f"  {test_name}:")
            for key, value in test_results.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        # Pattern-specific tests
        print("\nPattern-Specific Tests:")
        for key, value in validation["pattern_specific_tests"].items():
            print(f"  {key}: {value:.4f}")


def main() -> None:
    """Main demonstration function."""
    print("Enhanced Market Pattern Generator Demo")
    print("=" * 50)

    # Initialize generator
    generator = MarketPatternGenerator(base_price=100.0, base_volatility=0.02, seed=42)

    print("Generator initialized with seed=42")
    print(f"statsmodels available: {STATSMODELS_AVAILABLE}")

    # Set up plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    try:
        # Run demonstrations
        print("\n1. Pattern Comparison...")
        plot_pattern_comparison(generator)

        print("\n2. ARIMA Trends...")
        demonstrate_arima_trends(generator)

        print("\n3. Volatility Clustering...")
        demonstrate_volatility_clustering(generator)

        print("\n4. Microstructure Effects...")
        demonstrate_microstructure_effects(generator)

        print("\n5. Correlated Assets...")
        demonstrate_correlated_assets(generator)

        print("\n6. Regime Detection...")
        demonstrate_regime_detection(generator)

        print("\n7. Pattern Validation...")
        demonstrate_validation(generator)

        print("\nDemo completed successfully!")
        print("Check the generated PNG files for visualizations.")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
