from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .feature_extractor import FeatureExtractor
from .regime_classifier import RegimeClassifier
from .rule_engine import RuleEngine


@dataclass
class RiskMetrics:
    """Risk metrics data class."""
    portfolio_var: float = 0.0
    portfolio_cvar: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    leverage: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    beta: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_var: float = 0.05
    max_drawdown: float = 0.10
    max_leverage: float = 3.0
    max_correlation: float = 0.8
    max_portfolio_var: float = 0.05
    max_position_size: float = 0.1
    max_sector_exposure: float = 0.3
    max_daily_trades: int = 100
    max_daily_volume: float = 1000000.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15


class RiskManager:
    """Main risk manager class."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.risk_limits = self.limits  # Alias for backward compatibility
        self.current_metrics = RiskMetrics()

    def update_metrics(self, metrics: RiskMetrics) -> None:
        """Update current risk metrics."""
        self.current_metrics = metrics

    def check_limits(self) -> bool:
        """Check if current metrics exceed limits."""
        return (
            self.current_metrics.portfolio_var <= self.limits.max_var and
            self.current_metrics.current_drawdown <= self.limits.max_drawdown and
            self.current_metrics.leverage <= self.limits.max_leverage and
            self.current_metrics.correlation_risk <= self.limits.max_correlation
        )


class Config:
    """Simple config class for risk management."""

    def __init__(self) -> None:
        self.bull_multiplier = 1.5
        self.bear_multiplier = 0.8
        self.bull_stop_loss = 0.05
        self.bear_stop_loss = 0.03
        self.active_strategies = ["bull", "bear"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bull_multiplier": self.bull_multiplier,
            "bear_multiplier": self.bear_multiplier,
            "bull_stop_loss": self.bull_stop_loss,
            "bear_stop_loss": self.bear_stop_loss,
            "active_strategies": self.active_strategies
        }


class RiskManagementModule:
    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor(data_source=None)
        self.regime_classifier = RegimeClassifier()
        self.config = Config()
        self.rule_engine = RuleEngine(self.config.to_dict())

    def run(self) -> None:
        # Step 1: Extract features
        features = self.feature_extractor.get_features()

        # Step 2: Classify market regime
        regime = self.regime_classifier.classify_regime(features)

        # Step 3: Apply risk rules based on the classified regime
        print(f"Current regime: {regime}")


if __name__ == "__main__":
    risk_management_module = RiskManagementModule()
    risk_management_module.run()
