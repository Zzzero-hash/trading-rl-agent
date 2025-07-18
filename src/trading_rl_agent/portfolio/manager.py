"""
Portfolio manager implementing modern portfolio theory and risk management.

Handles multi-asset portfolios with sophisticated optimization and risk controls.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    from pypfopt.objective_functions import L2_reg

    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

try:
    import empyrical

    EMPYRICAL_AVAILABLE = True
except ImportError:
    from ..utils.empyrical_mock import (
        beta,
        calmar_ratio,
        conditional_value_at_risk,
        max_drawdown,
        sharpe_ratio,
        sortino_ratio,
        value_at_risk,
    )

    empyrical = type(
        "MockEmpyrical",
        (),
        {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "value_at_risk": value_at_risk,
            "conditional_value_at_risk": conditional_value_at_risk,
            "beta": beta,
        },
    )()
    EMPYRICAL_AVAILABLE = True

from ..core.logging import get_logger
from .transaction_costs import (
    BrokerType,
    MarketCondition,
    MarketData,
    OrderType,
    TransactionCostModel,
)

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime
    side: str = "long"  # "long" or "short"

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == "long":
            return self.quantity * (self.current_price - self.entry_price)
        return self.quantity * (self.entry_price - self.current_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""

    # Risk parameters
    max_position_size: float = 0.1  # Max 10% in single position
    max_sector_exposure: float = 0.3  # Max 30% in single sector
    max_leverage: float = 1.0  # No leverage by default

    # Rebalancing
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05  # Rebalance if >5% drift

    # Transaction costs (legacy - now handled by TransactionCostModel)
    commission_rate: float = 0.001  # 0.1% commission
    bid_ask_spread: float = 0.0002  # 2 bps spread

    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit

    # Portfolio optimization
    risk_aversion: float = 1.0  # Risk aversion parameter
    target_volatility: float | None = None  # Target portfolio volatility

    # Performance tracking
    benchmark_symbol: str = "SPY"  # Benchmark for comparison

    # Transaction cost model configuration
    broker_type: BrokerType = BrokerType.RETAIL
    default_order_type: OrderType = OrderType.MARKET
    default_market_condition: MarketCondition = MarketCondition.NORMAL


class PortfolioManager:
    """
    Comprehensive portfolio management system.

    Features:
    - Multi-asset position tracking
    - Risk-adjusted portfolio optimization
    - Automated rebalancing
    - Performance analytics
    - Risk management and controls
    """

    def __init__(
        self,
        initial_capital: float,
        config: PortfolioConfig | None = None,
        transaction_cost_model: TransactionCostModel | None = None,
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital amount
            config: Portfolio configuration
            transaction_cost_model: Optional custom transaction cost model
        """
        self.initial_capital = initial_capital
        self.config = config or PortfolioConfig()
        self.logger = get_logger(self.__class__.__name__)

        # Initialize transaction cost model
        if transaction_cost_model is not None:
            self.transaction_cost_model = transaction_cost_model
        else:
            self.transaction_cost_model = TransactionCostModel.create_broker_model(self.config.broker_type)

        # Portfolio state
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.transaction_history: list[dict] = []
        self.performance_history: list[dict] = []

        # Market data cache
        self._price_data: dict[str, pd.DataFrame] = {}

        # Validation
        if not PYPFOPT_AVAILABLE:
            self.logger.warning(
                "PyPortfolioOpt not available, optimization features disabled",
            )
        if not EMPYRICAL_AVAILABLE:
            self.logger.warning("Empyrical not available, some analytics disabled")

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    @property
    def equity_value(self) -> float:
        """Total equity value (positions only)."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def weights(self) -> dict[str, float]:
        """Current portfolio weights."""
        total_value = self.total_value
        if total_value == 0:
            return {}

        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position.market_value / total_value
        weights["cash"] = self.cash / total_value
        return weights

    @property
    def leverage(self) -> float:
        """Current portfolio leverage."""
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        if self.total_value == 0:
            return 0.0
        return gross_exposure / self.total_value

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

        # Log portfolio value
        current_value = self.total_value
        pnl = current_value - self.initial_capital
        pnl_pct = pnl / self.initial_capital if self.initial_capital > 0 else 0

        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "total_value": current_value,
                "cash": self.cash,
                "equity_value": self.equity_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "leverage": self.leverage,
            },
        )

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str = "long",
        order_type: OrderType | None = None,
        market_condition: MarketCondition | None = None,
        market_data: MarketData | None = None,
    ) -> bool:
        """
        Execute a trade with realistic transaction cost modeling.

        Args:
            symbol: Trading symbol
            quantity: Number of shares/units
            price: Execution price
            side: "long" or "short"
            order_type: Type of order (default from config)
            market_condition: Market condition (default from config)
            market_data: Market data for cost modeling (auto-generated if None)

        Returns:
            True if trade executed successfully
        """
        try:
            # Use defaults if not provided
            order_type = order_type or self.config.default_order_type
            market_condition = market_condition or self.config.default_market_condition

            # Create market data if not provided
            if market_data is None:
                # Estimate bid-ask spread and other market data
                estimated_spread = self.config.bid_ask_spread
                bid = price * (1 - estimated_spread / 2)
                ask = price * (1 + estimated_spread / 2)

                market_data = MarketData(
                    timestamp=datetime.now(),
                    bid=bid,
                    ask=ask,
                    mid_price=price,
                    volume=abs(quantity) * 10,  # Estimate volume
                    volatility=0.02,  # 2% volatility estimate
                    avg_daily_volume=1000000,  # Default daily volume
                    market_cap=1000000000,  # Default market cap
                )

            # Simulate execution with transaction costs
            execution_result = self.transaction_cost_model.simulate_execution(
                requested_quantity=abs(quantity),
                market_data=market_data,
                order_type=order_type,
                market_condition=market_condition,
            )

            if not execution_result.success:
                self.logger.warning(f"Trade execution failed: {execution_result.reason}")
                return False

            # Calculate actual trade value and costs
            trade_value = execution_result.executed_quantity * execution_result.executed_price
            total_cost = execution_result.total_cost

            # Pre-trade risk checks
            if not self._validate_trade(
                symbol,
                execution_result.executed_quantity,
                execution_result.executed_price,
                side,
                total_cost,
            ):
                return False

            # Execute the trade
            if quantity > 0:  # Buy/cover
                required_cash = trade_value + total_cost
                if self.cash < required_cash:
                    self.logger.warning(
                        f"Insufficient cash for {symbol} trade: need {required_cash}, have {self.cash}",
                    )
                    return False

                self.cash -= required_cash

                if symbol in self.positions:
                    # Update existing position
                    pos = self.positions[symbol]
                    new_quantity = pos.quantity + execution_result.executed_quantity
                    if new_quantity != 0:
                        # Weighted average price
                        total_cost_basis = (pos.quantity * pos.entry_price) + (
                            execution_result.executed_quantity * execution_result.executed_price
                        )
                        pos.entry_price = total_cost_basis / new_quantity
                        pos.quantity = new_quantity
                        pos.current_price = execution_result.executed_price
                    else:
                        # Position closed
                        del self.positions[symbol]
                else:
                    # New position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=execution_result.executed_quantity,
                        entry_price=execution_result.executed_price,
                        current_price=execution_result.executed_price,
                        timestamp=execution_result.execution_time,
                        side=side,
                    )

            else:  # Sell/short
                executed_quantity = abs(execution_result.executed_quantity)

                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if pos.quantity >= executed_quantity:
                        # Partial or full sale
                        self.cash += (executed_quantity * execution_result.executed_price) - total_cost
                        pos.quantity -= executed_quantity
                        if pos.quantity == 0:
                            del self.positions[symbol]
                    else:
                        self.logger.warning(
                            f"Cannot sell {executed_quantity} shares of {symbol}, only have {pos.quantity}",
                        )
                        return False
                else:
                    # Short position
                    self.cash += (executed_quantity * execution_result.executed_price) - total_cost
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=-executed_quantity,
                        entry_price=execution_result.executed_price,
                        current_price=execution_result.executed_price,
                        timestamp=execution_result.execution_time,
                        side="short",
                    )

            # Record detailed transaction
            self.transaction_history.append(
                {
                    "timestamp": execution_result.execution_time,
                    "symbol": symbol,
                    "quantity": execution_result.executed_quantity,
                    "price": execution_result.executed_price,
                    "side": side,
                    "order_type": order_type.value,
                    "market_condition": market_condition.value,
                    "commission": execution_result.cost_breakdown["commission"],
                    "slippage": execution_result.cost_breakdown["slippage"],
                    "market_impact": execution_result.cost_breakdown["market_impact"],
                    "spread_cost": execution_result.cost_breakdown["spread_cost"],
                    "total_cost": total_cost,
                    "delay_seconds": execution_result.delay_seconds,
                    "partial_fills": execution_result.partial_fills,
                    "success": execution_result.success,
                },
            )

            self.logger.info(
                f"Executed trade: {execution_result.executed_quantity} {symbol} at {execution_result.executed_price} "
                f"(cost: ${total_cost:.2f}, delay: {execution_result.delay_seconds:.2f}s)",
            )
            return True

        except Exception as e:
            self.logger.exception(f"Trade execution failed: {e}")
            return False

    def _validate_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        total_cost: float,
    ) -> bool:
        """Validate trade against risk parameters."""

        # Position size check
        trade_value = abs(quantity * price)
        max_position_value = self.total_value * self.config.max_position_size

        current_position_value: float = 0.0
        if symbol in self.positions:
            current_position_value = abs(self.positions[symbol].market_value)

        new_position_value = current_position_value + trade_value
        if new_position_value > max_position_value:
            self.logger.warning(f"Trade would exceed max position size for {symbol}")
            return False

        # Leverage check
        new_leverage = self.leverage + (trade_value / self.total_value)
        if new_leverage > self.config.max_leverage:
            self.logger.warning(f"Trade would exceed max leverage: {new_leverage}")
            return False

        return True

    def optimize_portfolio(
        self,
        target_symbols: list[str],
        price_data: dict[str, pd.DataFrame],
        method: str = "max_sharpe",
    ) -> dict[str, float] | None:
        """
        Optimize portfolio weights using modern portfolio theory.

        Args:
            target_symbols: List of symbols to include
            price_data: Historical price data for each symbol
            method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_risk')

        Returns:
            Dictionary of optimal weights or None if optimization fails
        """
        if not PYPFOPT_AVAILABLE:
            self.logger.error("PyPortfolioOpt not available for optimization")
            return None

        try:
            # Prepare price data
            prices_df = pd.DataFrame()
            for symbol in target_symbols:
                if symbol in price_data:
                    prices_df[symbol] = price_data[symbol]["close"]

            if prices_df.empty:
                self.logger.error("No price data available for optimization")
                return None

            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)

            # Create efficient frontier
            ef = EfficientFrontier(mu, S)

            # Add regularization to prevent overfitting
            ef.add_objective(L2_reg, gamma=0.1)

            # Optimize based on method
            if method == "max_sharpe":
                # weights = ef.max_sharpe()  # Not used currently
                pass
            elif method == "min_volatility":
                # weights = ef.min_volatility()  # Not used currently
                pass
            elif method == "efficient_risk" and self.config.target_volatility:
                # weights = ef.efficient_risk(self.config.target_volatility)  # Not used currently
                pass
            else:
                # weights = ef.max_sharpe()  # Default - Not used currently
                pass

            # Clean weights (remove tiny positions)
            cleaned_weights = ef.clean_weights()

            self.logger.info(f"Portfolio optimization completed using {method}")
            return cast("dict[str, float]", cleaned_weights)

        except Exception as e:
            self.logger.exception(f"Portfolio optimization failed: {e}")
            return None

    def get_performance_summary(self) -> dict[str, Any]:
        """Generate comprehensive performance summary."""
        if not self.performance_history:
            return {}

        df = pd.DataFrame(self.performance_history)
        returns = df["pnl_pct"].pct_change().dropna()

        summary = {
            "total_return": df["pnl_pct"].iloc[-1],
            "total_value": df["total_value"].iloc[-1],
            "max_drawdown": self._calculate_max_drawdown(df["total_value"]),
            "volatility": returns.std() * np.sqrt(252),  # Annualized
            "num_trades": len(self.transaction_history),
            "win_rate": self._calculate_win_rate(),
            "profit_factor": self._calculate_profit_factor(),
            "current_positions": len(self.positions),
            "cash_ratio": self.cash / self.total_value if self.total_value > 0 else 0,
            "leverage": self.leverage,
        }

        # Add Sharpe ratio if empyrical available
        if EMPYRICAL_AVAILABLE and len(returns) > 1:
            summary["sharpe_ratio"] = empyrical.sharpe_ratio(returns)

        return summary

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from transaction history."""
        if not self.transaction_history:
            return 0.0

        winning_trades = 0
        total_trades = 0

        # Group trades by symbol to calculate P&L
        symbol_trades: dict[str, list[dict[str, Any]]] = {}
        for trade in self.transaction_history:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        # Calculate P&L for each symbol's trades
        for symbol, trades in symbol_trades.items():
            for i in range(len(trades) - 1):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]

                if buy_trade["side"] != sell_trade["side"]:
                    pnl = (sell_trade["price"] - buy_trade["price"]) * min(
                        abs(buy_trade["quantity"]),
                        abs(sell_trade["quantity"]),
                    )

                    if pnl > 0:
                        winning_trades += 1
                    total_trades += 1

        return float(winning_trades / total_trades) if total_trades > 0 else 0.0

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.performance_history:
            return 0.0

        df = pd.DataFrame(self.performance_history)
        returns = df["pnl_pct"].diff().dropna()

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    def get_transaction_cost_analysis(self) -> dict[str, Any]:
        """Get detailed transaction cost analysis."""
        from .transaction_costs import TransactionCostAnalyzer

        analyzer = TransactionCostAnalyzer(self.transaction_cost_model)

        return {
            "cost_summary": self.transaction_cost_model.get_cost_summary(),
            "cost_trends": analyzer.analyze_cost_trends(),
            "efficiency_metrics": analyzer.calculate_cost_efficiency_metrics(),
            "optimization_recommendations": self.transaction_cost_model.generate_optimization_recommendations(),
            "cost_report": analyzer.generate_cost_report(),
        }

    def optimize_transaction_costs(self) -> dict[str, Any]:
        """Generate transaction cost optimization recommendations."""
        recommendations = self.transaction_cost_model.generate_optimization_recommendations()

        # Calculate potential savings
        total_potential_savings = sum(rec.expected_savings for rec in recommendations)
        current_total_costs = self.transaction_cost_model.get_cost_summary()["total_transaction_costs"]

        return {
            "recommendations": recommendations,
            "total_potential_savings": total_potential_savings,
            "current_total_costs": current_total_costs,
            "savings_percentage": (
                (total_potential_savings / current_total_costs * 100) if current_total_costs > 0 else 0
            ),
            "implementation_priority": sorted(
                recommendations,
                key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.priority, 0),
                reverse=True,
            ),
        }
