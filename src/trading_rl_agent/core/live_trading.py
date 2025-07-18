"""
Live Trading Framework - Core real-time trading execution system.

This module provides the foundation for live trading with:
- Real-time data feeds
- Order execution and management
- Risk management integration
- Performance monitoring
- Model inference integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..agents.hybrid import HybridAgent
from ..data.live_feed import LiveDataFeed
from ..models.cnn_lstm import CNNLSTMModel
from ..risk.manager import RiskLimits, RiskManager
from ..utils.metrics import calculate_sharpe_ratio


@dataclass
class TradingConfig:
    """Configuration for live trading session."""

    # Data configuration
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    data_source: str = "yfinance"  # yfinance, alpaca, etc.
    update_interval: int = 60  # seconds

    # Model configuration
    model_path: str | None = None
    cnn_lstm_path: str | None = None
    agent_type: str = "hybrid"  # hybrid, ppo, sac

    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit

    # Execution
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_pct: float = 0.001  # 0.1% commission

    # Portfolio
    initial_capital: float = 100000.0
    max_drawdown: float = 0.15  # 15% max drawdown


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float) -> None:
        """Update position with current price."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    order_type: str = "market"  # market, limit, stop
    status: str = "pending"  # pending, filled, cancelled
    timestamp: datetime = field(default_factory=datetime.now)


class TradingSession:
    """Manages a single trading session."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_feed = LiveDataFeed(config.symbols, config.data_source)
        risk_limits = RiskLimits(
            max_position_size=config.max_position_size,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            max_drawdown=config.max_drawdown,
        )
        self.risk_manager = RiskManager(risk_limits)

        # Portfolio state
        self.cash = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.portfolio_value = config.initial_capital

        # Models
        self.cnn_lstm_model: CNNLSTMModel | None = None
        self.rl_agent: HybridAgent | None = None

        # Performance tracking
        self.trades_history: list[dict[str, Any]] = []
        self.portfolio_history: list[dict[str, Any]] = []

        self._load_models()

    def _load_models(self) -> None:
        """Load trained models."""
        try:
            if self.config.cnn_lstm_path and Path(self.config.cnn_lstm_path).exists():
                # Create model with default input dimension (will be overridden by state dict)
                self.cnn_lstm_model = CNNLSTMModel(input_dim=50)  # Default input dimension
                self.cnn_lstm_model.load_state_dict(torch.load(self.config.cnn_lstm_path))
                self.logger.info(f"Loaded CNN+LSTM model from {self.config.cnn_lstm_path}")

            # Load RL agent based on type
            if self.config.agent_type == "ppo":
                raise NotImplementedError("PPO agent is not yet implemented for live trading.")
            if self.config.agent_type == "hybrid":
                self.rl_agent = HybridAgent(cnn_lstm_model=self.cnn_lstm_model, state_dim=50, action_dim=3)
                if self.config.model_path and Path(self.config.model_path).exists():
                    self.rl_agent.load(self.config.model_path)

        except Exception:
            self.logger.exception("Failed to load models")

    async def start(self) -> None:
        """Start the trading session."""
        self.logger.info("Starting live trading session...")

        try:
            await self.data_feed.connect()

            while True:
                # Get latest market data
                market_data = await self.data_feed.get_latest_data()

                # Update portfolio with current prices
                self._update_portfolio(market_data)

                # Generate trading signals
                signals = await self._generate_signals(market_data)

                # Execute trades based on signals and risk management
                await self._execute_trades(signals, market_data)

                # Log performance
                self._log_performance()

                # Check risk limits
                portfolio_weights = {
                    symbol: (pos.quantity * pos.current_price) / self.portfolio_value
                    for symbol, pos in self.positions.items()
                }
                if self.risk_manager.check_risk_limits(portfolio_weights, self.portfolio_value):
                    self.logger.warning("Risk limits exceeded, stopping trading")
                    break

                # Wait for next update
                await asyncio.sleep(self.config.update_interval)

        except Exception:
            self.logger.exception("Trading session error")
        finally:
            await self.data_feed.disconnect()
            self._save_session_data()

    def _update_portfolio(self, market_data: dict[str, float]) -> None:
        """Update portfolio with current market prices."""
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_price(market_data[symbol])
                total_value += position.quantity * position.current_price

        self.portfolio_value = total_value

    async def _generate_signals(self, market_data: dict[str, float]) -> dict[str, str]:
        """Generate trading signals using loaded models."""
        signals = {}

        for symbol in market_data:
            try:
                # Get features for the symbol
                features = await self.data_feed.get_features(symbol)

                if self.rl_agent is not None and features is not None:
                    # Get action from RL agent
                    state = torch.FloatTensor(features).unsqueeze(0)
                    action = self.rl_agent.select_action(state)

                    # Convert action to signal
                    if action == 0:
                        signals[symbol] = "buy"
                    elif action == 1:
                        signals[symbol] = "sell"
                    else:
                        signals[symbol] = "hold"
                else:
                    signals[symbol] = "hold"

            except Exception:
                self.logger.exception(f"Error generating signal for {symbol}")
                signals[symbol] = "hold"

        return signals

    async def _execute_trades(self, signals: dict[str, str], market_data: dict[str, float]) -> None:
        """Execute trades based on signals and risk management."""
        for symbol, signal in signals.items():
            if signal == "hold":
                continue

            current_price = market_data[symbol]

            # Calculate position size
            position_size = self.risk_manager.calculate_kelly_position_size(
                expected_return=0.02,  # Placeholder
                win_rate=0.55,  # Placeholder
                avg_win=0.05,  # Placeholder
                avg_loss=0.03,  # Placeholder
            )

            if position_size <= 0:
                continue

            # Create and execute order
            order = Order(
                symbol=symbol,
                side=signal,
                quantity=position_size * self.portfolio_value / current_price,
                price=current_price,
            )

            success = await self._execute_order(order)
            if success:
                self.orders.append(order)
                self._update_positions(order)

    async def _execute_order(self, order: Order) -> bool:
        """Execute a trading order."""
        try:
            # Simulate order execution
            await asyncio.sleep(0.05)  # Simulate network latency
            order.status = "filled"
            self.logger.info(f"Executed order: {order}")
            return True
        except Exception:
            self.logger.exception("Failed to execute order")
            order.status = "cancelled"
            return False

    def _update_positions(self, order: Order) -> None:
        """Update positions after a trade."""
        if order.side == "buy":
            if order.symbol in self.positions:
                # Update existing position
                pos = self.positions[order.symbol]
                new_quantity = pos.quantity + order.quantity
                new_entry_price = (pos.entry_price * pos.quantity + order.price * order.quantity) / new_quantity
                pos.quantity = new_quantity
                pos.entry_price = new_entry_price
            else:
                # Create new position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=order.price,
                    entry_time=datetime.now(),
                )
            self.cash -= order.quantity * order.price
        elif order.side == "sell":
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.realized_pnl += (order.price - pos.entry_price) * order.quantity
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self.positions[order.symbol]
                self.cash += order.quantity * order.price

    def _log_performance(self) -> None:
        """Log current performance metrics."""
        # This function needs to be updated to calculate returns and sharpe ratio
        # For now, it will just log the current portfolio value
        self.logger.info(
            f"Portfolio Value: ${self.portfolio_value:,.2f} | "
            f"Cash: ${self.cash:,.2f} | "
            f"Positions: {len(self.positions)} | "
            f"Total Return: {0:.2f}%",  # Placeholder for total return
        )

        # Log portfolio history
        if len(self.portfolio_history) > 0:
            returns = pd.Series([p["portfolio_value"] for p in self.portfolio_history]).pct_change()
            sharpe_ratio = calculate_sharpe_ratio(returns)
        else:
            sharpe_ratio = 0.0

        self.portfolio_history.append(
            {
                "timestamp": datetime.now(),
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
                "total_return": 0,  # Placeholder for total return
                "sharpe_ratio": sharpe_ratio,
            },
        )

    def _save_session_data(self) -> None:
        """Save session trade history and performance to disk."""
        output_dir = Path("live_trading_sessions") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trades and portfolio history
        pd.DataFrame(self.trades_history).to_csv(output_dir / "trades.csv", index=False)
        pd.DataFrame(self.portfolio_history).to_csv(output_dir / "portfolio.csv", index=False)
        self.logger.info(f"Saved session data to {output_dir}")


class LiveTradingEngine:
    """Manages multiple trading sessions."""

    def __init__(self) -> None:
        self.sessions: list[TradingSession] = []
        self.logger = logging.getLogger(__name__)

    def create_session(self, config: TradingConfig) -> TradingSession:
        """Create and add a new trading session."""
        session = TradingSession(config)
        self.sessions.append(session)
        self.logger.info(f"Created new trading session for symbols: {config.symbols}")
        return session

    async def start_all_sessions(self) -> None:
        """Start all trading sessions concurrently."""
        await asyncio.gather(*(session.start() for session in self.sessions))

    def stop_all_sessions(self) -> None:
        """Stop all trading sessions."""
        for session in self.sessions:
            # This needs to be implemented by cancelling the asyncio task
            pass
