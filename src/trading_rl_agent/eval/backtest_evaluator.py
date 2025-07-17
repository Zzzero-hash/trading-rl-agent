"""
Unified Backtest Evaluator

This module provides a comprehensive backtesting framework that integrates
with the model evaluation pipeline, including realistic transaction cost
modeling, slippage simulation, and detailed performance attribution.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from ..core.unified_config import BacktestConfig
from ..portfolio.manager import PortfolioConfig, PortfolioManager
from ..utils.metrics import (
    calculate_beta,
    calculate_calmar_ratio,
    calculate_expected_shortfall,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    calculate_win_rate,
)
from .metrics_calculator import MetricsCalculator
from .statistical_tests import StatisticalTests

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TransactionCostModel:
    """Configurable transaction cost model for realistic backtesting."""

    # Commission structure
    commission_rate: float = 0.001  # 0.1% commission
    min_commission: float = 1.0  # Minimum commission per trade
    max_commission: float = 1000.0  # Maximum commission per trade

    # Slippage model
    slippage_rate: float = 0.0001  # 0.01% slippage
    slippage_model: str = "linear"  # linear, square_root, volume_based

    # Market impact model
    market_impact_rate: float = 0.00005  # 0.005% market impact
    impact_model: str = "linear"  # linear, square_root

    # Bid-ask spread
    bid_ask_spread: float = 0.0002  # 2 bps spread

    def calculate_total_cost(
        self,
        trade_value: float,
        trade_volume: float,
        avg_daily_volume: float = 1000000.0,
    ) -> dict[str, float]:
        """
        Calculate total transaction costs for a trade.

        Args:
            trade_value: Dollar value of the trade
            trade_volume: Number of shares/units traded
            avg_daily_volume: Average daily volume for market impact

        Returns:
            Dictionary with cost breakdown
        """
        # Commission
        commission = max(self.min_commission, min(trade_value * self.commission_rate, self.max_commission))

        # Slippage
        if self.slippage_model == "linear":
            slippage = trade_value * self.slippage_rate
        elif self.slippage_model == "square_root":
            slippage = trade_value * self.slippage_rate * np.sqrt(trade_volume / 1000)
        else:
            slippage = trade_value * self.slippage_rate

        # Market impact
        volume_ratio = trade_volume / avg_daily_volume if avg_daily_volume > 0 else 0
        if self.impact_model == "linear":
            market_impact = trade_value * self.market_impact_rate * volume_ratio
        elif self.impact_model == "square_root":
            market_impact = trade_value * self.market_impact_rate * np.sqrt(volume_ratio)
        else:
            market_impact = trade_value * self.market_impact_rate

        # Bid-ask spread
        spread_cost = trade_value * self.bid_ask_spread

        total_cost = commission + slippage + market_impact + spread_cost

        return {
            "commission": commission,
            "slippage": slippage,
            "market_impact": market_impact,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "cost_pct": total_cost / trade_value if trade_value > 0 else 0,
        }


@dataclass
class TradeRecord:
    """Record of a single trade for detailed analysis."""

    timestamp: datetime
    symbol: str
    side: str  # buy, sell, short, cover
    quantity: float
    price: float
    trade_value: float
    costs: dict[str, float]
    portfolio_value: float
    cash: float
    positions: dict[str, float]

    # Performance tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "trade_value": self.trade_value,
            "commission": self.costs.get("commission", 0.0),
            "slippage": self.costs.get("slippage", 0.0),
            "market_impact": self.costs.get("market_impact", 0.0),
            "spread_cost": self.costs.get("spread_cost", 0.0),
            "total_cost": self.costs.get("total_cost", 0.0),
            "cost_pct": self.costs.get("cost_pct", 0.0),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "cumulative_return": self.cumulative_return,
        }


@dataclass
class BacktestResult:
    """Comprehensive backtest results with detailed analysis."""

    # Basic performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    profit_factor: float
    win_rate: float

    # Risk metrics
    volatility: float
    beta: float
    information_ratio: float
    tracking_error: float

    # Trading statistics
    num_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Cost analysis
    total_commission: float
    total_slippage: float
    total_market_impact: float
    total_spread_cost: float
    total_transaction_costs: float
    cost_drag: float  # Cost impact on returns

    # Portfolio evolution
    equity_curve: pd.Series
    returns_series: pd.Series
    drawdown_series: pd.Series

    # Trade history
    trades: list[TradeRecord]

    # Configuration
    config: BacktestConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "expected_shortfall": self.expected_shortfall,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "volatility": self.volatility,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
            "num_trades": self.num_trades,
            "avg_trade_return": self.avg_trade_return,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_market_impact": self.total_market_impact,
            "total_spread_cost": self.total_spread_cost,
            "total_transaction_costs": self.total_transaction_costs,
            "cost_drag": self.cost_drag,
        }


class BacktestEvaluator:
    """
    Unified backtesting evaluator that integrates with model evaluation pipeline.

    Features:
    - Realistic transaction cost modeling
    - Slippage and market impact simulation
    - Detailed trade-by-trade analysis
    - Performance attribution reports
    - Risk-adjusted performance metrics
    - Automated model comparison workflows
    """

    def __init__(
        self,
        config: BacktestConfig,
        transaction_cost_model: TransactionCostModel | None = None,
    ):
        """
        Initialize the backtest evaluator.

        Args:
            config: Backtesting configuration
            transaction_cost_model: Optional custom transaction cost model
        """
        self.config = config
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel(
            commission_rate=config.commission_rate,
            slippage_rate=config.slippage_rate,
        )

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.statistical_tests = StatisticalTests()

        # Portfolio manager
        portfolio_config = PortfolioConfig(
            max_position_size=config.max_position_size,
            max_leverage=config.max_leverage,
            commission_rate=config.commission_rate,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
        )
        self.portfolio_manager = PortfolioManager(
            initial_capital=config.initial_capital,
            config=portfolio_config,
        )

        # Results storage
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self.returns_series: list[float] = []

        logger.info(f"Initialized BacktestEvaluator with {config.initial_capital} initial capital")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_signals: pd.Series,
        benchmark_returns: pd.Series | None = None,
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """
        Run a comprehensive backtest with detailed analysis.

        Args:
            data: Historical price data with OHLCV columns
            strategy_signals: Series of trading signals (1=long, -1=short, 0=hold)
            benchmark_returns: Optional benchmark returns for comparison
            strategy_name: Name of the strategy for reporting

        Returns:
            Comprehensive backtest results
        """
        console.print(f"[bold blue]Running backtest for {strategy_name}...[/bold blue]")

        # Reset portfolio and results
        self._reset_backtest()

        # Validate inputs
        self._validate_inputs(data, strategy_signals)

        # Run the backtest
        self._execute_backtest(data, strategy_signals)

        # Calculate comprehensive results
        results = self._calculate_results(benchmark_returns, strategy_name)

        console.print(f"[bold green]✅ Backtest complete for {strategy_name}[/bold green]")
        return results

    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Compare multiple strategies using the same backtest framework.

        Args:
            data: Historical price data
            strategies: Dictionary of {strategy_name: signals} pairs
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary of backtest results for each strategy
        """
        console.print("[bold blue]Comparing strategies...[/bold blue]")

        results = {}
        for strategy_name, signals in strategies.items():
            result = self.run_backtest(data, signals, benchmark_returns, strategy_name)
            results[strategy_name] = result

        # Generate comparison summary
        self._print_comparison_summary(results)

        return results

    def generate_performance_report(
        self,
        results: BacktestResult,
        output_path: Path | None = None,
    ) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            results: Backtest results
            output_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report = self._create_performance_report(results)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            console.print(f"[blue]Report saved to {output_path}[/blue]")

        return report

    def _reset_backtest(self) -> None:
        """Reset backtest state for a new run."""
        self.trades.clear()
        self.equity_curve.clear()
        self.returns_series.clear()

        # Reset portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config.initial_capital,
            config=self.portfolio_manager.config,
        )

    def _validate_inputs(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """Validate input data and signals."""
        if len(data) != len(signals):
            raise ValueError("Data and signals must have the same length")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if signals.dtype not in [np.int64, np.float64]:
            raise ValueError("Signals must be numeric")

    def _execute_backtest(self, data: pd.DataFrame, signals: pd.Series) -> None:
        """Execute the backtest simulation."""
        initial_value = self.config.initial_capital
        current_value = initial_value

        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = row["close"]
            volume = row["volume"]

            # Update portfolio with current prices
            self.portfolio_manager.update_prices({"symbol": price})

            # Execute trades based on signals
            if signal != 0:
                self._execute_signal(signal, price, volume, timestamp)

            # Record portfolio state
            current_value = self.portfolio_manager.total_value
            self.equity_curve.append(current_value)

            # Calculate period return
            if i > 0:
                period_return = (current_value - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
                self.returns_series.append(period_return)
            else:
                self.returns_series.append(0.0)

    def _execute_signal(
        self,
        signal: float,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> None:
        """Execute a trading signal with realistic costs."""
        # Determine trade direction and size
        if signal > 0:  # Long position
            side = "buy"
            quantity = self._calculate_position_size(price, signal)
        else:  # Short position
            side = "sell"
            quantity = -self._calculate_position_size(price, abs(signal))

        if quantity == 0:
            return

        # Calculate transaction costs
        trade_value = abs(quantity * price)
        costs = self.transaction_cost_model.calculate_total_cost(
            trade_value=trade_value,
            trade_volume=abs(quantity),
            avg_daily_volume=volume,
        )

        # Execute trade
        success = self.portfolio_manager.execute_trade(
            symbol="symbol",  # Using generic symbol for single-asset backtest
            quantity=quantity,
            price=price,
            side=side,
        )

        if success:
            # Record trade
            trade_record = TradeRecord(
                timestamp=timestamp,
                symbol="symbol",
                side=side,
                quantity=abs(quantity),
                price=price,
                trade_value=trade_value,
                costs=costs,
                portfolio_value=self.portfolio_manager.total_value,
                cash=self.portfolio_manager.cash,
                positions={k: v.market_value for k, v in self.portfolio_manager.positions.items()},
            )
            self.trades.append(trade_record)

    def _calculate_position_size(self, price: float, signal_strength: float) -> float:
        """Calculate position size based on signal strength and risk management."""
        available_capital = self.portfolio_manager.cash
        max_position_value = available_capital * self.config.max_position_size

        # Scale position by signal strength
        position_value = max_position_value * min(abs(signal_strength), 1.0)

        return position_value / price if price > 0 else 0.0

    def _calculate_results(
        self,
        benchmark_returns: pd.Series | None,
        strategy_name: str,
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        # Convert to pandas series
        returns_series = pd.Series(self.returns_series)
        equity_series = pd.Series(self.equity_curve)

        # Calculate basic metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]

        # Risk-adjusted metrics
        sharpe_ratio = calculate_sharpe_ratio(returns_series)
        sortino_ratio = calculate_sortino_ratio(returns_series)
        calmar_ratio = calculate_calmar_ratio(returns_series)
        max_drawdown = calculate_max_drawdown(returns_series)
        var_95 = calculate_var(returns_series, 0.95)
        expected_shortfall = calculate_expected_shortfall(returns_series, 0.95)

        # Trading metrics
        profit_factor = calculate_profit_factor(returns_series)
        win_rate = calculate_win_rate(returns_series)

        # Risk metrics
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        beta = calculate_beta(returns_series, benchmark_returns) if benchmark_returns is not None else 0.0
        information_ratio = (
            calculate_information_ratio(returns_series, benchmark_returns) if benchmark_returns is not None else 0.0
        )
        tracking_error = (returns_series - benchmark_returns).std() if benchmark_returns is not None else 0.0

        # Trading statistics
        num_trades = len(self.trades)
        avg_trade_return = returns_series.mean() if len(returns_series) > 0 else 0.0

        # Win/loss analysis
        positive_returns = returns_series[returns_series > 0]
        negative_returns = returns_series[returns_series < 0]

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0
        largest_win = returns_series.max() if len(returns_series) > 0 else 0.0
        largest_loss = returns_series.min() if len(returns_series) > 0 else 0.0

        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(returns_series > 0)
        max_consecutive_losses = self._calculate_max_consecutive(returns_series < 0)

        # Cost analysis
        total_commission = sum(trade.costs.get("commission", 0) for trade in self.trades)
        total_slippage = sum(trade.costs.get("slippage", 0) for trade in self.trades)
        total_market_impact = sum(trade.costs.get("market_impact", 0) for trade in self.trades)
        total_spread_cost = sum(trade.costs.get("spread_cost", 0) for trade in self.trades)
        total_transaction_costs = sum(trade.costs.get("total_cost", 0) for trade in self.trades)

        # Cost drag on returns
        cost_drag = total_transaction_costs / self.config.initial_capital if self.config.initial_capital > 0 else 0.0

        # Create drawdown series
        drawdown_series = self._calculate_drawdown_series(equity_series)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            profit_factor=profit_factor,
            win_rate=win_rate,
            volatility=volatility,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_market_impact=total_market_impact,
            total_spread_cost=total_spread_cost,
            total_transaction_costs=total_transaction_costs,
            cost_drag=cost_drag,
            equity_curve=equity_series,
            returns_series=returns_series,
            drawdown_series=drawdown_series,
            trades=self.trades,
            config=self.config,
        )

    def _calculate_max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive occurrences of a condition."""
        if len(condition) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for value in condition:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity_series.expanding().max()
        return (equity_series - peak) / peak

    def _print_comparison_summary(self, results: dict[str, BacktestResult]) -> None:
        """Print comparison summary table."""
        table = Table(title="Strategy Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Total Return", style="green")
        table.add_column("Sharpe Ratio", style="blue")
        table.add_column("Max Drawdown", style="red")
        table.add_column("Win Rate", style="yellow")
        table.add_column("Num Trades", style="magenta")

        for strategy_name, result in results.items():
            table.add_row(
                strategy_name,
                f"{result.total_return:.2%}",
                f"{result.sharpe_ratio:.2f}",
                f"{result.max_drawdown:.2%}",
                f"{result.win_rate:.2%}",
                str(result.num_trades),
            )

        console.print(table)

    def _create_performance_report(self, results: BacktestResult) -> str:
        """Create a comprehensive performance report."""
        return f"""
# Backtest Performance Report

## Strategy Overview
- **Total Return**: {results.total_return:.2%}
- **Sharpe Ratio**: {results.sharpe_ratio:.2f}
- **Sortino Ratio**: {results.sortino_ratio:.2f}
- **Calmar Ratio**: {results.calmar_ratio:.2f}
- **Maximum Drawdown**: {results.max_drawdown:.2%}

## Risk Metrics
- **Volatility (Annualized)**: {results.volatility:.2%}
- **Value at Risk (95%)**: {results.var_95:.2%}
- **Expected Shortfall (95%)**: {results.expected_shortfall:.2%}
- **Beta**: {results.beta:.2f}
- **Information Ratio**: {results.information_ratio:.2f}

## Trading Statistics
- **Number of Trades**: {results.num_trades}
- **Win Rate**: {results.win_rate:.2%}
- **Profit Factor**: {results.profit_factor:.2f}
- **Average Win**: {results.avg_win:.2%}
- **Average Loss**: {results.avg_loss:.2%}
- **Largest Win**: {results.largest_win:.2%}
- **Largest Loss**: {results.largest_loss:.2%}
- **Max Consecutive Wins**: {results.max_consecutive_wins}
- **Max Consecutive Losses**: {results.max_consecutive_losses}

## Transaction Cost Analysis
- **Total Commission**: ${results.total_commission:.2f}
- **Total Slippage**: ${results.total_slippage:.2f}
- **Total Market Impact**: ${results.total_market_impact:.2f}
- **Total Spread Cost**: ${results.total_spread_cost:.2f}
- **Total Transaction Costs**: ${results.total_transaction_costs:.2f}
- **Cost Drag on Returns**: {results.cost_drag:.2%}

## Configuration
- **Initial Capital**: ${results.config.initial_capital:,.2f}
- **Commission Rate**: {results.config.commission_rate:.3%}
- **Slippage Rate**: {results.config.slippage_rate:.3%}
- **Max Position Size**: {results.config.max_position_size:.1%}
- **Max Leverage**: {results.config.max_leverage:.1f}
"""
