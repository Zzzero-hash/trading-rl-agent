"""
Alpaca Markets Integration

Comprehensive integration with Alpaca Markets for real-time data streaming,
paper trading order execution, and portfolio monitoring.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import GetOrdersRequest

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by Alpaca."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca Markets integration."""

    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    use_v2_api: bool = True
    paper_trading: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    websocket_timeout: int = 30
    order_timeout: int = 60
    cache_dir: str = "data/alpaca_cache"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key are required")

        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class MarketData:
    """Market data structure for real-time updates."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


@dataclass
class OrderRequest:
    """Order request structure."""

    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType
    time_in_force: str = "day"
    limit_price: float | None = None
    stop_price: float | None = None
    client_order_id: str | None = None
    extended_hours: bool = False


@dataclass
class PortfolioPosition:
    """Portfolio position structure."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str
    timestamp: datetime


class AlpacaError(Exception):
    """Base exception for Alpaca integration errors."""


class AlpacaConnectionError(AlpacaError):
    """Raised when connection to Alpaca fails."""


class AlpacaOrderError(AlpacaError):
    """Raised when order execution fails."""


class AlpacaDataError(AlpacaError):
    """Raised when data retrieval fails."""


class AlpacaIntegration:
    """
    Comprehensive Alpaca Markets integration for real-time data and paper trading.

    Features:
    - Real-time market data streaming
    - Paper trading order execution
    - Portfolio monitoring
    - Configuration management
    - Error handling and retry logic
    """

    def __init__(self, config: AlpacaConfig):
        """
        Initialize Alpaca integration.

        Args:
            config: Alpaca configuration object
        """
        self.config = config
        self._init_clients()
        self._stream: StockDataStream | None = None
        self._stream_connected = False
        self._data_callbacks: list[Callable] = []
        self._order_callbacks: list[Callable] = []
        self._portfolio_callbacks: list[Callable] = []
        self._last_portfolio_update: datetime | None = None
        self._portfolio_cache: dict[str, Any] = {}

        logger.info(f"Alpaca integration initialized (Paper trading: {config.paper_trading})")

    def _init_clients(self) -> None:
        """Initialize Alpaca API clients."""
        try:
            self.trading_client = TradingClient(
                self.config.api_key,
                self.config.secret_key,
                paper=self.config.paper_trading,
            )
            self.data_client = StockHistoricalDataClient(self.config.api_key, self.config.secret_key)
            logger.info("Using Alpaca V2 SDK")
        except Exception as e:
            raise AlpacaConnectionError(f"Failed to initialize Alpaca clients: {e}") from e

    def validate_connection(self) -> bool:
        """
        Validate connection to Alpaca Markets.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            account = self.trading_client.get_account()
            logger.info(f"Alpaca connection validated. Account: {account.id}")
            return True
        except APIError as e:
            logger.exception(f"Alpaca connection validation failed: {e}")
            return False

    def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary containing account details
        """
        try:
            account = self.trading_client.get_account()
            return account.dict()  # type: ignore
        except APIError as e:
            logger.exception(f"Failed to get account info: {e}")
            raise AlpacaConnectionError(f"Failed to get account info: {e}") from e

    def get_historical_data(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Get historical market data.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data frequency ('1Min', '5Min', '15Min', '30Min', '1Hour', '1Day')
            adjustment: Price adjustment ('raw', 'split', 'dividend', 'all')

        Returns:
            DataFrame with historical OHLCV data
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        timeframe_mapping = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe_mapping.get(timeframe, TimeFrame.Day),
                start=pd.to_datetime(start_date),
                end=pd.to_datetime(end_date),
            )
            bars = self.data_client.get_stock_bars(request_params).df
            if not bars.empty:
                bars.reset_index(inplace=True)
                bars.rename(
                    columns={"timestamp": "date", "volume": "volume"},
                    inplace=True,
                )
                from .utils import normalize_timestamps

                bars = normalize_timestamps(bars, timestamp_column="date")
                return bars.sort_values(["symbol", "date"]).reset_index(drop=True)
            return pd.DataFrame()
        except APIError as e:
            logger.exception(f"Historical data retrieval error: {e}")
            raise AlpacaDataError(f"Failed to get historical data: {e}") from e

    def _normalize_timestamp_column(self, timestamp_col: pd.Series, timezone: str = "America/New_York") -> pd.Series:
        """
        Normalize timestamp column to consistent timezone format.

        Args:
            timestamp_col: Series of timestamps
            timezone: Target timezone for normalization

        Returns:
            Normalized timestamp series
        """
        from .utils import _normalize_timestamp_series
        return _normalize_timestamp_series(timestamp_col, timezone)

    def get_real_time_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get real-time quotes for symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary with real-time quote data
        """
        try:
            quotes = self.data_client.get_latest_stock_quotes(symbols)
            return {
                symbol: {
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                    "spread": float(quote.ask_price - quote.bid_price),
                    "spread_pct": float((quote.ask_price - quote.bid_price) / quote.bid_price * 100),
                }
                for symbol, quote in quotes.items()
            }
        except APIError as e:
            logger.exception(f"Real-time quotes error: {e}")
            raise AlpacaDataError(f"Failed to get real-time quotes: {e}") from e

    def place_order(self, order_request: OrderRequest) -> dict[str, Any]:
        """
        Place a trading order.

        Args:
            order_request: Order request object

        Returns:
            Dictionary with order details
        """
        try:
            # Prepare order parameters
            order_params = {
                "symbol": order_request.symbol,
                "qty": order_request.qty,
                "side": AlpacaOrderSide(order_request.side.value),
                "type": AlpacaOrderType(order_request.order_type.value),
                "time_in_force": TimeInForce(order_request.time_in_force),
                "extended_hours": order_request.extended_hours,
                "client_order_id": order_request.client_order_id,
            }
            if order_request.limit_price:
                order_params["limit_price"] = order_request.limit_price
            if order_request.stop_price:
                order_params["stop_price"] = order_request.stop_price

            # Place order with retry logic
            last_exception = None
            for attempt in range(self.config.max_retries):
                try:
                    order = self.trading_client.submit_order(order_data=order_params)
                    # Wait for order to be processed
                    order = self._wait_for_order_fill(order.id)
                    return order.dict()  # type: ignore
                except APIError as e:
                    last_exception = e
                    if attempt == self.config.max_retries - 1:
                        raise last_exception from e
                    logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config.retry_delay)

            # This should never be reached, but mypy needs it
            raise AlpacaOrderError("All order attempts failed")

        except Exception as e:
            logger.exception(f"Order placement error: {e}")
            raise AlpacaOrderError(f"Failed to place order: {e}") from e

    def _wait_for_order_fill(self, order_id: str, timeout: int | None = None) -> Order:
        """
        Wait for order to be filled.

        Args:
            order_id: Order ID to monitor
            timeout: Timeout in seconds (uses config default if None)

        Returns:
            Updated order object
        """
        timeout = timeout or self.config.order_timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                if order.status in ["filled", "canceled", "rejected"]:
                    return order
                time.sleep(1)
            except APIError as e:
                logger.warning(f"Error checking order status: {e}")
                time.sleep(1)

        raise AlpacaOrderError(f"Order {order_id} did not fill within {timeout} seconds")

    def get_positions(self) -> list[PortfolioPosition]:
        """
        Get current portfolio positions.

        Returns:
            List of portfolio positions
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                PortfolioPosition(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_plpc=float(pos.unrealized_plpc),
                    side=pos.side,
                    timestamp=datetime.now(),
                )
                for pos in positions
            ]
        except APIError as e:
            logger.exception(f"Failed to get positions: {e}")
            raise AlpacaDataError(f"Failed to get positions: {e}") from e

    def get_portfolio_value(self) -> dict[str, float]:
        """
        Get current portfolio value and performance metrics.

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            account = self.trading_client.get_account()
            positions = self.get_positions()
            total_unrealized_pl = sum(pos.unrealized_pl for pos in positions)
            total_market_value = sum(pos.market_value for pos in positions)

            return {
                "total_equity": float(account.equity),
                "total_market_value": total_market_value,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "total_unrealized_pl": total_unrealized_pl,
                "total_unrealized_pl_pct": (
                    (total_unrealized_pl / float(account.equity)) * 100 if float(account.equity) > 0 else 0
                ),
                "day_trade_count": account.daytrade_count,
                "position_count": len(positions),
            }

        except APIError as e:
            logger.exception(f"Failed to get portfolio value: {e}")
            raise AlpacaDataError(f"Failed to get portfolio value: {e}") from e

    def start_data_stream(self, symbols: list[str], callback: Callable | None = None) -> None:
        """
        Start real-time data streaming.

        Args:
            symbols: List of symbols to stream
            callback: Optional callback function for data updates
        """
        if callback:
            self._data_callbacks.append(callback)

        try:
            self._stream = StockDataStream(self.config.api_key, self.config.secret_key)

            # Subscribe to trade updates
            async def trade_handler(trade: Any) -> None:
                await self._handle_trade_update(trade)

            async def bar_handler(bar: Any) -> None:
                await self._handle_bar_update(bar)

            self._stream.subscribe_trades(trade_handler, *symbols)
            self._stream.subscribe_bars(bar_handler, *symbols)
            self._stream.run()
            self._stream_connected = True
            logger.info(f"Started data stream for {len(symbols)} symbols")

        except Exception as e:
            logger.exception(f"Failed to start data stream: {e}")
            raise AlpacaConnectionError(f"Failed to start data stream: {e}") from e

    def stop_data_stream(self) -> None:
        """Stop real-time data streaming."""
        if self._stream is not None and self._stream_connected:
            try:
                self._stream.stop()
                self._stream_connected = False
                logger.info("Stopped data stream")
            except Exception as e:
                logger.warning(f"Error stopping data stream: {e}")

    async def _handle_trade_update(self, trade: Any) -> None:
        """Handle real-time trade updates."""
        try:
            trade_data = {
                "symbol": trade.symbol,
                "price": float(trade.price),
                "size": int(trade.size),
                "timestamp": trade.timestamp,
                "exchange": trade.exchange,
                "id": trade.id,
            }

            # Notify callbacks
            for callback in self._data_callbacks:
                try:
                    callback("trade", trade_data)
                except Exception as e:
                    logger.exception(f"Error in trade callback: {e}")

        except Exception as e:
            logger.exception(f"Error handling trade update: {e}")

    async def _handle_bar_update(self, bar: Any) -> None:
        """Handle real-time bar updates."""
        try:
            bar_data = MarketData(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, "vwap") else None,
                trade_count=(int(bar.trade_count) if hasattr(bar, "trade_count") else None),
            )

            # Notify callbacks
            for callback in self._data_callbacks:
                try:
                    callback("bar", bar_data)
                except Exception as e:
                    logger.exception(f"Error in bar callback: {e}")

        except Exception as e:
            logger.exception(f"Error handling bar update: {e}")

    def add_data_callback(self, callback: Callable) -> None:
        """Add a callback function for data updates."""
        self._data_callbacks.append(callback)

    def add_order_callback(self, callback: Callable) -> None:
        """Add a callback function for order updates."""
        self._order_callbacks.append(callback)

    def add_portfolio_callback(self, callback: Callable) -> None:
        """Add a callback function for portfolio updates."""
        self._portfolio_callbacks.append(callback)

    def get_order_history(
        self,
        status: str | None = None,
        limit: int = 500,
        after: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get order history.

        Args:
            status: Filter by order status
            limit: Maximum number of orders to return
            after: Filter orders after this date
            until: Filter orders until this date

        Returns:
            List of order dictionaries
        """
        try:
            request = GetOrdersRequest(status=status, limit=limit, after=after, until=until)
            orders = self.trading_client.get_orders(filter=request)
            return [order.dict() for order in orders]
        except APIError as e:
            logger.exception(f"Failed to get order history: {e}")
            raise AlpacaDataError(f"Failed to get order history: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if order was canceled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order {order_id}")
            return True
        except APIError as e:
            logger.exception(f"Failed to cancel order {order_id}: {e}")
            raise AlpacaOrderError(f"Failed to cancel order: {e}") from e

    def cancel_all_orders(self) -> list[str]:
        """
        Cancel all open orders.

        Returns:
            List of canceled order IDs
        """
        try:
            canceled_orders = self.trading_client.cancel_orders()
            order_ids = [order.id for order in canceled_orders]
            logger.info(f"Canceled {len(order_ids)} orders")
            return order_ids
        except APIError as e:
            logger.exception(f"Failed to cancel all orders: {e}")
            raise AlpacaOrderError(f"Failed to cancel all orders: {e}") from e

    def get_asset_info(self, symbol: str) -> dict[str, Any]:
        """
        Get asset information.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with asset information
        """
        try:
            asset = self.trading_client.get_asset(symbol)
            return asset.dict()  # type: ignore
        except APIError as e:
            logger.exception(f"Failed to get asset info for {symbol}: {e}")
            raise AlpacaDataError(f"Failed to get asset info: {e}") from e

    def __enter__(self) -> "AlpacaIntegration":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - cleanup resources."""
        self.stop_data_stream()


def create_alpaca_config_from_env() -> AlpacaConfig:
    """
    Create Alpaca configuration from environment variables.

    Returns:
        AlpacaConfig object
    """
    # Try to get configuration from the unified config system first
    try:
        from trade_agent.core.unified_config import UnifiedConfig

        config = UnifiedConfig()
        if config.alpaca_api_key and config.alpaca_secret_key:
            return AlpacaConfig(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                base_url=config.alpaca_base_url or "https://paper-api.alpaca.markets",
                data_url=config.alpaca_data_url or "https://data.alpaca.markets",
                use_v2_api=config.alpaca_use_v2,
                paper_trading=config.alpaca_paper_trading,
                max_retries=config.alpaca_max_retries,
                retry_delay=config.alpaca_retry_delay,
                websocket_timeout=config.alpaca_websocket_timeout,
                order_timeout=config.alpaca_order_timeout,
                cache_dir=config.alpaca_cache_dir or "data/alpaca_cache",
            )
    except Exception:
        pass

    # Fallback to direct environment variable access
    return AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        use_v2_api=os.getenv("ALPACA_USE_V2", "true").lower() == "true",
        paper_trading=os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true",
        max_retries=int(os.getenv("ALPACA_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("ALPACA_RETRY_DELAY", "1.0")),
        websocket_timeout=int(os.getenv("ALPACA_WEBSOCKET_TIMEOUT", "30")),
        order_timeout=int(os.getenv("ALPACA_ORDER_TIMEOUT", "60")),
        cache_dir=os.getenv("ALPACA_CACHE_DIR", "data/alpaca_cache"),
    )
