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

try:
    from alpaca_trade_api.entity import Order
    from alpaca_trade_api.rest import REST
    from alpaca_trade_api.stream import Stream

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    # Define fallback types for type hints
    Order = type("Order", (), {})
    REST = type("REST", (), {})
    Stream = type("Stream", (), {})
    logging.getLogger(__name__).warning(
        "Alpaca Trade API not available. Install with: pip install alpaca-trade-api",
    )

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide

    ALPACA_V2_AVAILABLE = True
except ImportError:
    ALPACA_V2_AVAILABLE = False
    # Define fallback types for type hints
    StockHistoricalDataClient = type("StockHistoricalDataClient", (), {})
    TradingClient = type("TradingClient", (), {})
    AlpacaOrderSide = type("AlpacaOrderSide", (), {})
    logging.getLogger(__name__).warning(
        "Alpaca V2 SDK not available. Install with: pip install alpaca-py",
    )

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
        self._stream: Any = None
        self._stream_connected = False
        self._data_callbacks: list[Callable] = []
        self._order_callbacks: list[Callable] = []
        self._portfolio_callbacks: list[Callable] = []
        self._last_portfolio_update: datetime | None = None
        self._portfolio_cache: dict[str, Any] = {}

        logger.info(f"Alpaca integration initialized (Paper trading: {config.paper_trading})")

    def _init_clients(self) -> None:
        """Initialize Alpaca API clients."""
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca Trade API required: pip install alpaca-trade-api")

        # Initialize V1 API client (fallback)
        try:
            self.rest_api = REST(
                self.config.api_key,
                self.config.secret_key,
                self.config.base_url,
                api_version="v2",
            )
        except TypeError:
            # Handle case where REST is a mock that doesn't accept arguments
            self.rest_api = REST()

        # Initialize V2 API clients if available
        if ALPACA_V2_AVAILABLE and self.config.use_v2_api:
            self.trading_client = TradingClient(
                self.config.api_key,
                self.config.secret_key,
                paper=self.config.paper_trading,
            )
            self.data_client = StockHistoricalDataClient(self.config.api_key, self.config.secret_key)
            logger.info("Using Alpaca V2 SDK")
        else:
            self.trading_client = None
            self.data_client = None
            logger.info("Using Alpaca V1 API")

    def validate_connection(self) -> bool:
        """
        Validate connection to Alpaca Markets.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            account = self.rest_api.get_account()
            logger.info(f"Alpaca connection validated. Account: {account.id}")
            return True
        except Exception as e:
            logger.exception(f"Alpaca connection validation failed: {e}")
            return False

    def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary containing account details
        """
        try:
            account = self.rest_api.get_account()
            return {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "regt_buying_power": float(account.regt_buying_power),
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at,
                "trade_suspended_by_user": account.trade_suspended_by_user,
                "multiplier": account.multiplier,
                "shorting_enabled": account.shorting_enabled,
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_maintenance_margin": float(account.last_maintenance_margin),
                "sma": float(account.sma),
                "daytrade_count": account.daytrade_count,
            }
        except Exception as e:
            logger.exception(f"Failed to get account info: {e}")
            raise AlpacaConnectionError(f"Failed to get account info: {e}") from e

    def get_historical_data(
        self,
        symbols: list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        timeframe: str = "1Day",
        adjustment: str = "raw",
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
        try:
            all_data = []

            for symbol in symbols:
                try:
                    bars = self.rest_api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date,
                        end=end_date,
                        adjustment=adjustment,
                    ).df

                    if not bars.empty:
                        bars["symbol"] = symbol
                        bars.reset_index(inplace=True)
                        bars.rename(
                            columns={"timestamp": "date", "volume": "volume"},
                            inplace=True,
                        )
                        all_data.append(bars)

                except Exception as e:
                    logger.warning(f"Failed to get historical data for {symbol}: {e}")
                    continue

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                # Handle timezone-aware datetime conversion consistently using utility function
                from .utils import normalize_timestamps
                combined_data = normalize_timestamps(combined_data, timestamp_column="date")
                return combined_data.sort_values(["symbol", "date"]).reset_index(drop=True)

            return pd.DataFrame()

        except Exception as e:
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
            quotes = {}
            for symbol in symbols:
                quote = self.rest_api.get_latest_quote(symbol)
                quotes[symbol] = {
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                    "spread": float(quote.ask_price - quote.bid_price),
                    "spread_pct": float((quote.ask_price - quote.bid_price) / quote.bid_price * 100),
                }
            return quotes
        except Exception as e:
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
                "side": order_request.side.value,
                "type": order_request.order_type.value,
                "time_in_force": order_request.time_in_force,
                "extended_hours": order_request.extended_hours,
            }

            # Add optional parameters
            if order_request.limit_price:
                order_params["limit_price"] = order_request.limit_price
            if order_request.stop_price:
                order_params["stop_price"] = order_request.stop_price
            if order_request.client_order_id:
                order_params["client_order_id"] = order_request.client_order_id

            # Place order with retry logic
            last_exception = None
            for attempt in range(self.config.max_retries):
                try:
                    order = self.rest_api.submit_order(**order_params)

                    # Wait for order to be processed
                    order = self._wait_for_order_fill(order.id)

                    return {
                        "order_id": order.id,
                        "client_order_id": order.client_order_id,
                        "symbol": order.symbol,
                        "qty": float(order.qty),
                        "side": order.side,
                        "type": order.type,
                        "status": order.status,
                        "filled_at": order.filled_at,
                        "filled_avg_price": (float(order.filled_avg_price) if order.filled_avg_price else None),
                        "filled_qty": float(order.filled_qty),
                        "submitted_at": order.submitted_at,
                        "limit_price": (float(order.limit_price) if order.limit_price else None),
                        "stop_price": (float(order.stop_price) if order.stop_price else None),
                    }

                except Exception as e:
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
                order = self.rest_api.get_order(order_id)
                if order.status in ["filled", "canceled", "rejected"]:
                    return order
                time.sleep(1)
            except Exception as e:
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
            positions = self.rest_api.list_positions()
            portfolio_positions = []

            for position in positions:
                portfolio_positions.append(
                    PortfolioPosition(
                        symbol=position.symbol,
                        qty=float(position.qty),
                        avg_entry_price=float(position.avg_entry_price),
                        current_price=float(position.current_price),
                        market_value=float(position.market_value),
                        unrealized_pl=float(position.unrealized_pl),
                        unrealized_plpc=float(position.unrealized_plpc),
                        side=position.side,
                        timestamp=datetime.now(),
                    )
                )

            return portfolio_positions

        except Exception as e:
            logger.exception(f"Failed to get positions: {e}")
            raise AlpacaDataError(f"Failed to get positions: {e}") from e

    def get_portfolio_value(self) -> dict[str, float]:
        """
        Get current portfolio value and performance metrics.

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            account = self.rest_api.get_account()
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

        except Exception as e:
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
            self._stream = Stream(
                self.config.api_key,
                self.config.secret_key,
                base_url=self.config.base_url,
                data_feed="iex",  # Use IEX for free data
            )

            # Subscribe to trade updates
            if self._stream is not None:
                self._stream.subscribe_trade_updates(self._handle_trade_update)

                # Subscribe to bar updates
                for symbol in symbols:
                    self._stream.subscribe_bars(self._handle_bar_update, symbol)

                # Start streaming in a separate thread
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

    def _handle_trade_update(self, trade: Any) -> None:
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

    def _handle_bar_update(self, bar: Any) -> None:
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
            orders = self.rest_api.list_orders(status=status, limit=limit, after=after, until=until)

            return [
                {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "type": order.type,
                    "status": order.status,
                    "filled_at": order.filled_at,
                    "filled_avg_price": (float(order.filled_avg_price) if order.filled_avg_price else None),
                    "filled_qty": float(order.filled_qty),
                    "submitted_at": order.submitted_at,
                    "limit_price": (float(order.limit_price) if order.limit_price else None),
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                }
                for order in orders
            ]

        except Exception as e:
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
            self.rest_api.cancel_order(order_id)
            logger.info(f"Canceled order {order_id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to cancel order {order_id}: {e}")
            raise AlpacaOrderError(f"Failed to cancel order: {e}") from e

    def cancel_all_orders(self) -> list[str]:
        """
        Cancel all open orders.

        Returns:
            List of canceled order IDs
        """
        try:
            canceled_orders = self.rest_api.cancel_all_orders()
            order_ids = [order.id for order in canceled_orders]
            logger.info(f"Canceled {len(order_ids)} orders")
            return order_ids
        except Exception as e:
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
            asset = self.rest_api.get_asset(symbol)
            return {
                "id": asset.id,
                "class": asset.asset_class,
                "exchange": asset.exchange,
                "symbol": asset.symbol,
                "name": asset.name,
                "status": asset.status,
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
                "min_order_size": (float(asset.min_order_size) if asset.min_order_size else None),
                "min_trade_increment": (float(asset.min_trade_increment) if asset.min_trade_increment else None),
                "price_increment": (float(asset.price_increment) if asset.price_increment else None),
            }
        except Exception as e:
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
