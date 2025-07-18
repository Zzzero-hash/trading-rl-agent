#!/usr/bin/env python3
"""
Alpaca Integration Demo

Comprehensive demonstration of the AlpacaIntegration class features:
- Real-time data streaming
- Paper trading order execution
- Portfolio monitoring
- Configuration management
- Error handling and retry logic

Usage:
    python alpaca_integration_demo.py [--config CONFIG_FILE] [--symbols SYMBOLS]
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_rl_agent.data.alpaca_integration import (
    AlpacaIntegration,
    AlpacaConfig,
    OrderRequest,
    OrderType,
    OrderSide,
    MarketData,
    create_alpaca_config_from_env
)
from trading_rl_agent.configs.alpaca_config import (
    get_alpaca_config,
    validate_alpaca_environment,
    AlpacaConfigManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlpacaDemo:
    """Demo class for showcasing Alpaca integration features."""
    
    def __init__(self, config_file: str = None, symbols: List[str] = None):
        """
        Initialize the demo.
        
        Args:
            config_file: Optional path to configuration file
            symbols: List of symbols to trade/monitor
        """
        self.config_file = config_file
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA"]
        self.alpaca = None
        self.running = False
        
        # Demo state
        self.trade_count = 0
        self.data_updates = 0
        self.portfolio_updates = 0
    
    def setup_configuration(self) -> AlpacaConfig:
        """Set up Alpaca configuration."""
        logger.info("Setting up Alpaca configuration...")
        
        try:
            if self.config_file:
                # Load from file
                config_manager = AlpacaConfigManager()
                config_model = config_manager.load_config(self.config_file)
                
                # Convert to AlpacaConfig
                config = AlpacaConfig(
                    api_key=config_model.api_key,
                    secret_key=config_model.secret_key,
                    base_url=config_model.base_url,
                    data_url=config_model.data_url,
                    use_v2_api=config_model.use_v2_api,
                    paper_trading=config_model.paper_trading,
                    max_retries=config_model.max_retries,
                    retry_delay=config_model.retry_delay,
                    websocket_timeout=config_model.websocket_timeout,
                    order_timeout=config_model.order_timeout,
                    cache_dir=config_model.cache_dir
                )
            else:
                # Load from environment
                if not validate_alpaca_environment():
                    logger.error("Alpaca environment not properly configured")
                    logger.info("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
                    sys.exit(1)
                
                config = create_alpaca_config_from_env()
            
            logger.info(f"Configuration loaded successfully (Paper trading: {config.paper_trading})")
            return config
            
        except Exception as e:
            logger.error(f"Failed to setup configuration: {e}")
            raise
    
    def initialize_alpaca(self, config: AlpacaConfig):
        """Initialize Alpaca integration."""
        logger.info("Initializing Alpaca integration...")
        
        try:
            self.alpaca = AlpacaIntegration(config)
            
            # Validate connection
            if not self.alpaca.validate_connection():
                raise Exception("Failed to validate Alpaca connection")
            
            logger.info("Alpaca integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca integration: {e}")
            raise
    
    def demo_account_info(self):
        """Demonstrate account information retrieval."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Account Information")
        logger.info("="*50)
        
        try:
            account_info = self.alpaca.get_account_info()
            
            logger.info("Account Details:")
            logger.info(f"  Account ID: {account_info['id']}")
            logger.info(f"  Status: {account_info['status']}")
            logger.info(f"  Currency: {account_info['currency']}")
            logger.info(f"  Cash: ${account_info['cash']:,.2f}")
            logger.info(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            logger.info(f"  Buying Power: ${account_info['buying_power']:,.2f}")
            logger.info(f"  Pattern Day Trader: {account_info['pattern_day_trader']}")
            logger.info(f"  Day Trade Count: {account_info['daytrade_count']}")
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
    
    def demo_historical_data(self):
        """Demonstrate historical data retrieval."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Historical Data Retrieval")
        logger.info("="*50)
        
        try:
            # Get last 30 days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            logger.info(f"Fetching historical data for {self.symbols} from {start_date.date()} to {end_date.date()}")
            
            data = self.alpaca.get_historical_data(
                symbols=self.symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1Day"
            )
            
            if not data.empty:
                logger.info(f"Retrieved {len(data)} data points")
                logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
                logger.info(f"Symbols: {data['symbol'].unique().tolist()}")
                
                # Show sample data
                logger.info("\nSample data:")
                print(data.head(10).to_string(index=False))
            else:
                logger.warning("No historical data retrieved")
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
    
    def demo_real_time_quotes(self):
        """Demonstrate real-time quotes."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Real-time Quotes")
        logger.info("="*50)
        
        try:
            logger.info(f"Fetching real-time quotes for {self.symbols}")
            
            quotes = self.alpaca.get_real_time_quotes(self.symbols)
            
            for symbol, quote in quotes.items():
                logger.info(f"\n{symbol}:")
                logger.info(f"  Bid: ${quote['bid_price']:.2f} (Size: {quote['bid_size']})")
                logger.info(f"  Ask: ${quote['ask_price']:.2f} (Size: {quote['ask_size']})")
                logger.info(f"  Spread: ${quote['spread']:.2f} ({quote['spread_pct']:.2f}%)")
                logger.info(f"  Timestamp: {quote['timestamp']}")
                
        except Exception as e:
            logger.error(f"Failed to get real-time quotes: {e}")
    
    def demo_portfolio_monitoring(self):
        """Demonstrate portfolio monitoring."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Portfolio Monitoring")
        logger.info("="*50)
        
        try:
            # Get portfolio value
            portfolio_value = self.alpaca.get_portfolio_value()
            
            logger.info("Portfolio Summary:")
            logger.info(f"  Total Equity: ${portfolio_value['total_equity']:,.2f}")
            logger.info(f"  Total Market Value: ${portfolio_value['total_market_value']:,.2f}")
            logger.info(f"  Cash: ${portfolio_value['cash']:,.2f}")
            logger.info(f"  Buying Power: ${portfolio_value['buying_power']:,.2f}")
            logger.info(f"  Unrealized P&L: ${portfolio_value['total_unrealized_pl']:,.2f}")
            logger.info(f"  Unrealized P&L %: {portfolio_value['total_unrealized_pl_pct']:.2f}%")
            logger.info(f"  Position Count: {portfolio_value['position_count']}")
            logger.info(f"  Day Trade Count: {portfolio_value['day_trade_count']}")
            
            # Get positions
            positions = self.alpaca.get_positions()
            
            if positions:
                logger.info("\nCurrent Positions:")
                for position in positions:
                    logger.info(f"  {position.symbol}:")
                    logger.info(f"    Quantity: {position.qty}")
                    logger.info(f"    Avg Entry: ${position.avg_entry_price:.2f}")
                    logger.info(f"    Current Price: ${position.current_price:.2f}")
                    logger.info(f"    Market Value: ${position.market_value:,.2f}")
                    logger.info(f"    Unrealized P&L: ${position.unrealized_pl:,.2f} ({position.unrealized_plpc:.2f}%)")
                    logger.info(f"    Side: {position.side}")
            else:
                logger.info("No current positions")
                
        except Exception as e:
            logger.error(f"Failed to get portfolio information: {e}")
    
    def demo_paper_trading(self):
        """Demonstrate paper trading order execution."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Paper Trading Order Execution")
        logger.info("="*50)
        
        if not self.alpaca.config.paper_trading:
            logger.warning("Paper trading is disabled. Skipping order execution demo.")
            return
        
        try:
            # Get current quotes to determine order prices
            quotes = self.alpaca.get_real_time_quotes([self.symbols[0]])
            if not quotes:
                logger.warning("No quotes available for order execution demo")
                return
            
            symbol = self.symbols[0]
            quote = quotes[symbol]
            
            # Create a small market order (1 share)
            order_request = OrderRequest(
                symbol=symbol,
                qty=1.0,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                time_in_force="day",
                client_order_id=f"demo_order_{int(time.time())}"
            )
            
            logger.info(f"Placing market order for 1 share of {symbol}")
            logger.info(f"Current bid: ${quote['bid_price']:.2f}, ask: ${quote['ask_price']:.2f}")
            
            # Place the order
            order_result = self.alpaca.place_order(order_request)
            
            logger.info("Order executed successfully:")
            logger.info(f"  Order ID: {order_result['order_id']}")
            logger.info(f"  Symbol: {order_result['symbol']}")
            logger.info(f"  Quantity: {order_result['qty']}")
            logger.info(f"  Side: {order_result['side']}")
            logger.info(f"  Type: {order_result['type']}")
            logger.info(f"  Status: {order_result['status']}")
            logger.info(f"  Filled Quantity: {order_result['filled_qty']}")
            if order_result['filled_avg_price']:
                logger.info(f"  Filled Average Price: ${order_result['filled_avg_price']:.2f}")
            logger.info(f"  Submitted At: {order_result['submitted_at']}")
            
            self.trade_count += 1
            
            # Wait a moment, then place a sell order
            time.sleep(2)
            
            # Place a limit sell order slightly above market
            sell_price = quote['ask_price'] * 1.01  # 1% above ask
            
            sell_order_request = OrderRequest(
                symbol=symbol,
                qty=1.0,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                time_in_force="day",
                limit_price=sell_price,
                client_order_id=f"demo_sell_order_{int(time.time())}"
            )
            
            logger.info(f"\nPlacing limit sell order for 1 share of {symbol} at ${sell_price:.2f}")
            
            sell_order_result = self.alpaca.place_order(sell_order_request)
            
            logger.info("Sell order placed successfully:")
            logger.info(f"  Order ID: {sell_order_result['order_id']}")
            logger.info(f"  Limit Price: ${sell_order_result['limit_price']:.2f}")
            logger.info(f"  Status: {sell_order_result['status']}")
            
            self.trade_count += 1
            
        except Exception as e:
            logger.error(f"Failed to execute paper trading demo: {e}")
    
    def data_stream_callback(self, data_type: str, data: Any):
        """Callback for real-time data stream."""
        self.data_updates += 1
        
        if data_type == "bar":
            logger.info(f"BAR UPDATE [{self.data_updates}]: {data.symbol} - Close: ${data.close:.2f}, Volume: {data.volume}")
        elif data_type == "trade":
            logger.info(f"TRADE UPDATE [{self.data_updates}]: {data['symbol']} - Price: ${data['price']:.2f}, Size: {data['size']}")
    
    def demo_real_time_streaming(self, duration: int = 30):
        """Demonstrate real-time data streaming."""
        logger.info("\n" + "="*50)
        logger.info(f"DEMO: Real-time Data Streaming ({duration} seconds)")
        logger.info("="*50)
        
        try:
            logger.info(f"Starting real-time data stream for {self.symbols}")
            logger.info("Press Ctrl+C to stop streaming early")
            
            # Add callback
            self.alpaca.add_data_callback(self.data_stream_callback)
            
            # Start streaming
            self.alpaca.start_data_stream(self.symbols)
            
            # Monitor for specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(1)
                
                # Show periodic updates
                if int(time.time() - start_time) % 10 == 0:
                    logger.info(f"Streaming for {int(time.time() - start_time)}s, received {self.data_updates} updates")
            
            # Stop streaming
            self.alpaca.stop_data_stream()
            
            logger.info(f"Streaming completed. Received {self.data_updates} data updates.")
            
        except KeyboardInterrupt:
            logger.info("\nStreaming stopped by user")
            self.alpaca.stop_data_stream()
        except Exception as e:
            logger.error(f"Failed to start data streaming: {e}")
            self.alpaca.stop_data_stream()
    
    def demo_order_history(self):
        """Demonstrate order history retrieval."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Order History")
        logger.info("="*50)
        
        try:
            # Get recent orders
            orders = self.alpaca.get_order_history(limit=10)
            
            if orders:
                logger.info(f"Recent orders ({len(orders)}):")
                for order in orders:
                    logger.info(f"  {order['symbol']} - {order['side']} {order['qty']} @ {order['type']}")
                    logger.info(f"    Status: {order['status']}, ID: {order['order_id']}")
                    if order['filled_avg_price']:
                        logger.info(f"    Filled: {order['filled_qty']} @ ${order['filled_avg_price']:.2f}")
                    logger.info(f"    Submitted: {order['submitted_at']}")
                    logger.info("")
            else:
                logger.info("No recent orders found")
                
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
    
    def demo_asset_information(self):
        """Demonstrate asset information retrieval."""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Asset Information")
        logger.info("="*50)
        
        try:
            for symbol in self.symbols[:2]:  # Show first 2 symbols
                logger.info(f"\nAsset information for {symbol}:")
                
                asset_info = self.alpaca.get_asset_info(symbol)
                
                logger.info(f"  Name: {asset_info['name']}")
                logger.info(f"  Exchange: {asset_info['exchange']}")
                logger.info(f"  Class: {asset_info['class']}")
                logger.info(f"  Status: {asset_info['status']}")
                logger.info(f"  Tradable: {asset_info['tradable']}")
                logger.info(f"  Marginable: {asset_info['marginable']}")
                logger.info(f"  Shortable: {asset_info['shortable']}")
                logger.info(f"  Easy to Borrow: {asset_info['easy_to_borrow']}")
                logger.info(f"  Fractionable: {asset_info['fractionable']}")
                
                if asset_info['min_order_size']:
                    logger.info(f"  Min Order Size: {asset_info['min_order_size']}")
                if asset_info['price_increment']:
                    logger.info(f"  Price Increment: ${asset_info['price_increment']:.4f}")
                    
        except Exception as e:
            logger.error(f"Failed to get asset information: {e}")
    
    def run_full_demo(self):
        """Run the complete demo."""
        logger.info("Starting Alpaca Integration Demo")
        logger.info("="*60)
        
        try:
            # Setup and initialization
            config = self.setup_configuration()
            self.initialize_alpaca(config)
            
            # Run demo components
            self.demo_account_info()
            self.demo_historical_data()
            self.demo_real_time_quotes()
            self.demo_portfolio_monitoring()
            self.demo_asset_information()
            self.demo_paper_trading()
            self.demo_order_history()
            self.demo_real_time_streaming(duration=30)
            
            # Final summary
            logger.info("\n" + "="*60)
            logger.info("DEMO SUMMARY")
            logger.info("="*60)
            logger.info(f"  Trades executed: {self.trade_count}")
            logger.info(f"  Data updates received: {self.data_updates}")
            logger.info(f"  Portfolio updates: {self.portfolio_updates}")
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            if self.alpaca:
                self.alpaca.stop_data_stream()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpaca Integration Demo")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL", "TSLA"],
                       help="Symbols to trade/monitor")
    parser.add_argument("--stream-only", action="store_true",
                       help="Only run the real-time streaming demo")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration of streaming demo in seconds")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = AlpacaDemo(config_file=args.config, symbols=args.symbols)
    
    if args.stream_only:
        # Only run streaming demo
        config = demo.setup_configuration()
        demo.initialize_alpaca(config)
        demo.demo_real_time_streaming(duration=args.duration)
    else:
        # Run full demo
        demo.run_full_demo()


if __name__ == "__main__":
    main()