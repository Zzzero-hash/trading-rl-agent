# Alpaca Markets Integration

Comprehensive integration with Alpaca Markets for real-time data streaming, paper trading order execution, and portfolio monitoring.

## Features

- **Real-time Data Streaming**: Live market data via WebSocket connections
- **Paper Trading**: Risk-free order execution in a simulated environment
- **Portfolio Monitoring**: Real-time portfolio tracking and performance metrics
- **Configuration Management**: Flexible configuration via environment variables or YAML files
- **Error Handling**: Robust error handling with retry logic
- **Historical Data**: Efficient historical data retrieval with caching
- **Order Management**: Complete order lifecycle management
- **Asset Information**: Detailed asset metadata and trading rules

## Quick Start

### 1. Install Dependencies

```bash
pip install alpaca-trade-api alpaca-py websockets
```

### 2. Set Up Alpaca Account

1. Create an account at [Alpaca Markets](https://app.alpaca.markets/)
2. Get your API key and secret key from the dashboard
3. Enable paper trading for testing

### 3. Configure Environment Variables

```bash
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_PAPER_TRADING="true"
```

### 4. Basic Usage

```python
from trading_rl_agent.data.alpaca_integration import AlpacaIntegration, AlpacaConfig
from trading_rl_agent.configs.alpaca_config import create_alpaca_config_from_env

# Load configuration from environment
config = create_alpaca_config_from_env()

# Initialize integration
alpaca = AlpacaIntegration(config)

# Get account information
account_info = alpaca.get_account_info()
print(f"Account: {account_info['id']}, Cash: ${account_info['cash']:,.2f}")

# Get real-time quotes
quotes = alpaca.get_real_time_quotes(["AAPL", "MSFT"])
for symbol, quote in quotes.items():
    print(f"{symbol}: ${quote['bid_price']:.2f} - ${quote['ask_price']:.2f}")
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Your Alpaca API key | Required |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key | Required |
| `ALPACA_BASE_URL` | Trading API base URL | `https://paper-api.alpaca.markets` |
| `ALPACA_DATA_URL` | Data API base URL | `https://data.alpaca.markets` |
| `ALPACA_PAPER_TRADING` | Enable paper trading | `true` |
| `ALPACA_USE_V2` | Use Alpaca V2 SDK | `true` |
| `ALPACA_MAX_RETRIES` | Maximum retry attempts | `3` |
| `ALPACA_RETRY_DELAY` | Retry delay in seconds | `1.0` |
| `ALPACA_WEBSOCKET_TIMEOUT` | WebSocket timeout | `30` |
| `ALPACA_ORDER_TIMEOUT` | Order timeout | `60` |
| `ALPACA_CACHE_DIR` | Cache directory | `data/alpaca_cache` |
| `ALPACA_DATA_FEED` | Data feed (iex/sip) | `iex` |
| `ALPACA_EXTENDED_HOURS` | Allow extended hours | `false` |
| `ALPACA_MAX_POSITION_SIZE` | Max position size | `10000.0` |
| `ALPACA_MAX_DAILY_TRADES` | Max daily trades | `100` |
| `ALPACA_LOG_LEVEL` | Logging level | `INFO` |

### Configuration File

Create a YAML configuration file:

```yaml
# alpaca_config.yaml
api_key: "your_api_key_here"
secret_key: "your_secret_key_here"
paper_trading: true
data_feed: "iex"
max_position_size: 10000.0
max_daily_trades: 100
```

Load from file:

```python
from trading_rl_agent.configs.alpaca_config import get_alpaca_config

config = get_alpaca_config("alpaca_config.yaml")
alpaca = AlpacaIntegration(config)
```

## Real-time Data Streaming

### Basic Streaming

```python
def data_callback(data_type: str, data):
    if data_type == "bar":
        print(f"Bar: {data.symbol} - Close: ${data.close:.2f}")
    elif data_type == "trade":
        print(f"Trade: {data['symbol']} - Price: ${data['price']:.2f}")

# Add callback
alpaca.add_data_callback(data_callback)

# Start streaming
alpaca.start_data_stream(["AAPL", "MSFT", "GOOGL"])

# Stop streaming when done
alpaca.stop_data_stream()
```

### Context Manager Usage

```python
with AlpacaIntegration(config) as alpaca:
    alpaca.add_data_callback(data_callback)
    alpaca.start_data_stream(["AAPL"])
    # Streaming automatically stops when exiting context
```

## Paper Trading

### Place Orders

```python
from trading_rl_agent.data.alpaca_integration import OrderRequest, OrderType, OrderSide

# Market order
order_request = OrderRequest(
    symbol="AAPL",
    qty=10.0,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    time_in_force="day"
)

result = alpaca.place_order(order_request)
print(f"Order executed: {result['order_id']}")

# Limit order
limit_order = OrderRequest(
    symbol="AAPL",
    qty=5.0,
    side=OrderSide.SELL,
    order_type=OrderType.LIMIT,
    limit_price=150.00,
    time_in_force="day"
)

result = alpaca.place_order(limit_order)
```

### Order Types

- **Market Orders**: Execute immediately at current market price
- **Limit Orders**: Execute only at specified price or better
- **Stop Orders**: Execute when price reaches stop level
- **Stop Limit Orders**: Combination of stop and limit orders

### Order Management

```python
# Get order history
orders = alpaca.get_order_history(limit=50)

# Cancel specific order
alpaca.cancel_order("order_id_here")

# Cancel all open orders
canceled_orders = alpaca.cancel_all_orders()
```

## Portfolio Monitoring

### Portfolio Overview

```python
# Get portfolio value and metrics
portfolio = alpaca.get_portfolio_value()
print(f"Total Equity: ${portfolio['total_equity']:,.2f}")
print(f"Cash: ${portfolio['cash']:,.2f}")
print(f"Unrealized P&L: ${portfolio['total_unrealized_pl']:,.2f}")

# Get current positions
positions = alpaca.get_positions()
for position in positions:
    print(f"{position.symbol}: {position.qty} shares, P&L: ${position.unrealized_pl:,.2f}")
```

### Real-time Portfolio Updates

```python
def portfolio_callback(portfolio_data):
    print(f"Portfolio Update: ${portfolio_data['total_equity']:,.2f}")

alpaca.add_portfolio_callback(portfolio_callback)
```

## Historical Data

### Get Historical Data

```python
from datetime import datetime, timedelta

# Get last 30 days of daily data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

data = alpaca.get_historical_data(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1Day",
    adjustment="raw"
)

print(f"Retrieved {len(data)} data points")
```

### Available Timeframes

- `1Min`: 1-minute bars
- `5Min`: 5-minute bars
- `15Min`: 15-minute bars
- `30Min`: 30-minute bars
- `1Hour`: 1-hour bars
- `1Day`: Daily bars

### Price Adjustments

- `raw`: No adjustments
- `split`: Split adjustments only
- `dividend`: Dividend adjustments only
- `all`: Both split and dividend adjustments

## Error Handling

### Custom Exceptions

```python
from trading_rl_agent.data.alpaca_integration import (
    AlpacaError,
    AlpacaConnectionError,
    AlpacaOrderError,
    AlpacaDataError
)

try:
    alpaca.place_order(order_request)
except AlpacaOrderError as e:
    print(f"Order failed: {e}")
except AlpacaConnectionError as e:
    print(f"Connection error: {e}")
except AlpacaDataError as e:
    print(f"Data error: {e}")
```

### Retry Logic

The integration includes automatic retry logic for failed operations:

```python
# Configure retry behavior
config = AlpacaConfig(
    api_key="your_key",
    secret_key="your_secret",
    max_retries=5,
    retry_delay=2.0
)
```

## Demo Script

Run the comprehensive demo to see all features in action:

```bash
# Run full demo
python examples/alpaca_integration_demo.py

# Run with custom symbols
python examples/alpaca_integration_demo.py --symbols AAPL MSFT GOOGL

# Run only streaming demo
python examples/alpaca_integration_demo.py --stream-only --duration 60

# Use configuration file
python examples/alpaca_integration_demo.py --config alpaca_config.yaml
```

## Integration with Existing Code

### Use with ProfessionalDataProvider

The AlpacaIntegration class can be used alongside the existing `ProfessionalDataProvider`:

```python
from trading_rl_agent.data.professional_feeds import ProfessionalDataProvider
from trading_rl_agent.data.alpaca_integration import AlpacaIntegration

# Use existing provider for historical data
provider = ProfessionalDataProvider(provider="alpaca")

# Use new integration for real-time features
alpaca = AlpacaIntegration(config)
alpaca.start_data_stream(["AAPL", "MSFT"])
```

### Use with Portfolio Manager

```python
from trading_rl_agent.portfolio.manager import PortfolioManager

# Initialize portfolio manager with Alpaca integration
portfolio_manager = PortfolioManager()
alpaca = AlpacaIntegration(config)

# Get real-time portfolio data
portfolio_data = alpaca.get_portfolio_value()
portfolio_manager.update_portfolio(portfolio_data)
```

## Best Practices

### Security

1. **Never commit API keys** to version control
2. Use environment variables or secure configuration files
3. Enable paper trading for development and testing
4. Use separate API keys for paper and live trading

### Performance

1. **Cache historical data** to avoid repeated API calls
2. Use appropriate timeframes for your use case
3. Implement rate limiting for high-frequency operations
4. Monitor API usage and stay within limits

### Error Handling

1. Always implement proper error handling
2. Use retry logic for transient failures
3. Log errors for debugging
4. Implement circuit breakers for critical failures

### Risk Management

1. Set position size limits
2. Implement daily trade limits
3. Monitor portfolio exposure
4. Use stop-loss orders for risk control

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify API credentials
   - Check network connectivity
   - Ensure correct base URL

2. **Order Failures**
   - Verify symbol is tradable
   - Check account status and buying power
   - Ensure order parameters are valid

3. **Data Streaming Issues**
   - Check WebSocket connection
   - Verify data feed subscription
   - Monitor for rate limits

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("trading_rl_agent.data.alpaca_integration").setLevel(logging.DEBUG)
```

### API Limits

- **Paper Trading**: 200 requests/minute
- **Live Trading**: 200 requests/minute
- **Data API**: Varies by subscription
- **WebSocket**: Unlimited connections

## Support

For issues and questions:

1. Check the [Alpaca API documentation](https://alpaca.markets/docs/)
2. Review error logs and debug output
3. Test with the demo script
4. Verify configuration and credentials

## License

This integration is part of the trading_rl_agent project and follows the same license terms.