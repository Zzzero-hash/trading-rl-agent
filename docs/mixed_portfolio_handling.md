# Mixed Portfolio Handling: Crypto vs Traditional Markets

## Overview

This document explains how the trading system handles the critical challenge of mixed portfolios containing both crypto assets (24/7 trading) and traditional assets (specific trading hours and holidays).

## The Problem

### Traditional Markets

- **Trading Hours**: 9:30 AM - 4:00 PM ET (NYSE/NASDAQ)
- **Days**: Monday through Friday
- **Holidays**: New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas Day
- **Extended Hours**: 4:00 AM - 8:00 PM ET (limited liquidity)

### Crypto Markets

- **Trading Hours**: 24/7, 365 days per year
- **No Holidays**: Continuous trading
- **Global**: Multiple timezones, no single market close

### The Challenge

When combining crypto and traditional assets in a portfolio, we face a fundamental data structure problem:

1. **Different Data Frequencies**: Crypto has data for weekends/holidays, traditional assets don't
2. **Timestamp Misalignment**: Same timestamp may have crypto data but no traditional data
3. **Feature Engineering Issues**: Technical indicators and features become inconsistent
4. **Model Training Problems**: Inconsistent input dimensions and temporal relationships

## Solution: Market Calendar System

### Core Components

#### 1. Market Calendar (`src/trade_agent/data/market_calendar.py`)

The `TradingCalendar` class provides comprehensive market schedule management:

```python
from trade_agent.data.market_calendar import get_trading_calendar

calendar = get_trading_calendar()

# Check if market is open
is_open = calendar.is_market_open("AAPL", datetime.now())  # False on weekends/holidays
is_open = calendar.is_market_open("BTC-USD", datetime.now())  # Always True

# Get next market open
next_open = calendar.get_next_market_open("AAPL", datetime.now())
```

#### 2. Asset Classification

Automatic classification of assets by market type:

```python
from trade_agent.data.market_calendar import classify_portfolio_assets

symbols = ["AAPL", "BTC-USD", "SPY", "ETH-USD"]
classification = classify_portfolio_assets(symbols)

# Result:
# {
#     "crypto": ["BTC-USD", "ETH-USD"],
#     "traditional": ["AAPL", "SPY"],
#     "mixed": True
# }
```

#### 3. Timestamp Alignment

The key function that solves the mixed portfolio problem:

```python
# Align timestamps for mixed portfolios with specified strategy
aligned_data = calendar.align_data_timestamps(data, symbols, alignment_strategy="last_known_value")
```

### Alignment Strategies

The system supports multiple strategies for aligning crypto data to traditional market hours:

#### 1. Last Known Value (Recommended)

- **Strategy**: Uses last known crypto values until the next actual crypto data point
- **Pros**: Preserves OHLC integrity, avoids misleading forward-filling, sets volume to zero during market closure
- **Best for**: Model training, backtesting, risk management, portfolio analysis

#### 2. Forward Fill

- **Strategy**: Simple forward-filling of all OHLCV data
- **Pros**: Simple implementation, preserves all original data points
- **Cons**: Misleading volume data, may create artificial price continuity
- **Best for**: Quick prototyping, simple analysis

#### 3. Interpolate

- **Strategy**: Interpolates between crypto data points
- **Pros**: Smooth price transitions, may capture intraday movements
- **Cons**: Creates artificial price points, may not reflect actual trading
- **Best for**: High-frequency analysis, smooth visualizations

#### For Traditional Assets

- Filter data to market hours only
- Remove weekends and holidays
- Ensure consistent trading day alignment

#### Example Alignment Process

**Before Alignment:**

```
Timestamp           | AAPL | BTC-USD
2024-01-01 09:30   | 150  | 45000
2024-01-01 10:00   | 151  | 45100
2024-01-01 16:00   | 152  | 45200
2024-01-01 20:00   | NaN  | 45300  # AAPL closed, BTC still trading
2024-01-02 09:30   | 153  | 45400
```

**After Alignment:**

```
Timestamp           | AAPL | BTC-USD | Data_Source
2024-01-01 09:30   | 150  | 45000   | traditional_market
2024-01-01 10:00   | 151  | 45100   | traditional_market
2024-01-01 16:00   | 152  | 45200   | traditional_market
2024-01-02 09:30   | 153  | 45400   | traditional_market
```

## Integration Points

### 1. Data Pipeline Integration

The market calendar is automatically integrated into the main data pipeline with `last_known_value` as the default strategy:

```python
# In src/trade_agent/data/pipeline.py
data = provider.get_market_data(
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    include_features=True,
    align_mixed_portfolio=True,  # Automatic alignment (default: True)
    alignment_strategy="last_known_value",  # Default strategy
)
```

**Automatic Behavior:**

- Crypto detection is automatic
- `last_known_value` strategy is applied by default
- No manual configuration required
- Seamless integration in data pipeline

### 2. Configuration Integration

Added configuration options in `src/trade_agent/core/unified_config.py` with `last_known_value` as the default:

```yaml
data:
  # Market calendar and mixed portfolio handling
  align_mixed_portfolios: true
  alignment_strategy: "last_known_value" # Default strategy
  market_timezone: "America/New_York"
  include_extended_hours: false
```

**Default Configuration:**

- `align_mixed_portfolios: true` - Automatic alignment enabled
- `alignment_strategy: "last_known_value"` - Recommended strategy by default
- Automatic crypto detection and classification

### 3. Professional Feeds Integration

The `ProfessionalDataProvider` automatically detects crypto and applies the `last_known_value` strategy by default:

```python
# Automatic detection and alignment with default strategy
provider = ProfessionalDataProvider("yahoo")
data = provider.get_market_data(
    symbols=["AAPL", "BTC-USD", "SPY"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    align_mixed_portfolio=True,  # Default: True
    alignment_strategy="last_known_value",  # Default: last_known_value
)
```

**Automatic Features:**

- Crypto detection is automatic
- `last_known_value` strategy applied by default
- Volume set to zero during market closure
- Metadata tracking for alignment methods used

## Usage Examples

### Basic Usage

```python
from trade_agent.data.professional_feeds import ProfessionalDataProvider

# Mixed portfolio with automatic alignment (default behavior)
provider = ProfessionalDataProvider("yahoo")
data = provider.get_market_data(
    symbols=["AAPL", "BTC-USD", "SPY", "ETH-USD"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    # align_mixed_portfolio=True,  # Default: True
    # alignment_strategy="last_known_value",  # Default: last_known_value
)
```

**No Configuration Required:**

- Crypto detection is automatic
- `last_known_value` strategy is the default
- Alignment is applied automatically for mixed portfolios

### Advanced Usage

```python
from trade_agent.data.market_calendar import get_trading_calendar, classify_portfolio_assets

# Manual portfolio classification
symbols = ["AAPL", "BTC-USD", "SPY", "ETH-USD"]
classification = classify_portfolio_assets(symbols)

if classification["mixed"]:
    print(f"Mixed portfolio detected: {len(classification['crypto'])} crypto, {len(classification['traditional'])} traditional")

    # Manual alignment
    calendar = get_trading_calendar()
    aligned_data = calendar.align_data_timestamps(data, symbols)
```

### Configuration

```yaml
# configs/unified_config.yaml
data:
  align_mixed_portfolios: true
  market_timezone: "America/New_York"
  include_extended_hours: false

  symbols:
    - AAPL
    - BTC-USD
    - SPY
    - ETH-USD
```

## Benefits

### 1. Consistent Data Structure

- All assets have the same timestamp frequency
- No missing data points for traditional assets
- Consistent feature engineering

### 2. Improved Model Training

- Consistent input dimensions
- Proper temporal relationships
- No data leakage from future crypto data

### 3. Accurate Backtesting

- Realistic market conditions
- Proper holiday handling
- Consistent portfolio rebalancing

### 4. Live Trading Compatibility

- Same data structure for training and inference
- Proper market hours enforcement
- Risk management alignment

## Strategy Selection Guide

### When to Use Each Strategy

#### Last Known Value (Default)

Use this strategy for:

- **Model Training**: Preserves data integrity and prevents data leakage
- **Backtesting**: Most realistic representation of actual trading conditions
- **Live Trading**: Consistent with training data and risk management
- **Portfolio Analysis**: Accurate representation of mixed portfolio behavior

```python
# Recommended for most use cases
data = provider.get_market_data(
    symbols=symbols,
    alignment_strategy="last_known_value"  # Default and recommended
)
```

#### Forward Fill

Use this strategy for:

- **Quick Prototyping**: Fastest to implement for initial testing
- **Simple Analysis**: When volume accuracy isn't critical
- **Legacy Systems**: When you need to maintain existing behavior

```python
# For quick prototyping
data = provider.get_market_data(
    symbols=symbols,
    alignment_strategy="forward_fill"
)
```

#### Interpolate

Use this strategy for:

- **High-Frequency Analysis**: When you need smooth data transitions
- **Research Applications**: When interpolation makes mathematical sense
- **Visualizations**: When you need smooth price charts

```python
# For research and high-frequency analysis
data = provider.get_market_data(
    symbols=symbols,
    alignment_strategy="interpolate"
)
```

### Strategy Comparison Example

```python
from trade_agent.data.market_calendar import get_trading_calendar

calendar = get_trading_calendar()

# Compare strategies
strategies = ["last_known_value", "forward_fill", "interpolate"]
results = {}

for strategy in strategies:
    aligned_data = calendar.align_data_timestamps(data, symbols, strategy)
    results[strategy] = aligned_data

    # Analyze results
    zero_volume_count = len(aligned_data[aligned_data["volume"] == 0])
    print(f"{strategy}: {len(aligned_data)} rows, {zero_volume_count} zero volume")
```

## Best Practices

### 1. Always Enable Alignment for Mixed Portfolios

```python
# Recommended
data = provider.get_market_data(
    symbols=symbols,
    align_mixed_portfolio=True,  # Always True for mixed portfolios
)
```

### 2. Use Appropriate Timeframes

```python
# For mixed portfolios, use daily data for consistency
data = provider.get_market_data(
    symbols=symbols,
    timeframe="1Day",  # Daily data works best for mixed portfolios
)
```

### 3. Monitor Alignment Results

```python
# Check alignment results
if not data.empty:
    crypto_count = len(data[data["symbol"].isin(crypto_symbols)])
    traditional_count = len(data[data["symbol"].isin(traditional_symbols)])
    print(f"Aligned data: {crypto_count} crypto, {traditional_count} traditional")
```

### 4. Handle Edge Cases

```python
# Check for data quality issues
missing_data = data[data["close"].isna()]
if not missing_data.empty:
    print(f"Warning: {len(missing_data)} rows with missing data")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Ensure market_calendar is properly imported
from trade_agent.data.market_calendar import get_trading_calendar
```

#### 2. Timezone Issues

```python
# Use consistent timezone
calendar = get_trading_calendar(timezone="America/New_York")
```

#### 3. Data Quality Issues

```python
# Check for missing data after alignment
if data["close"].isna().any():
    print("Warning: Missing data detected after alignment")
```

### Debug Mode

Enable debug logging to see alignment details:

```python
import logging
logging.getLogger("trade_agent.data.market_calendar").setLevel(logging.DEBUG)
```

## Future Enhancements

### 1. Extended Hours Support

- Include pre-market and after-hours data
- Configurable extended hours windows

### 2. International Markets

- Support for European and Asian markets
- Multi-timezone holiday calendars

### 3. Real-time Alignment

- Live data alignment for real-time trading
- Streaming data support

### 4. Advanced Alignment Strategies

- Volume-weighted alignment
- VWAP-based alignment
- Custom alignment rules

## Conclusion

The market calendar system provides a robust solution for handling mixed crypto/traditional portfolios. By automatically aligning timestamps and ensuring consistent data structures, it enables accurate model training, backtesting, and live trading across different asset types.

The system is designed to be:

- **Automatic**: No manual intervention required
- **Configurable**: Flexible settings for different use cases
- **Robust**: Handles edge cases and data quality issues
- **Extensible**: Easy to add new asset types and markets

This foundation enables the trading system to handle the complexity of modern multi-asset portfolios while maintaining data integrity and model consistency.
