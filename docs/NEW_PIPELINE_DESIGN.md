# Auto-Processing Data Pipeline - New Design

## Overview

The trading RL agent data pipeline has been redesigned to eliminate the `data/raw` folder dependency and provide seamless auto-processing. The new approach downloads, processes, and standardizes data in one flow, creating ready-to-use datasets directly in organized output directories.

## Key Improvements

### ❌ Old Approach (Deprecated)

```bash
# Multi-step process with raw folder storage
trade-agent data pipeline --download --symbols 'AAPL,GOOGL'  # → data/raw/
trade-agent data pipeline --sentiment                        # → data/sentiment/
trade-agent data pipeline --process                          # → data/processed/
```

**Problems:**

- Raw files persist, taking up storage space
- Multi-step workflow, complex to manage
- Manual coordination between steps required
- Raw folder becomes cluttered over time

### ✅ New Approach (Recommended)

```bash
# Single command, complete auto-processing
trade-agent data pipeline --run --symbols 'AAPL,GOOGL'      # → data/dataset_TIMESTAMP/
```

**Benefits:**

- No raw folder needed - data processed in memory
- One command creates complete, ready-to-use datasets
- Unique dataset directories for each run
- Multiple export formats (CSV, Parquet, Feather)
- Comprehensive metadata and performance tracking

## New Directory Structure

```
data/
├── dataset_20250123_143025/          # Auto-generated unique dataset
│   ├── dataset.csv                   # Main dataset (CSV format)
│   ├── dataset.parquet              # Compressed format for large datasets
│   ├── dataset.feather              # Fast I/O format
│   ├── data_standardizer.pkl        # Reusable standardizer
│   └── metadata.json               # Complete dataset metadata
├── dataset_20250123_151200/          # Another dataset run
│   └── ...
└── custom_tech_analysis/             # Named dataset
    └── ...
```

## Usage Examples

### Basic Auto-Processing

```bash
# Quick start with auto-selected symbols
trade-agent data pipeline --run

# Custom symbols
trade-agent data pipeline --run --symbols "AAPL,GOOGL,MSFT,TSLA"
```

### Advanced Processing Options

```bash
# Technical analysis focused dataset
trade-agent data pipeline --run \
  --features technical \
  --dataset-name "tech_analysis_v1" \
  --method robust

# Full feature set with sentiment analysis
trade-agent data pipeline --run \
  --features full \
  --sentiment-days 14 \
  --sentiment-sources "news,social"

# High-performance batch processing
trade-agent data pipeline --run \
  --max-symbols 100 \
  --workers 16 \
  --no-sentiment \
  --formats "csv,parquet"
```

### Production Deployment

```bash
# Scheduled dataset generation
trade-agent data pipeline --run \
  --dataset-name "daily_$(date +%Y%m%d)" \
  --max-symbols 200 \
  --workers 32 \
  --cache \
  --formats "parquet,feather"
```

## Command Options

| Option                       | Description                                       | Default                  |
| ---------------------------- | ------------------------------------------------- | ------------------------ |
| `--run`                      | Execute complete auto-processing pipeline         | Required                 |
| `--symbols`                  | Comma-separated stock symbols                     | Auto-selected            |
| `--dataset-name`             | Custom dataset name                               | Auto-generated timestamp |
| `--method`                   | Standardization method (robust, standard, minmax) | `robust`                 |
| `--features`                 | Feature set (basic, technical, full, custom)      | `full`                   |
| `--sentiment/--no-sentiment` | Include sentiment analysis                        | `true`                   |
| `--sentiment-days`           | Days back for sentiment analysis                  | `7`                      |
| `--sentiment-sources`        | Sentiment sources (news,social,scrape)            | `news,social`            |
| `--workers`                  | Number of parallel workers                        | `8`                      |
| `--cache/--no-cache`         | Use intelligent caching                           | `true`                   |
| `--formats`                  | Export formats (csv,parquet,feather)              | `csv`                    |
| `--max-symbols`              | Maximum auto-selected symbols                     | `50`                     |

## Feature Sets

### Basic Features

- Core OHLCV data
- Simple moving averages (5, 10, 20)
- Basic volume indicators

### Technical Features

- All basic features plus:
- RSI, MACD, Bollinger Bands
- ATR, Stochastic oscillators
- Candlestick patterns
- Volume-price indicators

### Full Features

- All technical features plus:
- Advanced momentum indicators
- Market microstructure features
- Multi-timeframe analysis
- Sentiment integration (if enabled)

### Custom Features

- User-defined feature sets via config file
- Configurable indicators and parameters
- Custom transformations and engineering

## Dataset Metadata

Each dataset includes comprehensive metadata in `metadata.json`:

```json
{
  "dataset_name": "dataset_20250123_143025",
  "created_at": "2025-01-23T14:30:25.123456",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "start_date": "2024-01-01",
  "end_date": "2025-01-23",
  "processing_method": "robust",
  "feature_set": "technical",
  "include_sentiment": true,
  "sentiment_days": 7,
  "sentiment_sources": ["news", "social"],
  "row_count": 15420,
  "column_count": 89,
  "export_formats": ["csv", "parquet"],
  "file_paths": {
    "csv": "data/dataset_20250123_143025/dataset.csv",
    "parquet": "data/dataset_20250123_143025/dataset.parquet"
  }
}
```

## Performance Optimizations

### Intelligent Caching

- Automatic caching of download and sentiment data
- Cache invalidation based on data freshness
- Significant speed improvements for repeated runs

### Parallel Processing

- Multi-threaded symbol downloading
- Parallel sentiment analysis
- Configurable worker pools

### Memory Efficiency

- Stream processing without raw file storage
- Efficient data structures and algorithms
- Automatic garbage collection

## Migration from Old Pipeline

### Step 1: Update Commands

Replace multi-step commands with single `--run` command:

```bash
# Old
trade-agent data pipeline --download --symbols 'AAPL'
trade-agent data pipeline --sentiment
trade-agent data pipeline --process

# New
trade-agent data pipeline --run --symbols 'AAPL'
```

### Step 2: Update Scripts

If you have automation scripts, update them:

```python
# Old approach
from trade_agent.data.prepare import prepare_data
prepare_data(input_path=Path("data/raw"), output_dir=Path("data/processed"))

# New approach
from trade_agent.data.prepare import create_auto_processed_dataset
result = create_auto_processed_dataset(
    symbols=["AAPL", "GOOGL"],
    start_date="2024-01-01",
    end_date="2025-01-23",
    dataset_name="my_dataset"
)
```

### Step 3: Clean Up Old Data

The raw folder is no longer needed:

```bash
# Review and backup if needed
ls -la data/raw/

# Remove when ready
rm -rf data/raw/
```

## Best Practices

### Dataset Naming

- Use descriptive names for important datasets
- Include version numbers for iterative development
- Use timestamps for automated/scheduled runs

### Feature Selection

- Start with `technical` features for most trading applications
- Use `full` features for comprehensive analysis
- Use `basic` features for rapid prototyping

### Resource Management

- Adjust `--workers` based on system capabilities
- Use `--cache` for development, disable for production if needed
- Monitor memory usage with large symbol lists

### Production Deployment

- Use `parquet` format for large datasets
- Implement dataset retention policies
- Monitor storage usage and performance metrics

## Troubleshooting

### Common Issues

1. **Memory errors with large symbol lists**
   - Reduce `--max-symbols` or `--workers`
   - Use `--no-sentiment` to reduce memory usage

2. **Slow performance**
   - Enable `--cache` for repeated runs
   - Increase `--workers` (up to CPU cores)
   - Use `--no-sentiment` for faster processing

3. **Sentiment analysis failures**
   - Check network connectivity
   - Use `--no-sentiment` as fallback
   - Verify API keys if using external services

### Legacy Support

The old pipeline commands are still supported but deprecated:

```bash
# Still works but shows deprecation warning
trade-agent data pipeline --download --symbols 'AAPL'
```

Use `--run` for all new implementations.
