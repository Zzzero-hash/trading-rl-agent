#!/bin/sh
# Fast tests without heavy dependencies
echo "🏃 Running fast tests..."
/opt/conda/bin/python3.12 -m pytest tests/test_data_pipeline.py tests/test_trading_env.py tests/test_features.py -v
