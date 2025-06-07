#!/bin/sh
# ML tests requiring PyTorch
echo "🤖 Running ML tests..."
/opt/conda/bin/python3.12 -m pytest tests/test_cnn_lstm.py tests/test_sac_agent.py tests/test_td3_agent.py -v
