#!/bin/sh
# All tests including Ray integration
echo "🎯 Running all tests..."
/opt/conda/bin/python3.12 -m pytest tests/ -v --tb=short
