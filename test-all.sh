#!/bin/sh
# All tests including Ray integration
echo "🎯 Running all tests..."
python3 -m pytest tests/ -v --tb=short
